# encoding=utf-8
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import BaseFairseqModel
from torch.nn.init import orthogonal_

from dep_nat.util import get_dep_mat, DepHeadTree
from nat_base.layer import build_relative_embeddings, BlockedDecoderLayer
from nat_base.util import new_arange, get_base_mask, set_key_value, get_key_value


def cos_loss(hidden1, hidden2):
    r = -F.cosine_similarity(hidden1, hidden2, dim=-1)
    return r.mean().exp()


class DepEncoder(nn.Module):
    """ 对输入进行编码 使用自注意力 """

    def __init__(self, args):
        super().__init__()
        num_layers = 4
        rel_keys = build_relative_embeddings(args)
        rel_vals = build_relative_embeddings(args)

        self.layers = nn.ModuleList(
            [BlockedDecoderLayer(args, no_encoder_attn=False, relative_keys=rel_keys, relative_vals=rel_vals
                                 ) for _ in range(num_layers)]
        )

    def forward(self, encoder_out, hidden_state, decoder_padding_mask):
        for layer in self.layers:
            hidden_state, layer_attn, _ = layer(
                hidden_state,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
        return hidden_state.transpose(0, 1)


class BiaffineAttentionDependency(nn.Module):
    def __init__(self, input_dim, head_tree=None, dropout=0.1):
        super().__init__()

        mlp_input_dim = input_dim
        self.mlp_dim = 500
        self.dropout = dropout
        self.arc_head_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout))

        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout))

        self.W_arc = nn.Parameter(orthogonal_(
            torch.empty(self.mlp_dim + 1, self.mlp_dim)
        ), requires_grad=True)

        self.head_tree = head_tree

        self.dropout = nn.Dropout(self.dropout)

    def get_reference(self, sample_ids, **kwargs):
        # -1表示不用的节点，0表示根节点。
        tree = self.head_tree.get_sentences(sample_ids, training=self.training)
        size = max(len(v) for v in tree)
        head_label = numpy.empty(shape=(len(tree), size + 2))
        head_label.fill(-1)
        for i, value in enumerate(tree):
            head_label[i][1:len(value) + 1] = value

        head_label = torch.from_numpy(head_label).long()

        return head_label

    def compute_loss(self, outputs, targets):
        logits = F.log_softmax(outputs, dim=-1)
        losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
        loss = losses.mean()

        return loss

    def forward_classifier(self, hidden_state, return_hidden=False):
        h_arc_dep = self.arc_dep_mlp(hidden_state)  # batch * max_trg * mlp_dim
        h_arc_head = self.arc_head_mlp(hidden_state)  # batch * max_trg * mlp_dim

        batch_size, max_trg_len, decoder_dim = h_arc_head.size()

        arc_dep = torch.cat((h_arc_dep, torch.ones(batch_size, max_trg_len, 1).to(h_arc_dep.device)),
                            dim=-1).to(h_arc_head.dtype)

        head_dep_result = arc_dep.matmul(self.W_arc).matmul(h_arc_head.transpose(1, 2))

        return head_dep_result


class DepHeadClassifier(BaseFairseqModel):

    def __init__(self, args, relative_dep_mat=None, dep_file="", **kwargs):

        super().__init__()
        self.args = args

        self.relative_dep_mat = relative_dep_mat
        self.encoder = DepEncoder(args)
        self.dropout = nn.Dropout(args.dropout)

        self.mlp_input_dim = args.decoder_embed_dim
        self.padding_idx = -1
        self.dep_file = dep_file
        self.head_tree = DepHeadTree(valid_subset=self.args.valid_subset, dep_file=dep_file)

        self.biaffine_parser = BiaffineAttentionDependency(input_dim=self.mlp_input_dim, head_tree=self.head_tree,
                                                           dropout=args.dropout)

        self.weight = getattr(args, "weight", 1)
        self.noglancing = getattr(args, "noglancing", False)

    def _forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out, **kwargs):
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.encoder(encoder_out, hidden_state, decoder_padding_mask)

        hidden_state2 = self.dropout(hidden_state)

        output = self.biaffine_parser.forward_classifier(hidden_state2)  # [b, tgt_len, tgt_len]
        return output, hidden_state

    def get_random_mask_output(self, mask_length=None, target_token=None, hidden_state=None, reference_embedding=None,
                               reference_mask=None):
        # mask_length大，说明模型性能不行，所以多提供reference  ==> mask_length代表的是reference的数目
        hidden_state = hidden_state.transpose(0, 1)
        reference_embedding = reference_embedding.transpose(0, 1)

        target_score = target_token.clone().float().uniform_()
        target_score.masked_fill_(~reference_mask, 2.0)

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]
        non_mask = ~mask
        full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)

        full_embedding = torch.cat((hidden_state.unsqueeze(-1), reference_embedding.unsqueeze(-1)), dim=-1)
        output_embedding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)

        return ~mask, output_embedding.transpose(0, 1)

    def get_mask_num(self, label, reference, reference_mask, update_nums):
        ratio = 0.5 - 0.2 / 300000 * update_nums
        diff = ((label != reference) & reference_mask).sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def get_noglancing_mask_num(self, reference, reference_mask, update_nums):
        ratio = (0.5 - 0.4 / 300000 * update_nums) * 0.5
        diff = (reference_mask).sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def forward_classifier(self, hidden_state=None, reference=None, target_tokens=None, ref_embedding=None,
                           **kwargs):

        if self.training:
            with torch.no_grad():
                reference_mask = reference != self.padding_idx
                if self.noglancing:
                    mask_length = self.get_noglancing_mask_num(reference, reference_mask,
                                                               kwargs.get('update_num', 300000))
                else:
                    score, _ = self._forward_classifier(hidden_state=hidden_state, **kwargs)
                    score = score.detach()
                    label = self.get_head(score)
                    mask_length = self.get_mask_num(label, reference, reference_mask, kwargs.get('update_num', 300000))
            _, hidden_state = self.get_random_mask_output(mask_length, target_tokens, hidden_state,
                                                          ref_embedding, reference_mask=get_base_mask(target_tokens))
            score, _ = self._forward_classifier(hidden_state=hidden_state, **kwargs)
        else:
            score, _ = self._forward_classifier(hidden_state=hidden_state, **kwargs)
        return score, hidden_state

    def get_head(self, score):
        return score.argmax(-1)

    def get_label(self, score, target_tokens=None, **kwargs):
        all_head = self.get_head(score.detach())

        mask = get_base_mask(target_tokens)
        dep_mat = get_dep_mat(all_head, mask)

        return dep_mat

    def inference_accuracy(self, get_mat=False, compute_accuracy=True, **kwargs):
        dep_mat = None
        score, predict, head_ref, mix_hidden = self.predict(**kwargs)
        loss = self.compute_loss(predict, head_ref)
        return loss, dep_mat, mix_hidden

    def compute_accuracy(self, predict, target):
        all = len(predict)
        head = self.get_head(predict)
        correct = (target == head).sum().item()
        return all, correct

    def compute_loss(self, outputs, targets, teacher_score=None, other=None, **kwargs):
        # 计算损失的肯定是依存树预测损失
        b_loss = self.biaffine_parser.compute_loss(outputs, targets) * self.weight
        loss = {"dep_classifier": {"loss": b_loss}}
        return loss

    def get_reference(self, sample_ids, **kwargs):
        head_ref = self.biaffine_parser.get_reference(sample_ids, **kwargs)
        return head_ref

    def predict(self, sample=None, target_tokens=None, **kwargs):

        decoder_padding_mask = target_tokens.eq(1)

        sample_ids = sample['id'].cpu().tolist()
        reference = self.get_reference(sample_ids, target_tokens=target_tokens).to(target_tokens.device)
        score, mix_hidden = self.forward_classifier(target_tokens=target_tokens, sample=sample,
                                                    decoder_padding_mask=decoder_padding_mask,
                                                    reference=reference,
                                                    **kwargs)

        reference_mask = reference != self.padding_idx

        predict = score[reference_mask]
        dependency_mat = reference[reference_mask]

        return score, predict, dependency_mat, mix_hidden

    def inference(self, hidden_state, position_embedding, target_tokens, **kwargs):

        # if kwargs.get('use_oracle_mat', False):
        #     sample = kwargs['sample']
        #     sample_ids = sample['id'].cpu().tolist()
        #     reference = sample["target"]
        #     dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
        #                                                               contain_eos=True)
        #
        #     return dependency_mat

        score, _ = self._forward_classifier(hidden_state, position_embedding,
                                            target_tokens=target_tokens,
                                            decoder_padding_mask=target_tokens.eq(1),
                                            **kwargs)
        _label = self.get_label(score, target_tokens=target_tokens, **kwargs)

        # count the head and save
        # save_head = self.get_head(score).cpu().tolist()
        # with open("/home/data_ti5_c/wangdq/code/dep_nat/_joint_head.log", 'a') as f:
        #     mask = get_base_mask(target_tokens).long().sum(-1).cpu().tolist()
        #     for l, h in zip(mask, save_head):
        #         f.write(",".join([str(hh) for hh in h[1: l + 1]]) + '\n')

        if kwargs.get("eval_accuracy", False):
            sample = kwargs['sample']
            sample_ids = sample['id'].cpu().tolist()
            head_ref = self.get_reference(sample_ids, target_tokens=target_tokens).to(score.device)

            # ref = head_ref.cpu().tolist()
            # save_head = self.get_head(score).cpu().tolist()
            # file1 = "/home/wangdq/code/dep_nat/joint_ref.log"
            # file2 = "/home/wangdq/code/dep_nat/joint.log"
            # with open(file1, 'a') as f1, open(file2, 'a') as f2:
            #     mask = get_base_mask(target_tokens).long().sum(-1).cpu().tolist()
            #     for l, h, r in zip(mask, save_head, ref):
            #         f1.write(" ".join([str(hh) for hh in r[1: l + 1]]) + '\n')
            #         f2.write(' '.join([str(hh) for hh in h[1: l + 1]]) + '\n')

            head_mask = head_ref != self.padding_idx

            predict = score[head_mask]
            target = head_ref[head_mask]
            predict = self.get_head(predict)
            all = len(predict)
            correct = (target == predict).sum().item()

            set_key_value("all", all)
            set_key_value("correct", correct)

            reference = sample["target"]
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                      contain_eos=True)

            def _count(i):
                predict_i = (predict == i).long().sum().item()  # 1 是相关
                target_i = (target == i).long().sum().item()
                correct_i = ((predict == i) & (target == i)).long().sum().item()
                return predict_i, target_i, correct_i

            predict = _label
            target = dependency_mat
            name = ["pad", "positive", "negative", "same"]

            for i in [1, 2]:  # 0 pad 1 相关 2 不相关 3 相似
                predict_i, target_i, correct_i = _count(i)
                set_key_value("predict_" + name[i], predict_i)
                set_key_value("target_" + name[i], target_i)
                set_key_value("correct_" + name[i], correct_i)

            print(get_key_value())

        return _label
