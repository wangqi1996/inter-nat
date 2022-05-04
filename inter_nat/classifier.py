# encoding=utf-8
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_

from fairseq.models import BaseFairseqModel
from inter_nat.util import Tree, get_dep_mat
from nat_base.layer import build_relative_embeddings, BlockedDecoderLayer
from nat_base.util import new_arange, get_base_mask


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
        self.tree_pad = -1

    def get_reference(self, sample_ids):
        tree = self.head_tree.get_sentences(sample_ids, training=self.training)  # 包含bos和eos
        size = max(len(v) for v in tree)
        head_label = numpy.empty(shape=(len(tree), size))
        head_label.fill(self.tree_pad)
        for i, value in enumerate(tree):
            value[0] = value[-1] = self.tree_pad  # don't predict the head of <bos> and <eos>
            head_label[i][:len(value)] = value

        head_label = torch.from_numpy(head_label).long()

        return head_label

    def compute_loss(self, outputs, targets):
        """
        :param outputs: [batsh, seq_len, seq_len]
        :param targets: [batch, seq_len]
        :return:
        """
        mask = targets.ne(self.tree_pad)
        outputs = outputs[mask]
        targets = targets[mask]
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


class Encoder(BaseFairseqModel):

    def __init__(self, args, layer_num=4):
        super().__init__()
        self.layers = nn.ModuleList()
        rel_keys = build_relative_embeddings(args)
        rel_vals = build_relative_embeddings(args)

        for i in range(layer_num):
            self.layers.extend([BlockedDecoderLayer(
                args,
                no_encoder_attn=False,
                relative_keys=rel_keys,
                relative_vals=rel_vals,
            )])

    def forward(self, x, encoder_out, padding_mask):
        for idx, layer in enumerate(self.layers):
            x, _, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=padding_mask,
                need_attn=False,
                need_head_weights=False,
                # dependency_mat=dep_mat
            )
        return x


class RelationClassifier(BaseFairseqModel):

    def __init__(self, args, dep_file="", dict=None):
        super().__init__()

        self.args = args
        self.encoder = Encoder(args)
        self.dropout = nn.Dropout(args.dropout)

        self.head_tree = Tree(valid_subset=self.args.valid_subset, dep_file=dep_file)

        self.biaffine_parser = BiaffineAttentionDependency(input_dim=args.decoder_embed_dim, head_tree=self.head_tree,
                                                           dropout=args.dropout)

        self.weight = getattr(args, "weight", 1)
        self.token_pad = dict.pad()
        self.token_bos = dict.bos()
        self.token_eos = dict.eos()
        self.tree_pad = self.biaffine_parser.tree_pad

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

        return output_embedding.transpose(0, 1)

    def get_mask_num(self, label, reference, reference_mask, update_nums, sample_ids):
        ratio = 0.5 - 0.2 / 300000 * update_nums
        try:
            diff = ((label != reference) & reference_mask).long().sum(-1).detach()
        except:
            from nat_base.nat_task import logger
            logger.info("self.training", self.training)
            logger.info("label.shape", label.shape)
            logger.info("reference.shape", reference.shape)
            logger.info("sample_ids", sample_ids)
            diff = reference_mask.long().sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def _forward_classifier(self, hidden_state, decoder_padding_mask, encoder_out):
        # 不dropout的好
        hidden_state = self.encoder.forward(hidden_state, encoder_out, decoder_padding_mask)
        hidden_state = hidden_state.transpose(0, 1)

        score = self.biaffine_parser.forward_classifier(hidden_state)  # [b, tgt_len, tgt_len]
        return score

    def forward(self, sample=None, hidden_state=None, ref_embedding=None, encoder_out=None, **kwargs):
        sample_ids = sample['id'].cpu().tolist()
        oracle_head = self.biaffine_parser.get_reference(sample_ids).to(hidden_state.device)
        oracle_token = sample['target']
        pad_mask = oracle_token.eq(self.token_pad)

        if self.training:
            with torch.no_grad():
                score = self._forward_classifier(hidden_state, pad_mask, encoder_out)
                label = self.get_head(score, oracle_token)

                mask_length = self.get_mask_num(label, oracle_head, oracle_head.ne(self.tree_pad),
                                                kwargs.get('update_num', 300000), sample_ids)

            hidden_state = self.get_random_mask_output(mask_length, oracle_token, hidden_state,
                                                       ref_embedding, reference_mask=get_base_mask(oracle_token))

        score = self._forward_classifier(hidden_state, pad_mask, encoder_out)

        b_loss = self.biaffine_parser.compute_loss(score, oracle_head) * self.weight
        loss = {"relation": {"loss": b_loss}}

        return loss

    def get_head(self, score, target_tokens):
        # 1. the score between the token and eos/pad token is 0
        mask = target_tokens.eq(self.token_eos) | target_tokens.eq(self.token_pad)
        score.masked_fill_(mask.unsqueeze(1), float('-inf'))
        head = score.argmax(-1)
        return head

    def inference(self, hidden_state=None, target_tokens=None, encoder_out=None, **kwargs):
        score = self._forward_classifier(hidden_state, target_tokens.eq(self.token_pad), encoder_out)
        head = self.get_head(score, target_tokens)

        # 2. reset the head of bos and eos
        head[:, 0] = 0
        target_length = target_tokens.ne(self.token_pad).long().sum(-1)
        for index, l in enumerate(target_length.cpu().tolist()):
            head[index][l - 1] = l - 1

        mat = get_dep_mat(head, target_tokens.ne(self.token_pad))

        return mat
