# encoding=utf-8
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_

from dep_nat import DepHeadClassifier


class BinaryBiaffine(nn.Module):
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

    def get_reference(self, sample_ids):

        # -1表示不用的节点，0表示根节点。
        tree = self.head_tree.get_sentences(sample_ids, training=self.training)
        size = max(len(v) for v in tree)
        head_label = numpy.empty(shape=(len(tree), size + 2, size + 2))
        head_label.fill(0)
        for i, value in enumerate(tree):
            head_label[i][1:len(value) + 1, 1:len(value) + 1] = value

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

        if not return_hidden:
            return head_dep_result
        else:
            return head_dep_result, hidden_state


class BinaryClassifier(DepHeadClassifier):

    def __init__(self, args, relative_dep_mat=None, dep_file="", **kwargs):

        super().__init__(args, relative_dep_mat, dep_file, **kwargs)

        self.biaffine_parser = BinaryBiaffine(input_dim=self.mlp_input_dim, head_tree=self.head_tree,
                                              dropout=args.dropout)
        self.padding_idx = 0

    def get_mask_num(self, label, reference, reference_mask, update_nums):
        if self.tune:
            ratio = 0.5
        else:
            ratio = 0.5 - 0.2 / 300000 * update_nums
        diff = ((label != reference) & reference_mask).sum(-1).mean(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def get_head(self, score):
        # 2表示相关
        return (score > 0.5).long() + 1

    def get_label(self, score, target_tokens=None, **kwargs):
        dep_mat = self.get_head(score.detach())
        return dep_mat

    def inference_accuracy(self, get_mat=False, compute_accuracy=True, **kwargs):
        dep_mat = None
        score, predict, head_ref, mix_hidden = self.predict(**kwargs)
        loss = self.compute_loss(predict, head_ref)
        if get_mat:
            dep_mat = self.get_label(score, **kwargs)

        if compute_accuracy:
            if self.training:
                all, correct = self.compute_accuracy(predict, head_ref)
                return loss, dep_mat, all, correct
            else:
                return loss, dep_mat, 0, 0

        return loss, dep_mat
