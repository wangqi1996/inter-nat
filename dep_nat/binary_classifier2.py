# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from dep_nat import DepHeadClassifier
from nat_base.util import get_base_mask


class BinaryBiaffine2(nn.Module):
    def __init__(self, input_dim, head_tree=None, dropout=0.1, relative_dep_mat=None):

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

        self.classifier = nn.Linear(self.mlp_dim * 2, 2)

        self.head_tree = head_tree
        self.dropout = nn.Dropout(self.dropout)
        self.relative_mat = relative_dep_mat

    def get_reference(self, sample_ids, target_tokens):

        head_label = self.relative_mat.get_dependency_mat(sample_ids, target_tokens, training=self.training)

        return head_label

    def compute_loss(self, outputs, targets):
        loss = F.nll_loss(outputs.log_softmax(-1), targets.to(outputs.device) - 1, reduction='none').mean()

        return loss

    def forward_classifier(self, hidden_state, return_hidden=False):
        h_arc_dep = self.arc_dep_mlp(hidden_state)  # batch * max_trg * mlp_dim
        h_arc_head = self.arc_head_mlp(hidden_state)  # batch * max_trg * mlp_dim

        batch_size, tgt_len, _ = h_arc_head.size()

        head_dep_result = torch.cat((h_arc_dep[:, :, None, :].repeat(1, 1, tgt_len, 1),
                                     h_arc_head[:, None, :, :].repeat(1, tgt_len, 1, 1)), dim=-1)

        head_dep_result = self.classifier(head_dep_result)

        if not return_hidden:
            return head_dep_result
        else:
            return head_dep_result, hidden_state


class BinaryClassifier2(DepHeadClassifier):

    def __init__(self, args, relative_dep_mat=None, dep_file="", **kwargs):
        super().__init__(args, relative_dep_mat, dep_file, **kwargs)

        self.biaffine_parser = BinaryBiaffine2(input_dim=self.mlp_input_dim, head_tree=self.head_tree,
                                               dropout=args.dropout, relative_dep_mat=relative_dep_mat)
        self.padding_idx = 0

    def get_mask_num(self, label, reference, reference_mask, update_nums):
        ratio = 0.5 - 0.2 / 300000 * update_nums
        diff = (((label != reference) & reference_mask).sum(-1) != 0).sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def get_head(self, score):
        # 2表示相关
        score = score.argmax(-1) + 1
        return score

    def get_label(self, score, target_tokens=None, **kwargs):
        dep_mat = self.get_head(score.detach())
        mask = get_base_mask(target_tokens)
        dep_mat.masked_fill_(~mask.unsqueeze(-1), 0)
        dep_mat.masked_fill_(~mask.unsqueeze(-2), 0)
        return dep_mat

