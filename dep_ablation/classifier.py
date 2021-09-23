# encoding=utf-8
import torch

from dep_nat.classifier import DepHeadClassifier


class DepHeadClassifier2(DepHeadClassifier):

    def __init__(self, args, relative_dep_mat=None, dep_file="", **kwargs):
        super(DepHeadClassifier2, self).__init__(args, relative_dep_mat, dep_file, **kwargs)

        self.glat_training = getattr(self.args, "glat_training", False)

    def forward_classifier(self, hidden_state=None, reference=None, target_tokens=None, ref_embedding=None,
                           **kwargs):

        if self.training and self.glat_training:
            with torch.no_grad():
                score, _ = self._forward_classifier(hidden_state=hidden_state, **kwargs)
                score = score.detach()
                reference_mask = reference != self.padding_idx
                label = self.get_head(score)
                mask_length = self.get_mask_num(label, reference, reference_mask, kwargs.get('update_nums', 300000))
            _, hidden_state = self.get_random_mask_output(mask_length, target_tokens, hidden_state,
                                                          ref_embedding, reference_mask=reference_mask)
            score, hidden_state = self._forward_classifier(hidden_state=hidden_state, **kwargs)
        else:
            score, hidden_state = self._forward_classifier(hidden_state=hidden_state, **kwargs)
        return score, hidden_state
