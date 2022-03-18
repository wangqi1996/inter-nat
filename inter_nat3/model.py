# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch

from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from inter_nat.classifier import RelationClassifier
from inter_nat.model import RelationBasedLayer, InterNAT
from inter_nat.util import ParentRelationMat
from nat_base.vanilla_nat import NATDecoder, NAT, nat_wmt_en_de, nat_iwslt14_de_en


class Test3Decoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.dep_file = getattr(self.args, "dep_file", "iwslt16")

        self.relative_dep_mat = ParentRelationMat(valid_subset=self.args.valid_subset, args=args,
                                                  dep_file=self.dep_file)

        self.dep_classifier = RelationClassifier(args=args, dep_file=self.dep_file, token_pad=self.padding_idx,
                                                 layer=self.layers[-1])

        # 对三角矩阵
        self._build_diag(200)

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
        return RelationBasedLayer(args, no_encoder_attn=no_encoder_attn, relative_keys=rel_keys,
                                  relative_vals=rel_vals, layer_id=0, **kwargs)

    def forward_relation(self, hidden_state, sample, encoder_out, target_tokens, **kwargs):
        if kwargs.get("generate", False):
            dependency_mat = self.dep_classifier.inference(sample=sample, hidden_state=hidden_state,
                                                           target_tokens=target_tokens,
                                                           encoder_out=encoder_out)
            return {}, dependency_mat
        else:
            ref_embedding, _, _ = self.forward_embedding(sample['target'])
            ref_embedding = ref_embedding.transpose(0, 1)
            loss = self.dep_classifier.forward(sample=sample, hidden_state=hidden_state,
                                               ref_embedding=ref_embedding, encoder_out=encoder_out,
                                               **kwargs)

            return loss, None

    def _build_diag(self, max_seq_len):
        self.max_seq_len = max_seq_len
        _mat = [1 for _ in range(self.max_seq_len)]
        _mat2 = [1 for _ in range(self.max_seq_len - 1)]
        self.max_dep_mat = torch.from_numpy(np.diag(_mat2, -1) + np.diag(_mat, 0) + np.diag(_mat2, 1)).cuda()

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            **unused
    ):
        x, decoder_padding_mask, position = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)
        x = x.transpose(0, 1)

        if not unused.get("use_oracle_mat", False):
            dep_loss, dep_mat = self.forward_relation(hidden_state=x,
                                                      target_tokens=prev_output_tokens,
                                                      encoder_out=encoder_out, **unused)
        else:
            dep_loss, dep_mat = None, None

        if dep_mat is not None:
            unused['dependency_mat'] = dep_mat
        else:
            unused['dependency_mat'] = self.get_dependency_mat(unused['sample'])

        for i, layer in enumerate(self.layers):
            if i == 1:
                # 替换成对三角矩阵
                batch_size, seq_len = prev_output_tokens.shape
                if seq_len > self.max_seq_len:
                    self._build_diag(seq_len)
                dep_mat = self.max_dep_mat[:seq_len, :seq_len].repeat(batch_size, 1, 1)
                dep_mat = dep_mat.masked_fill(decoder_padding_mask.unsqueeze(-1), 0)
                dep_mat = dep_mat.masked_fill(decoder_padding_mask.unsqueeze(-2), 0)
                unused['dependency_mat'] = dep_mat

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                prev_output_tokens=prev_output_tokens,
                **unused
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"dep_classifier_loss": dep_loss}

    def get_dependency_mat(self, sample, **kwargs):
        sample_ids = sample['id'].cpu().tolist()
        target_token = sample['target']
        dependency_mat = self.relative_dep_mat.get_relation_mat(sample_ids, target_token, training=self.training)
        return dependency_mat


model_name = "inter3"


@register_model(model_name)
class InterNAT3(InterNAT):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = Test3Decoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder


@register_model_architecture(model_name, model_name + '_wmt')
def inter_nat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def inter_nat_iwslt16_deen(args):
    nat_iwslt14_de_en(args)
