# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from dep_nat.dep_nat import DEPRelativeGLATDecoderLayer
from nat_base.util import get_base_mask
from nat_base.vanilla_nat import NATDecoder, NAT, nat_wmt_en_de, nat_iwslt16_de_en


def get_dep_mat(all_head, mask, dtype=torch.long):
    all_head.masked_fill_(~mask, 0)
    batch_size, tgt_len = all_head.shape

    flat_all_head = all_head.view(-1)
    add = torch.arange(0, batch_size * tgt_len * tgt_len, tgt_len).to(all_head.device)
    flat_all_head = flat_all_head + add
    same = add + torch.arange(0, tgt_len).repeat(batch_size).to(all_head.device)
    dep_mat = all_head.new_zeros((batch_size, tgt_len, tgt_len), dtype=dtype).fill_(2)

    dep_mat = dep_mat.view(-1)
    dep_mat[flat_all_head] = 1
    dep_mat[same] = 1

    dep_mat = dep_mat.view(batch_size, tgt_len, tgt_len)
    dep_mat.masked_fill_(~mask.unsqueeze(-1), 0)
    dep_mat.masked_fill_(~mask.unsqueeze(-2), 0)

    # 对称
    mask_1 = dep_mat == 1
    mask_1 = mask_1.transpose(-1, -2)
    dep_mat = dep_mat.masked_fill(mask_1, 1)

    return dep_mat


class LocalDecoder(NATDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(LocalDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.seq_len = 200
        _mat = [1 for _ in range(self.seq_len)]
        _mat2 = [1 for _ in range(self.seq_len - 1)]
        self.max_dep_mat = torch.from_numpy(np.diag(_mat2, -1) + np.diag(_mat, 0) + np.diag(_mat2, 1)).cuda() + 1

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
        return DEPRelativeGLATDecoderLayer(args, no_encoder_attn=no_encoder_attn, relative_keys=rel_keys,
                                           relative_vals=rel_vals, layer_id=layer_id, **kwargs)

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            **unused
    ):
        x, decoder_padding_mask, position = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        embedding = x.clone()
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # 构造mask矩阵
        seq_len, batch_size, _ = x.size()
        if seq_len > self.seq_len:
            _mat = [1 for _ in range(self.seq_len)]
            _mat2 = [1 for _ in range(self.seq_len - 1)]
            self.max_dep_mat = torch.from_numpy(np.diag(_mat2, -1) + np.diag(_mat, 0) + np.diag(_mat2, 1)) + 1
            self.seq_len = seq_len

        dep_mat = self.max_dep_mat[:seq_len, :seq_len].repeat(batch_size, 1, 1)
        mask = get_base_mask(prev_output_tokens)
        dep_mat.masked_fill_(~mask.unsqueeze(-1), 0)
        dep_mat.masked_fill_(~mask.unsqueeze(-2), 0)

        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                prev_output_tokens=prev_output_tokens,
                dependency_mat=dep_mat,
                **unused
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        other = {"attn": attn, "inner_states": inner_states, "embedding": embedding, "position_embedding": position}

        return x, other


SuperClass, model_name = NAT, "local"


@register_model(model_name)
class LocalNAT(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LocalDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner=True,
            **kwargs
        )

        losses = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True
            }
        }
        # length prediction
        if self.decoder.length_loss_factor > 0:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            losses["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }

        return losses


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
