# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from GLAT.GLAT import hamming_distance
from dep_nat.dep_nat import DEPNAT, DEPRelativeDecoder
from nat_base.util import get_base_mask, new_arange
from nat_base.vanilla_nat import nat_iwslt16_de_en


def get_random_mask_output(mask_length=None, reference=None, samples=None, encoder_out=None,
                           decoder_input=None, reference_embedding=None, unk=3):
    reference_mask = get_base_mask(reference)

    target_score = reference.clone().float().uniform_()
    target_score.masked_fill_(~reference_mask, 2.0)

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
    mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]

    return _mask(mask, reference_embedding, decoder_input, reference, unk)


def _mask(mask, reference_embedding, decoder_input, reference, unk):
    non_mask = ~mask
    full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)

    # 处理token
    predict_unk = reference.clone().detach()
    reference_mask = get_base_mask(reference)
    predict_unk.masked_fill_(reference_mask, unk)
    full_output_tokens = torch.cat((predict_unk.unsqueeze(-1), reference.unsqueeze(-1)), dim=-1)
    output_tokens = (full_output_tokens * full_mask).sum(-1).long()

    # 处理embedding
    full_embedding = torch.cat((decoder_input.unsqueeze(-1), reference_embedding.unsqueeze(-1)), dim=-1)
    output_emebdding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)

    return output_tokens, output_emebdding


def get_mask_num(reference, predict):
    distance = hamming_distance(reference, predict)
    ratio = 0.5

    mask_num = distance * ratio  # 使用reference的数目 == 不使用decoder input的数目
    return mask_num


class DEPGLATDecoder(DEPRelativeDecoder):

    def get_mask_output(self, mask_length=None, reference=None, samples=None, encoder_out=None, decoder_input=None):
        reference_embedding, _, _ = self.forward_embedding(reference)

        kwargs = {
            "mask_length": mask_length,
            "reference": reference,
            "samples": samples,
            "encoder_out": encoder_out,
            "decoder_input": decoder_input,
            "reference_embedding": reference_embedding,
            "unk": self.unk
        }
        return get_random_mask_output(**kwargs)

    def glancing_inference(self, x, dep_mat, sample, encoder_out, decoder_padding_mask, prev_output_tokens):
        embedding = x.transpose(0, 1)
        for i, layer in enumerate(self.layers):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                dependency_mat=dep_mat,
                prev_output_tokens=prev_output_tokens
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        logits = self.output_layer(x)
        _score, predict = logits.max(-1)

        reference = sample['target']
        mask_num = get_mask_num(reference, predict)

        output_token, output_embedding = self.get_mask_output(decoder_input=embedding, reference=reference,
                                                              mask_length=mask_num, samples=sample, encoder_out=None)
        return output_token, output_embedding.transpose(0, 1)

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
        attn = None
        inner_states = [x]

        # glancing预测依存树
        dep_loss, dep_mat, _ = self.forward_classifier(hidden_state=x, target_tokens=prev_output_tokens,
                                                       position_embedding=None,
                                                       encoder_out=encoder_out, **unused)

        if dep_mat is not None:
            unused['dependency_mat'] = dep_mat
        else:
            unused['dependency_mat'] = self.get_dependency_mat(unused['sample'])

        if not unused.get("generate", False):
            with torch.no_grad():
                prev_output_tokens, x = self.glancing_inference(x, unused['dependency_mat'], unused['sample'],
                                                                encoder_out, decoder_padding_mask, prev_output_tokens)

        for i, layer in enumerate(self.layers):

            # if i == 1 and self.ctc_loss:
            #     x = _maybe_upsample_3d(x, batch_first=False)
            #     decoder_padding_mask = _maybe_upsample_2d(decoder_padding_mask)

            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                prev_output_tokens=prev_output_tokens,
                **unused
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        other = {"attn": attn, "dep_classifier_loss": dep_loss, 'prev_output_tokens': prev_output_tokens}

        return x, other


SuperClass, model_name = DEPNAT, "glat_inter"


@register_model(model_name)
class GLATInter(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPGLATDecoder(args, tgt_dict, embed_tokens)
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
        prev_output_tokens = other['prev_output_tokens']
        losses = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": prev_output_tokens.eq(self.unk),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "ctc_mask": other.get("ctc_mask", None)
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

        dep_classifier_loss = other['dep_classifier_loss']
        losses.update(dep_classifier_loss)

        return losses


@register_model_architecture(model_name, model_name + '_iwslt')
def dep_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
