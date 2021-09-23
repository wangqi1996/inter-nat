# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from dep_nat.module import DepRelativeMultiheadAttention
from dep_nat.util import RelativeDepNoSubMat
from nat_base.layer import BlockedDecoderLayer
from nat_base.vanilla_nat import NATDecoder, NAT, nat_wmt_en_de, nat_iwslt16_de_en, nat_toy
from .classifier import DepHeadClassifier


def _maybe_upsample_3d(embedding, batch_first=True):
    def _us(x, s):
        batch_size, seq_len, dim = x.shape
        _x = x.unsqueeze(-2).expand(batch_size, seq_len, s, dim).reshape(batch_size, seq_len * s, dim)
        return _x

    if not batch_first:
        embedding = embedding.transpose(0, 1)
    r = _us(embedding, 2)
    if not batch_first:
        r = r.transpose(0, 1)
    return r


def _maybe_upsample_2d(tokens):
    def _us(x, s):
        batch_size, seq_len = x.shape
        _x = x.unsqueeze(-1).expand(batch_size, seq_len, s).reshape(batch_size, seq_len * s)
        return _x

    return _us(tokens, 2)


class DEPRelativeGLATDecoderLayer(BlockedDecoderLayer):

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False, layer_id=0, **kwargs):

        if layer_id == 0:
            return DepRelativeMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size
            )
        else:
            return super().build_self_attention(embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                                layer_id=layer_id, **kwargs)


class DEPRelativeDecoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.dep_file = getattr(self.args, "dep_file", "iwslt16")

        self.relative_dep_mat = None
        if args.valid_subset == "valid":
            self.relative_dep_mat = RelativeDepNoSubMat(valid_subset=self.args.valid_subset, args=args,
                                                        dep_file=self.dep_file)

        self.dep_classifier = DepHeadClassifier(relative_dep_mat=self.relative_dep_mat, args=args,
                                                dep_file=self.dep_file)

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
        x = x.transpose(0, 1)

        dep_loss, dep_mat, mix_hidden = self.forward_classifier(hidden_state=x, position_embedding=position,
                                                                target_tokens=prev_output_tokens,
                                                                encoder_out=encoder_out, **unused)

        if dep_mat is not None:
            unused['dependency_mat'] = dep_mat
        else:
            unused['dependency_mat'] = self.get_dependency_mat(unused['sample'])

        for i, layer in enumerate(self.layers):
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
        target_token = sample['prev_target']
        dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_token, training=self.training)
        return dependency_mat

    def forward_classifier(self, hidden_state, sample, **kwargs):

        if kwargs.get("generate", False):
            # 生成
            dependency_mat = self.dep_classifier.inference(sample=sample, hidden_state=hidden_state, **kwargs)
            return {}, dependency_mat, None
        else:
            ref_embedding, _, _ = self.forward_embedding(sample['target'])
            ref_embedding = ref_embedding.transpose(0, 1)
            loss, dep_mat, mix_hidden = self.dep_classifier.inference_accuracy(sample=sample, hidden_state=hidden_state,
                                                                               ref_embedding=ref_embedding,
                                                                               **kwargs)

            return loss, dep_mat, mix_hidden


SuperClass, model_name = NAT, "dep"


@register_model(model_name)
class DEPNAT(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPRelativeDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder

    @staticmethod
    def add_args(parser):
        SuperClass.add_args(parser)
        parser.add_argument('--use-oracle-mat', action="store_true")
        parser.add_argument('--dep-file', type=str, default="iwslt16")  # wmt16
        parser.add_argument('--tune', action="store_true")
        parser.add_argument('--weight', type=float, default=1)
        parser.add_argument('--noglancing', action="store_true")

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
                "nll_loss": True,
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


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)


@register_model_architecture(model_name, model_name + '_toy')
def dep_relative_glat_toy(args):
    nat_toy(args)
