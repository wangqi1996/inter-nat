# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from dep_ablation.classifier import DepHeadClassifier2
from dep_nat.util import RelativeDepNoSubMat
from nat_base.layer import BlockedDecoderLayer
from nat_base.vanilla_nat import NATDecoder, NAT, nat_wmt_en_de, nat_iwslt16_de_en


class DEPRelativeDecoder2(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.dep_file = getattr(self.args, "dep_file", "iwslt16")
        self.tune = getattr(args, "tune", False)

        self.relative_dep_mat = RelativeDepNoSubMat(valid_subset=self.args.valid_subset, args=args,
                                                    dep_file=self.dep_file, only_valid=self.tune)
        self.dep_classifier = DepHeadClassifier2(relative_dep_mat=self.relative_dep_mat, args=args,
                                                 dep_file=self.dep_file)

        self.use_oracle_mat = getattr(self.args, "use_oracle_mat", False)

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
        return BlockedDecoderLayer(args, no_encoder_attn=no_encoder_attn, relative_keys=rel_keys,
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

        dep_loss, dep_mat = self.forward_classifier(hidden_state=x, position_embedding=position,
                                                    target_tokens=prev_output_tokens, encoder_out=encoder_out, **unused)
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
                **unused
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        other = {"attn": attn, "inner_states": inner_states, "embedding": embedding, "position_embedding": position,
                 "dep_classifier_loss": dep_loss}

        return x, other

    def forward_classifier(self, hidden_state, sample, **kwargs):

        if kwargs.get("generate", False):
            _ = self.dep_classifier.inference(sample=sample, hidden_state=hidden_state,
                                              compute_accuracy=False,
                                              ref_embedding=None, get_mat=False,
                                              **kwargs)
            return {}, None
        else:
            ref_embedding, _, _ = self.forward_embedding(sample['target'])
            ref_embedding = ref_embedding.transpose(0, 1)
            loss, dep_mat = self.dep_classifier.inference_accuracy(sample=sample, hidden_state=hidden_state,
                                                                   compute_accuracy=False,
                                                                   ref_embedding=ref_embedding, get_mat=False,
                                                                   **kwargs)

        return loss, dep_mat


SuperClass, model_name = NAT, "dep_ablation"


@register_model(model_name)
class DEPNAT2(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPRelativeDecoder2(args, tgt_dict, embed_tokens)
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
        parser.add_argument('--add-classifier-position', type=str, default="cat")  # none
        parser.add_argument('--dep-encoder-layers', default=2, type=int)
        parser.add_argument('--glat-training', action="store_true")

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

        dep_classifier_loss = other['dep_classifier_loss']
        losses.update(dep_classifier_loss)

        return losses


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
