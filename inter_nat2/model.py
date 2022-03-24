# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from inter_nat.classifier import RelationClassifier
from nat_base.util import get_base_mask
from nat_base.vanilla_nat import NATDecoder, NAT, nat_wmt_en_de, nat_iwslt14_de_en


class TestDecoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.dep_file = getattr(self.args, "dep_file", "iwslt16")

        self.dep_classifier = RelationClassifier(args=args, dep_file=self.dep_file, token_pad=self.padding_idx)

    def forward_relation(self, hidden_state, sample, encoder_out, target_tokens, **kwargs):
        if kwargs.get("generate", False):
            if kwargs.get("write_tree", "") != "":
                score = self.dep_classifier._forward_classifier(hidden_state, target_tokens.eq(self.dictionary.pad()),
                                                                encoder_out)
                head = score.argmax(-1)
                predict_path = os.path.join(kwargs['write_tree'], "predict.tree")

                sample_ids = sample['id']
                reference_tree = self.dep_classifier.biaffine_parser.get_reference(sample_ids)
                reference_path = os.path.join(kwargs['write_tree'], "reference.tree")

                mask = target_tokens.ne(self.dictionary.pad())
                with open(predict_path, 'a') as f_pre, open(reference_path, 'a') as f_ref:
                    for index, h in enumerate(head):
                        h_str = ",".join([str(i) for i in h[mask[index]].cpu().tolist()])
                        f_pre.write(h_str + '\n')
                        h_str = ",".join([str(i) for i in reference_tree[index][mask[index]].cpu().tolist()])
                        f_ref.write(h_str + '\n')

            if kwargs.get("write_reference_pairs", "") != "":
                sample_ids = sample['id']
                reference_tree = self.dep_classifier.biaffine_parser.get_reference(sample_ids)
                mask = get_base_mask(sample['target'])
                with open(kwargs['write_reference_pairs'], 'a') as f:
                    for index, tree in enumerate(reference_tree):
                        token = self.dictionary.string(sample['target'][index][mask[index]]).split(' ')
                        tree = tree[mask[index]]
                        content = []
                        for i, t in enumerate(tree):
                            if t == 0:
                                content.append(token[i])
                            else:
                                content.append(token[i] + ',' + token[t - 1])
                        content = str(sample_ids[index].item()) + ";" + ";".join(content)
                        f.write(content + '\n')
            return None
        else:
            ref_embedding, _, _ = self.forward_embedding(sample['target'])
            ref_embedding = ref_embedding.transpose(0, 1)
            loss = self.dep_classifier.forward(sample=sample, hidden_state=hidden_state,
                                               ref_embedding=ref_embedding, encoder_out=encoder_out,
                                               **kwargs)

            return loss

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

        dep_loss = self.forward_relation(hidden_state=x, encoder_out=encoder_out, target_tokens=prev_output_tokens,
                                         **unused)

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


SuperClass, model_name = NAT, "test"


@register_model(model_name)
class TestNAT(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TestDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder

    @staticmethod
    def add_args(parser):
        SuperClass.add_args(parser)
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
def test_nat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def test_nat_iwslt16_deen(args):
    nat_iwslt14_de_en(args)
