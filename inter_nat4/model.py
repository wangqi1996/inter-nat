# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from inter_nat import RelationClassifier
from inter_nat.model import InterNAT, RelationBasedDecoder
from inter_nat.util import get_dep_mat
from nat_base.util import get_base_mask
from nat_base.vanilla_nat import NAT, nat_wmt_en_de, nat_iwslt14_de_en


class RelationClassifier2(RelationClassifier):

    def _forward_classifier(self, hidden_state, decoder_padding_mask, encoder_out):
        # 不dropout的好
        hidden_state = self.encoder.forward(hidden_state, encoder_out, decoder_padding_mask)

        score = self.biaffine_parser.forward_classifier(hidden_state.transpose(0, 1))  # [b, tgt_len, tgt_len]
        return score, hidden_state

    def forward(self, sample=None, hidden_state=None, encoder_out=None, **kwargs):
        sample_ids = sample['id'].cpu().tolist()
        oracle_head = self.biaffine_parser.get_reference(sample_ids).to(hidden_state.device)
        oracle_token = sample['target']

        pad_mask = oracle_token.eq(self.token_pad)
        score, hidden_state = self._forward_classifier(hidden_state, pad_mask, encoder_out)

        b_loss = self.biaffine_parser.compute_loss(score, oracle_head) * self.weight
        loss = {"relation": {"loss": b_loss}}
        return loss, hidden_state

    def inference(self, hidden_state=None, target_tokens=None, encoder_out=None, **kwargs):
        score, hidden_state = self._forward_classifier(hidden_state, target_tokens.eq(self.token_pad), encoder_out)
        head = self.get_head(score, target_tokens)

        # 2. reset the head of bos and eos
        head[:, 0] = 0
        target_length = target_tokens.ne(self.token_pad).long().sum(-1)
        for index, l in enumerate(target_length.cpu().tolist()):
            head[index][l - 1] = l - 1

        mat = get_dep_mat(head, target_tokens.ne(self.token_pad))

        return mat, hidden_state


class RelationClassifierDecoder(RelationBasedDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.dep_classifier = RelationClassifier2(args=args, dep_file=self.dep_file, dict=self.dictionary)

    def forward_relation(self, hidden_state, sample, encoder_out, target_tokens, **kwargs):
        if kwargs.get("generate", False):
            dependency_mat, hidden_state = self.dep_classifier.inference(sample=sample,
                                                                         hidden_state=hidden_state,
                                                                         target_tokens=target_tokens,
                                                                         encoder_out=encoder_out)
            return {}, dependency_mat, hidden_state
        else:
            loss, hidden_state = self.dep_classifier.forward(sample=sample, hidden_state=hidden_state,
                                                             encoder_out=encoder_out, **kwargs)

            return loss, None, hidden_state

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            x=None, decoder_padding_mask=None,
            **unused
    ):
        # 直接传了x进来
        x = x.transpose(0, 1)
        dep_loss, dep_mat, x = self.forward_relation(hidden_state=x,
                                                     target_tokens=prev_output_tokens,
                                                     encoder_out=encoder_out, **unused)

        if dep_mat is not None and not unused.get("use_oracle_mat", False):
            unused['dependency_mat'] = dep_mat  # [0 , 1] value
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

    def _forward(
            self,
            normalize,
            encoder_out,
            prev_output_tokens,
            step=0,
            inner=False,
            **unused):

        features, other = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            **unused
        )
        decoder_out = self.output_layer(features)
        decoder_out = F.log_softmax(
            decoder_out, -1) if normalize else decoder_out
        if inner:
            return decoder_out, other
        else:
            return decoder_out

    def forward(
            self,
            normalize,
            encoder_out,
            prev_output_tokens,
            step=0,
            inner=False,
            **unused):
        x, decoder_padding_mask, position = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        # valid & test
        if not self.training:
            return self._forward(normalize, encoder_out, prev_output_tokens, step, inner, x=x,
                                 decoder_padding_mask=decoder_padding_mask, **unused)
        else:
            unused['generate'] = True
            with torch.no_grad():
                features, other = self.extract_features(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    x=x,
                    decoder_padding_mask=decoder_padding_mask,
                    **unused
                )

                decoder_out = self.output_layer(features)  # [batch_size, seq_len, vocab_size]
                predict_tokens = decoder_out.argmax(-1)
                oracle_tokens = unused['sample']['target']
                mask_length = self.dep_classifier.get_mask_num(predict_tokens, oracle_tokens,
                                                               get_base_mask(oracle_tokens),
                                                               unused.get('update_num', 300000))

            x = self.glat_mask(oracle_tokens, x, mask_length)

            unused['generate'] = False
            return self._forward(normalize, encoder_out, prev_output_tokens, step, inner, x=x,
                                 decoder_padding_mask=decoder_padding_mask, **unused)

    def glat_mask(self, oracle_token, decoder_input, mask_length):
        oracle_embedding, _, _ = self.forward_embedding(oracle_token, add_position=True)
        oracle_mask = get_base_mask(oracle_token)
        output_embedding = self.dep_classifier.get_random_mask_output(mask_length, oracle_token,
                                                                      decoder_input.transpose(0, 1),
                                                                      oracle_embedding.transpose(0, 1), oracle_mask)

        return output_embedding.transpose(0, 1)


SuperClass, model_name = NAT, "inter44"


@register_model(model_name)
class Inter44NAT(InterNAT):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = RelationClassifierDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder


@register_model_architecture(model_name, model_name + '_wmt')
def inter_nat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def inter_nat_iwslt16_deen(args):
    nat_iwslt14_de_en(args)
