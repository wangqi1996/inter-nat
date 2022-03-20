# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from inter_nat.classifier import BiaffineAttentionDependency
from inter_nat.model import InterNAT, RelationBasedLayer
from inter_nat.util import ParentRelationMat, Tree, get_dep_mat
from nat_base.layer import BlockedDecoderLayer
from nat_base.util import get_base_mask, new_arange
from nat_base.vanilla_nat import NATDecoder, NAT, nat_wmt_en_de, nat_iwslt14_de_en


class RelationClassifierDecoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.dep_file = getattr(self.args, "dep_file", "iwslt16")

        self.relative_dep_mat = ParentRelationMat(valid_subset=self.args.valid_subset, args=args,
                                                  dep_file=self.dep_file)

        self.head_tree = Tree(valid_subset=self.args.valid_subset, dep_file=self.dep_file)

        self.biaffine_parser = BiaffineAttentionDependency(input_dim=args.decoder_embed_dim, head_tree=self.head_tree,
                                                           dropout=args.dropout)
        self.tree_pad = self.biaffine_parser.tree_pad

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
        # 前四层用来预测依存树
        if layer_id < 4:
            return BlockedDecoderLayer(
                args,
                no_encoder_attn=no_encoder_attn,
                relative_keys=rel_keys,
                relative_vals=rel_vals,
            )
        # 集成关系矩阵
        elif layer_id == 4:
            return RelationBasedLayer(args, no_encoder_attn=no_encoder_attn, relative_keys=rel_keys,
                                      relative_vals=rel_vals, layer_id=0, **kwargs)

        # 剩下的是decoder
        else:
            return BlockedDecoderLayer(
                args,
                no_encoder_attn=no_encoder_attn,
                relative_keys=rel_keys,
                relative_vals=rel_vals,
            )

    def forward_relation(self, hidden_state, sample, encoder_out, target_tokens, **kwargs):
        score = self.biaffine_parser.forward_classifier(hidden_state.transpose(0, 1))
        if kwargs.get("generate", False):
            head = score.argmax(-1)
            dependency_mat = get_dep_mat(head, target_tokens.ne(self.padding_idx))
            return {}, dependency_mat
        else:
            sample_ids = sample['id'].cpu().tolist()
            oracle_head = self.biaffine_parser.get_reference(sample_ids).to(hidden_state.device)

            _mask = oracle_head != self.tree_pad
            b_loss = self.biaffine_parser.compute_loss(score[_mask], oracle_head[_mask]) * self.args.weight
            loss = {"relation": {"loss": b_loss}}
            return loss, None

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            x=None, decoder_padding_mask=None,
            **unused
    ):
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            # 在第4层，计算依存树和依存树预测的loss
            if i == 4:
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

    def forward(
            self,
            normalize,
            encoder_out,
            prev_output_tokens,
            step=0,
            inner=False,
            **unused):
        x, decoder_padding_mask, position = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        if unused.get("generate", False) is True:
            features, other = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                x=x,
                decoder_padding_mask=decoder_padding_mask,
                **unused
            )
            decoder_out = self.output_layer(features)
            decoder_out = F.log_softmax(
                decoder_out, -1) if normalize else decoder_out
            if inner:
                return decoder_out, other
            else:
                return decoder_out
        else:
            self.eval()
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
                mask_length = self.get_mask_num(predict_tokens, oracle_tokens, oracle_tokens.ne(self.padding_idx),
                                                unused.get('update_num', 300000))

            self.train()
            # 就是用dropout之后的
            x = self.glat_mask(oracle_tokens, x, mask_length)

            unused['generate'] = False
            features, other = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                x=x,
                decoder_padding_mask=decoder_padding_mask,
                **unused
            )
            decoder_out = self.output_layer(features)
            decoder_out = F.log_softmax(
                decoder_out, -1) if normalize else decoder_out
            if inner:
                return decoder_out, other
            else:
                return decoder_out

    def glat_mask(self, oracle_token, decoder_input, mask_length):
        oracle_embedding, _, _ = self.forward_embedding(oracle_token, add_position=True)
        oracle_mask = get_base_mask(oracle_token)
        target_score = oracle_token.clone().float().uniform_()
        target_score.masked_fill_(~oracle_mask, 2.0)

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]

        non_mask = ~mask
        full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)

        full_embedding = torch.cat((decoder_input.unsqueeze(-1), oracle_embedding.unsqueeze(-1)), dim=-1)
        output_emebdding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)

        return output_emebdding

    def get_mask_num(self, label, reference, reference_mask, update_nums):
        ratio = 0.5 - 0.2 / 300000 * update_nums
        diff = ((label != reference) & reference_mask).sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def get_dependency_mat(self, sample, **kwargs):
        sample_ids = sample['id'].cpu().tolist()
        target_token = sample['target']
        dependency_mat = self.relative_dep_mat.get_relation_mat(sample_ids, target_token, training=self.training)
        return dependency_mat


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
