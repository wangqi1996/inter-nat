# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from dep_nat.binary_classifier import BinaryClassifier
from dep_nat.dep_nat import DEPRelativeDecoder, DEPNAT
from dep_nat.util import CoTree
from nat_base.vanilla_nat import nat_wmt_en_de, nat_iwslt16_de_en


class CoDecoder(DEPRelativeDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.relative_dep_mat = None
        if not self.tune:
            self.relative_dep_mat = CoTree(valid_subset=self.args.valid_subset, args=args,
                                           dep_file=self.dep_file, only_valid=self.tune, ratio=args.co_ratio)

        self.dep_classifier = BinaryClassifier(relative_dep_mat=self.relative_dep_mat, args=args,
                                               dep_file=self.dep_file)


SuperClass, model_name = DEPNAT, "co"


@register_model(model_name)
class CoNAT(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = CoDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)


@register_model_architecture(model_name, model_name + '_iwslt')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
