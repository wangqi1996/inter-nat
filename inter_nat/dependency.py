# encoding=utf-8

"""
依存树处理流程
1. 首先使用detokenizer，将tokenize data --> detokenize data
2. 然后调用stanza库解析依存关系
3. 将token级别的依存关系 --> bpe code 级别
"""

import argparse
import os

from fairseq import utils
from fairseq.data import Dictionary, indexed_dataset, data_utils

# DATASET = "/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/"
# save_dir = "/home/wangdq/dependency/iwslt16-deen/"

DATASET = "/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq/"
save_dir = "/home/wangdq/dependency/wmt-ende/"

split = "train"
os.makedirs(save_dir, exist_ok=True)

source_lang = "en"
target_lang = "de"

if target_lang == "de":
    resource = "/home/data_ti5_c/wangdq/model/tree/de/"

if target_lang == "en":
    resource = "/home/data_ti5_c/wangdq/model/tree/stanza_resources/"

if target_lang == "ro":
    resource = "/home/data_ti5_c/wangdq/model/tree/ro/"

if target_lang == "es":
    resource = "/home/data_ti5_c/wangdq/model/tree/es/"

if target_lang == "fr":
    resource = "/home/data_ti5_c/wangdq/model/tree/fr/"


def get_parser():
    parser = argparse.ArgumentParser(
        description='writes text from binarized file to stdout')
    # fmt: off
    parser.add_argument('--dataset-impl', help='dataset implementation',
                        choices=indexed_dataset.get_available_dataset_impl())
    parser.add_argument('--dict', metavar='FP', help='dictionary containing known words',
                        default=DATASET + "dict." + target_lang + ".txt")

    parser.add_argument('--batch_size', type=int, default=20480)

    parser.add_argument('--save', type=str,
                        default=save_dir + split + ".tree")
    parser.add_argument('--input', metavar='FP', help='binarized file to read',
                        default=DATASET + split + "." + source_lang + "-" + target_lang + "." + target_lang)

    return parser


class DependencyTree():

    def __init__(self, args):

        # dependency_tree
        import stanza
        self.dependency_tree = stanza.Pipeline(lang=target_lang, dir=resource,
                                               processors='tokenize,pos,lemma,depparse', tokenize_no_ssplit=True,
                                               tokenize_pretokenized=True)

        # dataset
        self.dictionary = Dictionary.load(args.dict) if args.dict is not None else None
        self.dataset = data_utils.load_indexed_dataset(
            args.input,
            self.dictionary,
            dataset_impl=args.dataset_impl,
            default='lazy',
        )

        from sacremoses import MosesDetokenizer, MosesTokenizer
        self.detok = MosesDetokenizer(target_lang)
        self.tok = MosesTokenizer(target_lang)

        self.args = args

    def process_text(self, text):

        # 返回raw text 和bpe text
        nopad_text = utils.strip_pad(text, self.dictionary.pad())
        raw_text = self.detok.unescape_xml(self.dictionary.string(nopad_text, "@@ "))  # 移出bep+转义
        bpe_text = self.dictionary.string(nopad_text)

        return raw_text, bpe_text

    def get_parent(self, dependency_tree):

        parent = [0 for _ in range(len(dependency_tree) + 1)]  # [bos] + 每个词
        token_list = []
        for token in dependency_tree:
            id = int(token['id'])
            parent[id] = token['head']
            token_list.append(token['text'])

        return parent, token_list

    def convert_head_token_to_bpe(self, word_head, token_list, bpe_text):
        if len(token_list) != len(bpe_text):
            index_mapping = self.word_index_to_bpe(bpe_text)

            bpe_head = [0 for _ in range(len(bpe_text))]  # 每个bpe的父节点
            for word_index, head in enumerate(word_head):
                if head == 0:
                    head_bpe_index = 0
                else:
                    head_bpe_index = index_mapping[head][0]  # 取头结点对应的节点信息

                first_bpe_index = (index_mapping[word_index][0])
                for bpe_index in index_mapping[word_index]:
                    bpe_head[bpe_index] = first_bpe_index
                bpe_head[first_bpe_index] = head_bpe_index
        else:
            bpe_head = word_head

        assert len(bpe_text) == len(bpe_head)
        dependency_layers = [str(d) for d in bpe_head]
        dependency_str = ','.join(dependency_layers) + ',' + str(
            len(bpe_head)) + '\n'  # the head of "eos" is "eos" itself.
        return dependency_str

    def word_index_to_bpe(self, bpe_text):

        word_index = 0
        mapping = {0: [0], }  # word_index: bpe_index
        for token_index, token in enumerate(bpe_text):
            if token_index == 0:
                continue
            if bpe_text[token_index - 1][-2:] == "@@":
                mapping[word_index].append(token_index)
            else:
                word_index += 1
                mapping[word_index] = [token_index]
        return mapping

    def get_dependency_head(self, sample_list):
        bpe_list = []
        raw_list = []
        dependency_input = ""
        result = []

        for text in sample_list:
            # 获取输入
            raw_text, bpe_text = self.process_text(text)
            dependency_input += (raw_text + '\n\n')  # \n\n是stanza指定的换行符
            bpe_list.append(bpe_text.split(' '))
            raw_list.append(raw_text.split(' '))

        # 依存树解析
        dependency_output = self.dependency_tree(dependency_input.rstrip()).to_dict()

        # 后处理依存树
        for index, dependency in enumerate(dependency_output):
            parent, token_list = self.get_parent(dependency)
            assert len(token_list) == len(raw_list[index]), u"解析树更改了tokenize"

            dependency_str = self.convert_head_token_to_bpe(parent, ["bos", ] + token_list, ["bos", ] + bpe_list[index])

            result.append(dependency_str)
        return result

    def main(self, args):
        sample = []

        with open(args.save, "w") as f:
            for index, tensor_line in enumerate(self.dataset):
                sample.append(tensor_line)

                if len(sample) > args.batch_size:
                    dependency_list = self.get_dependency_head(sample)
                    f.writelines(dependency_list)
                    sample = []
                    print("write lines: ", index)
            dependency_list = self.get_dependency_head(sample)
            f.writelines(dependency_list)
            print("write lines: ", index)


def main():
    parser = get_parser()
    args = parser.parse_args()

    DependencyTree(args).main(args)


if __name__ == '__main__':
    main()
