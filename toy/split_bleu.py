import argparse

import sacrebleu
from sacremoses import MosesDetokenizer


def main(args):
    detok = MosesDetokenizer('en')

    translations = {}
    with open(args.input) as f:
        for line in f:
            if line.startswith('H'):
                line = line.rstrip().split('\t')
                if len(line) == 2:
                    id, score = line
                    translation = ""
                else:
                    id, score, translation = line
                id = int(id[2:])
                translations[id] = detok.detokenize(translation.rstrip().split(' '), return_str=True)

    hyps = [translations[i] for i in sorted(translations)]
    dir_name = "/home/data_ti5_c/wangdq/data/dataset/toy4/test"
    ref = [open(dir_name + str(i) + ".en") for i in range(1, 4)]
    score = sacrebleu.corpus_bleu(hyps, ref_streams=ref, lowercase=True, tokenize="zh").score
    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')

    args = parser.parse_args()
    main(args)
