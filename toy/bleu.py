# encoding=utf-8

import sacrebleu
from sacremoses import MosesDetokenizer

NGRAM_ORDER = 1


def compute_bleu2():
    dir_name = "/home/data_ti5_c/wangdq/code/dep_nat/"
    source = dir_name + "AT/generate-test.txt"
    fref1 = "/home/wangdq/dataset/dev/orig/test2006.de"
    fref2 = "/home/wangdq/dataset/dev/orig/test2006.fr"
    fref3 = "/home/wangdq/dataset/dev/orig/test2006.es"
    detok = MosesDetokenizer('en')

    translations = {}
    with open(source) as f:
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

    bleu = 0
    ref = []
    with open(fref1) as f_ref1, open(fref2) as f_ref2, open(fref3) as f_ref3:
        for src, ref1, ref2, ref3 in zip(hyps, f_ref1, f_ref2, f_ref3):
            b1 = sacrebleu.sentence_bleu(src, [ref1]).score
            b2 = sacrebleu.sentence_bleu(src, [ref2]).score
            b3 = sacrebleu.sentence_bleu(src, [ref3]).score
            b = max(b1, b2, b3)
            bleu += b
            if b == 0:
                print("123")
                b1 = sacrebleu.sentence_chrf(src, [ref1]).score
                b2 = sacrebleu.sentence_chrf(src, [ref2]).score
                b3 = sacrebleu.sentence_chrf(src, [ref3]).score
                b = max(b1, b2, b3)
            if b == b1:
                ref.append(ref1)
            elif b == b2:
                ref.append(ref2)
            elif b == b3:
                ref.append(ref3)

    print(bleu / len(hyps))
    b = sacrebleu.corpus_bleu(hyps, [ref])
    print(b.score)


if __name__ == '__main__':
    compute_bleu2()
