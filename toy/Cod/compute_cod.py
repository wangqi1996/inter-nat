# encoding=utf-8
import argparse
import math
import os


def _align(src, trg, align, result: dict):
    src = src.strip().split(' ')
    trg = trg.strip().split(' ')
    for a in align.strip().split(' '):
        a, t = a.strip().split('-')
        s_token = src[int(a)]
        t_token = trg[int(t)]
        result.setdefault(s_token, {})
        result[s_token].setdefault(t_token, 0)
        result[s_token][t_token] += 1

    return result


def _cod(result: dict):
    _cross_entropy = 0
    src_cod = {}
    for s, t_dict in result.items():
        p = []
        for t, value in t_dict.items():
            p.append(value)
        _sum = sum(p)
        p = [pp / _sum * math.log(pp / _sum) for pp in p]
        r = sum(p)
        _cross_entropy += r
        src_cod[s] = -r

    return -_cross_entropy / len(result), src_cod


def Cod(args):
    dir = args.dir
    source_file = os.path.join(dir, args.source)
    target_file = os.path.join(dir, args.target)
    align_file = os.path.join(dir, args.align)
    result = {}
    with open(source_file) as f_src, open(target_file) as f_trg, open(align_file) as f_align:
        for src, trg, align in zip(f_src, f_trg, f_align):
            result = _align(src, trg, align, result)

    cod, src_dict_cod = _cod(result)
    sentence_cod, sentences = 0, 0
    with open(source_file) as f_src:
        for src in f_src:
            _sentence_cod, tokens = 0, 0
            for s in src.strip().split(' '):
                if s not in src_dict_cod:
                    continue
                _sentence_cod += src_dict_cod[s]
                tokens += 1
            if _sentence_cod > 0:
                sentence_cod += (_sentence_cod / tokens)
                sentences += 1

    sentence_cod /= sentences
    print("token-level: %.4f" % cod)
    print("sentence-level: %.4f" % sentence_cod)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="/home/wangdq/latent_glat_fast_align/rawdata")
    parser.add_argument('--source', type=str, default="source.txt")
    parser.add_argument('--target', type=str, default="word.txt")
    parser.add_argument('--align', type=str, default="train-align")

    args = parser.parse_args()
    Cod(args)
