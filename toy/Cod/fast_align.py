# encoding=utf-8
import argparse
import math


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
    for s, t_dict in result.items():
        p = []
        for t, value in t_dict.items():
            p.append(value)
        _sum = sum(p)
        p = [pp / _sum * math.log(pp / _sum) for pp in p]
        _cross_entropy += sum(p)

    return -_cross_entropy / len(result)


def Cod(args):
    dir = args.dir
    source_file = dir + args.source
    target_file = dir + args.target
    align_file = dir + args.align
    result = {}
    with open(source_file) as f_src, open(target_file) as f_trg, open(align_file) as f_align:
        for src, trg, align in zip(f_src, f_trg, f_align):
            result = _align(src, trg, align, result)

    cod = _cod(result)
    print(cod)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--source', type=str, default="source-target")
    parser.add_argument('--target', type=str, default="target.txt")
    parser.add_argument('--align', type=str, default="source-target-align")

    args = parser.parse_args()
    Cod(args)
