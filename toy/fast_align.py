# encoding=utf-8


"""
https://github.com/clab/fast_align/issues/33
python /home/data_ti5_c/wangdq/code/dep_nat/scripts/fast_align.py

1. 训练
/home/data_ti5_c/wangdq/code/fast_align/build/fast_align -i train.en-de -d -o -v  -p fwd_params >fwd_align 2>fwd_err
/home/data_ti5_c/wangdq/code/fast_align/build/fast_align -i train.en-de -d -o -v -r -p rev_params >rev_align 2>rev_err


2. inference
# fwd_params=/home/wangdq/dataset/train/ende/fwd_params
# fwd_err=/home/wangdq/dataset/train/ende/fwd_err
# rev_params=/home/wangdq/dataset/train/ende/rev_params
# rev_err=/home/wangdq/dataset/train/ende/rev_err
# python2 /home/data_ti5_c/wangdq/code/fast_align/build/force_align.py fwd_params fwd_err rev_params rev_err grow-diag-final-and <train.en-de >out


grep ^H multi_text/AT.txt | cut -f3- > dep_pred.txt
grep ^S multi_text/AT.txt | cut -f2- > dep_src.txt
"""

import math


def process_data():
    dir = "/home/wangdq/latent_glat_fast_align/distill/"
    file1 = dir + "source.txt"
    file2 = dir + "word.txt"
    result_file = dir + "source-word"

    result = []
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        for src, trg in zip(f1, f2):
            line = src.strip() + " ||| " + trg.strip() + '\n'
            result.append(line)

    with open(result_file, 'w') as f:
        f.writelines(result)


#
# class Aligner:
#
#     def __init__(self, rev_params, rev_err, heuristic='grow-diag-final-and'):
#
#         build_root = "/home/data_ti5_c/wangdq/code/fast_align/build/"
#         fast_align = os.path.join(build_root, 'fast_align')
#
#         (rev_T, rev_m) = self.read_err(rev_err)
#
#         self.rev_cmd = [fast_align, '-i', '-', '-d', '-T', rev_T, '-m', rev_m, '-f', rev_params, '-r']
#
#     def new_align(self, align_filename):
#         rev_filename = os.path.join(os.path.dirname(align_filename), "backward.align")
#         rev_cmd = ' '.join(self.rev_cmd) + ' <' + align_filename + ' ' + '>' + rev_filename
#         print(os.system(rev_cmd))
#
#     def read_err(self, err):
#         (T, m) = ('', '')
#         for line in open(err):
#             if 'expected target length' in line:
#                 m = line.split()[-1]
#             elif 'final tension' in line:
#                 T = line.split()[-1]
#         return (T, m)
#
#
# def force_align_main(rev_params, rev_err, token_filename):
#     aligner = Aligner(rev_params, rev_err, )
#
#     score_dict = aligner.new_align(token_filename)
#     return score_dict


# def fast_align():
#     key = "code"
#     dir = "/home/wangdq/checkpoint_best_avg_align/train/source_" + key + "_dir/"
#     source_file = dir + "source.test.txt"
#     target_file = dir + key + ".test.txt"
#     align_file = dir + "rev_align"
#     result_file = dir + "source-code-test"
#
#     result = []
#     with open(source_file, 'r') as f1, open(target_file, 'r') as f2:
#         for src, trg in zip(f1, f2):
#             line = src.strip() + " ||| " + trg.strip() + '\n'
#             result.append(line)
#
#     with open(result_file, 'w') as f:
#         f.writelines(result)
#
#     rev_params, rev_err = dir + "rev_params", dir + "rev_err"
#
#     force_align_main(rev_params, rev_err, result_file)
#
#     Cod1(dir + "backward.align")
#     print("done")


#
def Cod1(file):
    result = {}
    with open(file) as f:
        for line in f:
            src, trg, align, score = line.strip().split(' ||| ')
            result = _align(src, trg, align, result)
    cod = _cod(result)
    print(cod)


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


def Cod2():
    dir = "/home/wangdq/checkpoint_best_avg_align_new/source_word_dir/train/"
    source_file = dir + "source.txt"
    target_file = dir + "word.txt"
    align_file = dir + "train-align"
    result = {}
    with open(source_file) as f_src, open(target_file) as f_trg, open(align_file) as f_align:
        for src, trg, align in zip(f_src, f_trg, f_align):
            result = _align(src, trg, align, result)

    cod = _cod(result)
    print(cod)


if __name__ == '__main__':
    process_data()
    # Cod2()
    # fast_align()
