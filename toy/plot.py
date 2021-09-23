# encoding=utf-8

"""
grep ^H test/generate-test.txt | cut -f3- > multi_text/AT.pre
"""
import math

probability = {}

import ternary


def plot(points1, points2, points3):
    scale = 1
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    tax.scatter(points2, marker='x', color='#ea5455', s=100, linewidth=2)
    tax.scatter(points3, marker='*', color='#ffd460', s=100, linewidth=2)
    tax.scatter(points1, marker='v', color='#3490de', s=100, linewidth=2)

    fontsize = 45
    tax.right_corner_label('DE', fontsize=fontsize)
    tax.top_corner_label('FR', fontsize=fontsize, position=(-0.06, 1.12, 0))
    tax.left_corner_label('ES', fontsize=fontsize)

    tax.get_axes().axis('off')

    tax.show()

    tax.savefig("/home/data_ti5_c/wangdq/code/dep_nat/test.pdf")


def read_freq(lang):
    file = "/home/data_ti5_c/wangdq/model/fairseq/dep_nat_preject/multi-language/train/en" + lang + "/train." + lang + '.dict'
    print(file)
    result = {}
    with open(file, 'r') as f:
        for line in f:
            token = line.strip().split('\t')
            result[token[0]] = float(token[1])
    return result


def freq(lang):
    train_file = "/home/wangdq/dataset/train." + lang
    result = {}
    with open(train_file, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            for t in tokens:
                result.setdefault(t, 0)
                result[t] += 1

        _sum = sum(result.values())
        result = {k: v / _sum for k, v in result.items()}
        result2 = [str(k) + '\t' + str(v / _sum) + '\n' for k, v in result.items()]
        with open(train_file + '.dict', 'w') as f:
            f.writelines(result2)
        return result


def distance(a, b):
    def square(i):
        return (a[i] - b[i]) * (a[i] - b[i])

    return math.sqrt(square(0) + square(1) + square(2))


def count():
    # 20358
    file = "/home/data_ti5_c/wangdq/code/dep_nat/multi_text/AT.pre"
    # file = "/home/data_ti5_c/wangdq/model/fairseq/dep_nat_preject/multi-language/dev/test.trg"

    de_freq = read_freq("de")
    fr_freq = read_freq("fr")
    es_freq = read_freq("es")

    de_point = []
    fr_point = []
    es_point = []
    de_d, fr_d, es_d = 0, 0, 0
    with open(file, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            lp_de, lp_es, lp_fr = 0, 0, 0
            for t in tokens:
                p_de = de_freq.get(t, 0)
                p_es = es_freq.get(t, 0)
                p_fr = fr_freq.get(t, 0)
                p = p_de + p_es + p_fr
                if p != 0:
                    p_de, p_es, p_fr = p_de / p, p_es / p, p_fr / p
                    lp_de += p_de
                    lp_es += p_es
                    lp_fr += p_fr
                else:
                    lp_fr += 1 / 3
                    lp_de += 1 / 3
                    lp_es += 1 / 3

            lp_de /= len(tokens)
            lp_fr /= len(tokens)
            lp_es /= len(tokens)

            _max = max(lp_de, lp_es, lp_fr)
            if _max == lp_de:
                de_point.append((lp_de, lp_fr, lp_es))
                de_d += distance((1, 0, 0), (lp_de, lp_fr, lp_es))
            elif _max == lp_fr:
                fr_point.append((lp_de, lp_fr, lp_es))
                fr_d += distance((0, 1, 0), (lp_de, lp_fr, lp_es))
            elif _max == lp_es:
                es_point.append((lp_de, lp_fr, lp_es))
                es_d += distance((0, 0, 1), (lp_de, lp_fr, lp_es))

    print(len(de_point))
    print(len(fr_point))
    print(len(es_point))
    # print(de_d / len(de_point))
    # print(fr_d / len(fr_point))
    # print(es_d / len(es_point))
    print((de_d + fr_d + es_d) / (len(de_point) + len(fr_point) + len(es_point)))

    plot(de_point, fr_point, es_point)


if __name__ == '__main__':
    count()
