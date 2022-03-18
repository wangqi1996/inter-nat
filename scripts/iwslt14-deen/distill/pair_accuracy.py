# encoding=utf-8


"""
compute the tree accuracy
"""


def compute_accuracy(predict_path, reference_path):
    with open(predict_path) as f:
        predict = {}
        for line in f:
            id = int(line.split('\t')[0][2:])
            predict[id] = line.strip().split('\t')[-1].split(' ')

    with open(reference_path) as f:
        reference = {}
        for line in f:
            id = int(line.split(';')[0])
            reference[id] = line.strip().split(';')[1:]

    correct, all = 0, 0

    for id in reference.keys():
        pre = predict[id]
        ref = reference[id]

        all += len(ref)
        for r in ref:
            f = True
            for rr in r.split(','):
                if rr not in pre:
                    f = False
                    break
            if f:
                correct += 1

    print(correct, all, correct * 1.0 / all)


if __name__ == '__main__':
    # predict_path = "/home/wangdq/test/hypo"
    # reference_path = "/home/wangdq/predict/pair.log"
    import sys

    predict_path = sys.argv[1]
    reference_path = sys.argv[2]
    compute_accuracy(predict_path, reference_path)
