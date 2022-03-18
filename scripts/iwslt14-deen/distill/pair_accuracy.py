# encoding=utf-8


"""
compute the tree accuracy
"""


def compute_accuracy(predict_path, reference_path):
    with open(predict_path) as f:
        predict = f.readlines()

    predict = [[int(p) for p in pp.split(',')] for pp in predict]

    with open(reference_path) as f:
        reference = f.readlines()

    reference = [[int(p) for p in pp.split(',')] for pp in reference]

    correct, all = 0, 0
    for pp, rr in zip(predict, reference):
        for p, r in zip(pp, rr):
            if p == r:
                correct += 1
            all += 1

    print(correct, all, correct * 1.0 / all)


if __name__ == '__main__':
    predict_path = "/home/wangdq/predict/tree/predict.tree"
    reference_path = "/home/wangdq/predict/tree/reference.tree"
    compute_accuracy(predict_path, reference_path)
