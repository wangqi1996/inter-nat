# encoding=utf-8
from collections import defaultdict

dirname = "/home/wangdq/dependency/test/"

import os
import random


def load_dependency_by_length(hint="test"):
    dependency = defaultdict(list)
    from shutil import copyfile
    copyfile("/home/wangdq/dependency/iwslt16-deen-bak/test.tree", dirname + "test.tree")
    os.system("wc -l /home/wangdq/dependency/test/test.tree")
    train_filename = os.path.join(dirname, "train.tree")
    with open(train_filename) as f:
        for line in f:
            length = len(line.strip().split(','))
            dependency[length].append(line)

    test_filename = os.path.join(dirname, "test.tree")
    new_dependency = {}
    for k, v in dependency.items():
        new_dependency[k] = random.sample(v, 1)[0]

    content = []
    with open(test_filename) as f:
        for id, line in enumerate(f):
            length = len(line.strip().split(","))

            if length not in dependency:
                print(id)
                print(length)
                content.append("0,1\n")
                continue

            content.append(new_dependency[length])

    with open(test_filename + hint, 'w') as f:
        f.writelines(content)


if __name__ == '__main__':
    load_dependency_by_length('')
