# encoding=utf-8
import argparse


def process_data(args):
    file1 = args.dir + args.source
    file2 = args.dir + args.target
    result_file = args.dir + args.result

    result = []
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        for src, trg in zip(f1, f2):
            line = src.strip() + " ||| " + trg.strip() + '\n'
            result.append(line)

    with open(result_file, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--source', type=str, default="source.txt")
    parser.add_argument('--target', type=str, default="target.txt")
    parser.add_argument('--result', type=str, default="source-target")

    args = parser.parse_args()
    process_data(args)
