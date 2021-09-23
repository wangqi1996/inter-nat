# encoding=utf-8
import argparse
import os


def process_data(args):
    dir = args.dir
    source_file = os.path.join(dir, args.source)
    target_file = os.path.join(dir, args.target)
    result_file = os.path.join(dir, args.result)

    result = []
    with open(source_file, 'r') as f_source_file, open(target_file, 'r') as f_target_file:
        for src, trg in zip(f_source_file, f_target_file):
            line = src.strip() + " ||| " + trg.strip() + '\n'
            result.append(line)

    with open(result_file, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--source', type=str, default="source.txt")
    parser.add_argument('--target', type=str, default="target.txt")
    parser.add_argument('--result', type=str, default="source-target")

    args = parser.parse_args()
    process_data(args)
