# encoding=utf-8


import random

samples_num = 2002000
dataset_file = '/home/wangdq/toy/'
valid_num = 1000

debug = False
if debug:
    samples_num = 100
    dataset_file = '../scripts/toy'
    valid_num = 10

dict = ['1', '2', '3', '4', '5']
mapping = {'1': '1',
           '2': '22',
           '3': '333',
           '4': '4444',
           '5': '55555'}

len_0 = 4
pad_0 = {
    0: ('', '0000'),
    1: ('0', '000'),
    2: ('00', '00'),
    3: ('000', '0'),
    4: ('0000', '')
}

lang1 = 'de'  # 反正也无所谓
lang2 = 'en'
source_len = 30

source_file = dataset_file + 'src'
target_file = dataset_file + 'trg'

train_src = dataset_file + 'train.' + lang1
train_trg = dataset_file + 'train.' + lang2
test_src = dataset_file + 'test.' + lang1
test_trg = dataset_file + 'test.' + lang2
valid_src = dataset_file + 'valid.' + lang1
valid_trg = dataset_file + 'valid.' + lang2


def construct_toy_dataset():
    index = 0
    source = []
    target = []
    source_key = []
    with open(source_file, 'w') as f_source, open(target_file, 'w') as f_target:
        while index < samples_num:
            sample = ''
            for j in range(source_len):
                sample += random.choice(dict)
            if sample not in source_key:
                source_key.append(sample)
                result = ''
                for s in sample:
                    result += mapping.get(s)
                front_0 = random.randint(0, 4)
                _pad_0_f, _pad_0_t = pad_0[front_0]
                result = _pad_0_f + result + _pad_0_t

                source.append(sample + '\n')
                target.append(result + '\n')
                index += 1

            if len(source) > 700:
                f_source.writelines(source)
                f_target.writelines(target)

                source = []
                target = []
                print(index)

        if len(source) > 0:
            f_source.writelines(source)
            f_target.writelines(target)


def construct_toy_dataset2():
    source = []
    target = []
    # 两重循环
    with open(source_file, 'w') as f_source, open(target_file, 'w') as f_target:
        for key1 in dict:
            for key2 in dict:
                for key3 in dict:
                    source_key = []
                    index = 0
                    while index < samples_num / 125:
                        sample = key1 + key2 + key3
                        for j in range(3, source_len):
                            sample += random.choice(dict)
                        if sample not in source_key:
                            source_key.append(sample)
                            result = ''
                            for s in sample:
                                result += mapping.get(s)
                            front_0 = random.randint(0, 4)
                            _pad_0_f, _pad_0_t = pad_0[front_0]
                            result = _pad_0_f + result + _pad_0_t

                            source.append(sample + '\n')
                            target.append(result + '\n')
                            index += 1

                        if len(source) > 700:
                            f_source.writelines(source)
                            f_target.writelines(target)

                            source = []
                            target = []
                            print(index)

        if len(source) > 0:
            f_source.writelines(source)
            f_target.writelines(target)


def split_dataset():
    train_source = []
    train_target = []

    test_source = []
    test_target = []

    valid_source = []
    valid_target = []

    sample_ids = random.sample(list(range(0, samples_num)), valid_num * 2)
    valid_id = [sample_ids[i * 2] for i in range(valid_num)]
    test_id = [sample_ids[i * 2 + 1] for i in range(valid_num)]
    print(valid_id)
    print(test_id)

    with open(source_file, 'r') as f_source, open(target_file, 'r') as f_target:
        with open(train_src, 'w') as f_train_src, open(train_trg, 'w') as f_train_trg:
            for index, (src, trg) in enumerate(zip(f_source, f_target)):
                if index in valid_id:
                    valid_source.append(src)
                    valid_target.append(trg)
                elif index in test_id:
                    test_source.append(src)
                    test_target.append(trg)
                else:
                    train_source.append(src)
                    train_target.append(trg)

                if len(train_source) > 700:
                    f_train_src.writelines(train_source)
                    f_train_trg.writelines(train_target)
                    train_source = []
                    train_target = []
                    print(index)

            if len(train_source) > 0:
                f_train_src.writelines(train_source)
                f_train_trg.writelines(train_target)

    with open(valid_src, 'w') as f_valid_src, open(valid_trg, 'w') as f_valid_trg:
        f_valid_src.writelines(valid_source)
        f_valid_trg.writelines(valid_target)

    with open(test_src, "w") as f_test_src, open(test_trg, 'w') as f_test_trg:
        f_test_src.writelines(test_source)
        f_test_trg.writelines(test_target)


def construct_dep_tree():
    split = "train"
    trg_file = dataset_file + split + '.' + lang2 + ".raw"
    dep_file = dataset_file + 'dependency_head_2.' + split + '.log'
    dep_result = []

    with open(dep_file, 'w') as f_dep, open(trg_file, 'r') as f_trg:
        for line in f_trg:
            line = line.strip()
            head = ['-1']
            pre_index = 0
            root = '0'
            for index, token in enumerate(line[1:]):
                # if token in ['0', '1']:
                #     head.append(root)
                #     pre_index = -1
                #     continue

                if token in ['0', '9']:
                    head.append('-1')
                    pre_index = -1
                    continue
                if token == '1':
                    head.append(root)
                    pre_index = -1
                    continue

                if pre_index == -1:
                    head.append(root)
                    pre_index = index + 1
                else:
                    head.append(str(pre_index))
                    if index - pre_index + 2 == int(token):
                        pre_index = -1
            dep_result.append(','.join(head) + '\n')

            if len(dep_result) > 700:
                f_dep.writelines(dep_result)
                dep_result = []

        if len(dep_result) > 0:
            f_dep.writelines(dep_result)


def tokenizer():
    src_result = []
    tgt_result = []

    def process(text):
        text = text.strip()
        text = ' '.join(text) + '\n'
        return text

    split = 'train'
    trg_file = dataset_file + split + '.' + lang2
    src_file = dataset_file + split + '.' + lang1

    new_src_file = src_file + '.tok'
    new_trg_file = trg_file + '.tok'
    with open(new_src_file, 'w') as f_n_src, open(new_trg_file, 'w') as f_n_trg:
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for src, trg in zip(f_src, f_trg):
                new_src = process(src)
                new_trg = process(trg)
                src_result.append(new_src)
                tgt_result.append(new_trg)

            if len(src_result) > 700:
                f_n_src.writelines(src_result)
                f_n_trg.writelines(tgt_result)
                src_result = []
                tgt_result = []

        if len(src_result) > 0:
            f_n_src.writelines(src_result)
            f_n_trg.writelines(tgt_result)


def replace_0():
    src_result = []
    tgt_result = []
    split = 'train'
    trg_file = dataset_file + split + '.' + lang2
    src_file = dataset_file + split + '.' + lang1

    def process(text: str):
        return text.replace('0', '9')

    new_src_file = src_file + '.tok'
    new_trg_file = trg_file + '.tok'
    with open(new_src_file, 'w') as f_n_src, open(new_trg_file, 'w') as f_n_trg:
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for src, trg in zip(f_src, f_trg):
                new_src = process(src)
                new_trg = process(trg)
                src_result.append(new_src)
                tgt_result.append(new_trg)

            if len(src_result) > 700:
                f_n_src.writelines(src_result)
                f_n_trg.writelines(tgt_result)
                src_result = []
                tgt_result = []

        if len(src_result) > 0:
            f_n_src.writelines(src_result)
            f_n_trg.writelines(tgt_result)


if __name__ == '__main__':
    # replace_0()
    construct_dep_tree()
# construct_toy_dataset2()
# construct_toy_dataset()
# split_dataset()
