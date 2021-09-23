# encoding=utf-8

dataset_file = '/home/wangdq/toy5/'
train_num = 2000001
valid_num = 1000
test_num = 1000
seq_len = 30
lang1 = 'de'  # 反正也无所谓
lang2 = 'en'

debug = False
if debug:
    train_num = 70
    valid_num = 10
    test_num = 10
    dataset_file = '../scripts/toy'
    lang1 = 'de.txt'  # 反正也无所谓
    lang2 = 'en.txt'

dict = ['1', '2', '3', '4', '5']
# mapping1 = {'1': 'a b', '2': 'd e', '3': 'i j', '4': 'l m', '5': 'x y'}
# mapping2 = {'1': 'b c', '2': 'e h', '3': 'j k', '4': 'm n', '5': 'y z'}
# mapping3 = {'1': 'a c', '2': 'd h', '3': 'i k', '4': 'l n', '5': 'x z'}

# mapping1 = {'1': 'a b c', '2': 'd e h', '3': 'i j k', '4': 'l m n', '5': 'x y z'}
# mapping2 = {'1': 'b c a', '2': 'e h d', '3': 'j k i', '4': 'm n l', '5': 'y z x'}
# mapping3 = {'1': 'c a b', '2': 'h d e', '3': 'k i j', '4': 'n l m', '5': 'z x y'}
# mapping4 = {'1': 'a c b', '2': 'd h e', '3': 'i k j', '4': 'l n m', '5': 'x z y'}
# mapping5 = {'1': 'b a c', '2': 'e d h', '3': 'j i k', '4': 'm l n', '5': 'y x z'}
# mapping6 = {'1': 'c b a', '2': 'h e d', '3': 'k j i', '4': 'n m l', '5': 'z y x'}

mapping1 = {'1': 'a b b', '2': 'd e e', '3': 'i j j', '4': 'l m m', '5': 'x y y'}
mapping2 = {'1': 'c c a', '2': 'h h d', '3': 'k k i', '4': 'n n l', '5': 'z z x'}
mapping3 = {'1': 'b a c', '2': 'e d h', '3': 'j i k', '4': 'm l n', '5': 'y x z'}

train_src = dataset_file + 'train.' + lang1
train_trg = dataset_file + 'train.' + lang2
test_src = dataset_file + 'test.' + lang1
test_trg = dataset_file + 'test.' + lang2
test_trg1 = dataset_file + 'test1.' + lang2
test_trg2 = dataset_file + 'test2.' + lang2
valid_src = dataset_file + 'valid.' + lang1
valid_trg = dataset_file + 'valid.' + lang2
valid_trg1 = dataset_file + 'valid1.' + lang2
valid_trg2 = dataset_file + 'valid2.' + lang2

index_mapping = {
    0: mapping1,
    1: mapping2,
    2: mapping3,
    # 3: mapping4,
    # 4: mapping5,
    # 5: mapping6
}


def get_trg(src, mapping):
    r = []
    for s in src.strip().split(' '):
        r.append(mapping[s])

    return " ".join(r) + '\n'


#
# def construct_toy_dataset():
#     sample_num = (train_num + test_num + valid_num) / 125
#     source = []
#     for key1 in dict:
#         for key2 in dict:
#             for key3 in dict:
#                 key = key1 + key2 + key3
#                 index = 0
#                 _source = []
#                 while index < sample_num:
#                     src = key
#                     for i in range(3, seq_len):
#                         src += random.choice(dict)
#                     if src not in _source:
#                         _source.append(src)
#                         index += 1
#                 source.extend(_source)
#
#     # split
#     train_source = []
#     train_target = []
#
#     test_source = []
#     test_target1 = []
#     test_target2 = []
#     test_target3 = []
#
#     valid_source = []
#     valid_target1 = []
#     valid_target2 = []
#     valid_target3 = []
#
#     sample_ids = random.sample(list(range(0, len(source))), valid_num * 2)
#     valid_id = [sample_ids[i * 2] for i in range(valid_num)]
#     test_id = [sample_ids[i * 2 + 1] for i in range(valid_num)]
#

#     index_mapping = {
#         0: mapping1,
#         1: mapping2,
#         2: mapping3
#     }
#     for index, src in enumerate(source):
#
#         if index in test_id:
#             trg1 = get_trg(src, mapping1)
#             trg2 = get_trg(src, mapping2)
#             trg3 = get_trg(src, mapping3)
#             src = " ".join(src) + '\n'
#             test_source.append(src)
#             test_target1.append(trg1)
#             test_target2.append(trg2)
#             test_target3.append(trg3)
#         else:
#             trg = get_trg(src, index_mapping[index % 3])
#             src = " ".join(src) + '\n'
#             if index in valid_id:
#                 valid_source.append(src)
#                 valid_target1.append(trg)
#             else:
#                 train_source.append(src)
#                 train_target.append(trg)
#
#     def write(file, content):
#         with open(file, 'w') as f:
#             f.writelines(content)
#
#     write(train_src, train_source)
#     write(train_trg, train_target)
#     write(valid_src, valid_source)
#     write(valid_trg, valid_target1)
#     write(test_src, test_source)
#     write(test_trg, test_target1)
#     write(test_trg1, test_target2)
#     write(test_trg2, test_target3)


# head_mapping = {
#     0: ','.join(['-1'] + ['0' for _ in range(89)]),
#     1: ','.join(['0' for _ in range(89)] + ['-1']),
#     2: ','.join(['0' for _ in range(44)] + ['-1'] + ['0' for _ in range(45)])
# }
#

head1 = ['-1', '0', '0']
for i in range(29):
    head1.extend(['0', str(i * 3 + 3), str(i * 3 + 3)])

head3 = ['1', '-1', '1']
for i in range(29):
    head3.extend([str(i * 3 + 3), '1', str(i * 3 + 3)])

head2 = ['0', '0', '-1']
for i in range(29):
    head2.extend([str(i * 3 + 3), str(i * 3 + 3), '2'])

head_mapping = {
    0: ','.join(head1),
    1: ','.join(head2),
    2: ','.join(head3)
}


# print(head_mapping)


def construct_dep_tree():
    split = "valid"
    trg_file = dataset_file + split + '.' + lang2
    dep_file = dataset_file + 'dependency_head_2.' + split + '.log'
    dep_result = []

    with open(dep_file, 'w') as f_dep, open(trg_file, 'r') as f_trg:
        for index, line in enumerate(f_trg):
            head = head_mapping[index % 3]
            dep_result.append(head + '\n')
            if len(dep_result) > 700:
                f_dep.writelines(dep_result)
                dep_result = []

        if len(dep_result) > 0:
            f_dep.writelines(dep_result)


def construct_toy_dataset2():
    split = "train"
    trg_file = dataset_file + split + '.' + lang2
    src_file = dataset_file + split + '.' + lang1
    target_tokens = []

    with open(trg_file, 'w') as f_trg, open(src_file, 'r') as f_src:
        for index, src in enumerate(f_src):
            trg = get_trg(src, index_mapping[index % 3])
            target_tokens.append(trg)

        if len(target_tokens) > 0:
            f_trg.writelines(target_tokens)


if __name__ == '__main__':
    construct_toy_dataset2()
    # construct_dep_tree()
