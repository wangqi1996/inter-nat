# encoding=utf-8


"""
grep ^H test/generate-test.txt | cut -f3- > pre.txt
grep ^T test3/generate-test.txt | cut -f2- > ref.txt

"""


def count_mode():
    file = ""
    head_9 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    tail_9 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

def accuracy():
    file2 = '/home/data_ti5_c/wangdq/code/dep_nat/pre.txt'
    file1 = '/home/data_ti5_c/wangdq/code/dep_nat/ref.txt'
    sen_all, sen_correct = 0, 0
    word_all, word_correct = 0, 0

    head_9 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    tail_9 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    def remove_9(text, count_c=True):
        text = text.strip()
        text = ''.join(text.split(' '))
        index = 0
        c = 0
        while text[index] == '9':
            index += 1
            c += 1
        if count_c:
            head_9[c] += 1

        index2 = len(text) - 1
        c = 0
        while text[index2] == '9':
            index2 -= 1
            c += 1
        if count_c:
            tail_9[c] += 1
        return text[index: index2 + 1]

    with open(file1, 'r') as f_ref, open(file2, 'r') as f_pre:
        for ref, pre in zip(f_ref, f_pre):

            ref = remove_9(ref, count_c=False)
            pre = remove_9(pre, count_c=True)
            if ref == pre:
                sen_correct += 1
            # else:
            #     print(pre)
            #     print(ref)
            sen_all += 1

            if len(ref) == len(pre):
                for t_ref, t_pre in zip(ref, pre):
                    if t_ref == t_pre:
                        word_correct += 1
                    word_all += 1
    print(sen_all)
    print(sen_correct)
    print(sen_correct / sen_all)

    print(word_all)
    print(word_correct)
    print(word_correct / word_all)

    print(head_9)
    print(tail_9)


def compare(a, b):
    t = """eval_bleu_args='{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'"""
    a = a.strip().replace(t, '').split(', ')
    b = b.strip().replace(t, '').split(', ')
    a_key = {}
    b_key = {}
    for aa in a:
        if len(aa.strip()) <= 0:
            continue
        a1, a2 = aa.strip().split('=')
        a_key[a1.strip()] = a2.strip()

    for bb in b:
        if len(bb.strip()) <= 0:
            continue
        b1, b2 = bb.strip().split('=')
        b_key[b1.strip()] = b2.strip()

    for a_k, a_v in a_key.items():
        if a_k in b_key and a_v == b_key[a_k]:
            continue
        if a_k in ['distributed_init_method', 'distributed_num_procs', 'distributed_world_size', 'nprocs_per_node',
                   'save_dir', 'tensorboard_logdir']:
            continue
        print(a_k, ' ', a_v)

    print('-------------------')

    for a_k, a_v in b_key.items():
        if a_k in a_key and a_v == a_key[a_k]:
            continue
        if a_k in ['distributed_init_method', 'distributed_num_procs', 'distributed_world_size', 'nprocs_per_node',
                   'save_dir', 'tensorboard_logdir']:
            continue
        print(a_k, ' ', a_v)


if __name__ == '__main__':
    accuracy()
