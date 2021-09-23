# encoding=utf-8

# GLove论文是10


def occurrence_matrix():
    window_size = -1

    file = "/home/wangdq/dataset/ende_raw/fairseq/train.en-de.de"
    file2 = "/home/wangdq/dataset/ende_raw/fairseq/test.en-de.de"
    matrix_file = '/home/wangdq/dataset/ende_raw/fairseq/occurrenct.matrix.train.all'
    matrix_file2 = '/home/wangdq/dataset/ende_raw/fairseq/occurrenct.matrix.test.all'
    count = {}
    KEY = "_qw_"

    def get_key(t, token):
        return t + KEY + token

    def add(t, token):
        key = get_key(t, token)
        count.setdefault(key, 0)
        count[key] += 1

    def get_token(key):
        return key.split(KEY)

    word_count = {}
    with open(file, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            for index, token in enumerate(tokens):
                if window_size == -1:
                    window_size = index
                for i in range(max(0, index - window_size - 1), index):
                    t = tokens[i]
                    # 两边都加
                    add(t, token)
                    add(token, t)
                word_count.setdefault(token, 0)
                word_count[token] += 1

    # new_count = {}
    # for key, value in count.items():
    #     k1, k2 = get_token(key)
    #     new_count[key] = "%.3f" % (value / min(word_count[k1], word_count[k2]))
    # count = new_count

    a = sorted(count.values())
    a = [str(aa) + '\n' for aa in a]
    with open(matrix_file + '.a', 'w') as f:
        f.writelines(a)

    with open(file, 'r') as f:
        result = []

        for line in f:
            line_result = []
            tokens = line.strip().split(' ')
            for token in tokens:
                a = []
                for t in tokens:
                    kk = get_key(t, token)
                    a.append(str(count.get(kk, 0)))
                line_result.append(','.join(a))
            result.append('\t'.join(line_result) + '\n')

    with open(matrix_file, 'w') as f:
        f.writelines(result)

    with open(file2, 'r') as f:
        result = []

        for line in f:
            line_result = []
            tokens = line.strip().split(' ')
            for token in tokens:
                a = []
                for t in tokens:
                    kk = get_key(t, token)
                    a.append(str(count.get(kk, 0)))
                line_result.append(','.join(a))
            result.append('\t'.join(line_result) + '\n')

    with open(matrix_file2, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    occurrence_matrix()
