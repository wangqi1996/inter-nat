"""
1. 预处理数据


SCRIPTS=/home/data_ti5_c/wangdq/code/mosesdecoder/scripts/
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=/home/data_ti5_c/wangdq/code/subword-nmt/subword_nmt
BPE_TOKENS=37000


# tokenizer
cat europarl-v7.de-en.en | perl $NORM_PUNC en | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l en > train.de-en.en
cat europarl-v7.fr-en.en | perl $NORM_PUNC en | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l en > train.fr-en.en
cat europarl-v7.es-en.en | perl $NORM_PUNC en | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l en > train.es-en.en
cat europarl-v7.de-en.de | perl $NORM_PUNC de | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l de > train.de-en.de
cat europarl-v7.fr-en.fr | perl $NORM_PUNC fr | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l fr > train.fr-en.fr
cat europarl-v7.es-en.es | perl $NORM_PUNC es | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l es > train.es-en.es

# 脚本


# bpe:
cat train.en train.de train.es train.fr > train
echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < train > dict.en.txt

python $BPEROOT/apply_bpe.py -c dict.en.txt < train.en > train.de-en.en
python $BPEROOT/apply_bpe.py -c dict.en.txt < train.de > train.de-en.de
python $BPEROOT/apply_bpe.py -c dict.en.txt < train.fr > train.fr-en.fr
python $BPEROOT/apply_bpe.py -c dict.en.txt < train.es > train.es-en.es

cp train.de-en.en train.fr-en.en
cp train.de-en.en train.es-en.en

# clean
perl $CLEAN -ratio 1.5 train.de-en en de train1 1 250
perl $CLEAN -ratio 1.5 train.fr-en en fr train2 1 250
perl $CLEAN -ratio 1.5 train.es-en en es train3 1 250

# 运行脚本
mv train1.en train.de-en.en
mv train1.de train.de-en.de
mv train2.en train.fr-en.en
mv train2.fr train.fr-en.fr
mv train3.en train.es-en.en
mv train3.es train.es-en.es

# 顺序
de fr es

"""


def process():
    source = []
    target = [[], [], []]

    file1 = '/home/wangdq/dataset/train.de-en.en'
    file2 = '/home/wangdq/dataset/train.es-en.en'
    file3 = '/home/wangdq/dataset/train.fr-en.en'
    t_file1 = '/home/wangdq/dataset/train.de-en.de'
    t_file2 = '/home/wangdq/dataset/train.es-en.es'
    t_file3 = '/home/wangdq/dataset/train.fr-en.fr'
    result_src = '/home/wangdq/dataset/train.en'
    result_trg = '/home/wangdq/dataset/train.'

    def process(s_file, t_file):
        d = {}
        with open(s_file, 'r') as f_src, open(t_file, 'r') as f_trg:
            for src, trg in zip(f_src, f_trg):
                d[src.strip()] = trg.strip() + '\n'
        return d

    f1 = process(file1, t_file1)
    f2 = process(file2, t_file2)
    f3 = process(file3, t_file3)

    for src, trg in f1.items():
        if src in f2 and src in f3:
            target[0].append(trg)
            target[1].append(f2[src])
            target[2].append(f3[src])
            src = src + '\n'
            source.append(src)

    with open(result_src, 'w') as f_src:
        f_src.writelines(source)
    with open(result_trg + 'de', 'w') as f_trg:
        f_trg.writelines(target[0])
    with open(result_trg + 'es', 'w') as f_trg:
        f_trg.writelines(target[1])
    with open(result_trg + 'fr', 'w') as f_trg:
        f_trg.writelines(target[2])


def process_valid():
    # file1 = '/home/wangdq/dataset/dev/test.de'
    # file2 = '/home/wangdq/dataset/dev/test.fr'
    # file3 = '/home/wangdq/dataset/dev/test.es'
    # file4 = '/home/wangdq/dataset/dev/test.trg'

    split = "test"
    file1 = '/home/wangdq/dataset/data-bin-ende/dependency_head_2.' + split + '.log'
    file2 = '/home/wangdq/dataset/data-bin-fren/dependency_head_2.' + split + '.log'
    file3 = '/home/wangdq/dataset/data-bin-esen/dependency_head_2.' + split + '.log'
    file4 = '/home/wangdq/dataset/dependency_head_2.' + split + '.log'

    result = []
    with open(file1) as f1, open(file2) as f2, open(file3) as f3:
        for index, (l1, l2, l3) in enumerate(zip(f1, f2, f3)):
            l = [l1, l2, l3]  # de fr es
            l = l[index % 3]
            result.append(l)

    with open(file4, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    process_valid()
