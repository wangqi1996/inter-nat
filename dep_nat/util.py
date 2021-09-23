# encoding=utf-8
import torch


def get_dep_mat(all_head, mask, dtype=torch.long):
    all_head.masked_fill_(~mask, 0)
    batch_size, tgt_len = all_head.shape

    flat_all_head = all_head.view(-1)
    add = torch.arange(0, batch_size * tgt_len * tgt_len, tgt_len).to(all_head.device)
    flat_all_head = flat_all_head + add
    same = add + torch.arange(0, tgt_len).repeat(batch_size).to(all_head.device)
    dep_mat = all_head.new_zeros((batch_size, tgt_len, tgt_len), dtype=dtype).fill_(2)

    dep_mat = dep_mat.view(-1)
    dep_mat[flat_all_head] = 1
    dep_mat[same] = 1

    dep_mat = dep_mat.view(batch_size, tgt_len, tgt_len)
    dep_mat.masked_fill_(~mask.unsqueeze(-1), 0)
    dep_mat.masked_fill_(~mask.unsqueeze(-2), 0)

    # 对称
    mask_1 = dep_mat == 1
    mask_1 = mask_1.transpose(-1, -2)
    dep_mat = dep_mat.masked_fill(mask_1, 1)

    return dep_mat


def load_dependency_head_tree(dependency_tree_path, convert=False, add_one=False, scale=2):
    """ convert表示是否弄成损失样式  """
    dependency_list = []
    print(dependency_tree_path)
    with open(dependency_tree_path, "r") as f:
        for line in f:
            heads = line.strip().split(',')
            if add_one:
                c = [int(i) + 1 for i in heads]  # add one
            else:
                c = [int(i) for i in heads]

            dependency_list.append(c)

    return dependency_list


file_mapping = {
    "iwslt16": "/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/",
    "iwslt16_raw": "/home/data_ti5_c/wangdq/data/dataset/iwslt16_raw/dependency/",
    "iwslt16_raw_middle": "/home/data_ti5_c/wangdq/data/dataset/iwslt16_raw_middle/dependency/",
    "iwslt16_raw_last": "/home/data_ti5_c/wangdq/data/dataset/iwslt16_raw_last/dependency/",
    "iwslt16_raw2": "/home/data_ti5_c/wangdq/data/dataset/iwslt16_raw/dependency/",
    "wmt14r_raw": "/home/data_ti5_c/wangdq/data/dataset/wmt14_ende_google_raw/dependency/",
    "nist": "/home/data_ti5_c/wangdq/data/dataset/nist_zh_en/dependency/",
    "nist_distill": "/home/data_ti5_c/wangdq/data/dataset/nist_zhen_distill/dependency/",
    "wmt16_enro_raw": "/home/data_ti5_c/wangdq/data/dataset/wmt16_enro_raw/dependency/",
    "wmt16_roen_raw": "/home/data_ti5_c/wangdq/data/dataset/wmt16_roen_raw/dependency/",
    "wmt16_enro_distill": "/home/data_ti5_c/wangdq/data/dataset/wmt16_enro_distill/dependency/",
    "wmt16_roen_distill": "/home/data_ti5_c/wangdq/data/dataset/wmt16_roen_distill/dependency/",
    "wmt14_ende_distill": "/home/data_ti5_c/wangdq/data/dataset/wmt14_ende_distill/dependency/",
    "wmt14_deen_distill": "/home/data_ti5_c/wangdq/data/dataset/wmt14_deen_distill/dependency/",
    "wmt14_raw": "/home/data_ti5_c/wangdq/data/dataset/fairseq_ende_distill/dependency/",
    "wmt14r_raw2": "/home/data_ti5_c/wangdq/data/dataset/wmt14_ende_google_raw2/dependency/",
    "wmt14_raw2": "/home/data_ti5_c/wangdq/data/test/wmt14_de_en/dependency_noKD/",
    "toy": "/home/data_ti5_c/wangdq/data/dataset/multi-language/",
    "toy2": "/home/data_ti5_c/wangdq/data/dataset/toy2/",
    "wmt14_enfr_raw": "/home/data_ti5_c/wangdq/data/dataset/wmt14_enfr_raw/dependency/"
}


class DepTree():

    def get_file_dir(self, dep_file):
        return file_mapping.get(dep_file, "")

    def __init__(self, valid_subset="valid", use_tree=True, only_valid=False, **kwargs):
        if use_tree:
            self.train_tree, self.valid_tree = self.get_dep_tree(valid_subset, only_valid, **kwargs)
        else:
            self.train_tree, self.valid_tree = None, None

    def get_dep_tree(self, valid_subset, only_valid=False, **kwargs):
        raise Exception("怎么是这个类呢？？？？？")

    def get_one_sentence(self, index, training):
        if training:
            return self.train_tree[index]
        else:
            return self.valid_tree[index]

    def get_sentences(self, index_list, training):
        tree = self.train_tree if training else self.valid_tree
        return [tree[id] for id in index_list]


class DepHeadTree(DepTree):

    def get_dep_tree(self, valid_subset="valid", only_valid=False, dep_file="", **kwargs):
        dir_name = self.get_file_dir(dep_file)

        if valid_subset == "test":
            only_valid = True

        if not only_valid:
            train_dependency_tree_head = load_dependency_head_tree(
                dependency_tree_path=dir_name + "dependency_head_2.train.log",
                add_one=True)  # head[i]=j， 节点i的父节点是节点j，i从0开始（下标） j从1开始。
        else:
            train_dependency_tree_head = None

        valid_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path=dir_name + "dependency_head_2." + str(
                valid_subset) + ".log",
            add_one=True)

        return train_dependency_tree_head, valid_dependency_tree_head


class RelativeDepNoSubMat(DepHeadTree):
    def get_dep_tree(self, valid_subset="valid", only_valid=False, args=None, dep_file="iwslt16", **kwargs):
        train_tree, valid_tree = super().get_dep_tree(valid_subset, only_valid, dep_file)
        train_relative_dependency_mat = self.process_mat(train_tree)
        valid_relative_dependency_mat = self.process_mat(valid_tree)

        return train_relative_dependency_mat, valid_relative_dependency_mat

    def process_mat(self, tree):
        if tree is None or len(tree) < 0:
            return None
        result = []
        for sample_id, head in enumerate(tree):
            head = torch.LongTensor([0] + head + [0])
            mask = head.new_ones(head.size()).bool().fill_(True)
            mask[0] = mask[-1] = False
            dep_mat = get_dep_mat(head.unsqueeze(0), mask.unsqueeze(0), dtype=torch.uint8).squeeze(0)
            result.append(dep_mat)

        return result

    def get_dependency_mat(self, sample_ids, reference, training=True, contain_eos=False):
        batch_size, seq_len = reference.size()
        dep_tensor = reference.new_zeros(size=reference.size()).unsqueeze(-1).repeat(1, 1, seq_len)  # pad=0

        relative_positions = self.get_sentences(sample_ids, training)
        for index, relative_dep_position in enumerate(relative_positions):
            length, _ = relative_dep_position.size()
            dep_tensor[index][:length, :length] = relative_dep_position.long()

        return dep_tensor


Comapping = {
    "iwslt16_raw": "/home/wangdq/dataset/iwslt16/occurrenct.matrix.",
    "iwslt16_raw2": "/home/data_ti5_c/wangdq/data/dataset/iwslt16_raw/occurrent/occurrenct.matrix.all.",
}


class CoTree(DepTree):

    def get_dep_tree(self, valid_subset="valid", only_valid=False, args=None, dep_file="iwslt16", **kwargs):
        self.ratio = kwargs.get("ratio", 0.2)
        print("ratio：", self.ratio)
        tree_file = Comapping.get(dep_file)
        if valid_subset == "test":
            only_valid = True
        if only_valid:
            train_mat = None
        else:
            train_mat = self.process_mat(tree_file + 'train')
        valid_mat = self.process_mat(tree_file + valid_subset)
        return train_mat, valid_mat

    def process_mat(self, tree_file):
        print(tree_file)
        mat = []
        with open(tree_file, 'r') as f:
            for line in f:
                r = line.strip().split('\t')
                r = [rr.strip().split(',') for rr in r]
                r = [[int(rrr) for rrr in rr] for rr in r]
                m = torch.tensor(r)
                number = max(round(len(r[0]) * self.ratio), 1) - 1
                sort_m, _ = m.sort(-1, descending=True)
                threshold = sort_m[:, number].unsqueeze(-1)
                m = (m >= threshold).int() + 1
                mat.append(m)
        return mat

    def get_dependency_mat(self, sample_ids, reference, training=True, **kwargs):
        batch_size, seq_len = reference.size()
        dep_tensor = reference.new_zeros(size=reference.size()).unsqueeze(-1).repeat(1, 1, seq_len)  # pad=0

        relative_positions = self.get_sentences(sample_ids, training)
        for index, relative_dep_position in enumerate(relative_positions):
            length, _ = relative_dep_position.size()
            dep_tensor[index][1:length + 1, 1:length + 1] = relative_dep_position.long()

        return dep_tensor
