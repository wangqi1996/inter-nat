# encoding=utf-8
import os
import torch


def get_dep_mat(all_head, mask, dtype=torch.long):
    """父节点"""
    all_head.masked_fill_(~mask, 0)
    batch_size, tgt_len = all_head.shape

    flat_all_head = all_head.view(-1)
    add = torch.arange(0, batch_size * tgt_len * tgt_len, tgt_len).to(all_head.device)
    flat_all_head = flat_all_head + add
    same = add + torch.arange(0, tgt_len).repeat(batch_size).to(all_head.device)
    dep_mat = all_head.new_zeros((batch_size, tgt_len, tgt_len), dtype=dtype).fill_(0)

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


# def get_dep_mat(all_head, mask, dtype=torch.uint8):
#     """ 祖父节点 """
#     all_head.masked_fill_(~mask, 0)
#     batch_size, tgt_len = all_head.shape
#
#     flat_all_head = all_head.view(-1)
#     add = torch.arange(0, batch_size * tgt_len * tgt_len, tgt_len).to(all_head.device)
#     flat_all_head = flat_all_head + add
#     dep_mat = all_head.new_zeros((batch_size, tgt_len, tgt_len), dtype=dtype).fill_(0)
#
#     dep_mat = dep_mat.view(-1)
#     dep_mat[flat_all_head] = 1
#
#     dep_mat = dep_mat.view(batch_size, tgt_len, tgt_len)
#     dep_mat.masked_fill_(~mask.unsqueeze(-1), 0)
#     dep_mat.masked_fill_(~mask.unsqueeze(-2), 0)
#     # 祖父节点
#     dep_mat = torch.matmul(dep_mat.float(), dep_mat.float()).to(dtype) + dep_mat
#
#     # 对角线 & 对称
#     eye_tensor = torch.eye(tgt_len, tgt_len).repeat(batch_size, 1, 1).to(dep_mat)
#     dep_mat = (dep_mat + eye_tensor)
#     dep_mat = dep_mat + dep_mat.transpose(-1, -2)
#     dep_mat = dep_mat.clip(0, 1)
#     dep_mat.masked_fill_(~mask.unsqueeze(-1), 0)
#     dep_mat.masked_fill_(~mask.unsqueeze(-2), 0)
#
#     return dep_mat


def load_dependency_head_tree(tree_path):
    dependency_list = []
    print(tree_path)
    with open(tree_path, "r") as f:
        for line in f:
            heads = line.strip().split(',')
            c = [int(i) for i in heads]  # contain the bos and eos token
            dependency_list.append(c)

    return dependency_list


DependencyFileMapping = {
    "iwslt16_deen_raw": "/home/wangdq/dependency/iwslt16-deen/",
}


class Tree():
    def __init__(self, valid_subset="valid", dep_file="", **kwargs):
        dir_name = self.get_file_dir(dep_file)

        if valid_subset != "test":
            self.train_tree = load_dependency_head_tree(os.path.join(dir_name, "train.tree"))
        else:
            self.train_tree = None

        self.valid_tree = load_dependency_head_tree(os.path.join(dir_name, valid_subset + ".tree"))

    def get_file_dir(self, dep_file):
        return DependencyFileMapping.get(dep_file, "")

    def get_sentences(self, index_list, training):
        tree = self.train_tree if training else self.valid_tree
        return [tree[id] for id in index_list]


class ParentRelationMat():
    def __init__(self, valid_subset="valid", dep_file="", **kwargs):
        tree = Tree(valid_subset, dep_file)
        self.train_mat = self.process_mat(tree.train_tree)
        self.valid_mat = self.process_mat(tree.valid_tree)

    def process_mat(self, tree):
        if tree is None or len(tree) <= 0:
            return None
        result = []
        for sample_id, head in enumerate(tree):
            head = torch.LongTensor(head)
            mask = head.new_ones(head.size()).bool().fill_(True)
            dep_mat = get_dep_mat(head.unsqueeze(0), mask.unsqueeze(0), dtype=torch.uint8).squeeze(0)
            result.append(dep_mat)

        return result

    def get_relation_mat(self, sample_ids, reference, training=True):
        batch_size, seq_len = reference.size()
        dep_tensor = torch.eye(seq_len, seq_len).repeat(batch_size, 1, 1).to(reference)

        mat = self.train_mat if training else self.valid_mat
        relations = [mat[id] for id in sample_ids]
        for index, relation in enumerate(relations):
            length, _ = relation.size()
            dep_tensor[index][:length, :length] = relation.long()

        return dep_tensor
