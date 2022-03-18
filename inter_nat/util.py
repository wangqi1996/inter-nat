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
    "iwslt16-deen-raw": "",
}


class Tree():
    def __init__(self, valid_subset="valid", only_valid=False, dep_file="", **kwargs):
        dir_name = self.get_file_dir(dep_file)

        if valid_subset == "test":
            only_valid = True

        if not only_valid:
            self.train_tree = load_dependency_head_tree(dir_name + "train.tree")
        else:
            self.train_tree = None

        self.valid_tree = load_dependency_head_tree(dir_name + "." + valid_subset + ".tree")

    def get_file_dir(self, dep_file):
        return DependencyFileMapping.get(dep_file, "")

    def get_sentences(self, index_list, training):
        tree = self.train_tree if training else self.valid_tree
        return [tree[id] for id in index_list]


class ParentRelationMat():
    def __init__(self, valid_subset="valid", only_valid=False, dep_file="", **kwargs):
        tree = Tree(valid_subset, only_valid, dep_file)
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
        dep_tensor = reference.new_zeros(size=reference.size()).unsqueeze(-1).repeat(1, 1, seq_len)

        mat = self.train_mat if training else self.valid_mat
        relations = [mat[id] for id in sample_ids]
        for index, relation in enumerate(relations):
            length, _ = relation.size()
            dep_tensor[index][:length, :length] = relation.long()

        return dep_tensor
