# encoding=utf-8
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
