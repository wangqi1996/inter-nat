# encoding=utf-8
import torch


def init_global_count_tokens():
    global KEY_VALUE
    KEY_VALUE = {}
    global KEY_VALUE_LIST
    KEY_VALUE_LIST = {}


def set_key_value(key, value):
    global KEY_VALUE

    r = KEY_VALUE.get(key, 0)
    KEY_VALUE[key] = value + r


def get_key_value():
    return KEY_VALUE


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def get_base_mask(target_tokens):
    """ mask=True 表示不是特殊字符"""
    pad = 0
    bos = 1
    eos = 2

    target_masks = target_tokens.ne(pad) & \
                   target_tokens.ne(bos) & \
                   target_tokens.ne(eos)

    return target_masks
