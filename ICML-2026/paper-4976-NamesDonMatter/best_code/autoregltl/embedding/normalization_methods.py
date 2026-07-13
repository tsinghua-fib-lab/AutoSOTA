import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


func_list = {
    "disabled": lambda x: x,
    "l2": partial(F.normalize, dim=-1),
}
register_func = lambda f: func_list.setdefault(f.__name__, f)


def get_normalization_method(name):
    return func_list[name]