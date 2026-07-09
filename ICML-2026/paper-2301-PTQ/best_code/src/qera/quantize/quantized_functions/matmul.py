from copy import deepcopy
from functools import partial

import torch

from ..quantizers import get_quantizer

# PyTorch has torch.matmul and torch.bmm for matrix multiplication
MATMUL_MAP = {"matmul": torch.matmul, "bmm": torch.bmm}


def generic_matmul_flexible(x, y, q_config, style="matmul"):
    matmul = MATMUL_MAP[style]

    x_quantizer_config = deepcopy(q_config["x_quantizer"])
    w_quantizer_config = deepcopy(q_config["w_quantizer"])

    x_quantizer = partial(get_quantizer(x_quantizer_config.pop("name")), **x_quantizer_config)
    w_quantizer = partial(get_quantizer(w_quantizer_config.pop("name")), **w_quantizer_config)

    x = x_quantizer(x)
    y = w_quantizer(y)

    product = matmul(x, y)
    return product


def matmul_flexible(x, y, q_config):
    return generic_matmul_flexible(x, y, q_config, style="matmul")


def bmm_flexible(x, y, q_config):
    return generic_matmul_flexible(x, y, q_config, style="bmm")
