import torch
from torch import nn


def init_variance_scaling_(weight, scale_dim: int):
    scale_dim = torch.tensor(scale_dim)
    nn.init.normal_(weight, std=1 / torch.sqrt(scale_dim))


def init_linear_(linear: nn.Linear):
    init_variance_scaling_(linear.weight, linear.in_features)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_gru_cell_(cell: nn.GRUCell, scale_dim: int = None):
    if scale_dim is None:
        ih_scale = cell.input_size
        hh_scale = cell.hidden_size
    else:
        ih_scale = hh_scale = scale_dim
    init_variance_scaling_(cell.weight_ih, ih_scale)
    init_variance_scaling_(cell.weight_hh, hh_scale)
    nn.init.ones_(cell.bias_ih)
    cell.bias_ih.data[-cell.hidden_size :] = 0.0
    # NOTE: these weights are not present in TF
    nn.init.zeros_(cell.bias_hh)
