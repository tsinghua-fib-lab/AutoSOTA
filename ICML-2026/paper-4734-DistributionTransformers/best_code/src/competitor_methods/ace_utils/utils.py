# Taken from https://github.com/acerbilab/amortized-conditioning-engine/blob/main/src/model/utils.py
import torch
import torch.nn as nn
import math


def mlp_weight_damper(module, depth, factor):
    """
    input nn.Sequential module list
    damper the weight by a factor to the power of (1 / number of layer)
    """
    factor = factor ** (1 / (depth - 1))
    for _, layer in enumerate(module):
        if isinstance(layer, nn.Linear):
            # change init mode here

            layer.weight = torch.nn.Parameter(layer.weight * factor)
            layer.bias = torch.nn.Parameter(layer.bias * factor)


def positional_encoding_init(seq_len, d, n):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d / 2)):
            period = ((d / 2) / (2 * torch.pi)) / (k + 1)
            phase = 0  # (2*torch.pi*(d/2))* k/(seq_len+1)
            P[k, 2 * i] = torch.sin((i / period) + phase)
            P[k, 2 * i + 1] = torch.cos((i / period) + phase)
    return torch.nn.Parameter(P)


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth - 2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class SkipMLP(nn.Module):
    def __init__(self, dim_in, dim_out, nonlinear_layer):
        super(SkipMLP, self).__init__()
        self.linear_layer = nn.Linear(dim_in, dim_out)
        self.nonlinear_layer = nonlinear_layer

    def forward(self, x):
        linear = self.linear_layer(x)
        out = self.nonlinear_layer(linear)
        out += linear  # Adding the skip connection
        return out


def build_mlp_with_linear_skipcon(
    dim_in, dim_hid, dim_out, depth, weight_damper_factor=0.1
):
    nonlinear_layer = build_mlp(dim_out, dim_hid, dim_out, depth - 1)
    mlp_weight_damper(nonlinear_layer, depth=depth, factor=weight_damper_factor)
    skip_mlp = SkipMLP(dim_in, dim_out, nonlinear_layer)

    return skip_mlp


class AttrDict(dict):
    """Custom dictionary class that supports attribute-style access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


def initialize_head(
    d_model: int,
    dim_feedforward: int,
    dim_y: int,
    single_head: bool,
    num_components: int,
) -> nn.Module:
    """
    Initializes the model with either a single head or multiple heads based on the `single_head` flag.

    Parameters:
    - d_model (int): The input dimension.
    - dim_feedforward (int): The dimension of the feedforward network.
    - dim_y (int): The output dimension.
    - single_head (bool): Flag to determine whether to initialize a single head or multiple heads.
    - num_components (int): The number of components if `single_head` is False.

    Returns:
    - nn.Module: The initialized model head(s).
    """
    if single_head & num_components > 1:
        output_dim = dim_y * 3
    else:
        output_dim = dim_y * 2

    if single_head:
        model = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, output_dim),
        )
    else:
        model = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, dim_y * 3),
                )
                for _ in range(num_components)
            ]
        )
    return model