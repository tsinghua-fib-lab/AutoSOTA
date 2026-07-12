import torch
from torch import nn
import numpy as np
# from torchmeta.modules import (MetaModule, MetaSequential)
# from torchmeta.modules.utils import get_subdict


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=8, is_first=False, is_last=False):   #%%%%%%%%%% default 10
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def siren_model(num_layers, input_dim, hidden_dim,out_dim,w0):
    layers = [SirenLayer(input_dim, hidden_dim, w0=w0, is_first=True)]
    for i in range(1, num_layers - 1):
        layers.append(SirenLayer(hidden_dim, hidden_dim, w0=w0))
    layers.append(SirenLayer(hidden_dim, out_dim, w0=w0, is_last=True))

    return nn.Sequential(*layers)