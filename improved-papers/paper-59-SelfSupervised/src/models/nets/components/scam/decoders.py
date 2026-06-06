import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple, Optional
from torch.func import functional_call, stack_module_state

def replace_name(name):
    return name.replace('.', '_')


class Decoder(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_series: int, 
                 ):
        super(Decoder, self).__init__()
        self.d_h = dim // num_series
        self.w = nn.Linear(dim, dim)
        self.linears = nn.ModuleList([nn.Linear(self.d_h, 1)] * num_series)
        params, _ = stack_module_state(self.linears)
        self.stack_params = nn.ParameterDict({replace_name(name): nn.Parameter(param) for name, param in params.items()})
        self.activation = nn.SiLU()

    def linear_call(self, params, x):
        return functional_call(self.linears[0], (params, ), (x,))

    def forward(self, x):
        x = self.activation(x)
        x = rearrange(x, 'b c l (n h) -> b c l n h', h=self.d_h)
        x_dec = torch.vmap(self.linear_call, (0, -2))(dict(self.stack_params), x)
        x_dec = rearrange(x_dec, 'n b c l d -> b c n (l d)')
        return x_dec