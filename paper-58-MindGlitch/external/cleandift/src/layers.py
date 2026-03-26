import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from .utils import zero_init


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm=None, dropout=0.0):
        super().__init__()
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond_norm=None, **kwargs):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip

class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)
    

def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)

class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([FeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x
    

def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)

class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_swiglu(x, self.weight, self.bias)

    
class Linear(nn.Linear):
    def forward(self, x):
        return super().forward(x)
    

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)