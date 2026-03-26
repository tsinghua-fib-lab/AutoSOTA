import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple
from dataclasses import dataclass

class SparseEmbedding(nn.Module):
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 num_emb_layers: int, 
                 dim: int, 
                 ):
        super(SparseEmbedding, self).__init__()
        self.length = input_size + output_size
        self.num_emb_layers = num_emb_layers 
        self.dim = dim
        self.register_buffer('L', 2 ** torch.arange(self.num_emb_layers))
        self.register_buffer('pad_L', (self.length + self.L - 1) // self.L * self.L )
        self.layers = nn.ModuleList([nn.Linear(self.L[t], self.dim) for t in range(self.num_emb_layers)])
        self.norms = nn.ModuleList([nn.InstanceNorm1d(self.pad_L[t] // self.L[t]) for t in range(self.num_emb_layers)])
        self.activation = nn.GELU()
        self.linear = nn.Linear(self.dim * self.num_emb_layers, self.dim)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B*C, 1, L)
        xs = []
        for t in range(self.L.shape[0]):
            x_t = F.pad(x, (0, self.pad_L[t] - self.length))
            x_t = x_t.reshape(B*C, -1, self.L[t])
            if x_t.shape[-1] > 1:
                x_t = self.norms[t](x_t)
            x_t = self.layers[t](x_t).repeat(1, self.L[t], 1)[:, :L, :]
            xs.append(x_t)
        x_emb = torch.cat(xs, dim=-1)
        x_emb = self.linear(self.activation(x_emb))
        out = rearrange(x_emb, '(b c) l d -> b c l d', c=C)
        return out

class ConvEmbedding(nn.Module):
    def __init__(self, 
                 num_emb_layers: int, 
                 dim: int, 
                 dim_multiplier: int, 
                 ):
        super(ConvEmbedding, self).__init__()
        self.num_layers = num_emb_layers
        self.dim = dim
        self.dim_mutliplier = dim_multiplier

        now_channel = self.dim_mutliplier
        self.convs = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.activation = nn.SiLU()
        for _ in range(self.num_layers):
            conv = nn.Conv1d(now_channel, now_channel * 2, kernel_size=3, stride=2, padding=1)
            self.convs.append(conv)
            self.norms.append(nn.InstanceNorm1d(now_channel))
            now_channel *= 2

        self.linear = nn.Linear((self.num_layers + 1) * self.dim_mutliplier, self.dim)

    def forward(self, x):
        B, C, L = x.shape
        if L % 2 ** self.num_layers != 0:
            x = F.pad(x, (0, 2 ** self.num_layers - L % 2 ** self.num_layers))
        x = x.reshape(B*C, 1, -1).repeat(1, self.dim_mutliplier, 1)
        xs = [rearrange(x, 'b c l -> b l c')]
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x)
            x = conv(x)
            x = self.activation(x)
            x_flatten = rearrange(x, 'b (a c) l -> b (l c) a', a=self.dim_mutliplier)
            xs.append(x_flatten)
        x_emb = torch.stack(xs, dim=-1)
        x_emb = rearrange(x_emb, '(b c) l a n -> b c l (a n)', c=C)
        if x_emb.shape[-1] % 2 ** self.num_layers != 0:
            x_emb = x_emb[..., :L, :]
        out = self.linear(x_emb)
        return out
