"""Transformer with pair-feature attention bias (bpps, structure adjacency/distances) and dynamic
relative-position bias, plus a depthwise-conv module per layer. PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPositionBias(nn.Module):
    """Per-head bias from an MLP of the signed, log-scaled relative distance (length-extrapolating)."""

    def __init__(self, heads, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, heads))

    def forward(self, L, device):
        r = torch.arange(L, device=device)
        d = (r[None, :] - r[:, None]).float()
        d = torch.sign(d) * torch.log1p(d.abs())
        return self.mlp(d[..., None]).permute(2, 0, 1)  # [H, L, L]


class Layer(nn.Module):
    def __init__(self, d, heads, pair_dim, kernel=7, dropout=0.1, n_adj=0):
        super().__init__()
        self.h, self.dh = heads, d // heads
        if n_adj:
            self.ln_g = nn.LayerNorm(d)
            self.gnn = nn.Linear(n_adj * d, d)
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.pair = nn.Linear(pair_dim, heads)
        self.ln2 = nn.LayerNorm(d)
        self.conv_in = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.conv_bn = nn.BatchNorm1d(d)
        self.conv_out = nn.Linear(d, d)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pair, pos, adj=None):
        B, L, D = x.shape
        if adj is not None:
            msg = torch.einsum("blmc,bmd->blcd", adj, self.ln_g(x)).reshape(B, L, -1)
            x = x + self.drop(self.gnn(msg))
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att + self.pair(pair).permute(0, 3, 1, 2) + pos[None]
        att = self.drop(att.softmax(-1))
        x = x + self.drop(self.out((att @ v).transpose(1, 2).reshape(B, L, D)))

        h = F.glu(self.conv_in(self.ln2(x)), dim=-1).transpose(1, 2)
        h = F.silu(self.conv_bn(self.dw(h))).transpose(1, 2)
        x = x + self.drop(self.conv_out(h))
        return x + self.drop(self.ff(self.ln3(x)))


class RNATransformer(nn.Module):
    def __init__(self, n_feat, n_adj, d=192, heads=6, layers=8, pair_dim=32, kernel=7, dropout=0.1, n_out=5,
                 gnn=False):
        super().__init__()
        self.gnn = gnn
        self.inp = nn.Sequential(nn.Linear(n_feat, d), nn.LayerNorm(d))
        self.pair = nn.Sequential(nn.Conv2d(n_adj, pair_dim, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(pair_dim, pair_dim, 3, padding=1))
        self.pos = nn.ModuleList([DynamicPositionBias(heads) for _ in range(layers)])
        self.layers = nn.ModuleList([Layer(d, heads, pair_dim, kernel, dropout, n_adj if gnn else 0)
                                     for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def forward(self, x, a):
        """x [B, L+2, F], a [B, L+2, L+2, C] -> [B, L, 5] (start/end tokens dropped)."""
        return self.head(self.encode(x, a))[:, 1:-1]

    def encode(self, x, a):
        """-> [B, L+2, d] final hidden states."""
        L = x.shape[1]
        pair = self.pair(a.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        h = self.inp(x)
        adj = a / (a.sum(2, keepdim=True) + 1) if self.gnn else None
        for layer, pos in zip(self.layers, self.pos):
            h = layer(h, pair, pos(L, x.device), adj)
        return self.ln(h)
