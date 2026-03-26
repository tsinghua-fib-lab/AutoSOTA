import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        norm = lambda x: RMSNorm(x)
        self.attn_norm = norm(embed_dim)
        self.ff_norm = norm(embed_dim)
        self.q_norm = norm(self.head_dim)
        self.k_norm = norm(self.head_dim)

    def forward(self, x, context_length):
        x = x.transpose(0, 1)
        B, L, _ = x.size()
        h = self.attn_norm(x)
        q, ff_h, ff_gate = self.q_proj(h).chunk(3, dim=-1)
        k, v = self.kv_proj(h[:, :context_length]).chunk(2, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * math.log2(context_length) / 10
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        attn = attn.reshape(B, L, self.num_heads * self.head_dim)
        ff_x = ff_h * F.silu(ff_gate)
        residual = self.ff_norm(self.out_proj(attn + ff_x))
        x_out = x + residual
        return x_out.transpose(0, 1)
