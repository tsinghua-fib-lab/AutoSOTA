import torch
import torch.nn as nn
import torch.nn.functional as F
from .masking import TriangularCausalMask
from .RoPE import RotaryEmbedding
from einops import rearrange
import numpy as np
from typing import Optional

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Multi_head_attention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 if_apply_rope: bool,
                 causal_mask: bool,
                 res_attn: bool):

        super(Multi_head_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causal_mask = causal_mask
        self.res_attn = res_attn
        self.if_apply_rope = if_apply_rope
        if self.if_apply_rope:
            self.RoPE = RotaryEmbedding(d_model // num_heads)
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, prev=None, is_causal=False):
        score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model // self.num_heads)
        if is_causal:
            mask = TriangularCausalMask(Q.size(-3), Q.size(-2)).mask.to(Q.device)
            score = score.masked_fill_(mask == 0, -1e9)
        if prev is not None:
            score = score + prev
        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, V)
        if prev is not None:
            return out, score
        else:
            return out

    def forward(self, x, prev=None):
        bs, L, d = x.shape
        Q = self.Q(x).view(bs, L, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
        K = self.K(x).view(bs, L, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
        V = self.V(x).view(bs, L, self.num_heads, self.d_model // self.num_heads).permute(0, 2, 1, 3)
        # [bs, num_heads, L, d_model // num_heads]
        # o = F.scaled_dot_product_attention(Q, K, V, is_causal=self.mask_flag)
        if self.if_apply_rope:
            Q = self.RoPE.rotate_queries_or_keys(Q.transpose(1, 2)).transpose(1, 2)
            K = self.RoPE.rotate_queries_or_keys(K.transpose(1, 2)).transpose(1, 2)
        if prev is not None:
            # o, score = F.scaled_dot_product_attention(Q, K, V, prev, is_causal=self.causal_mask)
            o, score = self.scaled_dot_product_attention(Q, K, V, prev, is_causal=self.causal_mask)
        else:
            # o = F.scaled_dot_product_attention(Q, K, V, is_causal=self.causal_mask)
            o = self.scaled_dot_product_attention(Q, K, V, prev, is_causal=self.causal_mask)
        o = o.reshape(bs, L, self.d_model)

        if prev is not None:
            return o, score
        else:
            return o

class Attn_layer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 norm_type: str,
                 if_apply_rope: bool,
                 causal_mask: bool,
                 res_attn: bool,
                 ):
        super(Attn_layer, self).__init__()


        self.attn = Multi_head_attention(d_model=d_model,
                                         num_heads=num_heads,
                                         if_apply_rope=if_apply_rope,
                                         causal_mask=causal_mask,
                                         res_attn=res_attn)
        if 'batch' in norm_type.lower():
            self.AttnNorm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.AttnNorm = nn.LayerNorm(d_model)

        self.activation = nn.GELU()

    def forward(self, x, prev=None):
        bs, l, d = x.shape
        # print(x.shape)
        src = self.AttnNorm(x)
        if prev is not None:
            src2, score = self.attn(src, prev)
        else:
            src2 = self.attn(src)
        src = src + src2

        if prev is not None:
            return src, score
        else:
            return src, None

class Attn_Block(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 norm_type,
                 if_apply_rope: bool,
                 causal_mask: bool,
                 res_attn: bool,
                 d_in: Optional[int] = None):
        super(Attn_Block, self).__init__()
        self.d_in = d_in
        if d_in is not None:
            self.in_proj = nn.Linear(d_in, d_model)
            self.out_proj = nn.Linear(d_model, d_in)
        self.res_attn = res_attn
        self.layers = nn.ModuleList([Attn_layer(d_model=d_model,
                                                     num_heads=num_heads,
                                                     norm_type=norm_type,
                                                     if_apply_rope=if_apply_rope,
                                                     causal_mask=causal_mask,
                                                     res_attn=res_attn) for _ in range(num_layers)])

    def forward(self, x):
        # x[bs, L, d]
        bs, L, d = x.shape
        if self.d_in is not None:
            x = self.in_proj(x)
        prev = None
        for layer in self.layers:
            if self.res_attn:
                x, prev = layer(x, prev)
            else:
                x, _ = layer(x)
        if self.d_in is not None:
            x = self.out_proj(x)
        return x