"""Attention and transformer blocks used by MASS.

This module provides self-attention, cross-attention, prior-attention, and task
query attention layers used to encode reference priors and fuse them into image
features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'Mlp',
    'Attention',
    'CrossAttention',
    'LayerNorm',
    'PreNorm',
    'TransformerBlock',
    'DualPreNorm',
    'PriorAttentionBlock',
    'MultiPriorAttentionBlock',
    'TaskQueryAttentionBlock'
]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format

        if self.data_format == "channels_last":
            # Use PyTorch's native LayerNorm for channels_last format
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)
            self.normalized_shape = normalized_shape
        else:
            # Custom implementation for channels_first format is more efficient
            # than permuting tensors for 3D medical images
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self.norm(x)
        else:
            # normalize along the spatial dimensions (D, H, W)
            # This is more memory-efficient than permute+norm+permute for large 3D volumes
            dims = tuple(range(2, x.dim()))  # (2, 3, 4) for 5D tensor
            u = x.mean(dim=dims, keepdim=True)
            s = (x - u).pow(2).mean(dim=dims, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            # Reshape weights and bias for proper broadcasting
            weight = self.weight.view(*((1, -1) + (1,) * (x.dim() - 2)))
            bias = self.bias.view(*((1, -1) + (1,) * (x.dim() - 2)))
            
            return weight * x + bias


class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        # Use PyTorch's native MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_drop,
            batch_first=True  # Important for our input format (B, L, C)
        )
        
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: B, L, C (batch, sequence length, features)
        # MultiheadAttention expects (batch, seq, features)
        attn_output, _ = self.mha(x, x, x, need_weights=False)
        attn_output = self.proj_drop(attn_output)
        return attn_output


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()

        # Use PyTorch's native MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        # x1: query, x2: key/value
        # both have shape (B, L, C)
        attn_output, _ = self.mha(x1, x2, x2, need_weights=False)
        attn_output = self.proj_drop(attn_output)
        return attn_output

class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0., qk_dim=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        
        # If qk_dim not specified, use standard attention
        self.qk_dim = qk_dim if qk_dim is not None else dim
        self.qk_head_dim = self.qk_dim // heads
        self.v_head_dim = dim // heads
        
        assert self.qk_dim % heads == 0, "qk_dim must be divisible by num_heads"
        assert dim % heads == 0, "dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(dim, self.qk_dim)
        self.k_proj = nn.Linear(dim, self.qk_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = self.qk_head_dim ** -0.5

    def forward(self, x):
        B, L, C = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # [B, L, qk_dim]
        k = self.k_proj(x)  # [B, L, qk_dim]
        v = self.v_proj(x)  # [B, L, dim]
        
        # Reshape for multi-head: [B, heads, L, head_dim]
        q = q.view(B, L, self.heads, self.qk_head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.qk_head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.v_head_dim).transpose(1, 2)
        
        # Attention: [B, heads, L, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = attn @ v
        
        # Reshape back: [B, L, dim]
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class EfficientCrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0., qk_dim=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        
        # If qk_dim not specified, use standard attention
        self.qk_dim = qk_dim if qk_dim is not None else dim
        self.qk_head_dim = self.qk_dim // heads
        self.v_head_dim = dim // heads
        
        assert self.qk_dim % heads == 0, "qk_dim must be divisible by num_heads"
        assert dim % heads == 0, "dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(dim, self.qk_dim)
        self.k_proj = nn.Linear(dim, self.qk_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = self.qk_head_dim ** -0.5

    def forward(self, x1, x2):
        # x1: query [B, L1, C]
        # x2: key/value [B, L2, C]
        B, L1, C = x1.shape
        L2 = x2.shape[1]
        
        # Project Q from x1, K,V from x2
        q = self.q_proj(x1)  # [B, L1, qk_dim]
        k = self.k_proj(x2)  # [B, L2, qk_dim]
        v = self.v_proj(x2)  # [B, L2, dim]
        
        # Reshape for multi-head
        q = q.view(B, L1, self.heads, self.qk_head_dim).transpose(1, 2)
        k = k.view(B, L2, self.heads, self.qk_head_dim).transpose(1, 2)
        v = v.view(B, L2, self.heads, self.v_head_dim).transpose(1, 2)
        
        # Attention: [B, heads, L1, L2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = attn @ v
        
        # Reshape back: [B, L1, dim]
        out = out.transpose(1, 2).contiguous().view(B, L1, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.layers = nn.ModuleList()
        
        for _ in range(depth):
            # Use PyTorch's TransformerEncoderLayer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=proj_drop,
                activation="gelu",
                batch_first=True,
                norm_first=True  # Pre-norm architecture
            )
            self.layers.append(encoder_layer)
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x





class DualPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class PriorAttentionBlock(nn.Module):
    """Bidirectional attention block for a single set of prior tokens."""
        
    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., expansion=4):
        super().__init__()
        
        dim = feat_dim
        mlp_dim = int(dim * expansion)

        # Update priors by aggregating from flattened image features.
        self.prior_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
        
        # Update image features by injecting task-prior information.
        self.feat_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
        
    def forward(self, x1, x2):
        # x1: flattened image features, x2: prior tokens.

        x2 = self.prior_aggregate_block(x2, x1) + x2
        x2 = self.prior_ffn(x2) + x2

        x1 = self.feat_aggregate_block(x1, x2) + x1
        x1 = self.feat_ffn(x1) + x1

        return x1, x2


class MultiPriorAttentionBlock(nn.Module):
    """Bidirectional attention block for fusing many class priors at once."""

    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., expansion=4, qk_dim=None):
        super().__init__()

        dim = feat_dim
        mlp_dim = int(dim * expansion)

        # Update priors by aggregating from flattened image features.
        if qk_dim is not None:
            self.prior_aggregate_block = DualPreNorm(dim, EfficientCrossAttention(dim, heads, dim_head, attn_drop, proj_drop, qk_dim))
            self.feat_aggregate_block = DualPreNorm(dim, EfficientCrossAttention(dim, heads, dim_head, attn_drop, proj_drop, qk_dim))
            self.prior_attn_block = PreNorm(dim, EfficientAttention(dim, heads, dim_head, attn_drop, proj_drop, qk_dim))
        else:
            self.prior_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
            self.feat_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
            self.prior_attn_block = PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop))

        self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
        self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
        self.prior_ffn_2 = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

    def forward(self, x1, x2):
        # x1: flattened image features, x2: concatenated class-prior tokens.

        x2 = self.prior_aggregate_block(x2, x1) + x2
        x2 = self.prior_ffn(x2) + x2
        x1 = self.feat_aggregate_block(x1, x2) + x1
        x1 = self.feat_ffn(x1) + x1
        # Let task tokens interact with one another after seeing image features.
        x2 = self.prior_attn_block(x2) + x2
        x2 = self.prior_ffn_2(x2) + x2

        return x1, x2


class TaskQueryAttentionBlock(nn.Module):
    """Self/cross-attention block used to turn reference masks into task tokens."""

    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., expansion=4, qk_dim=None):
        super().__init__()

        dim = feat_dim
        mlp_dim = int(dim * expansion)
    
        if qk_dim is not None:
            self.query_attn_block_1 = PreNorm(dim, EfficientAttention(dim, heads, dim_head, attn_drop, proj_drop, qk_dim))
            self.query_aggregate_block = DualPreNorm(dim, EfficientCrossAttention(dim, heads, dim_head, attn_drop, proj_drop, qk_dim))
            self.query_attn_block_2 = PreNorm(dim, EfficientAttention(dim, heads, dim_head, attn_drop, proj_drop, qk_dim))
        
        else:
            self.query_attn_block_1 = PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop))
            self.query_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
            self.query_attn_block_2 = PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop))
        
        
        self.query_ffn_1 = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
        self.query_aggregate_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
        self.query_ffn_2 = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

    def forward(self, x1, x2):
        # x1: mask-conditioned reference features, x2: task query tokens.
        x2 = self.query_attn_block_1(x2) + x2
        x2 = self.query_ffn_1(x2) + x2

        x2 = self.query_aggregate_block(x2, x1) + x2
        x2 = self.query_aggregate_ffn(x2) + x2
        
        x2 = self.query_attn_block_2(x2) + x2
        x2 = self.query_ffn_2(x2) + x2
        
        return x2
