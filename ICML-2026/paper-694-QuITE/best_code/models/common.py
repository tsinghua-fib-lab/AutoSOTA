import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class multiTimeAttention(nn.Module):
    """mTAND attention used as a baseline embedding (Table 5 in the paper)."""
    def __init__(self, input_dim, nq=128, embed_time=16, num_heads=1, npatch=1):
        super().__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = math.ceil(nq / npatch)
        self.linears = nn.ModuleList([
            nn.Linear(embed_time, embed_time),
            nn.Linear(embed_time, embed_time),
            nn.Linear(self.nhidden, self.embed_time_k),
        ])

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.permute(0, 3, 1, 2).unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.permute(0, 3, 1, 2).unsqueeze(-3), -1), p_attn

    def forward(self, query, key, value, mask=None, dropout=None, npatch=None):
        batch, _, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query = self.linears[0](query).view(query.size(0), -1, self.h, self.embed_time_k).transpose(1, 2).unsqueeze(1)
        key = self.linears[1](key).view(key.size(0), key.size(1), key.size(2), self.h, self.embed_time_k).transpose(1, 2).transpose(2, 3)
        x, _ = self.attention(query, key, value, mask, dropout)

        if npatch is not None:
            d_k = x.size(-1)
            patch_len = math.ceil(d_k / npatch)
            pad_len = patch_len * npatch - d_k
            if pad_len > 0:
                x = F.pad(x, (0, pad_len))
            x = x.view(batch, dim, self.h, npatch, patch_len)
            x = self.linears[-1](x).transpose(2, 3)
            x = x.reshape(batch, dim, npatch, -1)
        else:
            x = self.linears[-1](x)
            x = x.reshape(batch, dim, self.h * self.embed_time_k)
        return x


class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a),
        )

    def forward(self, x):
        x = x + self.Resnet(x)
        return self.Conv_1x1(x)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class TempBlock(nn.Module):
    """Temporal-mixing block used by TMix (both forecasting and classification)."""
    def __init__(self, args):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(args.npatch * args.hid_dim, args.npatch * args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.npatch * args.hid_dim, args.npatch * args.hid_dim),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return x + self.temporal(x)
