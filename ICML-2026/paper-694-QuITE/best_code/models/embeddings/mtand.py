"""mTAND baseline (Shukla & Marlin 2021, Table 5).

Multi-time attention to a fixed set of reference time points (CLS queries).
"""
import torch
import torch.nn as nn

from models.common import multiTimeAttention
from models.embeddings._base import LearnableTE


class MTANDEmbedding(nn.Module):
    def __init__(self, args, family='variate'):
        super().__init__()
        self.d_model = args.hid_dim
        self.device = args.device
        self.family = family

        self.te = LearnableTE(self.d_model)
        npatch = 1 if family == 'variate' else args.npatch
        self.attn = multiTimeAttention(args.ndim, 128, self.d_model, 1, npatch=npatch)

    def forward(self, X, time, mask):
        # CLS queries: 128 fixed reference time points in [0, 1]
        cls_t = torch.linspace(0., 1., 128, device=self.device)
        cls_query = self.te(cls_t.unsqueeze(0).unsqueeze(-1))

        if self.family == 'variate':
            time_emb = self.te(time.unsqueeze(-1))
            return self.attn(cls_query, time_emb, X, mask)              # (B, N, D)

        # patch family: flatten (B, M, L_in, N) → (B, M*L_in, N)
        B, M, L_in, N = X.shape
        X_flat = X.reshape(B, M * L_in, N)
        time_flat = time.reshape(B, M * L_in, N)
        mask_flat = mask.reshape(B, M * L_in, N)
        key = self.te(time_flat.unsqueeze(-1))

        out = self.attn(cls_query, key, X_flat, mask_flat, npatch=M)    # (B, N, M, D)
        return out.reshape(B * N, M, self.d_model)
