"""Non-irregular baseline embeddings (Table 5).

These wrap the original backbone embeddings (PatchTST / iTransformer style)
with optional time-conditioning:
  - 'vanilla': raw value embedding only (no time conditioning)
  - 'add'    : value + time embedding
  - 'concat' : value || time embedding
"""
import torch
import torch.nn as nn

from models.layers.embed import (
    PatchTST_Embedding, PatchTST_Embedding_add, PatchTST_Embedding_concat,
    DataEmbedding_inverted, DataEmbedding_inverted_add, DataEmbedding_inverted_concat,
)


class PatchBaselineEmbedding(nn.Module):
    """Embedding for patch-family backbones (PatchTST / PatchMixer / TMix)."""
    def __init__(self, args, mode='vanilla'):
        super().__init__()
        self.mode = mode
        if mode == 'add':
            self.layer = PatchTST_Embedding_add(args.maxlen, args.hid_dim, args.dropout)
        elif mode == 'concat':
            self.layer = PatchTST_Embedding_concat(args.maxlen, args.hid_dim, args.dropout)
        else:
            self.layer = PatchTST_Embedding(args.maxlen, args.hid_dim, args.dropout)

    def forward(self, X, time, mask):
        """X: (B, M, L_in, N), time: same, mask: same → (B*N, M, D)"""
        B, M, L_in, N = X.shape
        x_enc = X.permute(0, 3, 1, 2).reshape(B * N, M, L_in)
        if self.mode in ('add', 'concat'):
            t_enc = time.permute(0, 3, 1, 2).reshape(B * N, M, L_in)
            return self.layer(x_enc, t_enc)
        return self.layer(x_enc)


class VariateBaselineEmbedding(nn.Module):
    """Embedding for variate-family backbones (iTransformer / S-Mamba).

    `c_in` defaults to `args.maxlen` (the per-sample sequence length for
    variate-token backbones). For the TimeXer exogenous path, the input is the
    flattened patch-time axis with length `args.maxlen * args.npatch`, so the
    caller must pass `c_in=args.maxlen * args.npatch`.
    """
    def __init__(self, args, mode='vanilla', c_in=None):
        super().__init__()
        self.mode = mode
        c_in = c_in if c_in is not None else args.maxlen
        if mode == 'add':
            self.layer = DataEmbedding_inverted_add(c_in, args.hid_dim, args.dropout)
        elif mode == 'concat':
            self.layer = DataEmbedding_inverted_concat(c_in, args.hid_dim, args.dropout)
        else:
            self.layer = DataEmbedding_inverted(c_in, args.hid_dim, args.dropout)

    def forward(self, X, time, mask):
        """X: (B, L, N), time: same or (B, L) → (B, N, D)"""
        if self.mode in ('add', 'concat'):
            if time.dim() == 2:
                time = time.unsqueeze(-1).expand(-1, -1, X.size(-1))
            return self.layer(X, time)
        return self.layer(X, None)


class PatchMixerLinearEmbedding(nn.Module):
    """PatchMixer uses a plain Linear projection (vanilla case)."""
    def __init__(self, args):
        super().__init__()
        self.layer = nn.Linear(args.maxlen, args.hid_dim)

    def forward(self, X, time, mask):
        B, M, L_in, N = X.shape
        x_enc = X.permute(0, 3, 1, 2).reshape(B * N, M, L_in)
        return self.layer(x_enc)
