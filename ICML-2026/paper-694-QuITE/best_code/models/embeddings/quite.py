"""QuITE (mode='quite'): query-based irregular time series embedding (paper main method).

For each "structured target" (variable or patch), a learnable query token is
prepended to the irregular observation tokens and aggregated via a single
masked self-attention layer (Eq. 5-13 in the paper). The updated query token
is returned as the structured embedding.
"""
import torch
import torch.nn as nn

from models.modules import SelfAttentionBlock
from models.embeddings._base import LearnableTE


class QuITEEmbedding(nn.Module):
    """Query-based embedding.

    Args:
        args: must contain `hid_dim`, `device`, `nhead`, `dropout`, plus
              `ndim` if family='variate' or `npatch` if family='patch'.
        family: 'variate' (one query per variable) or 'patch'
                (one query per (patch, variable) pair).
    """
    def __init__(self, args, family='variate'):
        super().__init__()
        self.d_model = args.hid_dim
        self.device = args.device
        self.family = family

        self.val_emb = nn.Linear(1, self.d_model)
        self.te = LearnableTE(self.d_model)

        if family == 'variate':
            self.query_emb = nn.Embedding(args.ndim, self.d_model)
        elif family == 'patch':
            self.query_emb = nn.Embedding(args.npatch, self.d_model)
        else:
            raise ValueError(f"Unknown family: {family}")

        self.attn = SelfAttentionBlock(self.d_model, args.nhead, dropout=args.dropout)

    def forward(self, X, time, mask):
        return self._variate(X, time, mask) if self.family == 'variate' \
                                            else self._patch(X, time, mask)

    # ----------------------------------------------------------------
    def _variate(self, X, time, mask):
        """Variate-level aggregation.

        Inputs:
            X    : (B, L, N)
            time : (B, L, N)
            mask : (B, L, N)
        Output:
            (B, N, D)
        """
        B, L, N = X.shape
        x_emb = self.val_emb(X.unsqueeze(-1))           # (B, L, N, D)
        t_emb = self.te(time.unsqueeze(-1))             # (B, L, N, D)

        z = (x_emb + t_emb).permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        m = mask.permute(0, 2, 1).reshape(B * N, L)

        q = self.query_emb(torch.arange(N, device=self.device))
        q = q.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 1, self.d_model)

        z = torch.cat([q, z], dim=1)
        m = torch.cat([torch.ones(B * N, 1, device=self.device), m], dim=1)

        out = self.attn(z, attn_mask=m.unsqueeze(-1))
        return out[:, 0, :].reshape(B, N, self.d_model)

    # ----------------------------------------------------------------
    def _patch(self, X, time, mask):
        """Patch-level aggregation.

        Inputs (after permute to N-on-axis-2):
            X    : (B, M, L_in, N) → internally permuted to (B, M, N, L_in)
            time : same
            mask : same
        Output:
            (B*N, M, D)
        """
        B, M, L_in, N = X.shape
        X = X.permute(0, 1, 3, 2)               # (B, M, N, L_in)
        time = time.permute(0, 1, 3, 2)
        mask = mask.permute(0, 1, 3, 2)

        x_emb = self.val_emb(X.unsqueeze(-1))   # (B, M, N, L_in, D)
        t_emb = self.te(time.unsqueeze(-1))

        q = self.query_emb(torch.arange(M, dtype=torch.long, device=self.device))
        q = q.reshape(1, M, 1, 1, self.d_model).expand(B, -1, N, -1, -1)

        z = torch.cat([q, x_emb + t_emb], dim=3)                # (B, M, N, L_in+1, D)
        z = z.reshape(B * M * N, L_in + 1, self.d_model)

        m = torch.cat([torch.ones(B, M, N, 1, device=self.device), mask], dim=3)
        m = m.reshape(B * M * N, L_in + 1)

        out = self.attn(z, attn_mask=m.unsqueeze(-1))
        out = out.reshape(B, M, N, L_in + 1, self.d_model)
        return out[:, :, :, 0, :].transpose(1, 2).reshape(B * N, M, self.d_model)
