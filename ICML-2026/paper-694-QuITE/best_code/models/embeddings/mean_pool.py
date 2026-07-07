"""Mean Pooling baseline (Table 5).

Apply self-attention to the observation token sequence (no query token),
then average over the observation axis to obtain a structured token.
"""
import torch
import torch.nn as nn

from models.modules import SelfAttentionBlock
from models.embeddings._base import LearnableTE


class MeanPoolEmbedding(nn.Module):
    def __init__(self, args, family='variate'):
        super().__init__()
        self.d_model = args.hid_dim
        self.device = args.device
        self.family = family

        self.val_emb = nn.Linear(1, self.d_model)
        self.te = LearnableTE(self.d_model)
        self.attn = SelfAttentionBlock(self.d_model, args.nhead, dropout=args.dropout)

    def forward(self, X, time, mask):
        return self._variate(X, time, mask) if self.family == 'variate' \
                                            else self._patch(X, time, mask)

    def _variate(self, X, time, mask):
        """X: (B, L, N) → (B, N, D)"""
        B, L, N = X.shape
        x_emb = self.val_emb(X.unsqueeze(-1))                       # (B, L, N, D)
        t_emb = self.te(time.unsqueeze(-1))

        z = (x_emb + t_emb).permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        m = mask.permute(0, 2, 1).reshape(B * N, L)

        out = self.attn(z, attn_mask=m.unsqueeze(-1))
        out = out.reshape(B, N, L, self.d_model)
        return out.mean(dim=2)  # mean over L

    def _patch(self, X, time, mask):
        """X: (B, M, L_in, N) → (B*N, M, D)"""
        B, M, L_in, N = X.shape
        X = X.permute(0, 1, 3, 2)
        time = time.permute(0, 1, 3, 2)
        mask = mask.permute(0, 1, 3, 2)

        x_emb = self.val_emb(X.unsqueeze(-1))                       # (B, M, N, L_in, D)
        t_emb = self.te(time.unsqueeze(-1))

        z = (x_emb + t_emb).reshape(B * M * N, L_in, self.d_model)
        m = mask.reshape(B * M * N, L_in)

        out = self.attn(z, attn_mask=m.unsqueeze(-1))
        out = out.reshape(B, M, N, L_in, self.d_model).mean(dim=-2)  # mean over L_in
        return out.transpose(1, 2).reshape(B * N, M, self.d_model)
