import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# ============================================================
# Variate-token embeddings (iTransformer / S-Mamba family)
# ============================================================
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class DataEmbedding_inverted_add(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.time_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t):
        x = x.permute(0, 2, 1)
        t = t.permute(0, 2, 1)
        return self.dropout(self.value_embedding(x) + self.time_embedding(t))


class DataEmbedding_inverted_concat(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(2 * c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t):
        x = x.permute(0, 2, 1)
        t = t.permute(0, 2, 1)
        return self.dropout(self.value_embedding(torch.cat([x, t], dim=-1)))


# ============================================================
# Patch-token embeddings (PatchTST / PatchMixer / TMix family)
# ============================================================
class PatchTST_Embedding(nn.Module):
    def __init__(self, patch_len, d_model, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class PatchTST_Embedding_add(nn.Module):
    def __init__(self, patch_len, d_model, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.time_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        return self.dropout(
            self.value_embedding(x) + self.time_embedding(t) + self.position_embedding(x)
        )


class PatchTST_Embedding_concat(nn.Module):
    def __init__(self, patch_len, d_model, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(2 * patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        return self.dropout(
            self.value_embedding(torch.cat([x, t], dim=-1)) + self.position_embedding(x)
        )


# ============================================================
# Hybrid (TimeXer) embedding
# ============================================================
class TimeXer_EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.nvars = n_vars

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, self.nvars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, M, L_in = x.shape
        glb = self.glb_token.repeat((B, 1, 1, 1))

        x = x.reshape(B * N, M, L_in)
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, self.nvars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x)
