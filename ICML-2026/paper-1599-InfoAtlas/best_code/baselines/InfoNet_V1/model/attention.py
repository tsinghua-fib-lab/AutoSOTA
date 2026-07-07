from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    
    def __init__(
        self,
        q_dim:      int,
        kv_dim:     int,
        qk_out_dim: Optional[int] = None,
        v_out_dim:  Optional[int] = None,
        output_dim: Optional[int] = None,
        heads:      int = 1,
        dropout:    float = 0.0
        ):
        
        super().__init__()

        if qk_out_dim is None:
            qk_out_dim = q_dim
        if v_out_dim is None:
            v_out_dim  = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim

        self.heads       = heads
        self.qk_head_dim = qk_out_dim // heads
        self.v_head_dim  = v_out_dim // heads

        self.qeury = nn.Linear(q_dim, qk_out_dim)
        self.key   = nn.Linear(kv_dim, qk_out_dim)
        self.value = nn.Linear(kv_dim, v_out_dim)

        self.projection = nn.Linear(v_out_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
        ):
        
        batch = x_q.shape[0]
        query_len, key_len, value_len = x_q.shape[1], x_kv.shape[1], x_kv.shape[1]

        queries = self.qeury(x_q)
        keys    = self.key(x_kv)
        values  = self.value(x_kv)
        

        # [N, len, embed_size] --> [N, len, heads, head_dim]
        queries = queries.reshape(batch, query_len, self.heads, self.qk_head_dim)
        keys    = keys.reshape(batch, key_len, self.heads, self.qk_head_dim)
        values  = values.reshape(batch, value_len, self.heads, self.v_head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        if attention_mask is not None:
            energy = energy.masked_fill(attention_mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.qk_head_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch, query_len, self.heads * self.v_head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)

        out = self.projection(out)
        # (N, query_len, embed_size)
        return out


class SelfAttention(nn.Module):

    def __init__(
        self,
        q_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.0):

        super().__init__()

        self.norm = nn.LayerNorm(q_dim)
        self.attention = MultiHeadAttention(
            q_dim=q_dim,
            kv_dim=q_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=q_dim,
            heads=heads,
            dropout=dropout
        )

    def forward(self, x_q: torch.Tensor, attention_mask: torch.Tensor = None):
        x_q = self.norm(x_q)
        return self.attention(x_q=x_q, x_kv=x_q, attention_mask=attention_mask)


class CrossAttention(nn.Module):

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.0,
        ):

        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attention = MultiHeadAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=q_dim,
            heads=heads,
            dropout=dropout
        )

    def forward(self, x_q, x_kv, attention_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, attention_mask=attention_mask)