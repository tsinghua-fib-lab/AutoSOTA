import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_VAL = 1e4


class Attention(nn.Module):
    """Scaled Dot-Product Attention."""
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -MAX_VAL)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    """Residual connection followed by layer norm."""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class SelfAttentionBlock(nn.Module):
    """Self-attention with residual + LayerNorm (no FFN).

    Used by:
      - QuITE embeddings (models/embeddings/{quite,mean_pool}.py)
      - QuITE++ query-based patch embedding and patch-level encoder (Eq. 10-17)
    """
    def __init__(self, hidden, attn_heads, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, attn_mask=None):
        return self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=attn_mask))


class TransformerEncoderBlock(nn.Module):
    """Self-attention + FFN. Used by QuITE++ hierarchical encoder."""
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, attn_mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=attn_mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention + FFN. Used by QuITE++ decoder for future-time queries."""
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, query, key, value, attn_mask=None):
        x = self.input_sublayer(query, lambda _q: self.cross_attn(_q, key, value, mask=attn_mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x
