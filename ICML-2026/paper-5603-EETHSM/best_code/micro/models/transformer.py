import torch
import torch.nn as nn
import torch.nn.functional as F

# Causal mask: Prevent attending to future tokens
def generate_mask(sz, window=None):
    if not window:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    else:
        mask = (torch.triu(torch.ones(sz, sz)) - torch.triu(torch.ones(sz, sz), diagonal=window) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

####################################################################################

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
        # scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.head_dim = embed_dim // num_heads
        self.head_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        self.k_proj = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, _ = query.size()

        # Linear projections
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and run through final linear layer
        # attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        attn_output = attn_output.transpose(1, 2).sum(dim=2).view(B, T, self.embed_dim)
        output = self.out_proj(attn_output)
        return output


class TransformerHead(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, do_norm=True):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        if do_norm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            self.norm1 = nn.Identity(embed_dim)
            self.norm2 = nn.Identity(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention + residual + norm
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm, x_norm, x_norm, mask)
        # x = x + self.norm1(self.dropout(attn_out))
        x = x + self.dropout(attn_out)

        # Feedforward + residual + norm
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        # x = x + self.norm2(self.dropout(ff_out))
        x = x + self.dropout(ff_out)
        return x


class SimpleHandmadeTFLayer(nn.Module):
    def __init__(self, d_model, nhead, causal=True, do_norm=True):
        super().__init__()
        self.transformer_encoder = TransformerHead(d_model, nhead, d_model, dropout=0.2, do_norm=do_norm)
        self.d_model = d_model
        self.nhead = nhead
        self.causal = causal

    def forward(self, x, mask):
        # xx = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        xx = x  
        if self.causal:
            xx = self.transformer_encoder(xx, mask=mask)
        else:
            xx = self.transformer_encoder(xx)
        # return xx.permute(1, 0, 2) # Already includes the skip connection
        return xx # Already includes the skip connection

    # # Finds the average required memory for select data
    # def required_activation_memory(self, x, mask, thres=0.01):
    #     B, T, _ = x.size()

    #     # Linear projections
    #     mha = self.transformer_encoder.mha
    #     q = mha.q_proj(x).view(B, T, self.nhead, self.d_model // self.nhead).transpose(1, 2)
    #     k = mha.k_proj(x).view(B, T, self.nhead, self.d_model // self.nhead).transpose(1, 2)
    #     v = mha.v_proj(x).view(B, T, self.nhead, self.d_model // self.nhead).transpose(1, 2)

    #     attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
    #     # This calculation is saying 'we only need the tokens where the last token attends highly with it'
    #     # It's not exactly what we are going for, so we need to think about this more
    #     return (torch.sum(attn_weights[:, 0, -1, :] > thres) / x.shape[0] * self.d_model).item()

####################################################################################

# A simple wrapper for the PyTorch transformer encoder layers
class SimplePyTorchTFLayer(nn.Module):
    def __init__(self, d_model, nhead, causal=True):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.d_model = d_model
        self.nhead = nhead
        self.causal = causal

    def forward(self, x, mask):
        xx = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        # xx = x
        if self.causal:
            xx = self.transformer_encoder(xx, mask=mask)
        else:
            xx = self.transformer_encoder(xx)
        # return x + xx
        return x + xx.permute(1, 0, 2)  # (batch_size, seq_len, vocab_size)