from typing import Optional

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention, MultiHeadAttention

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        widening_factor: int = 1,
        dropout: int = 0.0):

        super().__init__()
        self.cross_attention = MultiHeadAttention(
            embed_dim=q_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.mlp = MLP(q_dim, widening_factor, dropout)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, attention_mask: torch.Tensor = None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        attention = self.cross_attention(query=x_q, key=x_kv, value=x_kv, attention_mask=attention_mask)
        attention = self.dropout(attention)
        x = x_q + attention
        x = x + self.mlp(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim: int,
        num_heads: int,
        widening_factor: int = 1,
        dropout: float = 0.0
        ):

        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            embed_dim=q_dim,
            kv_dim=q_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(q_dim, widening_factor, dropout)
        self.q_norm = nn.LayerNorm(q_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        x = self.q_norm(x)
        attention = self.self_attention(query=x, key=x, value=x, attention_mask=attention_mask)
        attention = self.dropout(attention)
        x = x + attention
        #x = attention
        x = x + self.mlp(x)
        
        return x


class MLP(nn.Module):

    def __init__(
        self,
        hidden_dim:int,
        widening_factor:int,
        dropout: float = 0.0
        ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, widening_factor * hidden_dim),
            nn.GELU(),
            nn.Linear(widening_factor * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, dimension=8):
        super(PositionalEmbedding, self).__init__()
        self.freq= dimension//2
        self.freq_bands = 2 ** torch.linspace(0, dimension//2 - 1, dimension//2)
    
    def forward(self, x):
        out = [x.unsqueeze(-1)] 
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x).unsqueeze(-1))
            out.append(torch.cos(freq * x).unsqueeze(-1))
        if len(out) % 2 == 1:
            out.pop()
        return torch.cat(out, dim=-1)
            
class MLP_4layer(nn.Module):

    def __init__(
        self,
        hidden_dim:int,
        dropout: float = 0.0
        ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)
