from typing import Optional

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        qk_out_dim: int,
        v_out_dim: int,
        heads: int,
        widening_factor: int = 1,
        dropout: int = 0.0):

        super().__init__()
        self.cross_attention = CrossAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            heads=heads,
            dropout=dropout
        )
        
        self.mlp = MLP(q_dim, widening_factor, dropout)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, attention_mask: torch.Tensor = None):
        attention = self.cross_attention(x_q=x_q, x_kv=x_kv, attention_mask=attention_mask)
        attention = self.dropout(attention)
        x = x_q + attention
        #x = attention
        x = x + self.mlp(x)
        
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim: int,
        qk_out_dim: Optional[int],
        v_out_dim: Optional[int],
        heads: int,
        widening_factor: int = 1,
        dropout: float = 0.0
        ):

        super().__init__()
        self.self_attention = SelfAttention(
            q_dim=q_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            heads=heads,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(q_dim, widening_factor, dropout)
        
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        attention = self.self_attention(x_q=x, attention_mask=attention_mask)
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