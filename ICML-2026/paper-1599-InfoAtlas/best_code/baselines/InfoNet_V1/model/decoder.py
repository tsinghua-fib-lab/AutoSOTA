import torch
import torch.nn as nn

from typing import Optional

from .attention_block import CrossAttentionBlock


class Decoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        q_dim: int,
        heads: int = 4,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        cross_attn_widening_factor: int = 1,
        dropout: float = 0.0
        ):

        super().__init__()
        self.cross_attn = CrossAttentionBlock(
            q_dim=q_dim,
            kv_dim=latent_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            heads=heads,
            widening_factor=cross_attn_widening_factor,
            dropout=dropout
        )
        
        if projection_dim is not None:
            self.projection = nn.Linear(q_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x_q: torch.Tensor, latents: torch.Tensor, query_mask: Optional[torch.Tensor] = None):
        
        output = self.cross_attn(
            x_q=x_q,
            x_kv=latents,
            attention_mask=query_mask
        )

        return self.projection(output)

