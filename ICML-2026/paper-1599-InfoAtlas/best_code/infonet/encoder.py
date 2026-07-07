from typing import Optional

import torch
import torch.nn as nn

from .attention_block import CrossAttentionBlock, SelfAttentionBlock
from .encoder2 import Encoder2

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_y: int,
        latent_num: int,
        latent_dim: int,
        resnet_widden_dim: int = 512,
        resnet_block_num: int = 4,
        cross_attn_heads: int = 4,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        cross_attn_widening_factor: int = 1,
        self_attn_heads: int = 4,
        self_attn_widening_factor: int = 1,
        num_self_attn_per_block: int = 8,
        num_self_attn_blocks: int = 1,
        dropout: float = 0.0,
        ):

        super().__init__()
        self.num_self_attn_blocks = num_self_attn_blocks

        self.latents = nn.Parameter(torch.randn(latent_num, latent_dim))

        self.cross_attn_block = CrossAttentionBlock(
            q_dim=latent_dim,
            kv_dim=input_dim_x+input_dim_y,
            num_heads=cross_attn_heads,
            dropout=dropout,
            widening_factor=cross_attn_widening_factor
        )

        self.self_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionBlock(
                    q_dim=latent_dim,
                    num_heads=self_attn_heads,
                    widening_factor=self_attn_widening_factor,
                    dropout=dropout
                ) for _ in range(num_self_attn_per_block)
            ]) for _ in range(num_self_attn_blocks)
        ])

        self.resnet_pre = Encoder2(input_dim=input_dim_x+input_dim_y, hidden_dim=512, output_dim=resnet_widden_dim, num_layers=resnet_block_num)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        '''
            Args:
                x: (B, M, C)
                mask: (B, M)
        '''

        b, *_= x.shape
        latents = self.latents.repeat(b, 1, 1)

        latents = self.cross_attn_block(
            x_q=latents,
            x_kv=x,
            attention_mask=attention_mask
        )
        for block in self.self_attn_blocks:
            for self_attn_layer in block:
                latents = self_attn_layer(latents)
        
        return latents