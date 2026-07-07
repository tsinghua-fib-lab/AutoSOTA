from typing import Optional

import torch
import torch.nn as nn

#from .attention_block import CrossAttentionBlock, SelfAttentionBlock
#from .attention import CrossAttention, SelfAttention, MultiHeadAttention

'''
class Encoder2(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_y: int,
        expand_dim: int=128,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        self_attn_heads: int = 4,
        self_attn_widening_factor: int = 1,
        num_self_attn_per_block: int = 8,
        num_self_attn_blocks: int = 1,
        dropout: float = 0.0,
        ):

        super().__init__()
        self.num_self_attn_blocks = num_self_attn_blocks

        self.self_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionBlock(
                    q_dim=expand_dim,
                    num_heads=self_attn_heads,
                    widening_factor=self_attn_widening_factor,
                    dropout=dropout
                ) for _ in range(num_self_attn_per_block)
            ]) for _ in range(num_self_attn_blocks)
        ])

        self.expand_mlp = nn.Sequential(
            nn.Linear(input_dim_x+input_dim_y, expand_dim),
            nn.LayerNorm(expand_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(expand_dim, expand_dim),
            nn.LayerNorm(expand_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(expand_dim, expand_dim)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        x = self.expand_mlp(x)

        for block in self.self_attn_blocks:
            for self_attn_layer in block:
                x = self_attn_layer(x)
        
        return x
'''

class Encoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=10):
        super(Encoder2, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            ))

        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.output_layer(x)
        for layer in self.layers:
            x = x + layer(x)
        
        return x

'''
input_dim = 32
hidden_dim = 2048
output_dim = 512
num_layers = 10

model = Encoder2(input_dim, hidden_dim, output_dim, num_layers=num_layers)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params / 1e6:.2f}M")  # ~100M parameters

batch_size, sequence_length = 32, 5000
x = torch.randn(batch_size, sequence_length, input_dim)
output = model(x)
print(f"Output Shape: {output.shape}")
'''