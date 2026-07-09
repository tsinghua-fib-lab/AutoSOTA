from torch import nn

from .attention import TimeWiseMultiheadAttention
from .rope import TimeAwareRotaryEmbedding
from .swiglu import SwiGLU
from .rmsnorm import RMSNorm


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, dropout, rotary_emb=None, RMS_norm=True, dim_head=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout

        self.norm1 = RMSNorm(embed_dim) if RMS_norm else nn.LayerNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim) if RMS_norm else nn.LayerNorm(embed_dim)

        self.attention: MultiHeadAttention = TimeWiseMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            rotary_emb=rotary_emb,
            dim_head=dim_head,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * mlp_hidden_dim),
            SwiGLU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        pre_norm_1 = self.norm1(inputs)
        hidden_state = inputs + self.attention(pre_norm_1).contiguous()

        pre_norm_2 = self.norm2(hidden_state)
        return hidden_state + self.mlp(pre_norm_2)


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_hidden_dim, dropout, dim_head=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.rotary_emb = TimeAwareRotaryEmbedding(
            embed_dim // num_heads,
            use_xpos=True,
            cache_if_possible=True,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    rotary_emb=self.rotary_emb,
                    dim_head=dim_head,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs, return_layer=-1):
        for layer_idx, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if layer_idx == return_layer:
                return inputs
        return inputs
