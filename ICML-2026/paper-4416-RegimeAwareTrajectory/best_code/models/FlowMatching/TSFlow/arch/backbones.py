import math

import torch
from einops import rearrange, repeat
# from linear_attention_transformer import LinearAttentionTransformer
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.arch.s4 import S4


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class S4Layer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        bidirectional=True,
    ):
        super().__init__()
        self.layer = S4(
            d_model=d_model,
            d_state=128,
            bidirectional=bidirectional,
            dropout=dropout,
            transposed=True,
            postact=None,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)
        # Dropout on the output of the layer
        z = self.dropout(z)
        # Residual connection
        x = z + x
        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)
        # Residual connection
        x = z + x
        return x, state


class S4Block(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        expand=2,
        num_features=0,
        setting="univariate",
        bidirectional=True,
        layer=0,
        target_dim=1,
    ):
        super().__init__()
        self.s4block = S4Layer(d_model, bidirectional=bidirectional)
        if target_dim > 1:
            raise NotImplementedError("TSFlow only supports univariate time series.")
            # self.transformer_layer = LinearAttentionTransformer(
            #     dim=d_model,
            #     depth=1,
            #     heads=8,
            #     max_seq_len=256,
            #     n_local_attn_heads=0,
            #     local_attn_window_size=0,
            # )
        self.time_linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_linear1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.out_linear2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.feature_encoder = nn.Conv2d(num_features, d_model, kernel_size=1)
        self.setting = setting
        self.target_dim = target_dim

    def forward(self, x, t, features=None, feat_embeddings=None):
        b, c, k, l = x.shape
        t = self.time_linear(t)
        t = rearrange(t, "b 1 c -> b c 1 1")
        s4_in = x + t
        out, _ = self.s4block(rearrange(s4_in, "b c k l -> (b k) c l"))
        if self.target_dim > 1:
            feature_in = rearrange(out, "(b k) c l -> (b l) k c", k=k, l=l)
            out = self.transformer_layer(feature_in)
            out = rearrange(out, "(b l) k c -> b c k l", l=l)
        else:
            out = rearrange(out, "(b k) c l -> b c k l", k=k)

        if features is not None:  # B C K L
            feature_out = self.feature_encoder(features)
            out = out + feature_out

        out = self.tanh(out) * self.sigm(out)
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


class BackboneModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        step_emb: int,
        num_residual_blocks: int,
        num_features: int,
        target_dim: int = 1,
        residual_block: str = "s4",
        dropout: float = 0.0,
        bidirectional: bool = True,
        init_skip: bool = True,
        feature_skip: bool = True,
    ):
        super().__init__()
        if residual_block == "s4":
            residual_block = S4Block
        else:
            raise ValueError(f"Unknown residual block {residual_block}")
        self.input_init = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.time_init = nn.Sequential(
            nn.Linear(step_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        def get_residual_blocks(hidden_dim, num_residual_blocks):
            residual_blocks = []
            for i in range(num_residual_blocks):
                residual_blocks.append(
                    residual_block(
                        hidden_dim,
                        num_features=num_features + feature_skip + (16 if target_dim > 1 else 0),
                        bidirectional=bidirectional,
                        dropout=dropout,
                        layer=i,
                        target_dim=target_dim,
                    )
                )
            return nn.ModuleList(residual_blocks)

        self.residual_blocks = get_residual_blocks(hidden_dim, num_residual_blocks)
        self.step_embedding = SinusoidalPositionEmbeddings(step_emb)
        self.init_skip = init_skip
        self.feature_skip = feature_skip
        self.feature_embedd = nn.Embedding(target_dim, 16)

    @typechecked
    def forward(
        self,
        t: TensorType[float],
        x_in: TensorType[float, "batch", "length", "num_series"],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        args: dict | None = None,
    ) -> TensorType[float, "batch", "length", "num_series"]:
        # if not self.training:
        B, L, K = x_in.shape
        x = x_in.unsqueeze(-1)
        if self.feature_skip and features is not None:
            features = torch.cat([features, x_in.unsqueeze(-1)], dim=-1)
        if len(t.shape) == 0:
            t = repeat(t, " -> b 1", b=x.shape[0])
        else:
            t = t[..., 0]
        t = self.step_embedding(t * 10000)

        feat_embeddings = None
        if features is not None:
            features = features.transpose(-1, 1)  # B, C, K, L

            if K > 1:
                feat_embeddings = repeat(
                    self.feature_embedd(torch.arange(K, device=features.device)),
                    "K C -> B C K L",
                    B=B,
                    L=L,
                )
                features = torch.cat([features, feat_embeddings], dim=1)

        t = self.time_init(t)
        x = self.input_init((x_in).unsqueeze(-1))  # B, L, H, 1   # B, L ,K, H
        x = x.transpose(-1, 1)  # B, H, 1, L  # B, H, K, L
        skips = []
        # univariate
        for i, layer in enumerate(self.residual_blocks):
            x, skip = layer(x, t, features)
            skips.append(skip)
        skip = rearrange(torch.stack(skips).sum(0), "b c k l -> b l k c")
        out = self.out_linear(skip)[..., 0]

        if self.init_skip:
            out = out - x_in
        return out
