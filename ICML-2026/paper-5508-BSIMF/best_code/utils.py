import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


LOG_VAR_MIN = -20.0
LOG_VAR_MAX = 20.0


def _log_normal_diag(
    x: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """Log-density of a diagonal-covariance Gaussian evaluated elementwise.

    Args:
        x, mean, log_var: broadcastable to the same shape (..., D)

    Returns:
        log_prob: same shape (..., D)
    """
    log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    var = torch.exp(log_var)
    return -0.5 * (
        (x - mean) ** 2 / (var + 1e-8)
        + math.log(2.0 * math.pi)
        + log_var
    )


# -----------------------
# Basic building blocks
# -----------------------

def plot_tsne_z(
    z: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    random_state: int = 0,
    perplexity: float = 30.0,
) -> None:
    """
    t-SNE visualization of the learned z space.

    Args:
        z: (N, d_Z) array of latent means (μ_z).
        labels: (N,) array of integers (e.g., true c, GMM clusters, prior clusters).
        title: plot title.
        out_path: where to save the PNG.
    """
    n = z.shape[0]
    per = min(perplexity, max(5, (n - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=per,
        init="random",
        random_state=random_state,
    )
    emb = tsne.fit_transform(z.astype(float))

    plt.figure(figsize=(5, 4))
    sc = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.7,
    )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


class MLP(nn.Module):
    """
    Simple multi-layer perceptron used as backbone or head networks.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer used inside ViT backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        x = x + self.dropout(self.mlp(x_norm))
        return x


class ViTBackbone(nn.Module):
    """
    Minimal Vision Transformer backbone operating on single-channel square images.
    It returns patch tokens and a global CLS token representation.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C=1, H=image_size, W=image_size)

        Returns:
            patch_tokens: (B, N_patches, D)
            cls_repr: (B, D)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, D)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1+N_patches, D)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        cls_repr = x[:, 0]      # (B, D)
        patch_tokens = x[:, 1:] # (B, N_patches, D)
        return patch_tokens, cls_repr


class CrossAttention(nn.Module):
    """
    Cross-attention CA(z, ViT(y)) used inside the mask network.
    ViT patch tokens as queries; z projected to embedding dim as a single key/value token.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N_patches, D) from ViT(y)
            z:      (B, d_Z)

        Returns:
            ca_tokens: (B, N_patches, D)
        """
        B, N, D = tokens.shape
        # Project z to embedding dim and use as a single key/value token
        z_proj = self.latent_proj(z).unsqueeze(1)  # (B, 1, D)

        q = self.norm(tokens)   # (B, N, D)
        k = self.norm(z_proj)   # (B, 1, D)
        v = z_proj              # (B, 1, D)

        attn_out, _ = self.attn(q, k, v, need_weights=False)  # (B, N, D)
        attn_out = self.dropout(attn_out)
        return attn_out
