import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encode scalar time with sinusoidal features."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding requires even dim")
        self.dim = dim
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return time embeddings."""
        # t: (B,) or (B,1)
        if t.ndim == 1:
            t = t[:, None]  # t is (B,1) now
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, device=device, dtype=t.dtype)
            / max(half - 1, 1)
        )  # freq is (half,)
        angles = t * freqs[None, :] * 2 * math.pi  # angles is (B, half)
        emb = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )  # emb is (B, dim)
        return emb


class VelocityMLP(nn.Module):
    """MLP velocity field conditioned on time."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        out_dim: int | None = None,
        time_embed_dim: int = 128,
        time_embedding: str = "sin",
        t_cond: str = "concat",
        dropout: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.time_embed_dim = time_embed_dim
        self.time_embedding = time_embedding
        self.t_cond = t_cond
        self.dropout = dropout
        self.norm = norm

        # time embedding
        if self.time_embedding == "sin":
            self.t_embed = SinusoidalTimeEmbedding(self.time_embed_dim)
        else:
            raise ValueError("Time embedding not recognized")
        layers = []
        feat_dim = self.hidden_dim
        in_dim0 = self.in_dim + (self.time_embed_dim if self.t_cond == "concat" else 0)
        layers.append(nn.Linear(in_dim0, feat_dim))
        for _ in range(self.depth - 1):
            layers.append(nn.SELU())  # Same as in the minibatch OT paper
            if self.norm == "layer":
                layers.append(nn.LayerNorm(feat_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(feat_dim, feat_dim))
        self.backbone = nn.Sequential(*layers)
        # Projection back to the original dimension
        self.head = nn.Sequential(nn.SELU(), nn.Linear(feat_dim, self.out_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity at positions and times."""
        t_emb = self.t_embed(t)
        if self.t_cond == "concat":
            h = torch.cat([x, t_emb], dim=-1)
            h = self.backbone(h)
        else:
            raise ValueError("Unknown conditioning type for the time")
        return self.head(h)


class ScoreMLP(VelocityMLP):
    """Score network with the same architecture as the velocity network."""

    pass
