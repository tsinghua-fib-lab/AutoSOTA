from typing import List, Sequence, Optional

import torch
import torch.nn as nn

from .embedding import SinusoidalPositionalEmbedding


class FiLMBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        activation: str = "SiLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.act = getattr(nn, activation)() if hasattr(nn, activation) else nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.cond_proj = nn.Linear(cond_dim, out_dim * 2)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.residual_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.in_proj.weight, a=0, mode="fan_in")
        nn.init.kaiming_normal_(self.out_proj.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h = self.in_proj(x)
        h = self.norm(h)

        emb = self.cond_proj(cond)
        gamma, beta = emb.chunk(2, dim=-1)
        h = h * (1 + gamma) + beta

        h = self.act(h)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h + residual


class CondMLP(nn.Module):
    """
    Conditional MLP that accepts an external condition embedding module.

    Args:
        c_embedder (nn.Module | None): A module that takes raw 'c' and outputs an embedding.
            If None, CondMLP assumes `c` is already an embedding with shape (B, x_dim).
            If it exposes an ``output_dim`` attribute, that will be used as the
            embedder output dim for optional projection.
        x_dim (int): Dimension of the state/input x.
        hidden_dims (List[int]): Dimensions of hidden layers in the backbone.
        time_emb_dim (int): Dimension of the internal time embedding.
    """

    def __init__(
        self,
        c_embedder: Optional[nn.Module] = None,
        x_dim: int = 1,
        c_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (128, 128, 128),
        time_emb_dim: int = 32,
        activation: str = "SiLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.c_embedder = c_embedder if c_embedder is not None else nn.Identity()
        if c_dim is not None and int(c_dim) != self.x_dim:
            raise ValueError(
                f"CondMLP requires c_dim == x_dim, got c_dim={int(c_dim)} and x_dim={self.x_dim}."
            )
        self.c_dim = self.x_dim
        inferred_embed_out_dim: Optional[int] = None
        if c_embedder is None:
            inferred_embed_out_dim = self.c_dim
        elif hasattr(c_embedder, "output_dim"):
            inferred_embed_out_dim = int(getattr(c_embedder, "output_dim"))
        elif isinstance(c_embedder, nn.Linear):
            inferred_embed_out_dim = int(c_embedder.out_features)

        if inferred_embed_out_dim is None:
            raise ValueError(
                "CondMLP could not infer c_embedder output dim. "
                "Please either give the embedder an 'output_dim' attribute or use nn.Linear."
            )

        self.c_proj: nn.Module
        if inferred_embed_out_dim != self.c_dim:
            self.c_proj = nn.Linear(inferred_embed_out_dim, self.c_dim)
        else:
            self.c_proj = nn.Identity()
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        cond_dim = int(time_emb_dim)
        self.input_proj = nn.Linear(self.x_dim + self.c_dim, hidden_dims[0])
        self.input_act = (
            getattr(nn, activation)() if hasattr(nn, activation) else nn.SiLU()
        )
        self.blocks = nn.ModuleList()
        curr_dim = hidden_dims[0]

        for h_dim in hidden_dims:
            self.blocks.append(
                FiLMBlock(
                    in_dim=curr_dim,
                    out_dim=h_dim,
                    cond_dim=cond_dim,
                    activation=activation,
                    dropout=dropout,
                )
            )
            curr_dim = h_dim
        self.final_norm = nn.LayerNorm(curr_dim)
        self.out_proj = nn.Linear(curr_dim, x_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input state [Batch, x_dim]
            t: Time steps [Batch]
            c: Raw Conditioning vector [Batch, ...] (Passed directly to c_embedder)
        """
        B = x.shape[0]
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(B, -1)
        if x.ndim != 2 or x.shape[-1] != self.x_dim:
            raise ValueError(
                f"Expected x with shape (B, x_dim) where x_dim={self.x_dim}, got x.shape={tuple(x.shape)}"
            )
        if not torch.is_tensor(t):
            t = torch.tensor([t] * B, device=x.device)
        elif t.ndim == 0:
            t = t.repeat(B)
        elif t.ndim == 2:
            t = t.squeeze(-1)
        if t.ndim != 1 or t.shape[0] != B:
            raise ValueError(
                f"Expected t with shape (B,), got t.shape={tuple(t.shape)}"
            )
        t_emb = self.time_embed(t)
        c_emb = self.c_embedder(c)
        if c_emb.ndim == 1:
            c_emb = c_emb.unsqueeze(0)
        if c_emb.ndim > 2:
            c_emb = c_emb.view(B, -1)
        if c_emb.ndim != 2 or c_emb.shape[0] != B:
            raise ValueError(
                f"Expected c_embedder(c) with shape (B, D), got c_emb.shape={tuple(c_emb.shape)}"
            )

        c_emb = self.c_proj(c_emb)

        if c_emb.shape[-1] != self.c_dim:
            raise ValueError(
                f"c_embedder output dim {c_emb.shape[-1]} != expected c_dim {self.c_dim} "
                f"(c_emb.shape={tuple(c_emb.shape)}, x.shape={tuple(x.shape)})"
            )

        x_in = torch.cat([x, c_emb], dim=-1)
        h = self.input_proj(x_in)
        h = self.input_act(h)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.final_norm(h)
        out = self.out_proj(h)

        if len(orig_shape) > 2:
            out = out.view(orig_shape)

        return out
