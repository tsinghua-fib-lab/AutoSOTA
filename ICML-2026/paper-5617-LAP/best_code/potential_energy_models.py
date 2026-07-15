# potential_energy_models.py

from __future__ import annotations
from typing import Union, Callable, Optional
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
Tensor = torch.Tensor
TimeLike = Union[float, Tensor]

# ============================================================
# Utility: time column helper
# ============================================================

def make_time_column(x: Tensor, t: TimeLike) -> Tensor:
    """
    Build a (N,1) column of times matching x's device/dtype.

    x : (N,d)
    t : float or tensor, allowed shapes:
        - float
        - 0D tensor
        - (N,)
        - (N,1)
        - (1,)
        - (1,1)
    """
    N = x.shape[0]
    if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
        tcol = torch.full((N, 1), float(t), device=x.device, dtype=x.dtype)
    else:
        tcol = t.to(device=x.device, dtype=x.dtype)
        if tcol.ndim == 1:
            if tcol.shape[0] == 1:
                tcol = tcol.view(1, 1).expand(N, 1)
            elif tcol.shape[0] == N:
                tcol = tcol.view(N, 1)
            else:
                raise ValueError(f"Incompatible t shape {tcol.shape} for N={N}")
        elif tcol.ndim == 2:
            if tcol.shape == (1, 1):
                tcol = tcol.expand(N, 1)
            elif tcol.shape == (N, 1):
                pass
            else:
                raise ValueError(f"Incompatible t shape {tcol.shape} for N={N}")
        else:
            raise ValueError(f"Unsupported t ndim={tcol.ndim}")
    return tcol

from dataclasses import dataclass
@dataclass(eq=False)
class BatchedSimpleAttentionFlow(nn.Module):
    """
    Simplified transformer flow with optional Time and CoM features.
    Includes EMA stability utility.
    """
    D: int = 2
    hidden_dim: int = 512
    num_flow_layers: int = 6
    num_heads: int = 1
    dropout: float = 0.0
    activation: nn.Module = nn.GELU

    # Feature Toggles
    use_time: bool = False  # Include Time embedding
    d_time: int = 16  # Dimension for time embedding if used

    ffn_dim: Optional[int] = 512
    use_spectral_norm: bool = False

    def __post_init__(self):
        super().__init__()

        # --- 1. Time Embedding Setup ---
        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, self.d_time),
                nn.SiLU(),
                nn.Linear(self.d_time, self.d_time)
            )
        else:
            self.time_mlp = None

        # --- 2. Calculate Input Dimension ---
        in_dim = self.D
        if self.use_time:
            in_dim += self.d_time

        # --- 3. Layers ---
        self.input_projection = nn.Linear(in_dim, self.hidden_dim)

        inner_dim = self.ffn_dim if self.ffn_dim is not None else 512

        self.flow_layers = nn.ModuleList()
        for _ in range(self.num_flow_layers):
            self.flow_layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.hidden_dim),
                'self_attn': nn.MultiheadAttention(
                    self.hidden_dim, self.num_heads,
                    dropout=self.dropout, batch_first=True
                ),
                'norm2': nn.LayerNorm(self.hidden_dim),
                'ffn': nn.Sequential(
                    spectral_norm(nn.Linear(self.hidden_dim, inner_dim)) if self.use_spectral_norm else nn.Linear(self.hidden_dim, inner_dim),
                    self.activation(),
                    nn.Dropout(self.dropout),
                    spectral_norm(nn.Linear(inner_dim, self.hidden_dim)) if self.use_spectral_norm else nn.Linear(inner_dim, self.hidden_dim),
                    nn.Dropout(self.dropout)
                )
            }))

        self.output_projection = nn.Linear(self.hidden_dim, 1)

    def forward(self, z: torch.Tensor, t: float | torch.Tensor | None = None):
        """
        z: (B, N, D) or (N, D)
        t: scalar float or (B, 1) tensor. Required if use_time=True.
        """
        if z.ndim == 2:
            z = z.unsqueeze(0)
        elif z.ndim != 3:
            raise ValueError(f"Expected z shape (N,D) or (B,N,D), got {z.shape}")

        B, N, _ = z.shape
        feats = [z]


        # Time Embedding
        if self.use_time:
            if t is None:
                raise ValueError("Model configured with use_time=True, but no t provided.")

            if isinstance(t, float):
                t_val = torch.full((B, 1), t, device=z.device, dtype=z.dtype)
            else:
                t_val = t.to(z.device, z.dtype)
                if t_val.ndim == 1:
                    t_val = t_val.view(B, 1)
                elif t_val.ndim == 0:
                    t_val = t_val.view(1, 1).expand(B, 1)

            t_emb = self.time_mlp(t_val).unsqueeze(1)
            t_tile = t_emb.expand(-1, N, -1)
            feats.append(t_tile)

        z_in = torch.cat(feats, dim=-1)
        x = self.input_projection(z_in)

        # --- Transformer Flow ---
        for layer in self.flow_layers:
            # Attention Block
            res = x
            x = layer['norm1'](x)
            x_attn, _ = layer['self_attn'](x, x, x) #attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
            x = res + x_attn

            # FFN Block
            res = x
            x = layer['norm2'](x)
            x = res + layer['ffn'](x)

        out = self.output_projection(x)
        return out.squeeze(-1)

    @torch.no_grad()
    def update_moving_average(self, source_model: nn.Module, eta: float = 0.999):
        """
        Updates the parameters of 'self' to be a moving average of 'source_model'.

        Formula:
            self_param = eta * self_param + (1 - eta) * source_param

        Args:
            source_model: The active training model.
            eta: Decay rate. Close to 1 (e.g., 0.999) means slow updates (stable).
                 Close to 0 means fast updates.
        """
        # Ensure models are on same device or handle transfer
        for target_p, source_p in zip(self.parameters(), source_model.parameters()):
            target_p.data.mul_(eta).add_(source_p.data, alpha=(1.0 - eta))

        # Also sync buffers (like batch norm stats), though LayerNorm has none usually
        for target_b, source_b in zip(self.buffers(), source_model.buffers()):
            target_b.data.mul_(eta).add_(source_b.data, alpha=(1.0 - eta))

# ============================================================
# Autograd helpers: ∇_x ψ and a(x,t) = -∇_x ψ
# ============================================================

def grad_wrt_x(
    scalar_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    create_graph: bool,
) -> Tensor:
    """
    Compute ∇_x scalar_fn(x).

    Parameters
    ----------
    scalar_fn : callable
        Takes x_req (N,d) -> (N,) scalar per particle.
    x : (N,d) tensor
    create_graph : bool
        - True  → keep graph so that gradients can flow back to model params.
        - False → no graph; cheaper, for evaluation / GIFs.
    """
    # print('shape:', x.shape)
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        f = scalar_fn(x_req)  # shape (..., N)
        g, = torch.autograd.grad(
            f,
            x_req,
            grad_outputs=torch.ones_like(f),
            create_graph=create_graph,
            retain_graph=create_graph,
        )
    return g if create_graph else g.detach()


import inspect

SDP_MATH_CFG = dict(
    enable_flash=False,
    enable_mem_efficient=False,
    enable_math=True,
)

def accel_from_potential(
        model: nn.Module,
        x: torch.Tensor,
        t: float,  # or Tensor
        *,
        create_graph: bool = True,
        max_force: float = None,
) -> torch.Tensor:
    """
    Robust version: Checks if 't' is explicitly named in the forward signature.
    """
    sig = inspect.signature(model.forward)
    has_time_arg = 't' in sig.parameters

    def psi_of_x(xx: torch.Tensor) -> torch.Tensor:
        # Force SDPA math kernel to get double-backward support
        if torch.cuda.is_available():
            with torch.backends.cuda.sdp_kernel(**SDP_MATH_CFG):
                if has_time_arg:
                    return model(xx, t)
                else:
                    return model(xx)
        else:
            # CPU: just call it
            if has_time_arg:
                return model(xx, t)
            else:
                return model(xx)

    from potential_energy_models import grad_wrt_x
    raw_force = -grad_wrt_x(psi_of_x, x, create_graph=create_graph)

    if max_force is not None and max_force > 0:
        norms = raw_force.norm(p=2, dim=-1, keepdim=True)
        scale = torch.clamp(max_force / (norms + 1e-6), max=1.0)
        return raw_force * scale

    return raw_force


def make_accel_from_potential(
        model: nn.Module,
        *,
        create_graph: bool,
        max_force: float = None,
):
    """
    Returns a callable accel(x,t) suitable for leapfrog_auto.
    """

    def accel(x, t):
        return accel_from_potential(
            model,
            x,
            t,
            create_graph=create_graph,
            max_force=max_force
        )

    return accel

import argparse
from typing import Any, Dict, Tuple

def build_model_and_kwargs(args: argparse.Namespace, d: int) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Architectures used by train.py.
    Returns: (model, model_kwargs_dict_for_logging)
    """

    # --- 2. Attention Flow Architecture ---
    if args.arch == "attn_flow":
        kwargs = dict(
            D=d,
            hidden_dim=args.attn_hidden_dim,
            num_flow_layers=args.attn_layers,
            num_heads=args.attn_heads,
            ffn_dim=args.ff_dim,
            dropout=args.dropout,
            use_time=args.use_time,
            d_time=args.d_time,
            use_spectral_norm=getattr(args, 'use_spectral_norm', False),
        )
        return BatchedSimpleAttentionFlow(**kwargs), {"arch": "attn_flow", **kwargs}



__all__ = [
    "BatchedSimpleAttentionFlow",
    "grad_wrt_x",
    "accel_from_potential",
    "make_accel_from_potential",
    "make_time_column",
]