"""C-Transform (hard argmin) for squared Euclidean cost.

Provides the non-entropic Kantorovich c-transform and differentiable semi-dual
objective for use in WPP (Wasserstein Patch Prior) and other semi-dual OT methods.

Mathematical formulation:
    c^ψ(x_i) = min_j [cost_scale * ||x_i - y_j||² - ψ_j]
    j*(i)    = argmin_j [same]

Semi-dual objective (weighted):
    L(x, ψ) = Σ_i a_i * c^ψ(x_i) + Σ_j b_j * ψ_j

Backward via Danskin's theorem at the minimizer j*(i):
    ∂L/∂x_i  = a_i * 2 * cost_scale * (x_i - y_{j*(i)})
    ∂L/∂ψ_j  = -Σ_{i: j*(i)=j} a_i  +  b_j

Functions:
    c_transform_fwd:  Non-differentiable building block (returns values + indices)
    c_transform_cost: Differentiable semi-dual objective (autograd.Function)
"""

from __future__ import annotations

from typing import Optional

import torch

from flash_sinkhorn.kernels._common import _validate_device
from flash_sinkhorn.kernels.c_transform_sqeuclid import c_transform_kernel


def c_transform_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    psi: torch.Tensor,
    *,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    autotune: bool = True,
    **kernel_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the c-transform (non-differentiable).

    Returns the c-transform values and argmin indices:
        c_i = min_j [cost_scale * ||x_i - y_j||² - ψ_j]
        j*_i = argmin_j [same]

    Args:
        x: Source points [n, d], CUDA
        y: Target points [m, d], CUDA
        psi: Dual potential [m], CUDA
        cost_scale: Cost scaling (1.0 for ||x-y||², 0.5 for ||x-y||²/2)
        allow_tf32: Enable TF32 for matmul
        autotune: Enable kernel autotuning
        **kernel_kwargs: Passed to c_transform_kernel (block_m, block_n, etc.)

    Returns:
        c_values: C-transform values [n], float32
        argmin_idx: Argmin indices [n], int64

    Raises:
        ValueError: If shapes are invalid or tensors not on same CUDA device
        NotImplementedError: If y.requires_grad (grad_y not supported)
    """
    # Shape validation
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [n, d], got shape {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D [m, d], got shape {y.shape}")
    if psi.ndim != 1:
        raise ValueError(f"psi must be 1D [m], got shape {psi.shape}")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError(f"x and y must have same feature dim, got {d} vs {d2}")
    if psi.shape[0] != m:
        raise ValueError(f"psi must have length {m}, got {psi.shape[0]}")
    if m == 0:
        raise ValueError("m must be > 0")

    # Device validation
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    _validate_device(x, [("y", y), ("psi", psi)])

    # Fail-fast: grad_y not supported
    if y.requires_grad:
        raise NotImplementedError(
            "grad_y not supported for c_transform_fwd. "
            "Detach y or use a solver that supports grad_y."
        )

    # Precompute bias = cost_scale * ||y||² - ψ
    y_f = y.float()
    psi_f = psi.float()
    bias = cost_scale * (y_f * y_f).sum(dim=1) - psi_f  # [m]

    # Kernel computes: min_j[-coord_scale * dot(x_i, y_j) + bias_j]
    min_inner, argmin_idx = c_transform_kernel(
        x, y, bias,
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        autotune=autotune,
        **kernel_kwargs,
    )

    # Add back alpha_i = cost_scale * ||x_i||² (factored out of the min)
    x_f = x.float()
    alpha = cost_scale * (x_f * x_f).sum(dim=1)  # [n]
    c_values = alpha + min_inner

    return c_values.detach(), argmin_idx.detach()


class _CTransformCostFunction(torch.autograd.Function):
    """Differentiable semi-dual objective via Danskin's theorem.

    Forward: L = (a * c^ψ(x)).sum() + (b * ψ).sum()
    Backward:
        ∂L/∂x_i = grad_output * a_i * 2 * cost_scale * (x_i - y_{j*(i)})
        ∂L/∂ψ_j = grad_output * (b_j - Σ_{i: j*(i)=j} a_i)
    """

    @staticmethod
    def forward(ctx, x, y, psi, cost_scale, allow_tf32, autotune, a, b):
        # Compute c-transform (non-differentiable)
        c_values, argmin_idx = c_transform_fwd(
            x, y, psi,
            cost_scale=cost_scale,
            allow_tf32=allow_tf32,
            autotune=autotune,
        )

        # Semi-dual objective: L = a · c^ψ(x) + b · ψ
        loss = (a * c_values).sum() + (b * psi.float()).sum()

        # Save for backward
        ctx.save_for_backward(x, y, argmin_idx, a, b)
        ctx.cost_scale = cost_scale

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, y, argmin_idx, a, b = ctx.saved_tensors
        cost_scale = ctx.cost_scale

        grad_x = None
        grad_psi = None

        # Guard: grad_y not implemented
        if ctx.needs_input_grad[1]:
            raise NotImplementedError(
                "grad_y not implemented for c_transform_cost. "
                "Detach y before passing to c_transform_cost."
            )

        if ctx.needs_input_grad[0]:
            # ∂L/∂x_i = grad_output * a_i * 2 * cost_scale * (x_i - y_{j*(i)})
            x_f = x.float()
            y_f = y.float()
            y_matched = y_f[argmin_idx]  # [n, d] — gather
            grad_x = grad_output * (a * 2.0 * cost_scale).unsqueeze(1) * (x_f - y_matched)

        if ctx.needs_input_grad[2]:
            # ∂L/∂ψ_j = grad_output * (b_j - Σ_{i: j*(i)=j} a_i)
            m = y.shape[0]
            # Scatter: accumulate a_i into bins defined by argmin_idx
            assigned_mass = torch.zeros(m, device=a.device, dtype=torch.float32)
            assigned_mass.scatter_add_(0, argmin_idx, a)
            grad_psi = grad_output * (b - assigned_mass)

        # Returns: grad_x, grad_y, grad_psi, grad_cost_scale, grad_allow_tf32, grad_autotune, grad_a, grad_b
        return grad_x, None, grad_psi, None, None, None, None, None


def c_transform_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    psi: torch.Tensor,
    *,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    autotune: bool = True,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable semi-dual objective for the c-transform.

    Computes:
        L(x, ψ) = Σ_i a_i * c^ψ(x_i) + Σ_j b_j * ψ_j

    where c^ψ(x_i) = min_j [cost_scale * ||x_i - y_j||² - ψ_j].

    Differentiable w.r.t. x and ψ via Danskin's theorem.
    Not differentiable w.r.t. y (raises NotImplementedError if y.requires_grad).

    Args:
        x: Source points [n, d], CUDA. May require grad.
        y: Target points [m, d], CUDA. Must NOT require grad.
        psi: Dual potential [m], CUDA. May require grad.
        cost_scale: Cost scaling (1.0 for ||x-y||², 0.5 for ||x-y||²/2)
        allow_tf32: Enable TF32 for matmul
        autotune: Enable kernel autotuning
        a: Source weights [n]. Default: uniform 1/n
        b: Target weights [m]. Default: uniform 1/m

    Returns:
        Scalar loss (differentiable w.r.t. x and psi)

    Raises:
        ValueError: If a or b requires grad, or shapes are invalid
        NotImplementedError: If y.requires_grad
    """
    n = x.shape[0]
    m = y.shape[0]

    # Default weights: uniform
    if a is None:
        a = torch.full((n,), 1.0 / n, device=x.device, dtype=torch.float32)
    if b is None:
        b = torch.full((m,), 1.0 / m, device=y.device, dtype=torch.float32)

    # Validate weights
    if a.shape != (n,):
        raise ValueError(f"a must have shape ({n},), got {a.shape}")
    if b.shape != (m,):
        raise ValueError(f"b must have shape ({m},), got {b.shape}")
    _validate_device(x, [("a", a), ("b", b)])
    if a.requires_grad:
        raise ValueError("a must not require grad (weights are non-differentiable)")
    if b.requires_grad:
        raise ValueError("b must not require grad (weights are non-differentiable)")

    # Ensure float32
    a = a.float()
    b = b.float()

    return _CTransformCostFunction.apply(
        x, y, psi, cost_scale, allow_tf32, autotune, a, b
    )
