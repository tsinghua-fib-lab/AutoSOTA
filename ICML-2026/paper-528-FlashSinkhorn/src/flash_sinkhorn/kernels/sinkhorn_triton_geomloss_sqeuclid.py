"""DEPRECATED: GeomLoss-style Sinkhorn solver - thin wrapper around FlashSinkhorn.

This module is DEPRECATED. Use sinkhorn_flashstyle_symmetric from
flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid instead for better performance.

This facade maintains backward compatibility for existing code that imports:
- sinkhorn_geomloss_online_potentials_sqeuclid (DEPRECATED → sinkhorn_flashstyle_symmetric)
- geomloss_to_ott_potentials (kept for convenience)
- ott_to_geomloss_potentials (kept for convenience)

The FlashSinkhorn implementation provides:
- 8-57% faster performance (depending on problem size)
- Same numerical results (verified by 21+ parity tests)
- Support for OTDD label cost, semi-unbalanced OT, early stopping
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch

# Import from _common for backward compatibility (re-export)
from flash_sinkhorn.kernels._common import dampening, epsilon_schedule, log_weights, max_diameter

# Import the new FlashSinkhorn implementation
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_symmetric,
)


def geomloss_to_ott_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert GeomLoss-style potentials to OTT-style potentials.

    GeomLoss convention corresponds to a plan:
      P = diag(a) * exp((f+g-C)/eps) * diag(b)
    OTT convention uses:
      P = exp((f_hat+g_hat-C)/eps)
    with:
      f_hat = f + eps*log(a), g_hat = g + eps*log(b).
    """
    loga = log_weights(a)
    logb = log_weights(b)
    eps_f = float(eps)
    return f.float() + eps_f * loga, g.float() + eps_f * logb


def ott_to_geomloss_potentials(
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert OTT-style potentials to GeomLoss-style potentials.

    OTT convention uses:
      P = exp((f_hat+g_hat-C)/eps)
    GeomLoss convention corresponds to a plan:
      P = diag(a) * exp((f+g-C)/eps) * diag(b)
    with:
      f = f_hat - eps*log(a), g = g_hat - eps*log(b).
    """
    loga = log_weights(a)
    logb = log_weights(b)
    eps_f = float(eps)
    return f_hat.float() - eps_f * loga, g_hat.float() - eps_f * logb


def sinkhorn_geomloss_online_potentials_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    blur: float = 0.05,
    scaling: float = 0.5,
    use_epsilon_scaling: bool = True,
    last_extrapolation: bool = True,
    allow_tf32: bool = True,
    eps: Optional[float] = None,
    n_iters: Optional[int] = None,
    diameter: Optional[float] = None,
    eps_list: Optional[Sequence[float]] = None,
    rho: Optional[float] = None,
    reach: Optional[float] = None,
    rho_x: Optional[float] = None,
    rho_y: Optional[float] = None,
    reach_x: Optional[float] = None,
    reach_y: Optional[float] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
    use_exp2: bool = True,
    return_prelast: bool = False,
    cost_scale: float = 1.0,
    # OTDD label-augmented cost parameters
    label_x: Optional[torch.Tensor] = None,
    label_y: Optional[torch.Tensor] = None,
    label_cost_matrix: Optional[torch.Tensor] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 0.0,
    # Early stopping parameters
    threshold: Optional[float] = None,
    check_every: int = 5,
    return_n_iters: bool = False,
    # Warm-start parameters (for compatibility, but NOT passed to FlashSinkhorn)
    f_init: Optional[torch.Tensor] = None,
    g_init: Optional[torch.Tensor] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
]:
    """DEPRECATED: Use sinkhorn_flashstyle_symmetric instead.

    GeomLoss-style symmetric Sinkhorn with online (streaming) LSE computation.
    This function now delegates to sinkhorn_flashstyle_symmetric which provides
    8-57% better performance with identical numerical results.

    Args:
        x: Source points, shape (n, d)
        y: Target points, shape (m, d)
        a: Source marginal weights, shape (n,)
        b: Target marginal weights, shape (m,)
        blur: Target blur (final eps = blur^2)
        scaling: Epsilon decay factor
        use_epsilon_scaling: Use exponential epsilon schedule
        last_extrapolation: Do final full update
        allow_tf32: Enable TF32 for matmul
        eps: Fixed regularization (if not using epsilon scaling)
        n_iters: Number of iterations
        diameter: Point cloud diameter (auto-computed if None)
        eps_list: Explicit epsilon schedule
        rho, reach: Unbalanced OT marginal penalty
        rho_x, rho_y: Semi-unbalanced marginal penalties
        reach_x, reach_y: Alternatives to rho_x, rho_y
        block_m, block_n, block_k, num_warps, num_stages: Manual kernel config (ignored)
        autotune: Enable autotuning
        use_exp2: Use exp2/log2 optimization
        return_prelast: Return pre-extrapolation potentials
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        label_x, label_y, label_cost_matrix: OTDD label cost
        lambda_x, lambda_y: Weights for Euclidean and label cost
        threshold, check_every: Early stopping parameters
        return_n_iters: Return iteration count
        f_init, g_init: Warm-start potentials (NOT supported by FlashSinkhorn - ignored)

    Returns:
        f, g: Converged potentials
        (optional) f_prelast, g_prelast: Pre-extrapolation potentials
        (optional) n_iters_used: Number of iterations

    .. deprecated:: 2.0.0
        Use :func:`sinkhorn_flashstyle_symmetric` instead for better performance.
    """
    warnings.warn(
        "sinkhorn_geomloss_online_potentials_sqeuclid is deprecated and will be removed "
        "in a future version. Use sinkhorn_flashstyle_symmetric from "
        "flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid for 8-57% better performance.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Warn if unsupported parameters are specified
    if f_init is not None or g_init is not None:
        warnings.warn(
            "f_init and g_init warm-start parameters are not supported by FlashSinkhorn "
            "and will be ignored. The solver starts from zero potentials.",
            UserWarning,
            stacklevel=2,
        )

    if any(x is not None for x in [block_m, block_n, block_k, num_warps]):
        warnings.warn(
            "Manual block sizes (block_m, block_n, block_k, num_warps) are ignored. "
            "FlashSinkhorn uses its own optimized block configurations.",
            UserWarning,
            stacklevel=2,
        )

    # Delegate to FlashSinkhorn
    result = sinkhorn_flashstyle_symmetric(
        x=x,
        y=y,
        a=a,
        b=b,
        blur=blur,
        scaling=scaling,
        use_epsilon_scaling=use_epsilon_scaling,
        last_extrapolation=last_extrapolation,
        cost_scale=cost_scale,
        eps=eps,
        n_iters=n_iters,
        diameter=diameter,
        eps_list=eps_list,
        rho=rho,
        reach=reach,
        rho_x=rho_x,
        rho_y=rho_y,
        reach_x=reach_x,
        reach_y=reach_y,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
        threshold=threshold,
        check_every=check_every,
        return_n_iters=return_n_iters,
        return_prelast=return_prelast,
        label_x=label_x,
        label_y=label_y,
        label_cost_matrix=label_cost_matrix,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
    )

    return result


# =============================================================================
# DEPRECATED INTERNAL FUNCTIONS (kept for backward compatibility)
# =============================================================================
# These functions are kept to avoid breaking code that imports them directly,
# but they now just raise deprecation warnings and provide minimal stubs.


def _default_block_sizes(
    d: int, dtype: torch.dtype, allow_tf32: bool
) -> Tuple[int, int, int, int]:
    """DEPRECATED: Internal function - FlashSinkhorn uses its own block sizes.

    This function is kept for backward compatibility only.
    """
    warnings.warn(
        "_default_block_sizes is deprecated. FlashSinkhorn uses optimized block sizes.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Return reasonable defaults for compatibility
    if dtype == torch.float32 and not allow_tf32:
        block_m = 128 if d >= 32 else 64
        block_n = 64 if d >= 32 else 64
        block_k = 32 if d >= 64 else 16
        num_warps = 4
        return block_m, block_n, block_k, num_warps
    # TF32 mode defaults
    block_m = 64
    block_n = 64
    block_k = 32 if d >= 64 else 16
    num_warps = 4
    return block_m, block_n, block_k, num_warps


def _geomloss_autotune_configs():
    """DEPRECATED: Internal function - FlashSinkhorn uses its own autotune configs."""
    warnings.warn(
        "_geomloss_autotune_configs is deprecated. FlashSinkhorn uses optimized configs.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []


# Stub for the kernel implementation (kept for imports but raises on use)
def _geomloss_symmetric_step_sqeuclid_impl(*args, **kwargs):
    """DEPRECATED: Use FlashSinkhorn kernels instead."""
    raise NotImplementedError(
        "_geomloss_symmetric_step_sqeuclid_impl is deprecated. "
        "Use sinkhorn_flashstyle_symmetric instead."
    )


# Alias
_geomloss_symmetric_step_sqeuclid = _geomloss_symmetric_step_sqeuclid_impl
_geomloss_symmetric_step_sqeuclid_autotune = _geomloss_symmetric_step_sqeuclid_impl
_geomloss_symmetric_step_sqeuclid_autotune_label = _geomloss_symmetric_step_sqeuclid_impl


__all__ = [
    # Main API (deprecated but maintained)
    "sinkhorn_geomloss_online_potentials_sqeuclid",
    # Utility functions (kept)
    "geomloss_to_ott_potentials",
    "ott_to_geomloss_potentials",
    # Re-exported from _common
    "dampening",
    "epsilon_schedule",
    "log_weights",
    "max_diameter",
    # Deprecated internal functions (kept for backward compat)
    "_default_block_sizes",
    "_geomloss_autotune_configs",
    "_geomloss_symmetric_step_sqeuclid_impl",
    "_geomloss_symmetric_step_sqeuclid",
    "_geomloss_symmetric_step_sqeuclid_autotune",
    "_geomloss_symmetric_step_sqeuclid_autotune_label",
]
