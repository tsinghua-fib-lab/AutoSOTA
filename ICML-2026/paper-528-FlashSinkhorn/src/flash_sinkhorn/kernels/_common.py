"""Common utilities shared across kernel modules."""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import triton


def _cache_key_bucket(size: int, bucket: int = 32) -> int:
    """Bucket a problem size for Triton autotune cache keys.

    Follows FlashAttention's seqlen // 32 pattern (Dao et al.).
    Groups nearby sizes to limit recompilations while still
    differentiating small vs large problems.
    """
    return size // bucket


def dampening(eps: float, rho: Optional[float]) -> float:
    """Compute dampening factor for unbalanced OT with KL marginal penalties.

    For balanced OT (rho=None), returns 1.0.
    For unbalanced OT, returns 1/(1 + eps/rho), which controls the strength
    of marginal relaxation.

    Args:
        eps: Entropy regularization parameter.
        rho: Marginal constraint penalty (None for balanced OT).

    Returns:
        Dampening factor in (0, 1] range.
    """
    if rho is None:
        return 1.0
    return 1.0 / (1.0 + eps / rho)


def log_weights(w: torch.Tensor) -> torch.Tensor:
    """Convert weights to log domain, handling zeros safely."""
    w = w.float()
    out = w.log()
    out = torch.where(w > 0, out, torch.full_like(out, -100000.0))
    return out


def max_diameter(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the maximum diameter (bounding box diagonal) of point clouds x and y."""
    x_f = x.float()
    y_f = y.float()
    mins = torch.stack((x_f.min(dim=0).values, y_f.min(dim=0).values)).min(dim=0).values
    maxs = torch.stack((x_f.max(dim=0).values, y_f.max(dim=0).values)).max(dim=0).values
    return (maxs - mins).norm().item()


def epsilon_schedule(
    diameter: float, blur: float, scaling: float, p: float = 2.0
) -> Sequence[float]:
    """Generate exponential epsilon decay schedule for Sinkhorn iterations.

    Matches GeomLoss's epsilon_schedule exactly: exponential cooling from
    diameter^p down to blur^p with decay factor scaling^p per iteration.

    Args:
        diameter: Maximum diameter of the point clouds.
        blur: Target blur (the regularization "blur" parameter from GeomLoss).
        scaling: Decay factor per iteration (0 < scaling < 1).
        p: Cost exponent (default 2 for squared Euclidean).

    Returns:
        Sequence of epsilon values from diameter^p down to blur^p.
    """
    if diameter <= 0:
        raise ValueError("diameter must be > 0.")
    if blur <= 0:
        raise ValueError("blur must be > 0.")
    if not (0.0 < scaling < 1.0):
        raise ValueError("scaling must be in (0, 1).")

    # Match GeomLoss: final eps = blur^p
    final_eps = blur ** p
    log_final = p * np.log(blur)

    eps_list = [diameter**p]
    eps_list += [
        float(np.exp(e))
        for e in np.arange(p * np.log(diameter), log_final, p * np.log(scaling))
    ]
    eps_list += [final_eps]
    return eps_list


# ---------------------------------------------------------------------------
# Shared helpers for apply kernels (used by apply_ott.py and apply_flash.py)
# ---------------------------------------------------------------------------


def _validate_device(
    x: torch.Tensor,
    tensors: Sequence[Tuple[str, Optional[torch.Tensor]]],
) -> None:
    """Validate that all tensors are on the same CUDA device as x.

    Args:
        x: Reference tensor (must be on CUDA)
        tensors: List of (name, tensor) pairs to validate

    Raises:
        ValueError: If any tensor is not on the same CUDA device as x
    """
    ref_device = x.device
    for name, tensor in tensors:
        if tensor is not None:
            if not tensor.is_cuda or tensor.device != ref_device:
                raise ValueError(
                    f"{name} must be on the same CUDA device as x ({ref_device}), "
                    f"got {tensor.device if tensor.is_cuda else 'CPU'}"
                )


def _default_block_sizes(n: int, m: int, d: int) -> Tuple[int, int, int]:
    """Default block sizes for apply kernels (vec, mat, mat5).

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D dot accumulator causes incorrect results.

    The key constraint is: BLOCK_K < D (must have at least 2 k iterations).
    - d >= 64: block_k = 32 -> at least 2 iterations
    - d >= 32: block_k = 16 -> at least 2 iterations
    - d < 32:  block_k = 16 -> minimum for tl.dot
    """
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    elif n >= 32:
        block_m = 32
    else:
        block_m = 16

    if m >= 128:
        block_n = 128
    elif m >= 64:
        block_n = 64
    elif m >= 32:
        block_n = 32
    else:
        block_n = 16

    # Choose BLOCK_K to ensure multiple k iterations (BLOCK_K < D)
    if d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot

    return block_m, block_n, block_k


def _apply_mat_autotune_configs() -> Sequence[triton.Config]:
    """Autotune configs for apply_plan_mat and mat5 kernels.

    Curated configs based on tuning experiments. Includes:
    - Small BLOCK_D (16-64): For typical d <= 64 dimensions
    - Medium BLOCK_D (128-256): For d up to 256
    - Large BLOCK_D (512-2048): For high-dimensional features (d > 256)

    Config pruning functions filter these at runtime based on actual D
    to avoid compiling wasteful or invalid configs.
    """
    return [
        # Best overall
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        # Good for small n
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        # Good for large n
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        # Fallback options
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=64 for d >= 64
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=128 for d >= 128
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=256 for d >= 256 (reduced BLOCK_M/N for register pressure)
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_D": 256}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=512 for d >= 512
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_D": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_D": 512}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=1024 for d >= 1024
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "BLOCK_D": 1024}, num_warps=4, num_stages=2),
        # Configs with BLOCK_D=2048 for d >= 2048
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "BLOCK_D": 2048}, num_warps=4, num_stages=2),
    ]


def _apply_mat_prune_configs(configs, named_args, **kwargs):
    """Prune apply_plan_mat autotune configs for given D.

    Unlike mat5, these kernels DO tile over D (2D grid), so BLOCK_D can be < D.
    We only prune excessively large BLOCK_D configs to reduce compile time.
    """
    D = named_args.get("D", 64)
    # Upper bound: BLOCK_D should be <= max(D, 64) to avoid wasteful configs
    # For small D, allow up to 64; for large D, allow up to D itself
    max_block_d = max(D, 64)

    pruned = [
        cfg for cfg in configs
        if cfg.kwargs.get("BLOCK_D", 16) <= max_block_d
    ]

    # Safety: if all configs pruned, keep smallest ones
    if not pruned:
        min_block_d = min(cfg.kwargs.get("BLOCK_D", 16) for cfg in configs)
        pruned = [cfg for cfg in configs if cfg.kwargs.get("BLOCK_D", 16) == min_block_d]

    return pruned
