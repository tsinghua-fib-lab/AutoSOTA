"""C-Transform (hard argmin) kernel for squared Euclidean cost.

Computes the non-entropic Kantorovich c-transform via streaming min + argmin:

    c_i = min_j [cost_scale * ||x_i - y_j||² - ψ_j]
    j*_i = argmin_j [cost_scale * ||x_i - y_j||² - ψ_j]

Factorization (same trick as the FlashSinkhorn LSE kernel):

    cost_scale * ||x-y||² - ψ = cost_scale*||x||² + (cost_scale*||y||² - ψ) - 2*cost_scale*(x·y)
                               = alpha_i + bias_j - coord_scale * dot(x_i, y_j)

where:
    alpha_i = cost_scale * ||x_i||²  (constant w.r.t. j, factors out of min)
    bias_j  = cost_scale * ||y_j||² - ψ_j
    coord_scale = 2 * cost_scale

The kernel computes min_j[-coord_scale * dot(x_i, y_j) + bias_j] via tiled streaming.
The Python wrapper adds alpha_i back to get the final c-transform values.

Tie-breaking: across tiles, smallest-j wins (strict < comparison).
Within a tile, tl.argmin selects the first minimum per Triton lane ordering.

Kernel outputs int32 indices (saves SRAM/registers). Python wrapper casts to int64.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._common import (
    _cache_key_bucket,
    _validate_device,
)
from flash_sinkhorn.kernels._triton_helpers import _tiled_dot


# =============================================================================
# AUTOTUNE CONFIGURATION
# =============================================================================

def _c_transform_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for c-transform kernel.

    Mirrors _flashsinkhorn_lse_autotune_configs() but without USE_EXP2.
    The c-transform kernel is simpler (no exp/log), so register pressure is lower.
    """
    configs = []
    for block_m in (128, 64, 32):
        for block_n in (128, 64, 32):
            for block_k in (32, 16):
                for num_warps in (8, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                            num_warps=num_warps,
                            num_stages=3,
                        )
                    )
    # Additional configs for large-d / small-n
    for block_m in (64, 32):
        for block_n in (64, 32):
            for block_k in (32, 16):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                        num_warps=4,
                        num_stages=2,
                    )
                )
    return configs


def _default_block_sizes(n: int, m: int, d: int) -> Tuple[int, int, int, int]:
    """Default block sizes for c-transform kernel (manual path).

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    """
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    else:
        block_m = 32

    if m >= 128:
        block_n = 128
    elif m >= 64:
        block_n = 64
    else:
        block_n = 32

    block_k = 32 if d >= 64 else 16

    num_warps = 8 if n >= 128 else 4
    return block_m, block_n, block_k, num_warps


# =============================================================================
# TRITON KERNEL
# =============================================================================

@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _c_transform_kernel_impl(
    x_ptr,           # Source coordinates: [n, d]
    y_ptr,           # Target coordinates: [m, d]
    bias_ptr,        # Pre-scaled bias: [m] = cost_scale * ||y||² - ψ
    out_val_ptr,     # Output min values: [n] float32
    out_idx_ptr,     # Output argmin indices: [n] int32
    n,               # Number of source points
    m,               # Number of target points
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_bias,
    stride_out_val,
    stride_out_idx,
    coord_scale,     # 2 * cost_scale
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Streaming min + argmin over cost_scale*||x_i - y_j||² - ψ_j.

    Computes per-row:
        min_val_i = min_j [-coord_scale * dot(x_i, y_j) + bias_j]
        min_idx_i = argmin_j [same]

    The caller adds alpha_i = cost_scale * ||x_i||² to get the full c-transform.

    Tie-breaking: strict < across tiles (smallest j wins).
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    # Running min accumulators
    min_val = tl.full([BLOCK_M], float("inf"), tl.float32)
    min_idx = tl.zeros([BLOCK_M], tl.int32)

    # Iterate over all targets (j dimension)
    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # Load pre-scaled bias: bias_j = cost_scale * ||y_j||² - ψ_j
        if EVEN_N:
            bias = tl.load(
                bias_ptr + offs_n * stride_bias,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            bias = tl.load(
                bias_ptr + offs_n * stride_bias,
                mask=mask_n,
                other=float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)

        # Compute x @ y.T via tiled matmul
        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # vals_ij = -coord_scale * dot(x_i, y_j) + bias_j
        vals = -coord_scale * dot + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, float("inf"))

        # Per-row min + argmin within this tile
        tile_min = tl.min(vals, axis=1)                  # [BLOCK_M]
        tile_argmin = tl.argmin(vals, axis=1)             # [BLOCK_M] local index
        tile_argmin = tile_argmin.to(tl.int32) + j0       # Convert to global index

        # Update running min (strict < for smallest-j tie-break)
        update_mask = tile_min < min_val
        min_val = tl.where(update_mask, tile_min, min_val)
        min_idx = tl.where(update_mask, tile_argmin, min_idx)

    tl.store(out_val_ptr + offs_m * stride_out_val, min_val, mask=mask_m)
    tl.store(out_idx_ptr + offs_m * stride_out_idx, min_idx, mask=mask_m)


# Autotuned version
_c_transform_kernel_autotune = triton.autotune(
    configs=_c_transform_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID"],
)(_c_transform_kernel_impl)


# =============================================================================
# PYTHON WRAPPER
# =============================================================================

def c_transform_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    *,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 3,
    autotune: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute streaming min + argmin over the factored cost.

    Computes per source point i:
        min_val_i = min_j [-coord_scale * dot(x_i, y_j) + bias_j]
        min_idx_i = argmin_j [same]

    This is the inner minimum only. The caller adds alpha_i = cost_scale * ||x_i||²
    to get the full c-transform values.

    Args:
        x: Source coordinates [n, d], CUDA
        y: Target coordinates [m, d], CUDA
        bias: Pre-scaled bias [m] = cost_scale * ||y||² - ψ
        cost_scale: Cost scaling (1.0 for ||x-y||², 0.5 for ||x-y||²/2)
        allow_tf32: Enable TF32 for matmul
        block_m, block_n, block_k: Manual block sizes (disables autotune)
        num_warps: Number of warps (disables autotune)
        num_stages: Pipeline stages
        autotune: Enable autotuning

    Returns:
        min_vals: Inner minimum values [n], float32
        argmin_idx: Argmin indices [n], int64 (cast from kernel int32)
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if bias.ndim != 1:
        raise ValueError("bias must be 1D tensor")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension")
    if bias.shape[0] != m:
        raise ValueError(f"bias must have length {m}, got {bias.shape[0]}")
    if m == 0:
        raise ValueError("m must be > 0")

    _validate_device(x, [("y", y), ("bias", bias)])

    # Dtype ID for autotuning cache (captured BEFORE fp32 cast)
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # Ensure contiguous float32
    x = x.contiguous().float()
    y = y.contiguous().float()
    bias = bias.contiguous().float()

    coord_scale = 2.0 * cost_scale
    out_val = torch.empty((n,), device=x.device, dtype=torch.float32)
    out_idx = torch.empty((n,), device=x.device, dtype=torch.int32)

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = autotune and not manual_blocks

    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        grid = (triton.cdiv(n, bm),)
        _c_transform_kernel_impl[grid](
            x, y, bias, out_val, out_idx,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            bias.stride(0),
            out_val.stride(0),
            out_idx.stride(0),
            float(coord_scale),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        _c_transform_kernel_autotune[grid](
            x, y, bias, out_val, out_idx,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            bias.stride(0),
            out_val.stride(0),
            out_idx.stride(0),
            float(coord_scale),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
        )

    launch = _launch_autotune if use_autotune else _launch_manual
    launch()

    # Cast int32 -> int64 for compatibility with gather/scatter_add
    return out_val, out_idx.to(torch.long)
