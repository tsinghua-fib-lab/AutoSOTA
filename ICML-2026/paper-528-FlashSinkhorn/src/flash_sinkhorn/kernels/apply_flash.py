"""FlashStyle apply kernels (shifted potentials, s_I cancellation).

This module contains streaming P @ V and P @ vec kernels that work with
SHIFTED potentials directly (f_hat = f - alpha, g_hat = g - beta), avoiding
the cost of converting between potential conventions.

Kernels:
- apply_plan_mat_flashstyle: P @ mat or P^T @ mat (2D grid, tiles over D)
- apply_plan_vec_flashstyle: P @ vec or P^T @ vec (1D grid, s_I cancellation)

Key insight: Compute score WITHOUT f_hat/g_hat in the tiled loop, then apply
row/column marginal correction at the end. This is the same numerical trick as
FlashAttention's online softmax with correction factor.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._common import (
    _apply_mat_autotune_configs,
    _apply_mat_prune_configs,
    _cache_key_bucket,
    _default_block_sizes,
    _validate_device,
)
from flash_sinkhorn.kernels._triton_helpers import _tiled_dot


# =============================================================================
# FlashStyle Mat Kernels (P @ mat, P^T @ mat)
# =============================================================================
#
# For axis=1 (P @ V):
#   score_ij = (coord_scale * x_i . y_j + g_hat_j + delta_j) / eps
#   where delta_j = eps * log(b_j)
#
#   After online LSE:
#     f_hat_plus_i = -eps * (m_i + log(s_i))  # One-step f update
#     r_i = a_i * exp((f_hat_i - f_hat_plus_i) / eps)  # Row marginal correction
#     out_i = r_i * (O_i / s_i)  # Normalized output
#           = a_i * exp(f_hat_i / eps + m_i) * O_i  # Simplified form
#


@triton.jit
def _apply_plan_axis1_mat_flashstyle_kernel(
    x_ptr,
    y_ptr,
    f_hat_ptr,      # Shifted f potential: f_hat = f - alpha
    g_hat_ptr,      # Shifted g potential: g_hat = g - beta
    log_a_ptr,      # Log source weights: log(a)
    log_b_ptr,      # Log target weights: log(b)
    mat_ptr,        # Matrix V to multiply: [m, D]
    scale_ptr,      # Optional per-row scale: [m] (for mat * scale[:, None])
    out_ptr,        # Output: [n, D]
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_mat0,
    stride_mat1,
    stride_scale,
    stride_out0,
    stride_out1,
    eps,
    coord_scale,    # 2 * cost_scale (to match Q @ K.T in FlashSinkhorn)
    CACHE_KEY_N,    # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,    # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """FlashStyle P @ V kernel with shifted potentials.

    Computes out[i, :] = sum_j P_ij * mat[j, :] where:
        P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)
             = a_i * b_j * exp((f_hat + alpha + g_hat + beta - alpha - beta + 2*cs*x.y) / eps)
             = a_i * b_j * exp((f_hat + g_hat + 2*cs*x.y) / eps)

    Algorithm:
    1. Compute score WITHOUT f_hat: S_ij = (coord_scale * x.y + g_hat + delta) / eps
       where delta = eps * log(b)
    2. Online LSE + weighted sum (FlashAttention-style):
       - m_i = max_j(S_ij)
       - s_i = sum_j exp(S_ij - m_i)
       - O_i = sum_j exp(S_ij - m_i) * mat[j, :]
    3. Row marginal correction at end:
       - out_i = a_i * exp(f_hat_i / eps + m_i) * O_i
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # Load f_hat and log_a for row marginal correction at end
    f_hat_i = tl.load(f_hat_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    log_a_i = tl.load(log_a_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    # Online accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # Load g_hat and log_b for bias computation
        g_hat_j = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0, eviction_policy="evict_first").to(tl.float32)
        log_b_j = tl.load(log_b_ptr + offs_n, mask=mask_n, other=0.0, eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul
        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Score WITHOUT f_hat: (coord_scale * x.y + g_hat + delta) / eps
        # where delta = eps * log(b), so delta/eps = log(b)
        # Thus: score = coord_scale * dot / eps + g_hat / eps + log(b)
        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = g_hat_j * inv_eps_log2 + log_b_j * log2e
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :])
        else:
            bias = g_hat_j * inv_eps + log_b_j
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        # Online LSE + weighted sum update
        block_max = tl.max(vals, axis=1)
        new_m = tl.maximum(m_i, block_max)
        new_m_neg_inf = new_m == -float("inf")
        if USE_EXP2:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_i - new_m))
            w = tl.where(
                new_m_neg_inf[:, None], 0.0, tl.exp2(vals - new_m[:, None])
            )
        else:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_i - new_m))
            w = tl.where(
                new_m_neg_inf[:, None], 0.0, tl.exp(vals - new_m[:, None])
            )

        # Load matrix tile and apply optional scale
        mat_tile = tl.load(
            mat_ptr + offs_n[:, None] * stride_mat0 + offs_d[None, :] * stride_mat1,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(scale_ptr + offs_n * stride_scale, mask=mask_n, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            mat_tile = mat_tile * scale[:, None]

        o_i = o_i * alpha[:, None] + tl.dot(w, mat_tile, allow_tf32=ALLOW_TF32)
        m_i = new_m

    # Row marginal correction:
    # out = a * exp(f_hat / eps + m) * O
    # In log space: log_scale = log(a) + f_hat / eps + m (for exp)
    #               log_scale_log2 = (log(a) + f_hat / eps + m) * log2e (for exp2)
    if USE_EXP2:
        log_scale = log_a_i + f_hat_i * inv_eps + m_i * ln2
        scale_factor = tl.exp2(log_scale * log2e)
    else:
        log_scale = log_a_i + f_hat_i * inv_eps + m_i
        scale_factor = tl.exp(log_scale)

    out = scale_factor[:, None] * o_i

    tl.store(
        out_ptr + offs_m[:, None] * stride_out0 + offs_d[None, :] * stride_out1,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _apply_plan_axis0_mat_flashstyle_kernel(
    x_ptr,
    y_ptr,
    f_hat_ptr,      # Shifted f potential: f_hat = f - alpha
    g_hat_ptr,      # Shifted g potential: g_hat = g - beta
    log_a_ptr,      # Log source weights: log(a)
    log_b_ptr,      # Log target weights: log(b)
    mat_ptr,        # Matrix V to multiply: [n, D]
    out_ptr,        # Output: [m, D]
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_mat0,
    stride_mat1,
    stride_out0,
    stride_out1,
    eps,
    coord_scale,    # 2 * cost_scale
    CACHE_KEY_N,    # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,    # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """FlashStyle P^T @ V kernel with shifted potentials.

    Computes out[j, :] = sum_i P_ij * mat[i, :] where:
        P_ij = a_i * b_j * exp((f_hat + g_hat + 2*cs*x.y) / eps)

    Algorithm:
    1. Compute score WITHOUT g_hat: S_ij = (coord_scale * x.y + f_hat + gamma) / eps
       where gamma = eps * log(a)
    2. Online LSE + weighted sum (reduce over i):
       - m_j = max_i(S_ij)
       - s_j = sum_i exp(S_ij - m_j)
       - O_j = sum_i exp(S_ij - m_j) * mat[i, :]
    3. Column marginal correction at end:
       - out_j = b_j * exp(g_hat_j / eps + m_j) * O_j
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # Load g_hat and log_b for column marginal correction at end
    g_hat_j = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    log_b_j = tl.load(log_b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # Online accumulators
    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    o_j = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    for i0 in range(0, n, BLOCK_M):
        i0 = tl.multiple_of(i0, BLOCK_M)
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        # Load f_hat and log_a for bias computation
        f_hat_i = tl.load(f_hat_ptr + offs_m, mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        log_a_i = tl.load(log_a_ptr + offs_m, mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul
        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Score WITHOUT g_hat: (coord_scale * x.y + f_hat + gamma) / eps
        # where gamma = eps * log(a), so gamma/eps = log(a)
        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = f_hat_i * inv_eps_log2 + log_a_i * log2e
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[:, None])
        else:
            bias = f_hat_i * inv_eps + log_a_i
            vals = dot * scaled_inv_eps + bias[:, None]
        vals = tl.where(mask_m[:, None], vals, -float("inf"))

        # Online LSE + weighted sum update (reduce over axis=0, i.e., over i)
        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m_j, block_max)
        new_m_neg_inf = new_m == -float("inf")
        if USE_EXP2:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_j - new_m))
            w = tl.where(
                new_m_neg_inf[None, :], 0.0, tl.exp2(vals - new_m[None, :])
            )
        else:
            alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_j - new_m))
            w = tl.where(
                new_m_neg_inf[None, :], 0.0, tl.exp(vals - new_m[None, :])
            )

        # Load matrix tile
        mat_tile = tl.load(
            mat_ptr + offs_m[:, None] * stride_mat0 + offs_d[None, :] * stride_mat1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        o_j = o_j * alpha[:, None] + tl.dot(tl.trans(w), mat_tile, allow_tf32=ALLOW_TF32)
        m_j = new_m

    # Column marginal correction:
    # out = b * exp(g_hat / eps + m) * O
    if USE_EXP2:
        log_scale = log_b_j + g_hat_j * inv_eps + m_j * ln2
        scale_factor = tl.exp2(log_scale * log2e)
    else:
        log_scale = log_b_j + g_hat_j * inv_eps + m_j
        scale_factor = tl.exp(log_scale)

    out = scale_factor[:, None] * o_j

    tl.store(
        out_ptr + offs_n[:, None] * stride_out0 + offs_d[None, :] * stride_out1,
        out,
        mask=mask_n[:, None] & mask_d[None, :],
    )


# Create autotuned versions of flashstyle mat kernels
_apply_plan_axis1_mat_flashstyle_kernel_autotune = triton.autotune(
    configs=_apply_mat_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D"],
    prune_configs_by={"early_config_prune": _apply_mat_prune_configs},
)(_apply_plan_axis1_mat_flashstyle_kernel)

_apply_plan_axis0_mat_flashstyle_kernel_autotune = triton.autotune(
    configs=_apply_mat_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D"],
    prune_configs_by={"early_config_prune": _apply_mat_prune_configs},
)(_apply_plan_axis0_mat_flashstyle_kernel)


def apply_plan_mat_flashstyle(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    mat: torch.Tensor,
    *,
    eps: float,
    axis: int,
    cost_scale: float = 1.0,
    scale: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    block_d: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,
) -> torch.Tensor:
    """Apply P or P^T to a matrix using shifted potentials (FlashSinkhorn style).

    This kernel works with SHIFTED potentials directly:
        f_hat = f - alpha  where alpha = cost_scale * ||x||^2
        g_hat = g - beta   where beta  = cost_scale * ||y||^2

    Computes:
      axis=1: out[i, :] = sum_j P_ij * mat[j, :]  (P @ V)
      axis=0: out[j, :] = sum_i P_ij * mat[i, :]  (P^T @ V)

    where P_ij = a_i * b_j * exp((f_hat + g_hat + 2*cost_scale*x.y) / eps)

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        f_hat: Shifted f potential [n] (f_hat = f - cost_scale * ||x||^2)
        g_hat: Shifted g potential [m] (g_hat = g - cost_scale * ||y||^2)
        log_a: Log source weights [n]
        log_b: Log target weights [m]
        mat: Matrix V [m, d] for axis=1, [n, d] for axis=0
        eps: Regularization parameter
        axis: 1 for P @ V, 0 for P^T @ V
        cost_scale: Scaling for cost (1.0 for ||x-y||^2, 0.5 for ||x-y||^2/2)
        scale: Optional per-row scale [m] for axis=1 (applies mat * scale[:, None])
        autotune: If True (default), use autotuned kernel configs

    Returns:
        out: Result matrix [n, d] for axis=1, [m, d] for axis=0

    Note:
        The shifted potentials can be obtained from standard potentials via:
            f_hat = f - cost_scale * (x ** 2).sum(dim=1)
            g_hat = g - cost_scale * (y ** 2).sum(dim=1)

        Or from FlashSinkhorn solvers which output shifted potentials directly.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or mat.ndim != 2:
        raise ValueError("x, y, mat must be 2D tensors.")
    if f_hat.ndim != 1 or g_hat.ndim != 1:
        raise ValueError("f_hat and g_hat must be 1D.")
    if log_a.ndim != 1 or log_b.ndim != 1:
        raise ValueError("log_a and log_b must be 1D.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [
        ("y", y), ("f_hat", f_hat), ("g_hat", g_hat),
        ("log_a", log_a), ("log_b", log_b), ("mat", mat), ("scale", scale)
    ])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f_hat.shape[0] != n or log_a.shape[0] != n:
        raise ValueError("f_hat and log_a must have shape (n,).")
    if g_hat.shape[0] != m or log_b.shape[0] != m:
        raise ValueError("g_hat and log_b must have shape (m,).")

    if axis == 1 and mat.shape != (m, d):
        raise ValueError("For axis=1, mat must have shape (m, d).")
    if axis == 0 and mat.shape != (n, d):
        raise ValueError("For axis=0, mat must have shape (n, d).")

    if scale is not None:
        if axis != 1:
            raise ValueError("scale is only supported for axis=1.")
        if scale.shape != (m,):
            raise ValueError("scale must have shape (m,).")

    eps_f = float(eps)
    coord_scale = 2.0 * cost_scale

    # Ensure float32 and contiguous
    f_hat = f_hat.float().contiguous()
    g_hat = g_hat.float().contiguous()
    log_a = log_a.float().contiguous()
    log_b = log_b.float().contiguous()
    mat = mat.float().contiguous()
    if scale is not None:
        scale = scale.float().contiguous()

    # Determine whether to use autotuning
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or block_d is not None
    )
    use_autotune = autotune and not manual_blocks

    if not use_autotune:
        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16
        if block_d is None:
            block_d = 16

    scale_ptr = scale if scale is not None else mat
    stride_scale = int(scale.stride(0)) if scale is not None else 0
    has_scale = scale is not None

    if axis == 1:
        out = torch.empty((n, d), device=x.device, dtype=torch.float32)
        if use_autotune:
            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]), triton.cdiv(d, meta["BLOCK_D"]))
            _apply_plan_axis1_mat_flashstyle_kernel_autotune[grid](
                x, y, f_hat, g_hat, log_a, log_b, mat, scale_ptr, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                mat.stride(0), mat.stride(1),
                stride_scale,
                out.stride(0), out.stride(1),
                eps_f,
                float(coord_scale),
                CACHE_KEY_N=_cache_key_bucket(n),
                CACHE_KEY_M=_cache_key_bucket(m),
                D=d,
                HAS_SCALE=has_scale,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
            )
        else:
            grid = (triton.cdiv(n, block_m), triton.cdiv(d, block_d))
            _apply_plan_axis1_mat_flashstyle_kernel[grid](
                x, y, f_hat, g_hat, log_a, log_b, mat, scale_ptr, out,
                n, m,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                mat.stride(0), mat.stride(1),
                stride_scale,
                out.stride(0), out.stride(1),
                eps_f,
                float(coord_scale),
                _cache_key_bucket(n), _cache_key_bucket(m),
                D=d,
                BLOCK_D=block_d,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                HAS_SCALE=has_scale,
                USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return out

    # axis == 0
    out = torch.empty((m, d), device=x.device, dtype=torch.float32)
    if use_autotune:
        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_N"]), triton.cdiv(d, meta["BLOCK_D"]))
        _apply_plan_axis0_mat_flashstyle_kernel_autotune[grid](
            x, y, f_hat, g_hat, log_a, log_b, mat, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            mat.stride(0), mat.stride(1),
            out.stride(0), out.stride(1),
            eps_f,
            float(coord_scale),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
        )
    else:
        grid = (triton.cdiv(m, block_n), triton.cdiv(d, block_d))
        _apply_plan_axis0_mat_flashstyle_kernel[grid](
            x, y, f_hat, g_hat, log_a, log_b, mat, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            mat.stride(0), mat.stride(1),
            out.stride(0), out.stride(1),
            eps_f,
            float(coord_scale),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            BLOCK_D=block_d,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


# =============================================================================
# FlashStyle Vec Kernels (P @ vec, P^T @ vec) with s_I Cancellation
# =============================================================================
#
# Key optimization: The normalizing sum s_I cancels in the final formula.
# For axis=1 (P @ vec): out_I = a_I * exp(f_hat_I/eps + m_I) * O_I
# For axis=0 (P^T @ vec): out_J = b_J * exp(g_hat_J/eps + m_J) * O_J
#
# This eliminates the need to track s_I/s_J, saving registers and FLOPs.
# =============================================================================


@triton.jit
def _apply_plan_axis1_vec_flashstyle_kernel(
    # Pointers
    x_ptr, y_ptr,
    f_hat_ptr, g_hat_ptr,
    log_a_ptr, log_b_ptr,
    vec_ptr, out_ptr,
    # Dimensions
    N, M,
    # Strides
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    # Scalar params
    eps: tl.constexpr,
    coord_scale: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Flags
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """FlashStyle P @ vec kernel with s_I cancellation.

    Computes: out[i] = sum_j P[i,j] * vec[j]
    where P[i,j] = a[i] * b[j] * exp((f_hat[i] + g_hat[j] + 2*cost_scale*x[i].y[j]) / eps)

    Using online max tracking and s_I cancellation:
    out_I = a_I * exp(f_hat_I/eps + m_I) * O_I
    where O_I = sum_j exp(S_Ij - m_I) * vec_j
    """
    pid_m = tl.program_id(0)

    # Row block indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    # Load row-specific values (f_hat, log_a)
    f_hat_i = tl.load(f_hat_ptr + offs_m, mask=mask_m, other=0.0)
    log_a_i = tl.load(log_a_ptr + offs_m, mask=mask_m, other=-float("inf"))

    # Initialize online accumulators
    LOG2E: tl.constexpr = 1.4426950408889634
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)  # Running max
    o_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running weighted sum

    # Iterate over column blocks
    for n0 in range(0, M, BLOCK_N):
        n0 = tl.multiple_of(n0, BLOCK_N)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < M

        # Load column-specific values (g_hat, log_b, vec)
        g_hat_j = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0, eviction_policy="evict_first")
        log_b_j = tl.load(log_b_ptr + offs_n, mask=mask_n, other=-float("inf"), eviction_policy="evict_first")
        vec_j = tl.load(vec_ptr + offs_n, mask=mask_n, other=0.0, eviction_policy="evict_first")

        # Compute bias: (g_hat_j + delta_j) / eps where delta_j = eps * log(b_j)
        # = g_hat_j/eps + log(b_j)
        bias_j = g_hat_j / eps + log_b_j

        # Compute dot product x @ y^T via tiled matmul
        dot = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, stride_x0, BLOCK_K):  # stride_x0 = D
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < stride_x0

            # Load x[i, k] and y[j, k]
            x_ik = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y_jk = tl.load(
                y_ptr + offs_n[:, None] * stride_y0 + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # Accumulate dot product
            dot += tl.dot(x_ik, tl.trans(y_jk), allow_tf32=ALLOW_TF32)

        # Score: S_ij = coord_scale * x_i.y_j / eps + bias_j
        # Note: score does NOT include f_hat_i (will be added in marginal correction)
        vals = coord_scale * dot / eps + bias_j[None, :]

        # Online max update
        block_max = tl.max(vals, axis=1)
        new_m = tl.maximum(m_i, block_max)

        # Compute exp weights
        if USE_EXP2:
            alpha = tl.exp2((m_i - new_m) * LOG2E)
            w = tl.exp2((vals - new_m[:, None]) * LOG2E)
        else:
            alpha = tl.exp(m_i - new_m)
            w = tl.exp(vals - new_m[:, None])

        # Update running weighted sum: o_i = o_i * alpha + sum_j(w_ij * vec_j)
        o_i = o_i * alpha + tl.sum(w * vec_j[None, :], axis=1)
        m_i = new_m

    # Final marginal correction: out_i = a_i * exp(f_hat_i/eps + m_i) * o_i
    # This is where s_i cancellation happens - we don't divide by s_i!
    # Note: m_i is in natural log space (not log2 space), so we convert entire expression
    log_scale = log_a_i + f_hat_i / eps + m_i
    if USE_EXP2:
        scale_factor = tl.exp2(log_scale * LOG2E)
    else:
        scale_factor = tl.exp(log_scale)

    out_i = scale_factor * o_i

    tl.store(out_ptr + offs_m, out_i, mask=mask_m)


@triton.jit
def _apply_plan_axis0_vec_flashstyle_kernel(
    # Pointers
    x_ptr, y_ptr,
    f_hat_ptr, g_hat_ptr,
    log_a_ptr, log_b_ptr,
    vec_ptr, out_ptr,
    # Dimensions
    N, M,
    # Strides
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    # Scalar params
    eps: tl.constexpr,
    coord_scale: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Flags
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """FlashStyle P^T @ vec kernel with s_J cancellation.

    Computes: out[j] = sum_i P[i,j] * vec[i]
    where P[i,j] = a[i] * b[j] * exp((f_hat[i] + g_hat[j] + 2*cost_scale*x[i].y[j]) / eps)

    Using online max tracking and s_J cancellation:
    out_J = b_J * exp(g_hat_J/eps + m_J) * O_J
    where O_J = sum_i exp(S_iJ - m_J) * vec_i
    """
    pid_n = tl.program_id(0)

    # Column block indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < M

    # Load column-specific values (g_hat, log_b)
    g_hat_j = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0)
    log_b_j = tl.load(log_b_ptr + offs_n, mask=mask_n, other=-float("inf"))

    # Initialize online accumulators
    LOG2E: tl.constexpr = 1.4426950408889634
    m_j = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)  # Running max
    o_j = tl.zeros([BLOCK_N], dtype=tl.float32)  # Running weighted sum

    # Iterate over row blocks
    for m0 in range(0, N, BLOCK_M):
        m0 = tl.multiple_of(m0, BLOCK_M)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N

        # Load row-specific values (f_hat, log_a, vec)
        f_hat_i = tl.load(f_hat_ptr + offs_m, mask=mask_m, other=0.0, eviction_policy="evict_first")
        log_a_i = tl.load(log_a_ptr + offs_m, mask=mask_m, other=-float("inf"), eviction_policy="evict_first")
        vec_i = tl.load(vec_ptr + offs_m, mask=mask_m, other=0.0, eviction_policy="evict_first")

        # Compute bias: (f_hat_i + delta_i) / eps where delta_i = eps * log(a_i)
        # = f_hat_i/eps + log(a_i)
        bias_i = f_hat_i / eps + log_a_i

        # Compute dot product x @ y^T via tiled matmul
        dot = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, stride_x0, BLOCK_K):  # stride_x0 = D
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < stride_x0

            # Load x[i, k] and y[j, k]
            x_ik = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y_jk = tl.load(
                y_ptr + offs_n[:, None] * stride_y0 + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # Accumulate dot product
            dot += tl.dot(x_ik, tl.trans(y_jk), allow_tf32=ALLOW_TF32)

        # Score: S_ij = coord_scale * x_i.y_j / eps + bias_i
        # Note: score does NOT include g_hat_j (will be added in marginal correction)
        # Transpose to get [BLOCK_N, BLOCK_M] for column-wise processing
        vals = tl.trans(coord_scale * dot / eps) + bias_i[None, :]  # [BLOCK_N, BLOCK_M]

        # Online max update (along rows, i.e., over source indices)
        block_max = tl.max(vals, axis=1)
        new_m = tl.maximum(m_j, block_max)

        # Compute exp weights
        if USE_EXP2:
            alpha = tl.exp2((m_j - new_m) * LOG2E)
            w = tl.exp2((vals - new_m[:, None]) * LOG2E)
        else:
            alpha = tl.exp(m_j - new_m)
            w = tl.exp(vals - new_m[:, None])

        # Update running weighted sum: o_j = o_j * alpha + sum_i(w_ji * vec_i)
        o_j = o_j * alpha + tl.sum(w * vec_i[None, :], axis=1)
        m_j = new_m

    # Final marginal correction: out_j = b_j * exp(g_hat_j/eps + m_j) * o_j
    # This is where s_j cancellation happens - we don't divide by s_j!
    # Note: m_j is in natural log space (not log2 space), so we convert entire expression
    log_scale = log_b_j + g_hat_j / eps + m_j
    if USE_EXP2:
        scale_factor = tl.exp2(log_scale * LOG2E)
    else:
        scale_factor = tl.exp(log_scale)

    out_j = scale_factor * o_j

    tl.store(out_ptr + offs_n, out_j, mask=mask_n)


def apply_plan_vec_flashstyle(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    vec: torch.Tensor,
    *,
    eps: float,
    axis: int,
    cost_scale: float = 1.0,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Apply transport plan to vector using FlashStyle with s_I cancellation.

    Computes P @ vec (axis=1) or P^T @ vec (axis=0) where:
    P[i,j] = a[i] * b[j] * exp((f_hat[i] + g_hat[j] + 2*cost_scale*x[i].y[j]) / eps)

    Key optimization: The normalizing sum s_I cancels algebraically, so we use:
    - axis=1: out_I = a_I * exp(f_hat_I/eps + m_I) * O_I
    - axis=0: out_J = b_J * exp(g_hat_J/eps + m_J) * O_J

    Args:
        x: Source points [n, d], fp16/fp32
        y: Target points [m, d], fp16/fp32
        f_hat: Shifted source potential [n], fp32
        g_hat: Shifted target potential [m], fp32
        log_a: Log source weights [n], fp32
        log_b: Log target weights [m], fp32
        vec: Vector to apply [m] for axis=1, [n] for axis=0
        eps: Regularization parameter
        axis: 1 for P @ vec, 0 for P^T @ vec
        cost_scale: Scaling for cost (0.5 for half_cost)
        allow_tf32: Use TF32 for dot products
        use_exp2: Use exp2 instead of exp
        block_m, block_n, block_k: Block sizes
        num_warps, num_stages: Triton tuning params

    Returns:
        out: Result vector [n] for axis=1, [m] for axis=0
    """
    n, d = x.shape
    m = y.shape[0]

    # Validate inputs
    assert f_hat.shape == (n,), f"f_hat shape {f_hat.shape} != ({n},)"
    assert g_hat.shape == (m,), f"g_hat shape {g_hat.shape} != ({m},)"
    assert log_a.shape == (n,), f"log_a shape {log_a.shape} != ({n},)"
    assert log_b.shape == (m,), f"log_b shape {log_b.shape} != ({m},)"

    # Ensure contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Convert eps to float32
    eps_f = float(eps)

    # Coordinate scale for FlashStyle: 2 * cost_scale
    coord_scale = 2.0 * cost_scale

    if axis == 1:
        # P @ vec: output shape [n]
        assert vec.shape == (m,), f"vec shape {vec.shape} != ({m},) for axis=1"
        out = torch.empty(n, device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(n, block_m),)
        _apply_plan_axis1_vec_flashstyle_kernel[grid](
            x, y,
            f_hat, g_hat,
            log_a, log_b,
            vec, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            eps_f,
            float(coord_scale),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # axis == 0: P^T @ vec
    assert vec.shape == (n,), f"vec shape {vec.shape} != ({n},) for axis=0"
    out = torch.empty(m, device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(m, block_n),)
    _apply_plan_axis0_vec_flashstyle_kernel[grid](
        x, y,
        f_hat, g_hat,
        log_a, log_b,
        vec, out,
        n, m,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps_f,
        float(coord_scale),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        USE_EXP2=use_exp2,
        ALLOW_TF32=allow_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
