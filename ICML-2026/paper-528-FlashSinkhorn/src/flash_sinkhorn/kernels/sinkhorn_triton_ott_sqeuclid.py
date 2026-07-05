"""DEPRECATED: OTT-style Sinkhorn solver - thin wrapper around FlashSinkhorn.

This module is DEPRECATED. Use sinkhorn_flashstyle_alternating from
flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid instead for better performance.

This facade maintains backward compatibility for existing code that imports:
- sinkhorn_potentials_sqeuclid (DEPRECATED → sinkhorn_flashstyle_alternating)
- apply_lse_kernel_sqeuclid (KEPT - no FlashSinkhorn equivalent for general LSE)
- update_potential (KEPT - thin wrapper over apply_lse_kernel_sqeuclid)
- apply_transport_from_potentials_sqeuclid (KEPT)

The FlashSinkhorn implementation provides:
- 7-34% faster performance at n ≥ 10,000 (with autotuning)
- Same numerical results (verified by parity tests)
- Support for semi-unbalanced OT and early stopping
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Import the new FlashSinkhorn implementation
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
)


# =============================================================================
# MAIN SOLVER (FACADE - delegates to FlashSinkhorn)
# =============================================================================


def sinkhorn_potentials_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    loga: torch.Tensor,
    logb: torch.Tensor,
    eps: float,
    n_iters: int,
    *,
    fused: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    cost_scale: float = 1.0,
    # Unbalanced OT parameters (supports semi-unbalanced)
    rho: Optional[float] = None,
    reach: Optional[float] = None,
    rho_x: Optional[float] = None,
    rho_y: Optional[float] = None,
    reach_x: Optional[float] = None,
    reach_y: Optional[float] = None,
    # Early stopping
    threshold: Optional[float] = None,
    check_every: int = 5,
    return_n_iters: bool = False,
    **kwargs,  # Absorb any unused parameters for backward compat
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DEPRECATED: Use sinkhorn_flashstyle_alternating instead.

    Compute Sinkhorn potentials using alternating (Gauss-Seidel) updates.

    This function now delegates to sinkhorn_flashstyle_alternating which provides
    7-34% better performance at n ≥ 10,000 with identical numerical results.

    Args:
        x: Source points, shape (n, d)
        y: Target points, shape (m, d)
        loga: Log of source marginal, shape (n,)
        logb: Log of target marginal, shape (m,)
        eps: Regularization strength
        n_iters: Number of Sinkhorn iterations
        fused: Ignored (FlashSinkhorn always uses optimized kernels)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable kernel autotuning
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        rho, reach: Unbalanced OT marginal penalty
        rho_x, rho_y, reach_x, reach_y: Semi-unbalanced marginal penalties
        threshold, check_every: Early stopping parameters
        return_n_iters: Return iteration count
        **kwargs: Additional arguments (absorbed for backward compat)

    Returns:
        f, g: Converged potentials in OTT convention (log marginals absorbed)
        (optional) n_iters_used: Number of iterations if return_n_iters=True

    Note:
        OTT convention: P_ij = exp((f_i + g_j - C_ij) / eps)
        Potentials include log marginals: f = eps*log(a) - eps*LSE[(g-C)/eps]

    .. deprecated:: 2.0.0
        Use :func:`sinkhorn_flashstyle_alternating` instead for better performance.
    """
    warnings.warn(
        "sinkhorn_potentials_sqeuclid is deprecated and will be removed in a future "
        "version. Use sinkhorn_flashstyle_alternating from "
        "flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid for 7-34% better performance.",
        DeprecationWarning,
        stacklevel=2,
    )

    if kwargs:
        # Warn about unused parameters
        warnings.warn(
            f"Unused parameters: {list(kwargs.keys())}. These are ignored by FlashSinkhorn.",
            UserWarning,
            stacklevel=2,
        )

    # Convert log marginals to marginals (FlashSinkhorn expects normalized weights)
    # OTT convention: loga = log(a), logb = log(b)
    a = torch.exp(loga)
    b = torch.exp(logb)

    # Delegate to FlashSinkhorn with OTT convention (log marginals absorbed)
    result = sinkhorn_flashstyle_alternating(
        x=x,
        y=y,
        a=a,
        b=b,
        eps=eps,
        n_iters=n_iters,
        cost_scale=cost_scale,
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
        ott_convention=True,  # OTT convention: log marginals absorbed into potentials
    )

    return result


# =============================================================================
# LSE KERNELS (KEPT - no direct FlashSinkhorn equivalent for general LSE)
# =============================================================================
# These kernels provide the raw LSE computation which is used by various
# other components (HVP, apply transport, etc.). FlashSinkhorn's LSE kernels
# are specialized for the shifted potential formulation.


@triton.jit
def _lse_axis1_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    vec_ptr,
    out_ptr,
    sgn_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_vec,
    stride_out,
    stride_sgn,
    eps,
    D: tl.constexpr,
    HAS_VEC: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    inv_eps = 1.0 / eps

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    # FlashAttention-style: do the online reduction in log2 space using exp2/log2.
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    for j0 in range(0, m, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
        logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        if HAS_VEC:
            vec = tl.load(vec_ptr + offs_n * stride_vec, mask=mask_n, other=0.0).to(
                tl.float32
            )
            vec_abs = tl.abs(vec)
            if USE_EXP2:
                log_vec = tl.where(vec_abs > 0, tl.log2(vec_abs), -float("inf"))
                vals = logits * log2e + log_vec[None, :]
            else:
                log_vec = tl.where(vec_abs > 0, tl.log(vec_abs), -float("inf"))
                vals = logits + log_vec[None, :]

            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            vec_sign = tl.where(
                vec > 0, 1.0, tl.where(vec < 0, -1.0, 0.0)
            )
            if USE_EXP2:
                s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(
                    vec_sign[None, :] * tl.exp2(vals - new_m[:, None]), axis=1
                )
            else:
                s_i = s_i * tl.exp(m_i - new_m) + tl.sum(
                    vec_sign[None, :] * tl.exp(vals - new_m[:, None]), axis=1
                )
            m_i = new_m
        else:
            if USE_EXP2:
                logits2 = logits * log2e
                block_max = tl.max(logits2, axis=1)
                new_m = tl.maximum(m_i, block_max)
                s_i = s_i * tl.exp2(m_i - new_m) + tl.sum(
                    tl.exp2(logits2 - new_m[:, None]), axis=1
                )
            else:
                block_max = tl.max(logits, axis=1)
                new_m = tl.maximum(m_i, block_max)
                s_i = s_i * tl.exp(m_i - new_m) + tl.sum(
                    tl.exp(logits - new_m[:, None]), axis=1
                )
            m_i = new_m

    if HAS_VEC:
        s_abs = tl.abs(s_i)
        lse = m_i + (tl.log2(s_abs) if USE_EXP2 else tl.log(s_abs))
        sgn = tl.where(s_i > 0, 1.0, tl.where(s_i < 0, -1.0, 0.0))
        lse = tl.where(s_abs > 0, lse, -float("inf"))
    else:
        lse = m_i + (tl.log2(s_i) if USE_EXP2 else tl.log(s_i))
        sgn = tl.full([BLOCK_M], 1.0, tl.float32)

    out = (eps * ln2) * lse if USE_EXP2 else eps * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)
    tl.store(sgn_ptr + offs_m * stride_sgn, sgn, mask=mask_m)


@triton.jit
def _lse_axis0_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    x2_ptr,
    y2_ptr,
    vec_ptr,
    out_ptr,
    sgn_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_vec,
    stride_out,
    stride_sgn,
    eps,
    D: tl.constexpr,
    HAS_VEC: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    inv_eps = 1.0 / eps

    g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    for i0 in range(0, n, BLOCK_M):
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            dot += tl.dot(x, y, allow_tf32=ALLOW_TF32)

        cost = x2[:, None] + y2[None, :] - 2.0 * dot
        logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_m[:, None], logits, -float("inf"))

        if HAS_VEC:
            vec = tl.load(vec_ptr + offs_m * stride_vec, mask=mask_m, other=0.0).to(
                tl.float32
            )
            vec_abs = tl.abs(vec)
            if USE_EXP2:
                log_vec = tl.where(vec_abs > 0, tl.log2(vec_abs), -float("inf"))
                vals = logits * log2e + log_vec[:, None]
            else:
                log_vec = tl.where(vec_abs > 0, tl.log(vec_abs), -float("inf"))
                vals = logits + log_vec[:, None]

            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            vec_sign = tl.where(
                vec > 0, 1.0, tl.where(vec < 0, -1.0, 0.0)
            )
            if USE_EXP2:
                s_j = s_j * tl.exp2(m_j - new_m) + tl.sum(
                    vec_sign[:, None] * tl.exp2(vals - new_m[None, :]), axis=0
                )
            else:
                s_j = s_j * tl.exp(m_j - new_m) + tl.sum(
                    vec_sign[:, None] * tl.exp(vals - new_m[None, :]), axis=0
                )
            m_j = new_m
        else:
            if USE_EXP2:
                logits2 = logits * log2e
                block_max = tl.max(logits2, axis=0)
                new_m = tl.maximum(m_j, block_max)
                s_j = s_j * tl.exp2(m_j - new_m) + tl.sum(
                    tl.exp2(logits2 - new_m[None, :]), axis=0
                )
            else:
                block_max = tl.max(logits, axis=0)
                new_m = tl.maximum(m_j, block_max)
                s_j = s_j * tl.exp(m_j - new_m) + tl.sum(
                    tl.exp(logits - new_m[None, :]), axis=0
                )
            m_j = new_m

    if HAS_VEC:
        s_abs = tl.abs(s_j)
        lse = m_j + (tl.log2(s_abs) if USE_EXP2 else tl.log(s_abs))
        sgn = tl.where(s_j > 0, 1.0, tl.where(s_j < 0, -1.0, 0.0))
        lse = tl.where(s_abs > 0, lse, -float("inf"))
    else:
        lse = m_j + (tl.log2(s_j) if USE_EXP2 else tl.log(s_j))
        sgn = tl.full([BLOCK_N], 1.0, tl.float32)

    out = (eps * ln2) * lse if USE_EXP2 else eps * lse
    tl.store(out_ptr + offs_n * stride_out, out, mask=mask_n)
    tl.store(sgn_ptr + offs_n * stride_sgn, sgn, mask=mask_n)


def _default_block_sizes(n, m, d):
    """Default block sizes for OTT forward kernel.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
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
        block_k = 32
    elif d >= 32:
        block_k = 16
    else:
        block_k = 16

    return block_m, block_n, block_k


def _lse_autotune_configs_axis1() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ]


def _lse_autotune_configs_axis0() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ]


_lse_axis1_kernel_autotune = triton.autotune(
    configs=_lse_autotune_configs_axis1(),
    key=["D", "ALLOW_TF32", "DTYPE_ID", "HAS_VEC", "USE_EXP2"],
)(_lse_axis1_kernel)

_lse_axis0_kernel_autotune = triton.autotune(
    configs=_lse_autotune_configs_axis0(),
    key=["D", "ALLOW_TF32", "DTYPE_ID", "HAS_VEC", "USE_EXP2"],
)(_lse_axis0_kernel)


def apply_lse_kernel_sqeuclid(
    x,
    y,
    f,
    g,
    eps,
    axis,
    vec=None,
    x2=None,
    y2=None,
    block_m=None,
    block_n=None,
    block_k=None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    backend: str = "triton",
    autotune: bool = False,
):
    """Compute log-sum-exp reduction along specified axis.

    This kernel computes:
        out[i] = eps * log(sum_j exp((f[i] + g[j] - C[i,j]) / eps * vec[j]))

    where C[i,j] = ||x[i] - y[j]||² is the squared Euclidean cost.

    Args:
        x, y: Point clouds [n, d] and [m, d]
        f, g: Potentials [n] and [m]
        eps: Regularization parameter
        axis: 0 or 1 (reduction axis)
        vec: Optional weighting vector
        x2, y2: Optional precomputed squared norms
        block_m, block_n, block_k: Manual block sizes
        num_warps, num_stages: Kernel launch config
        use_exp2: Use exp2/log2 optimization
        allow_tf32: Enable TF32
        backend: "triton" (only supported)
        autotune: Enable autotuning

    Returns:
        out: LSE result
        sgn: Sign of the weighted sum (all 1s if no vec)
    """
    if not x.is_cuda or not y.is_cuda or not f.is_cuda or not g.is_cuda:
        raise ValueError("apply_lse_kernel_sqeuclid requires CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D tensors.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")
    if vec is not None:
        if vec.ndim != 1:
            raise ValueError("vec must be a 1D tensor.")
        if axis == 1 and vec.shape[0] != m:
            raise ValueError("vec shape must match axis=1 reduction (m).")
        if axis == 0 and vec.shape[0] != n:
            raise ValueError("vec shape must match axis=0 reduction (n).")

    eps = float(eps)
    if x2 is None:
        x2 = (x.float() * x.float()).sum(dim=1)
    else:
        if x2.shape != (n,):
            raise ValueError("x2 must have shape (n,).")
        x2 = x2.float()
    if y2 is None:
        y2 = (y.float() * y.float()).sum(dim=1)
    else:
        if y2.shape != (m,):
            raise ValueError("y2 must have shape (m,).")
        y2 = y2.float()
    f = f.float()
    g = g.float()

    if backend != "triton":
        raise ValueError(f"Unknown backend={backend!r}. Only 'triton' is supported.")

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

    user_specified_tiles = (
        block_m is not None or block_n is not None or block_k is not None
    )
    user_specified_launch = num_warps is not None or num_stages is not None
    use_autotune = bool(autotune) and not user_specified_tiles and not user_specified_launch

    if not use_autotune:
        if num_warps is None:
            num_warps = 4
        if num_stages is None:
            num_stages = 2

        # Tuned defaults for strict fp32 (no TF32)
        if (
            x.dtype == torch.float32
            and not allow_tf32
            and d >= 32
            and not user_specified_tiles
            and not user_specified_launch
        ):
            block_m = 128
            block_k = 32
            num_stages = 3
            if axis == 1:
                block_n = 128
                num_warps = 8
            else:
                block_n = 64
                num_warps = 4

        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16

    if axis == 1:
        out = torch.empty((n,), device=x.device, dtype=torch.float32)
        sgn = torch.empty((n,), device=x.device, dtype=torch.float32)
        if vec is None:
            vec_ptr = x2
            has_vec = False
        else:
            vec = vec.float().contiguous()
            vec_ptr = vec
            has_vec = True

        if use_autotune:
            def grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)

            _lse_axis1_kernel_autotune[grid](
                x, y, f, g, x2, y2, vec_ptr, out, sgn,
                n, m,
                x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                f.stride(0), g.stride(0), x2.stride(0), y2.stride(0),
                vec_ptr.stride(0), out.stride(0), sgn.stride(0),
                eps, D=d, HAS_VEC=has_vec, USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32, DTYPE_ID=dtype_id,
            )
        else:
            grid = (triton.cdiv(n, block_m),)
            _lse_axis1_kernel[grid](
                x, y, f, g, x2, y2, vec_ptr, out, sgn,
                n, m,
                x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                f.stride(0), g.stride(0), x2.stride(0), y2.stride(0),
                vec_ptr.stride(0), out.stride(0), sgn.stride(0),
                eps, D=d, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                HAS_VEC=has_vec, USE_EXP2=use_exp2, ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id, num_warps=num_warps, num_stages=num_stages,
            )
        remove = f
    else:
        out = torch.empty((m,), device=x.device, dtype=torch.float32)
        sgn = torch.empty((m,), device=x.device, dtype=torch.float32)
        if vec is None:
            vec_ptr = x2
            has_vec = False
        else:
            vec = vec.float().contiguous()
            vec_ptr = vec
            has_vec = True

        if use_autotune:
            def grid(meta):
                return (triton.cdiv(m, meta["BLOCK_N"]),)

            _lse_axis0_kernel_autotune[grid](
                x, y, f, g, x2, y2, vec_ptr, out, sgn,
                n, m,
                x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                f.stride(0), g.stride(0), x2.stride(0), y2.stride(0),
                vec_ptr.stride(0), out.stride(0), sgn.stride(0),
                eps, D=d, HAS_VEC=has_vec, USE_EXP2=use_exp2,
                ALLOW_TF32=allow_tf32, DTYPE_ID=dtype_id,
            )
        else:
            grid = (triton.cdiv(m, block_n),)
            _lse_axis0_kernel[grid](
                x, y, f, g, x2, y2, vec_ptr, out, sgn,
                n, m,
                x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                f.stride(0), g.stride(0), x2.stride(0), y2.stride(0),
                vec_ptr.stride(0), out.stride(0), sgn.stride(0),
                eps, D=d, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                HAS_VEC=has_vec, USE_EXP2=use_exp2, ALLOW_TF32=allow_tf32,
                DTYPE_ID=dtype_id, num_warps=num_warps, num_stages=num_stages,
            )
        remove = g

    safe_remove = torch.where(torch.isfinite(remove), remove, torch.zeros_like(remove))
    out = out - safe_remove
    return out, sgn


def update_potential(x, y, f, g, log_marginal, eps, axis, **kwargs):
    """Compute potential update using apply_lse_kernel_sqeuclid.

    Args:
        x, y: Point clouds
        f, g: Current potentials
        log_marginal: Log marginal (loga for axis=1, logb for axis=0)
        eps: Regularization
        axis: Update axis (1 for f-update, 0 for g-update)
        **kwargs: Passed to apply_lse_kernel_sqeuclid

    Returns:
        Updated potential
    """
    lse, _ = apply_lse_kernel_sqeuclid(x, y, f, g, eps, axis, **kwargs)
    safe_lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    return eps * log_marginal - safe_lse


def apply_transport_from_potentials_sqeuclid(
    x, y, f, g, vec, eps, axis, **kwargs
):
    """Apply transport plan to a vector.

    Computes: (P @ vec) if axis=1, or (P.T @ vec) if axis=0
    where P_ij = exp((f_i + g_j - C_ij) / eps).

    Args:
        x, y: Point clouds
        f, g: Potentials
        vec: Vector to apply transport to
        eps: Regularization
        axis: 1 for P @ vec, 0 for P.T @ vec
        **kwargs: Passed to apply_lse_kernel_sqeuclid

    Returns:
        Result of transport plan application
    """
    lse_res, lse_sgn = apply_lse_kernel_sqeuclid(
        x, y, f, g, eps, axis, vec=vec, **kwargs
    )
    remove = f if axis == 1 else g
    lse_res = lse_res + remove
    return lse_sgn * torch.exp(lse_res / eps)


__all__ = [
    # Main solver (deprecated facade)
    "sinkhorn_potentials_sqeuclid",
    # LSE kernels (kept)
    "apply_lse_kernel_sqeuclid",
    "update_potential",
    "apply_transport_from_potentials_sqeuclid",
    # Autotune configs (for advanced users)
    "_lse_autotune_configs_axis0",
    "_lse_autotune_configs_axis1",
    "_default_block_sizes",
]
