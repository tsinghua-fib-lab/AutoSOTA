"""OTT-convention apply kernels (mat5 for HVP, deprecated vec/mat wrappers).

This module contains:
- mat5_sqeuclid: Active kernel for HVP computation (P_ij * (A_i . y_j) * y_j)
- apply_plan_vec_sqeuclid: Deprecated, delegates to FlashStyle
- apply_plan_mat_sqeuclid: Deprecated, delegates to FlashStyle

The mat5 kernel uses OTT-style potentials (f, g include absorbed log marginals).
The deprecated wrappers convert OTT potentials to shifted form and call FlashStyle.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._common import (
    _apply_mat_autotune_configs,
    _cache_key_bucket,
    _default_block_sizes,
    _validate_device,
)
from flash_sinkhorn.kernels._triton_helpers import _tiled_dot


# =============================================================================
# Mat5 kernel: compute (-4*cs/eps) * sum_j P_ij (A_i . y_j) y_j
# =============================================================================


@triton.jit
def _mat5_axis1_kernel(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    a_ptr,
    x2_ptr,
    y2_ptr,
    out_ptr,
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_a0,
    stride_a1,
    stride_x2,
    stride_y2,
    stride_out0,
    stride_out1,
    eps,
    scale,
    cost_scale,
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    inv_eps = 1.0 / eps

    f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0, eviction_policy="evict_first").to(tl.float32)
        y2 = tl.load(
            y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)

        dot_xy = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        dot_ay = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D

            xk = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            ak = tl.load(
                a_ptr + offs_m[:, None] * stride_a0 + offs_k[None, :] * stride_a1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            yk = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )

            dot_xy += tl.dot(xk, yk, allow_tf32=ALLOW_TF32)
            dot_ay += tl.dot(ak, yk, allow_tf32=ALLOW_TF32)

        cost = cost_scale * (x2[:, None] + y2[None, :] - 2.0 * dot_xy)
        if USE_EXP2:
            logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
        else:
            logits = (f[:, None] + g[None, :] - cost) * inv_eps
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, block_max)
        if USE_EXP2:
            alpha = tl.exp2(m_i - new_m)
            w = tl.exp2(logits - new_m[:, None])
        else:
            alpha = tl.exp(m_i - new_m)
            w = tl.exp(logits - new_m[:, None])

        tmp = w * dot_ay

        o = o * alpha[:, None]
        yv_t = tl.load(
            y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
            mask=mask_d[:, None]
            & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        yv = tl.trans(yv_t)  # (BLOCK_N, BLOCK_D)
        o += tl.dot(tmp, yv, allow_tf32=ALLOW_TF32)
        m_i = new_m

    if USE_EXP2:
        out = tl.exp2(m_i)[:, None] * o
    else:
        out = tl.exp(m_i)[:, None] * o
    out = out * scale

    tl.store(
        out_ptr + offs_m[:, None] * stride_out0 + offs_d[None, :] * stride_out1,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def _mat5_prune_configs(configs, named_args, **kwargs):
    """Prune mat5 autotune configs to valid range for given D.

    The mat5 kernel does NOT tile along the D dimension (only 1D grid),
    so it requires BLOCK_D >= D to process all feature dimensions.

    Also prunes excessively large BLOCK_D configs (BLOCK_D > 4*D) to:
    - Reduce compile time on first run
    - Avoid OutOfResources on smaller GPUs with limited shared memory
    """
    D = named_args.get("D", 64)
    # Lower bound: BLOCK_D must be >= D (kernel doesn't tile over D)
    # Upper bound: BLOCK_D should be <= 4*D (avoid wasteful configs)
    #   Exception: always allow at least one config (minimum BLOCK_D that fits)
    min_valid_block_d = D
    max_block_d = max(D * 4, 128)  # At least allow BLOCK_D=128 for small D

    pruned = [
        cfg for cfg in configs
        if min_valid_block_d <= cfg.kwargs.get("BLOCK_D", 16) <= max_block_d
    ]

    # Safety: if all configs pruned (shouldn't happen), keep smallest valid ones
    if not pruned:
        pruned = [cfg for cfg in configs if cfg.kwargs.get("BLOCK_D", 16) >= D]

    return pruned


# Create autotuned version of mat5 kernel with config pruning
_mat5_axis1_kernel_autotune = triton.autotune(
    configs=_apply_mat_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D"],  # Include D in key since BLOCK_D must >= D
    prune_configs_by={"early_config_prune": _mat5_prune_configs},
)(_mat5_axis1_kernel)


# =============================================================================
# Deprecated OTT-convention wrappers (delegate to FlashStyle)
# =============================================================================


def apply_plan_vec_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    vec: torch.Tensor,
    *,
    eps: float,
    axis: int,
    cost_scale: float = 1.0,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    log_a: Optional[torch.Tensor] = None,
    log_b: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,
) -> torch.Tensor:
    """Apply P or Pt to a vector without materializing P (streaming, stable).

    .. deprecated::
        Use :func:`apply_plan_vec_flashstyle` instead for better performance
        with FlashStyle shifted potentials.

    Computes:
      axis=1: out[i] = sum_j exp((f_i + g_j - cost_scale*C_ij)/eps) * vec[j]
      axis=0: out[j] = sum_i exp((f_i + g_j - cost_scale*C_ij)/eps) * vec[i]

    Args:
        f: OTT-style source potential [n] (includes absorbed log marginal)
        g: OTT-style target potential [m] (includes absorbed log marginal)
        log_a: Optional log source weights [n]. If None, assumes uniform (log(1/n)).
        log_b: Optional log target weights [m]. If None, assumes uniform (log(1/m)).
        cost_scale: Scaling for cost function. 1.0 for full ||x-y||^2, 0.5 for half.
        autotune: If True (default), use autotuned kernel configs for best performance.
                  If False, use manual block sizes (useful for reproducible benchmarks).
    """
    # Lazy import to avoid circular dependency (apply_ott -> apply_flash)
    from flash_sinkhorn.kernels.apply_flash import apply_plan_vec_flashstyle

    import warnings
    warnings.warn(
        "apply_plan_vec_sqeuclid is deprecated and will be removed in a future version. "
        "Use apply_plan_vec_flashstyle with shifted potentials instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (n,d) and (m,d).")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D.")
    if vec.ndim != 1:
        raise ValueError("vec must be 1D.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [("y", y), ("f", f), ("g", g), ("vec", vec), ("x2", x2), ("y2", y2)])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")

    if axis == 1 and vec.shape[0] != m:
        raise ValueError("vec must have shape (m,) for axis=1.")
    if axis == 0 and vec.shape[0] != n:
        raise ValueError("vec must have shape (n,) for axis=0.")

    # Convert OTT-style potentials to FlashStyle shifted potentials
    # Default to uniform marginals if not provided
    if log_a is None:
        log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
    if log_b is None:
        log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)

    # Compute squared norms for shift
    alpha = cost_scale * (x.float() ** 2).sum(dim=1)
    beta = cost_scale * (y.float() ** 2).sum(dim=1)

    # Convert OTT potentials to FlashStyle shifted potentials
    f_hat = f.float() - eps * log_a - alpha
    g_hat = g.float() - eps * log_b - beta

    # Call FlashStyle kernel
    return apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, vec,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        block_m=block_m if block_m is not None else 64,
        block_n=block_n if block_n is not None else 64,
        block_k=block_k if block_k is not None else 64,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def apply_plan_mat_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    mat: torch.Tensor,
    *,
    eps: float,
    axis: int,
    cost_scale: float = 1.0,
    scale: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    log_a: Optional[torch.Tensor] = None,
    log_b: Optional[torch.Tensor] = None,
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
    """Apply P or Pt to a matrix without materializing P (streaming, stable).

    .. deprecated::
        Use :func:`apply_plan_mat_flashstyle` instead for better performance
        with FlashStyle shifted potentials.

    Computes:
      axis=1: out[i, :] = sum_j exp((f_i + g_j - cost_scale*C_ij)/eps) * mat[j, :]
      axis=0: out[j, :] = sum_i exp((f_i + g_j - cost_scale*C_ij)/eps) * mat[i, :]

    If ``scale`` is provided (axis=1 only), the kernel uses `mat[j,:] * scale[j]`
    on the fly (avoids allocating `mat * scale[:,None]`).

    Args:
        f: OTT-style source potential [n] (includes absorbed log marginal)
        g: OTT-style target potential [m] (includes absorbed log marginal)
        log_a: Optional log source weights [n]. If None, assumes uniform (log(1/n)).
        log_b: Optional log target weights [m]. If None, assumes uniform (log(1/m)).
        cost_scale: Scaling for cost function. 1.0 for full ||x-y||^2, 0.5 for half.
        autotune: If True (default), use autotuned kernel configs for best performance.
                  If False, use manual block sizes (useful for reproducible benchmarks).
    """
    # Lazy import to avoid circular dependency (apply_ott -> apply_flash)
    from flash_sinkhorn.kernels.apply_flash import apply_plan_mat_flashstyle

    import warnings
    warnings.warn(
        "apply_plan_mat_sqeuclid is deprecated and will be removed in a future version. "
        "Use apply_plan_mat_flashstyle with shifted potentials instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or mat.ndim != 2:
        raise ValueError("x,y,mat must be 2D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [("y", y), ("f", f), ("g", g), ("mat", mat), ("scale", scale), ("x2", x2), ("y2", y2)])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")

    if axis == 1 and mat.shape != (m, d):
        raise ValueError("For axis=1, mat must have shape (m, d).")
    if axis == 0 and mat.shape != (n, d):
        raise ValueError("For axis=0, mat must have shape (n, d).")

    if scale is not None:
        if axis != 1:
            raise ValueError("scale is only supported for axis=1.")
        if scale.shape != (m,):
            raise ValueError("scale must have shape (m,).")

    # Convert OTT-style potentials to FlashStyle shifted potentials
    # OTT: P = exp((f_ott + g_ott - C) / eps)
    # FlashStyle: P = a * b * exp((f_hat + g_hat + 2*cs*xy) / eps)
    #
    # Conversion:
    #   f_ott = f_std + eps * log_a, where f_std is standard (GeomLoss) potential
    #   f_hat = f_std - alpha, where alpha = cost_scale * ||x||^2
    #   So: f_hat = f_ott - eps * log_a - alpha

    # Default to uniform marginals if not provided
    if log_a is None:
        log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
    if log_b is None:
        log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)

    # Compute squared norms for shift
    alpha = cost_scale * (x.float() ** 2).sum(dim=1)
    beta = cost_scale * (y.float() ** 2).sum(dim=1)

    # Convert OTT potentials to FlashStyle shifted potentials
    f_hat = f.float() - eps * log_a - alpha
    g_hat = g.float() - eps * log_b - beta

    # Call FlashStyle kernel
    return apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, mat,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        scale=scale,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        block_d=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
        autotune=autotune,
    )


def mat5_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    cost_scale: float = 1.0,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
    autotune: bool = True,
) -> torch.Tensor:
    """Compute Mat5 term for HVP: Mat5 = (-4*cost_scale/eps) * sum_j P_ij (A_i.y_j) y_j.

    Args:
        autotune: If True (default), use autotuned kernel configs for best performance.
                  If False, use manual block sizes (useful for reproducible benchmarks).
    """

    if not x.is_cuda:
        raise ValueError("CUDA required.")
    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")

    # Validate all tensors are on the same CUDA device
    _validate_device(x, [("y", y), ("f", f), ("g", g), ("A", A), ("x2", x2), ("y2", y2)])

    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")

    eps_f = float(eps)
    x2 = (x.float() * x.float()).sum(dim=1).contiguous() if x2 is None else x2.float().contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous() if y2 is None else y2.float().contiguous()

    f = f.float().contiguous()
    g = g.float().contiguous()
    A = A.float().contiguous()

    # Determine whether to use autotuning
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
    )
    use_autotune = autotune and not manual_blocks

    if not use_autotune:
        if block_m is None or block_n is None or block_k is None:
            block_m, block_n, block_k = _default_block_sizes(n, m, d)
        if block_k < 16:
            block_k = 16

    block_d = max(16, 1 << (int(d) - 1).bit_length())

    out = torch.empty((n, d), device=x.device, dtype=torch.float32)
    scale = -4.0 * cost_scale / eps_f

    if use_autotune:
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)
        _mat5_axis1_kernel_autotune[grid](
            x, y, f, g, A, x2, y2, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            f.stride(0), g.stride(0),
            A.stride(0), A.stride(1),
            x2.stride(0), y2.stride(0),
            out.stride(0), out.stride(1),
            eps_f,
            float(scale),
            float(cost_scale),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            USE_EXP2=use_exp2,
            ALLOW_TF32=allow_tf32,
        )
    else:
        grid = (triton.cdiv(n, block_m),)
        _mat5_axis1_kernel[grid](
            x, y, f, g, A, x2, y2, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            f.stride(0), g.stride(0),
            A.stride(0), A.stride(1),
            x2.stride(0), y2.stride(0),
            out.stride(0), out.stride(1),
            eps_f,
            float(scale),
            float(cost_scale),
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
