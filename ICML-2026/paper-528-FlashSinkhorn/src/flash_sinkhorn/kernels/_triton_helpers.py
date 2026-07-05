"""Shared Triton JIT helpers for FlashSinkhorn kernels.

All functions are @triton.jit and get fully inlined at compile time — zero overhead.

Helpers:
    _tiled_dot: Compute x @ y.T via tiled matmul (x[M,K] @ y[N,K].T -> [M,N])
    _online_softmax_rescale: Online softmax/LSE rescale step for axis=1 reduction
    _online_softmax_rescale_axis0: Same for axis=0 reduction
    _final_lse: Convert online (m_i, s_i) accumulators to final LSE values
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _tiled_dot(
    x_ptr,
    y_ptr,
    offs_m,
    offs_n,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    D: tl.constexpr,
    mask_m,
    mask_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Compute tiled x @ y.T -> [BLOCK_M, BLOCK_N] in fp32.

    Loads x as [BLOCK_M, BLOCK_K] and y transposed as [BLOCK_K, BLOCK_N],
    accumulating into an fp32 dot product. This is the standard layout for
    axis=1 reductions (iterating over j/columns).

    The y load uses offs_n on dimension 0 and offs_k on dimension 1,
    then transposes via indexing: y_ptr[offs_n, offs_k] -> [BLOCK_K, BLOCK_N].

    Args:
        x_ptr, y_ptr: Pointers to contiguous [*, D] tensors.
        offs_m: Row offsets [BLOCK_M] for x.
        offs_n: Column offsets [BLOCK_N] for y.
        stride_x0, stride_x1: Strides for x (row, col).
        stride_y0, stride_y1: Strides for y (row, col).
        D: Feature dimension (constexpr).
        mask_m: Boolean mask [BLOCK_M] for valid rows.
        mask_n: Boolean mask [BLOCK_N] for valid columns.
        BLOCK_K: Tile size for k-dimension.
        ALLOW_TF32: Whether to allow TF32 in tl.dot.

    Returns:
        dot: Accumulated x @ y.T result [BLOCK_M, BLOCK_N] in fp32.
    """
    dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k0 = tl.multiple_of(k0, BLOCK_K)
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        y_block = tl.load(
            y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)
    return dot


@triton.jit
def _online_softmax_rescale(
    vals,
    m_i,
    USE_EXP2: tl.constexpr,
):
    """Compute one step of the online softmax/LSE rescale.

    Given a block of logit values and the current running max, compute:
    - new_m: updated running max
    - alpha: rescale factor for previous accumulator (exp(old_max - new_max))
    - w: softmax weights for current block (exp(vals - new_max))

    This is the core of FlashAttention-style online reduction. The caller
    is responsible for accumulating: s_i = s_i * alpha + reduce(w, ...).

    Args:
        vals: Logit values [BLOCK_M, BLOCK_N] or [BLOCK_N, BLOCK_M].
        m_i: Current running max [BLOCK_M] or [BLOCK_N].
        USE_EXP2: If True, use exp2 (with log2e pre-scaling assumed in vals).

    Returns:
        new_m: Updated running max.
        alpha: Rescale factor for previous accumulator.
        w: Softmax weights for current block.
    """
    block_max = tl.max(vals, axis=1)
    new_m = tl.maximum(m_i, block_max)
    if USE_EXP2:
        alpha = tl.exp2(m_i - new_m)
        w = tl.exp2(vals - new_m[:, None])
    else:
        alpha = tl.exp(m_i - new_m)
        w = tl.exp(vals - new_m[:, None])
    return new_m, alpha, w


@triton.jit
def _online_softmax_rescale_axis0(
    vals,
    m_j,
    USE_EXP2: tl.constexpr,
):
    """Online softmax rescale for axis=0 (column-wise) reduction.

    Same as _online_softmax_rescale but reduces over axis=0 instead of axis=1.
    Used in g-update kernels and P^T application where we accumulate over rows.

    Args:
        vals: Logit values [BLOCK_M, BLOCK_N].
        m_j: Current running max [BLOCK_N].
        USE_EXP2: If True, use exp2.

    Returns:
        new_m: Updated running max [BLOCK_N].
        alpha: Rescale factor for previous accumulator [BLOCK_N].
        w: Softmax weights [BLOCK_M, BLOCK_N].
    """
    block_max = tl.max(vals, axis=0)
    new_m = tl.maximum(m_j, block_max)
    if USE_EXP2:
        alpha = tl.exp2(m_j - new_m)
        w = tl.exp2(vals - new_m[None, :])
    else:
        alpha = tl.exp(m_j - new_m)
        w = tl.exp(vals - new_m[None, :])
    return new_m, alpha, w


@triton.jit
def _final_lse(
    m_i,
    s_i,
    USE_EXP2: tl.constexpr,
):
    """Compute final LSE from online softmax accumulators.

    LSE = max + log(sum_exp). Guards against s_i == 0 to avoid log(0).

    Args:
        m_i: Running max [BLOCK].
        s_i: Running sum of exp [BLOCK].
        USE_EXP2: If True, use log2/exp2 path.

    Returns:
        lse: log-sum-exp values [BLOCK].
    """
    s_i_safe = tl.maximum(s_i, 1e-40)
    if USE_EXP2:
        ln2: tl.constexpr = 0.6931471805599453
        lse = (m_i + tl.log2(s_i_safe)) * ln2
    else:
        lse = m_i + tl.log(s_i_safe)
    return lse
