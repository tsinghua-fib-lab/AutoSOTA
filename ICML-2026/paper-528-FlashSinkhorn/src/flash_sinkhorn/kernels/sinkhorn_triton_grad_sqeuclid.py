from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._common import _cache_key_bucket
from flash_sinkhorn.kernels._triton_helpers import _tiled_dot
from flash_sinkhorn.kernels._common import log_weights


def _grad_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for gradient kernel.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D accumulator causes incorrect results.

    We use BLOCK_K in {16, 32} to ensure this for most common d values (64+).
    """
    configs = []
    for block_m, block_n in ((64, 64), (128, 64), (64, 128)):
        # IMPORTANT: Use BLOCK_K <= 32 to ensure multiple k iterations for common d
        for block_k in (16, 32):  # Removed 64 to avoid BLOCK_K >= D bug
            for num_warps in (4, 8):
                for num_stages in (2, 3):
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                            },
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


def _grad_large_d_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for large-D gradient kernel.

    The large-D kernel tiles over D and accumulates into global memory.
    Key tradeoffs:
    - Larger BLOCK_M/N: Better GPU occupancy, but more global memory traffic
    - Larger BLOCK_D: Fewer D-tiles, but more shared memory per tile
    - BLOCK_K: Controls inner dot product tiling (should be <= BLOCK_D)

    Shared memory budget per config (BLOCK_M=BLOCK_N assumed):
      w: BLOCK_M × BLOCK_N × 4 bytes
      o_tile: BLOCK_M × BLOCK_D × 4 bytes
      yv: BLOCK_D × BLOCK_N × 4 bytes
    """
    configs = []
    # Config 1: BLOCK_M=64, BLOCK_D=64 (48KB shared mem)
    for block_m, block_n in ((64, 64), (64, 32), (32, 64)):
        for block_d in (64, 128):
            for block_k in (32, 64):
                if block_k > block_d:
                    continue  # BLOCK_K must be <= BLOCK_D
                for num_warps in (4, 8):
                    for num_stages in (2, 3):  # Include stages=3 for large d
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_D": block_d,
                                    "BLOCK_K": block_k,
                                },
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )
    # Config 2: Smaller blocks for very large d (>1024)
    for block_m, block_n in ((32, 32), (32, 64)):
        for block_d in (128, 256):
            for block_k in (32, 64):
                if block_k > block_d:
                    continue
                for num_warps in (4, 8):
                    for num_stages in (2, 3):  # Include stages=3 for large d
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_D": block_d,
                                    "BLOCK_K": block_k,
                                },
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )
    return configs


@triton.jit
def _geomloss_grad_sqeuclid_impl(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    loga_ptr,
    logb_ptr,
    a_ptr,
    b_ptr,
    x2_ptr,
    y2_ptr,
    grad_scale_ptr,
    grad_x_ptr,
    grad_y_ptr,
    # OTDD label cost parameters (optional)
    label_x_ptr,  # int32 labels for x: [n] or None
    label_y_ptr,  # int32 labels for y: [m] or None
    W_ptr,        # Label cost matrix: [V, V] or None
    pid_offset,
    n,
    m,
    V,            # Number of classes (0 if no label cost)
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_loga,
    stride_logb,
    stride_a,
    stride_b,
    stride_x2,
    stride_y2,
    stride_grad_x0,
    stride_grad_x1,
    stride_grad_y0,
    stride_grad_y1,
    eps,
    cost_scale,  # Cost scaling: 1.0 for full ||x-y||², 0.5 for half ||x-y||²/2
    lambda_x,    # Weight for Euclidean cost (default 1.0)
    lambda_y,    # Weight for label cost (default 0.0)
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_LABEL_COST: tl.constexpr,  # Whether to use label cost (compile-time)
):
    pid = tl.program_id(0) + pid_offset
    inv_eps = 1.0 / eps
    grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
    blocks_x = tl.cdiv(n, BLOCK_M)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # grad_x: for each x_i, compute y_bar_i = E_{P(.|x_i)}[y] and return
    # grad_x_i = 2 * a_i * (x_i - y_bar_i) * grad_scale.
    if pid < blocks_x:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        a = tl.load(a_ptr + offs_m * stride_a, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(
            x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
        ).to(tl.float32)

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        o = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        for j0 in range(0, m, BLOCK_N):
            j0 = tl.multiple_of(j0, BLOCK_N)
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < m

            g = tl.load(
                g_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_g,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            logb = tl.load(
                logb_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_logb,
                mask=mask_m[:, None] & mask_n[None, :],
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            if USE_EXP2:
                logb = logb * log2e

            y2 = tl.load(
                y2_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_y2,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            dot = _tiled_dot(
                x_ptr, y_ptr, offs_m, offs_n,
                stride_x0, stride_x1, stride_y0, stride_y1,
                D, mask_m, mask_n,
                BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
            )

            # Compute Euclidean cost
            euclidean_cost = cost_scale * (x2[:, None] + y2 - 2.0 * dot)

            # Add label cost if enabled (OTDD-style augmented cost)
            if USE_LABEL_COST:
                # Load labels for this block
                label_i = tl.load(label_x_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
                label_j = tl.load(label_y_ptr + offs_n, mask=mask_n, other=0).to(tl.int32)
                # Compute flattened indices into W: W[label_i, label_j]
                w_idx = label_i[:, None] * V + label_j[None, :]
                # Gather from W (label cost matrix)
                w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                # Combined cost: lambda_x * euclidean + lambda_y * label
                # Apply cost_scale to w_cost for OTDD parity (OTDD divides label cost by p)
                cost = lambda_x * euclidean_cost + lambda_y * cost_scale * w_cost
            else:
                cost = euclidean_cost

            if USE_EXP2:
                vals = tl.fma(g - cost, inv_eps_log2, logb)
            else:
                vals = (g - cost) * inv_eps + logb
            # Out-of-bounds columns already have logb=-inf from the masked load.

            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            if USE_EXP2:
                alpha = tl.exp2(m_i - new_m)
                w = tl.exp2(vals - new_m[:, None])
            else:
                alpha = tl.exp(m_i - new_m)
                w = tl.exp(vals - new_m[:, None])

            l_i = l_i * alpha + tl.sum(w, axis=1)
            o = o * alpha[:, None]

            yv_t = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
                mask=mask_d[:, None] & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            yv = tl.trans(yv_t)
            o += tl.dot(w, yv, allow_tf32=ALLOW_TF32)
            m_i = new_m

        # Guard against l_i == 0 (can happen if all exp values underflow)
        # Use 1e-40 as minimum to avoid division by zero
        l_i_safe = tl.maximum(l_i, 1e-40)
        # For combined cost C = λ_x * ||x-y||² + λ_y * W, gradient w.r.t. x is:
        # ∂C/∂x = λ_x * 2 * cost_scale * (x - y)
        # Note: λ_y * W doesn't depend on x, so it doesn't contribute to gradient
        if USE_LABEL_COST:
            scale = (2.0 * lambda_x * cost_scale * grad_scale) * a
        else:
            scale = (2.0 * cost_scale * grad_scale) * a
        y_bar = o / l_i_safe[:, None]
        grad = (x - y_bar) * scale[:, None]
        tl.store(
            grad_x_ptr + offs_m[:, None] * stride_grad_x0 + offs_d[None, :] * stride_grad_x1,
            grad,
            mask=mask_m[:, None] & mask_d[None, :],
        )
        return

    # grad_y: for each y_j, compute x_bar_j = E_{P(.|y_j)}[x] and return
    # grad_y_j = 2 * b_j * (y_j - x_bar_j) * grad_scale.
    pid_y = pid - blocks_x
    offs_n = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    b = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(
        y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
    ).to(tl.float32)

    y_t = tl.load(
        y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
        mask=mask_d[:, None] & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
        other=0.0,
    ).to(tl.float32)
    y = tl.trans(y_t)

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    l_j = tl.zeros([BLOCK_N], tl.float32)
    o = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    for i0 in range(0, n, BLOCK_M):
        i0 = tl.multiple_of(i0, BLOCK_M)
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(
            f_ptr + offs_m[:, None] * stride_f + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        loga = tl.load(
            loga_ptr + offs_m[:, None] * stride_loga + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        if USE_EXP2:
            loga = loga * log2e
        x2 = tl.load(
            x2_ptr + offs_m[:, None] * stride_x2 + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Compute Euclidean cost
        euclidean_cost = cost_scale * (x2 + y2[None, :] - 2.0 * dot)

        # Add label cost if enabled (OTDD-style augmented cost)
        if USE_LABEL_COST:
            # Load labels for this block
            label_i = tl.load(label_x_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
            label_j = tl.load(label_y_ptr + offs_n, mask=mask_n, other=0).to(tl.int32)
            # Compute flattened indices into W: W[label_i, label_j]
            w_idx = label_i[:, None] * V + label_j[None, :]
            # Gather from W (label cost matrix)
            w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
            # Combined cost: lambda_x * euclidean + lambda_y * label
            # Apply cost_scale to w_cost for OTDD parity (OTDD divides label cost by p)
            cost = lambda_x * euclidean_cost + lambda_y * cost_scale * w_cost
        else:
            cost = euclidean_cost

        if USE_EXP2:
            vals = tl.fma(f - cost, inv_eps_log2, loga)
        else:
            vals = (f - cost) * inv_eps + loga
        # Out-of-bounds rows already have loga=-inf from the masked load.

        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m_j, block_max)
        if USE_EXP2:
            alpha = tl.exp2(m_j - new_m)
            w = tl.exp2(vals - new_m[None, :])
        else:
            alpha = tl.exp(m_j - new_m)
            w = tl.exp(vals - new_m[None, :])

        l_j = l_j * alpha + tl.sum(w, axis=0)
        o = o * alpha[:, None]

        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        o += tl.dot(tl.trans(w), xv, allow_tf32=ALLOW_TF32)
        m_j = new_m

    # Guard against l_j == 0 (can happen if all exp values underflow)
    # Use 1e-40 as minimum to avoid division by zero
    l_j_safe = tl.maximum(l_j, 1e-40)
    # For combined cost C = λ_x * ||x-y||² + λ_y * W, gradient w.r.t. y is:
    # ∂C/∂y = λ_x * 2 * cost_scale * (y - x)
    if USE_LABEL_COST:
        scale = (2.0 * lambda_x * cost_scale * grad_scale) * b
    else:
        scale = (2.0 * cost_scale * grad_scale) * b
    x_bar = o / l_j_safe[:, None]
    grad = (y - x_bar) * scale[:, None]
    tl.store(
        grad_y_ptr + offs_n[:, None] * stride_grad_y0 + offs_d[None, :] * stride_grad_y1,
        grad,
        mask=mask_n[:, None] & mask_d[None, :],
    )


@triton.jit
def _geomloss_grad_sqeuclid_large_d_impl(
    x_ptr,
    y_ptr,
    f_ptr,
    g_ptr,
    loga_ptr,
    logb_ptr,
    a_ptr,
    b_ptr,
    x2_ptr,
    y2_ptr,
    grad_scale_ptr,
    grad_x_ptr,
    grad_y_ptr,
    # OTDD label cost parameters (optional)
    label_x_ptr,  # int32 labels for x: [n] or None
    label_y_ptr,  # int32 labels for y: [m] or None
    W_ptr,        # Label cost matrix: [V, V] or None
    pid_offset,
    n,
    m,
    V,            # Number of classes (0 if no label cost)
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_loga,
    stride_logb,
    stride_a,
    stride_b,
    stride_x2,
    stride_y2,
    stride_grad_x0,
    stride_grad_x1,
    stride_grad_y0,
    stride_grad_y1,
    eps,
    cost_scale,  # Cost scaling: 1.0 for full ||x-y||², 0.5 for half ||x-y||²/2
    lambda_x,    # Weight for Euclidean cost (default 1.0)
    lambda_y,    # Weight for label cost (default 0.0)
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_LABEL_COST: tl.constexpr,  # Whether to use label cost (compile-time)
):
    pid = tl.program_id(0) + pid_offset
    inv_eps = 1.0 / eps
    grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
    blocks_x = tl.cdiv(n, BLOCK_M)

    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    # grad_x: for each x_i, compute y_bar_i = E_{P(.|x_i)}[y] and return
    # grad_x_i = 2 * a_i * (x_i - y_bar_i) * grad_scale.
    if pid < blocks_x:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        a = tl.load(a_ptr + offs_m * stride_a, mask=mask_m, other=0.0).to(tl.float32)
        x2 = tl.load(
            x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0
        ).to(tl.float32)

        for d0 in range(0, D, BLOCK_D):
            d0 = tl.multiple_of(d0, BLOCK_D)
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            tl.store(
                grad_x_ptr
                + offs_m[:, None] * stride_grad_x0
                + offs_d[None, :] * stride_grad_x1,
                0.0,
                mask=mask_m[:, None] & mask_d[None, :],
            )

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)

        for j0 in range(0, m, BLOCK_N):
            j0 = tl.multiple_of(j0, BLOCK_N)
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < m

            g = tl.load(
                g_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_g,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            logb = tl.load(
                logb_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_logb,
                mask=mask_m[:, None] & mask_n[None, :],
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            if USE_EXP2:
                logb = logb * log2e

            y2 = tl.load(
                y2_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_y2,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            dot = _tiled_dot(
                x_ptr, y_ptr, offs_m, offs_n,
                stride_x0, stride_x1, stride_y0, stride_y1,
                D, mask_m, mask_n,
                BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
            )

            # Compute Euclidean cost
            euclidean_cost = cost_scale * (x2[:, None] + y2 - 2.0 * dot)

            # Add label cost if enabled (OTDD-style augmented cost)
            if USE_LABEL_COST:
                # Load labels for this block
                label_i = tl.load(label_x_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
                label_j = tl.load(label_y_ptr + offs_n, mask=mask_n, other=0).to(tl.int32)
                # Compute flattened indices into W: W[label_i, label_j]
                w_idx = label_i[:, None] * V + label_j[None, :]
                # Gather from W (label cost matrix)
                w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                # Combined cost: lambda_x * euclidean + lambda_y * label
                # Apply cost_scale to w_cost for OTDD parity (OTDD divides label cost by p)
                cost = lambda_x * euclidean_cost + lambda_y * cost_scale * w_cost
            else:
                cost = euclidean_cost

            if USE_EXP2:
                vals = tl.fma(g - cost, inv_eps_log2, logb)
            else:
                vals = (g - cost) * inv_eps + logb
            # Out-of-bounds columns already have logb=-inf from the masked load.

            block_max = tl.max(vals, axis=1)
            new_m = tl.maximum(m_i, block_max)
            if USE_EXP2:
                alpha = tl.exp2(m_i - new_m)
                w = tl.exp2(vals - new_m[:, None])
            else:
                alpha = tl.exp(m_i - new_m)
                w = tl.exp(vals - new_m[:, None])

            l_i = l_i * alpha + tl.sum(w, axis=1)

            for d0 in range(0, D, BLOCK_D):
                d0 = tl.multiple_of(d0, BLOCK_D)
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                o_tile = tl.load(
                    grad_x_ptr
                    + offs_m[:, None] * stride_grad_x0
                    + offs_d[None, :] * stride_grad_x1,
                    mask=mask_m[:, None] & mask_d[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                yv_t = tl.load(
                    y_ptr
                    + offs_n[None, :] * stride_y0
                    + offs_d[:, None] * stride_y1,
                    mask=mask_d[:, None]
                    & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                yv = tl.trans(yv_t)
                o_tile = o_tile * alpha[:, None] + tl.dot(w, yv, allow_tf32=ALLOW_TF32)
                tl.store(
                    grad_x_ptr
                    + offs_m[:, None] * stride_grad_x0
                    + offs_d[None, :] * stride_grad_x1,
                    o_tile,
                    mask=mask_m[:, None] & mask_d[None, :],
                )
            m_i = new_m

        # Guard against l_i == 0 (can happen if all exp values underflow)
        # Use 1e-40 as minimum to avoid division by zero
        l_i_safe = tl.maximum(l_i, 1e-40)
        # For combined cost C = λ_x * ||x-y||² + λ_y * W, gradient w.r.t. x is:
        # ∂C/∂x = λ_x * 2 * cost_scale * (x - y)
        # Note: λ_y * W doesn't depend on x, so it doesn't contribute to gradient
        if USE_LABEL_COST:
            scale = (2.0 * lambda_x * cost_scale * grad_scale) * a
        else:
            scale = (2.0 * cost_scale * grad_scale) * a

        for d0 in range(0, D, BLOCK_D):
            d0 = tl.multiple_of(d0, BLOCK_D)
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            o_tile = tl.load(
                grad_x_ptr
                + offs_m[:, None] * stride_grad_x0
                + offs_d[None, :] * stride_grad_x1,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            y_bar = o_tile / l_i_safe[:, None]
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            grad = (x_tile - y_bar) * scale[:, None]
            tl.store(
                grad_x_ptr
                + offs_m[:, None] * stride_grad_x0
                + offs_d[None, :] * stride_grad_x1,
                grad,
                mask=mask_m[:, None] & mask_d[None, :],
            )
        return

    # grad_y: for each y_j, compute x_bar_j = E_{P(.|y_j)}[x] and return
    # grad_y_j = 2 * b_j * (y_j - x_bar_j) * grad_scale.
    pid_y = pid - blocks_x
    offs_n = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m

    b = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    y2 = tl.load(
        y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0
    ).to(tl.float32)

    for d0 in range(0, D, BLOCK_D):
        d0 = tl.multiple_of(d0, BLOCK_D)
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        tl.store(
            grad_y_ptr
            + offs_n[:, None] * stride_grad_y0
            + offs_d[None, :] * stride_grad_y1,
            0.0,
            mask=mask_n[:, None] & mask_d[None, :],
        )

    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    l_j = tl.zeros([BLOCK_N], tl.float32)

    for i0 in range(0, n, BLOCK_M):
        i0 = tl.multiple_of(i0, BLOCK_M)
        offs_m = i0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        f = tl.load(
            f_ptr + offs_m[:, None] * stride_f + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        loga = tl.load(
            loga_ptr + offs_m[:, None] * stride_loga + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        if USE_EXP2:
            loga = loga * log2e
        x2 = tl.load(
            x2_ptr + offs_m[:, None] * stride_x2 + offs_n[None, :] * 0,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Compute Euclidean cost
        euclidean_cost = cost_scale * (x2 + y2[None, :] - 2.0 * dot)

        # Add label cost if enabled (OTDD-style augmented cost)
        if USE_LABEL_COST:
            # Load labels for this block
            label_i = tl.load(label_x_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
            label_j = tl.load(label_y_ptr + offs_n, mask=mask_n, other=0).to(tl.int32)
            # Compute flattened indices into W: W[label_i, label_j]
            w_idx = label_i[:, None] * V + label_j[None, :]
            # Gather from W (label cost matrix)
            w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
            # Combined cost: lambda_x * euclidean + lambda_y * label
            # Apply cost_scale to w_cost for OTDD parity (OTDD divides label cost by p)
            cost = lambda_x * euclidean_cost + lambda_y * cost_scale * w_cost
        else:
            cost = euclidean_cost

        if USE_EXP2:
            vals = tl.fma(f - cost, inv_eps_log2, loga)
        else:
            vals = (f - cost) * inv_eps + loga
        # Out-of-bounds rows already have loga=-inf from the masked load.

        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m_j, block_max)
        if USE_EXP2:
            alpha = tl.exp2(m_j - new_m)
            w = tl.exp2(vals - new_m[None, :])
        else:
            alpha = tl.exp(m_j - new_m)
            w = tl.exp(vals - new_m[None, :])

        l_j = l_j * alpha + tl.sum(w, axis=0)

        for d0 in range(0, D, BLOCK_D):
            d0 = tl.multiple_of(d0, BLOCK_D)
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            o_tile = tl.load(
                grad_y_ptr
                + offs_n[:, None] * stride_grad_y0
                + offs_d[None, :] * stride_grad_y1,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            xv = tl.load(
                x_ptr + offs_m[:, None] * stride_x0 + offs_d[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            o_tile = o_tile * alpha[:, None] + tl.dot(tl.trans(w), xv, allow_tf32=ALLOW_TF32)
            tl.store(
                grad_y_ptr
                + offs_n[:, None] * stride_grad_y0
                + offs_d[None, :] * stride_grad_y1,
                o_tile,
                mask=mask_n[:, None] & mask_d[None, :],
            )
        m_j = new_m

    # Guard against l_j == 0 (can happen if all exp values underflow)
    # Use 1e-40 as minimum to avoid division by zero
    l_j_safe = tl.maximum(l_j, 1e-40)
    # For combined cost C = λ_x * ||x-y||² + λ_y * W, gradient w.r.t. y is:
    # ∂C/∂y = λ_x * 2 * cost_scale * (y - x)
    if USE_LABEL_COST:
        scale = (2.0 * lambda_x * cost_scale * grad_scale) * b
    else:
        scale = (2.0 * cost_scale * grad_scale) * b
    for d0 in range(0, D, BLOCK_D):
        d0 = tl.multiple_of(d0, BLOCK_D)
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        o_tile = tl.load(
            grad_y_ptr
            + offs_n[:, None] * stride_grad_y0
            + offs_d[None, :] * stride_grad_y1,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x_bar = o_tile / l_j_safe[:, None]
        y_t = tl.load(
            y_ptr + offs_n[None, :] * stride_y0 + offs_d[:, None] * stride_y1,
            mask=mask_d[:, None] & tl.broadcast_to(mask_n[None, :], (BLOCK_D, BLOCK_N)),
            other=0.0,
        ).to(tl.float32)
        y_tile = tl.trans(y_t)
        grad = (y_tile - x_bar) * scale[:, None]
        tl.store(
            grad_y_ptr
            + offs_n[:, None] * stride_grad_y0
            + offs_d[None, :] * stride_grad_y1,
            grad,
            mask=mask_n[:, None] & mask_d[None, :],
        )


def _default_block_sizes(
    d: int, dtype: torch.dtype, allow_tf32: bool
) -> Tuple[int, int, int, int]:
    """Default block sizes for gradient kernel.

    Returns starting block sizes. The main function will clamp BLOCK_M/N
    based on shared memory constraints (see SHARED_MEM_BUDGET_KB).

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D (single k iteration)
    combined with 2D accumulator causes incorrect results.
    """
    num_warps = 4
    block_m = 64  # Will be clamped by main function if needed
    block_n = 64

    # Choose BLOCK_K to ensure multiple k iterations (BLOCK_K < D)
    # Also BLOCK_K must be >= 16 for tl.dot
    if d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot

    return block_m, block_n, block_k, num_warps


# Shared memory budget for gradient kernel.
#
# The kernel uses three main arrays that may coexist in shared memory:
#   - o (accumulator): BLOCK_M × BLOCK_D × 4 bytes
#   - w (weights):     BLOCK_M × BLOCK_N × 4 bytes
#   - yv (y values):   BLOCK_N × BLOCK_D × 4 bytes
#
# Total footprint (assuming BLOCK_M = BLOCK_N):
#   total = (BLOCK_M² + 2 × BLOCK_M × BLOCK_D) × 4 bytes
#
# A100/H100 have ~164KB shared memory per SM, but Triton typically uses
# ~48-100KB per block. We use 96KB as a conservative budget.
SHARED_MEM_BUDGET_KB = 96


def _max_block_m_for_smem(block_d: int, budget_kb: int = SHARED_MEM_BUDGET_KB) -> int:
    """Compute max BLOCK_M that fits in shared memory budget.

    Solves: BLOCK_M × (BLOCK_M + 2 × BLOCK_D) × 4 <= budget
    Using quadratic formula: BLOCK_M = -BLOCK_D + sqrt(BLOCK_D² + budget/4)

    Args:
        block_d: The BLOCK_D tile size (power of 2, >= d)
        budget_kb: Shared memory budget in KB

    Returns:
        Maximum BLOCK_M that fits, rounded down to power of 2, capped at 64.

    Examples (with 96KB budget):
        | BLOCK_D | max_block_m | Effective BLOCK_M |
        |---------|-------------|-------------------|
        | 64      | 105         | 64 (cap)          |
        | 128     | 74          | 64                |
        | 256     | 44          | 32                |
        | 512     | 27          | 16                |
        | 1024    | 16          | 16                |
        | 2048    | 9           | -> large-D path   |
    """
    import math
    budget_bytes = budget_kb * 1024
    # Solve: BLOCK_M² + 2*BLOCK_D*BLOCK_M - budget/4 = 0
    # BLOCK_M = -BLOCK_D + sqrt(BLOCK_D² + budget/4)
    discriminant = block_d * block_d + budget_bytes // 4
    max_block = int(-block_d + math.sqrt(discriminant))

    # Round down to power of 2, cap at 64
    if max_block >= 64:
        return 64
    elif max_block >= 32:
        return 32
    elif max_block >= 16:
        return 16
    else:
        return max_block  # Will trigger large-D path if < 16


_geomloss_grad_sqeuclid_autotune = triton.autotune(
    configs=_grad_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID"],
)(_geomloss_grad_sqeuclid_impl)


# Autotuned version of the large-D gradient kernel
_geomloss_grad_sqeuclid_large_d_autotune = triton.autotune(
    configs=_grad_large_d_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID"],
)(_geomloss_grad_sqeuclid_large_d_impl)


def sinkhorn_geomloss_online_grad_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
    grad_scale: Optional[torch.Tensor] = None,
    compute_grad_x: bool = True,
    compute_grad_y: bool = True,
    cost_scale: float = 1.0,
    # OTDD label-augmented cost parameters
    label_x: Optional[torch.Tensor] = None,
    label_y: Optional[torch.Tensor] = None,
    label_cost_matrix: Optional[torch.Tensor] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D tensors.")
    if f.ndim != 1 or g.ndim != 1:
        raise ValueError("f and g must be 1D tensors.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have the same feature dimension.")
    if a.shape[0] != n or f.shape[0] != n:
        raise ValueError("a and f shapes must match x.")
    if b.shape[0] != m or g.shape[0] != m:
        raise ValueError("b and g shapes must match y.")

    x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    y2 = (y.float() * y.float()).sum(dim=1).contiguous()
    loga = log_weights(a).contiguous()
    logb = log_weights(b).contiguous()

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    elif x.dtype == torch.float32:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype for x/y: {x.dtype}")

    if block_m is None or block_n is None or block_k is None or num_warps is None:
        bm, bn, bk, nw = _default_block_sizes(d, x.dtype, allow_tf32)
        bm = bm if block_m is None else block_m
        bn = bn if block_n is None else block_n
        bk = bk if block_k is None else block_k
        nw = nw if num_warps is None else num_warps
    else:
        bm, bn, bk, nw = block_m, block_n, block_k, num_warps
    if bk < 16:
        bk = 16

    # Triton prefers power-of-2 ranges; use full-D tile when it fits in shared memory.
    block_d_full = max(16, 1 << (int(d) - 1).bit_length())

    # Shared memory constraint accounts for full footprint:
    #   o[BLOCK_M, BLOCK_D] + w[BLOCK_M, BLOCK_N] + yv[BLOCK_N, BLOCK_D]
    # See _max_block_m_for_smem() for the derivation.
    MIN_BLOCK = 16
    MIN_EFFICIENT_BLOCK = 32  # Minimum for good GPU utilization
    max_block_for_full = _max_block_m_for_smem(block_d_full)

    # Large-D path: tile over D and accumulate into global memory.
    # Use large-D path when:
    #   1. Full-D doesn't fit (max_block_for_full < MIN_BLOCK), OR
    #   2. Full-D forces inefficiently small blocks (d >= 256 and max_block < 32)
    # This trades memory bandwidth for better GPU occupancy at large d.
    use_global_o = (max_block_for_full < MIN_BLOCK) or (d >= 256 and max_block_for_full < MIN_EFFICIENT_BLOCK)
    if use_global_o:
        # Use smaller BLOCK_D and larger BLOCK_M/N for better GPU utilization.
        # Memory usage with BLOCK_M=BLOCK_N=64, BLOCK_D=64:
        #   w: 64×64×4 = 16KB, o_tile: 64×64×4 = 16KB, yv: 64×64×4 = 16KB
        #   Total: 48KB (safe)
        # For very large d (>1024), use BLOCK_D=128 with BLOCK_M=32
        if d > 1024:
            block_d = 128
            bm = 32
            bn = 32
        else:
            block_d = 64
            bm = 64
            bn = 64
        max_block_for_d = bm
        autotune = False
    else:
        block_d = block_d_full
        max_block_for_d = max_block_for_full
        # Disable autotune if any autotune config would overflow shared memory.
        # Autotune uses BLOCK_M/BLOCK_N up to 128, so if max_block_for_d < 128, disable it.
        if autotune and max_block_for_d < 128:
            autotune = False

    # Clamp BLOCK_M/N to max allowed by shared memory
    if bm > max_block_for_d:
        bm = max_block_for_d
    if bn > max_block_for_d:
        bn = max_block_for_d

    blocks_x = triton.cdiv(n, bm)
    blocks_y = triton.cdiv(m, bn)

    compute_grad_x = bool(compute_grad_x)
    compute_grad_y = bool(compute_grad_y)
    if not compute_grad_x and not compute_grad_y:
        return None, None

    if compute_grad_x and compute_grad_y:
        pid_offset = 0
        grid = (blocks_x + blocks_y,)
    elif compute_grad_x and not compute_grad_y:
        pid_offset = 0
        grid = (blocks_x,)
    else:
        # grad_y only: requires pid offset by blocks_x. Autotune can't vary this
        # with meta["BLOCK_M"], so keep a single (manual) config.
        if autotune:
            autotune = False
        pid_offset = blocks_x
        grid = (blocks_y,)

    grad_x = (
        torch.empty((n, d), device=x.device, dtype=torch.float32)
        if compute_grad_x
        else torch.empty((1, 1), device=x.device, dtype=torch.float32)
    )
    grad_y = (
        torch.empty((m, d), device=x.device, dtype=torch.float32)
        if compute_grad_y
        else torch.empty((1, 1), device=x.device, dtype=torch.float32)
    )

    if grad_scale is None:
        grad_scale = torch.ones((), device=x.device, dtype=torch.float32)
    else:
        if grad_scale.numel() != 1:
            raise ValueError("grad_scale must be a scalar tensor.")
        if grad_scale.device != x.device:
            raise ValueError("grad_scale must be on the same device as x.")
        grad_scale = grad_scale.to(dtype=torch.float32)

    # Handle OTDD label cost parameters
    use_label_cost = (
        label_x is not None
        and label_y is not None
        and label_cost_matrix is not None
        and lambda_y != 0.0
    )

    if use_label_cost:
        # Validate label tensors
        if label_x.shape[0] != n:
            raise ValueError(f"label_x must have length n={n}, got {label_x.shape[0]}")
        if label_y.shape[0] != m:
            raise ValueError(f"label_y must have length m={m}, got {label_y.shape[0]}")
        if label_cost_matrix.ndim != 2 or label_cost_matrix.shape[0] != label_cost_matrix.shape[1]:
            raise ValueError(f"label_cost_matrix must be square (V, V), got {label_cost_matrix.shape}")
        V = label_cost_matrix.shape[0]
        if label_x.max() >= V or label_x.min() < 0:
            raise ValueError(f"label_x values must be in [0, {V-1}]")
        if label_y.max() >= V or label_y.min() < 0:
            raise ValueError(f"label_y values must be in [0, {V-1}]")
        # Ensure contiguous and correct dtype
        label_x = label_x.to(dtype=torch.int32, device=x.device).contiguous()
        label_y = label_y.to(dtype=torch.int32, device=x.device).contiguous()
        W = label_cost_matrix.to(dtype=torch.float32, device=x.device).contiguous()
    else:
        # Provide dummy tensors (size 1) to satisfy Triton's pointer requirements
        # These are never accessed when USE_LABEL_COST=False
        V = 1
        label_x = torch.zeros(1, dtype=torch.int32, device=x.device)
        label_y = torch.zeros(1, dtype=torch.int32, device=x.device)
        W = torch.zeros(1, dtype=torch.float32, device=x.device)

    if use_global_o:
        kernel = _geomloss_grad_sqeuclid_large_d_autotune if autotune else _geomloss_grad_sqeuclid_large_d_impl
    else:
        kernel = _geomloss_grad_sqeuclid_autotune if autotune else _geomloss_grad_sqeuclid_impl
    if autotune:
        # Autotune uses a callable grid.
        def grid_fn(meta):
            if compute_grad_x and compute_grad_y:
                return (
                    triton.cdiv(n, meta["BLOCK_M"]) + triton.cdiv(m, meta["BLOCK_N"]),
                )
            # grad_x only, pid_offset=0.
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        kernel[grid_fn](
            x,
            y,
            f,
            g,
            loga,
            logb,
            a,
            b,
            x2,
            y2,
            grad_scale,
            grad_x,
            grad_y,
            # OTDD label cost pointers (None if not used)
            label_x,
            label_y,
            W,
            pid_offset,
            n,
            m,
            V,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            loga.stride(0),
            logb.stride(0),
            a.stride(0),
            b.stride(0),
            x2.stride(0),
            y2.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_y.stride(0),
            grad_y.stride(1),
            float(eps),
            float(cost_scale),
            float(lambda_x),
            float(lambda_y),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            BLOCK_D=block_d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
        )
    else:
        kernel[grid](
            x,
            y,
            f,
            g,
            loga,
            logb,
            a,
            b,
            x2,
            y2,
            grad_scale,
            grad_x,
            grad_y,
            # OTDD label cost pointers (None if not used)
            label_x,
            label_y,
            W,
            pid_offset,
            n,
            m,
            V,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            f.stride(0),
            g.stride(0),
            loga.stride(0),
            logb.stride(0),
            a.stride(0),
            b.stride(0),
            x2.stride(0),
            y2.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_y.stride(0),
            grad_y.stride(1),
            float(eps),
            float(cost_scale),
            float(lambda_x),
            float(lambda_y),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            BLOCK_D=block_d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
            num_warps=nw,
            num_stages=num_stages,
        )

    out_x = grad_x if compute_grad_x else None
    out_y = grad_y if compute_grad_y else None
    return out_x, out_y
