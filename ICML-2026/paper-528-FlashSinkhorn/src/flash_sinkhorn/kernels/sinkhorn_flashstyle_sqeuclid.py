"""FlashSinkhorn: Sinkhorn OT with FlashAttention-style streaming and shifted potentials.

This module implements a reformulated Sinkhorn algorithm that aligns exactly with
FlashAttention's interface, enabling potential future integration with optimized
FlashAttention kernels.

Mathematical Reformulation
--------------------------

Standard Sinkhorn f-update:
    f_i = -ε · LSE_j[(g_j - C_ij)/ε + log(b_j)]

where C_ij = ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2·x_i·y_j = α_i + β_j - 2·x_i·y_j

Reformulation with shifted potentials:
1. Define: Q = √(2·cost_scale)·X, K = √(2·cost_scale)·Y
   So: Q_i·K_j = 2·cost_scale·x_i·y_j

2. Define shifted potentials:
   f̂ = f - α  where α_i = cost_scale·||x_i||²
   ĝ = g - β  where β_j = cost_scale·||y_j||²

3. Define pre-scaled biases:
   δ = ε·log(b)  (for f-update)
   γ = ε·log(a)  (for g-update)

4. The f-update becomes:
   f̂_i = -ε · LSE_j[(Q_i·K_j + ĝ_j + δ_j)/ε]
       = -ε · LSE_j[Q_i·K_j/ε + u_j]

   where u_j = (ĝ_j + δ_j)/ε is the pre-scaled bias

This matches FlashAttention exactly:
- Score matrix: S = Q·K^T
- Softmax scale: 1/ε
- Additive bias: u = (ĝ + δ)/ε

Benefits over current implementation:
- 67% fewer bias vector loads per tile (1 vs 3)
- ~78% fewer elementwise operations per element
- Direct FlashAttention interface alignment
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._common import (
    _cache_key_bucket,
    log_weights,
)
from flash_sinkhorn.kernels._triton_helpers import (
    _final_lse,
    _online_softmax_rescale,
    _online_softmax_rescale_axis0,
    _tiled_dot,
)


# =============================================================================
# PRECOMPUTATION UTILITIES
# =============================================================================

def precompute_flashsinkhorn_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    cost_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute static bias components for FlashSinkhorn.

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        a: Source marginal weights [n]
        b: Target marginal weights [m]
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full ||x-y||², 0.5 for half ||x-y||²/2)

    Returns:
        alpha: Source squared norms [n] = cost_scale * ||x||²
        beta: Target squared norms [m] = cost_scale * ||y||²
        gamma: Scaled log source weights [n] = eps * log(a) (for g-update)
        delta: Scaled log target weights [m] = eps * log(b) (for f-update)

    Notes:
        - For cost_scale=1.0: full squared Euclidean C = ||x-y||²
        - For cost_scale=0.5: half squared Euclidean C = ||x-y||²/2 (GeomLoss default)
        - Use flashsinkhorn_lse() with raw x, y coordinates (not pre-scaled Q, K)
    """
    # Squared norms with cost scaling
    alpha = cost_scale * (x.float() ** 2).sum(dim=1)  # [n]
    beta = cost_scale * (y.float() ** 2).sum(dim=1)   # [m]

    # Scaled log marginals
    gamma = eps * log_weights(a)  # [n], for g-update
    delta = eps * log_weights(b)  # [m], for f-update

    return alpha, beta, gamma, delta


def compute_bias_f(
    g: torch.Tensor,
    beta: torch.Tensor,
    delta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute pre-scaled bias for f-update: u = (ĝ + δ)/ε.

    Args:
        g: Current g potential [m]
        beta: Target squared norms [m] = cost_scale * ||y||²
        delta: Scaled log target weights [m] = eps * log(b)
        eps: Regularization parameter

    Returns:
        u: Pre-scaled bias [m] = (g - beta + delta) / eps
    """
    g_hat = g - beta  # Shifted potential: ĝ = g - β
    return (g_hat + delta) / eps


def compute_bias_g(
    f: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute pre-scaled bias for g-update: v = (f̂ + γ)/ε.

    Args:
        f: Current f potential [n]
        alpha: Source squared norms [n] = cost_scale * ||x||²
        gamma: Scaled log source weights [n] = eps * log(a)
        eps: Regularization parameter

    Returns:
        v: Pre-scaled bias [n] = (f - alpha + gamma) / eps
    """
    f_hat = f - alpha  # Shifted potential: f̂ = f - α
    return (f_hat + gamma) / eps


# =============================================================================
# TRITON KERNELS
# =============================================================================

def _flashsinkhorn_lse_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for FlashSinkhorn alternating LSE kernel.

    EXPANDED for n=10000,d=64 on A100-80GB: num_warps=16, num_stages=4,
    BLOCK_M=256, more combos. Pruned to ~50 configs.

    CRITICAL: BLOCK_K must be < D (64). Triton compiler bug with BLOCK_K>=D.
    """
    configs = []

    # BLOCK_M=256: max SM utilization with num_warps=16
    for bn in (128, 64):
        for bk in (32, 16):
            for ns in (4, 3):
                configs.append(triton.Config(
                    {"BLOCK_M": 256, "BLOCK_N": bn, "BLOCK_K": bk},
                    num_warps=16, num_stages=ns))

    # BLOCK_M=128: large tiles with 8-16 warps
    for bn in (128, 64, 32):
        for bk in (32, 16):
            for nw in (16, 8):
                for ns in (4, 3):
                    if nw == 16 and bn < 64:
                        continue
                    configs.append(triton.Config(
                        {"BLOCK_M": 128, "BLOCK_N": bn, "BLOCK_K": bk},
                        num_warps=nw, num_stages=ns))

    # BLOCK_M=64: medium tiles
    for bn in (128, 64, 32):
        for bk in (32, 16):
            for nw in (8, 4):
                for ns in (4, 3):
                    configs.append(triton.Config(
                        {"BLOCK_M": 64, "BLOCK_N": bn, "BLOCK_K": bk},
                        num_warps=nw, num_stages=ns))

    # BLOCK_M=32: small tiles for fine-grained parallelism
    for bn in (64, 32):
        for bk in (32, 16):
            for ns in (3, 2):
                configs.append(triton.Config(
                    {"BLOCK_M": 32, "BLOCK_N": bn, "BLOCK_K": bk},
                    num_warps=4, num_stages=ns))

    # Lightweight fallbacks: num_warps=4, num_stages=2
    for bm in (64, 32):
        for bn in (64, 32):
            for bk in (32, 16):
                configs.append(triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                    num_warps=4, num_stages=2))

    return configs
@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _flashsinkhorn_lse_raw_kernel_impl(
    x_ptr,           # Source coordinates: [n, d] (NOT pre-scaled!)
    y_ptr,           # Target coordinates: [m, d] (NOT pre-scaled!)
    bias_ptr,        # Pre-scaled bias: [m] for f-update, [n] for g-update
    out_ptr,         # Output: shifted potential [n] or [m]
    n,               # Number of rows (sources for f-update, targets for g-update)
    m,               # Number of cols (targets for f-update, sources for g-update)
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_bias,
    stride_out,
    coord_scale,     # 2 * cost_scale (to scale x @ y.T to match Q @ K.T)
    eps,             # Regularization parameter (for output scaling)
    damping,         # Unbalanced OT damping factor (1.0 for balanced)
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Compute f̂_i = -ε · damping · LSE_j[coord_scale * x_i·y_j / ε + bias_j]

    This kernel computes a single shifted potential update using RAW coordinates.
    The coord_scale is applied INSIDE the kernel after the matmul, ensuring
    TF32 rounding behavior matches the fused kernel.

    KEY: Takes x, y directly (NOT pre-scaled Q, K). This ensures TF32 parity
    with the fused symmetric kernel which also uses raw coordinates.

    Args:
        x_ptr: Source coordinates [n, d] (NOT pre-scaled!)
        y_ptr: Target coordinates [m, d] (NOT pre-scaled!)
        bias_ptr: Pre-scaled bias [m] (or [n] for g-update)
        out_ptr: Output shifted potential [n] (or [m] for g-update)
        n, m: Dimensions
        coord_scale: 2 * cost_scale (to scale x·y to match Q·K)
        eps: Regularization parameter
        damping: Unbalanced OT factor (1.0 for balanced)
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    # Online LSE accumulators (FlashAttention style)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    # Constants for exp2/log2 optimization
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps = 1.0 / eps
    # Combined scale: coord_scale / eps (for x @ y.T -> scaled score / eps)
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # Iterate over all columns (j dimension)
    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # Load ONLY the pre-scaled bias (instead of g, logb, y² in current impl)
        # This is the key memory savings: 1 vector instead of 3
        # CRITICAL: Must cast to float32 for consistent precision
        if EVEN_N:
            bias = tl.load(bias_ptr + offs_n * stride_bias, eviction_policy="evict_first").to(tl.float32)
        else:
            bias = tl.load(bias_ptr + offs_n * stride_bias, mask=mask_n, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
        # This ensures TF32 rounding matches the fused kernel
        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Form logits: (coord_scale * x @ y.T) / eps + bias
        # = dot * (coord_scale / eps) + bias
        # This matches the fused kernel's computation exactly
        if USE_EXP2:
            # Use fma for better precision and potentially fused operations
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :] * log2e)
        else:
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        # Online LSE update (standard FlashAttention algorithm)
        new_m, alpha, w = _online_softmax_rescale(vals, m_i, USE_EXP2)
        s_i = s_i * alpha + tl.sum(w, axis=1)
        m_i = new_m

    # Final LSE computation
    lse = _final_lse(m_i, s_i, USE_EXP2)

    # Output: f̂ = -ε * damping * LSE
    # For balanced OT, damping = 1.0
    # For unbalanced OT, damping = 1/(1 + eps/rho)
    out = -eps * damping * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


# Create autotuned version of raw LSE kernel (uses raw x, y coordinates)
_flashsinkhorn_lse_raw_kernel_autotune = triton.autotune(
    configs=_flashsinkhorn_lse_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_lse_raw_kernel_impl)


@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _flashsinkhorn_lse_fused_kernel_impl(
    x_ptr,              # Source coordinates: [n, d]
    y_ptr,              # Target coordinates: [m, d]
    g_hat_ptr,          # Shifted potential: [m] for f-update (ĝ = g - β)
    log_w_ptr,          # Log marginal: [m] for f-update (log(b), NOT scaled!)
    out_ptr,            # Output: shifted potential [n]
    n,
    m,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_out,
    coord_scale,        # 2 * cost_scale
    eps,                # Regularization parameter
    damping,            # Unbalanced OT damping factor (1.0 for balanced)
    CACHE_KEY_N,        # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,        # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Fused LSE kernel that computes bias in SRAM (matches symmetric kernel interface).

    This computes: f̂_i = -ε · damping · LSE_j[coord_scale * x_i·y_j / ε + ĝ_j/ε + log(w_j)]

    KEY OPTIMIZATION: Load ĝ and log(w) separately and compute bias = ĝ/ε + log(w) in SRAM.
    This matches the symmetric kernel interface and eliminates Python kernel launch overhead.

    Interface matches symmetric kernel:
    - Takes log_w directly (not eps*log(w))
    - Computes: bias = g_hat * inv_eps + log_w (same as symmetric)
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n

    # Online LSE accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)

    # Constants
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    inv_eps = 1.0 / eps
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # Iterate over columns
    for j0 in range(0, m, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # FUSED BIAS COMPUTATION IN SRAM (matches symmetric kernel exactly):
        # Load ĝ and log(w), compute bias with exp2 pre-scaling if enabled
        if EVEN_N:
            g_hat = tl.load(g_hat_ptr + offs_n, eviction_policy="evict_first").to(tl.float32)
            log_w = tl.load(log_w_ptr + offs_n, eviction_policy="evict_first").to(tl.float32)
        else:
            g_hat = tl.load(g_hat_ptr + offs_n, mask=mask_n, other=0.0, eviction_policy="evict_first").to(tl.float32)
            log_w = tl.load(log_w_ptr + offs_n, mask=mask_n, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul
        dot = _tiled_dot(
            x_ptr, y_ptr, offs_m, offs_n,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_m, mask_n,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Form logits: (coord_scale * dot)/ε + ĝ/ε + log(w)
        # PRE-SCALE bias for exp2 (matches symmetric kernel - avoids extra multiply)
        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = g_hat * inv_eps_log2 + log_w * log2e  # Pre-scaled for exp2
            vals = tl.fma(dot, scaled_inv_eps_log2, bias[None, :])  # Use directly
        else:
            bias = g_hat * inv_eps + log_w
            vals = dot * scaled_inv_eps + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, -float("inf"))

        # Online LSE update
        new_m, alpha, w = _online_softmax_rescale(vals, m_i, USE_EXP2)
        s_i = s_i * alpha + tl.sum(w, axis=1)
        m_i = new_m

    # Final LSE
    lse = _final_lse(m_i, s_i, USE_EXP2)

    out = -eps * damping * lse
    tl.store(out_ptr + offs_m * stride_out, out, mask=mask_m)


# Create autotuned version of fused LSE kernel
_flashsinkhorn_lse_fused_kernel_autotune = triton.autotune(
    configs=_flashsinkhorn_lse_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_lse_fused_kernel_impl)


# =============================================================================
# FUSED SYMMETRIC STEP KERNEL (single kernel for both f and g updates)
# =============================================================================

@triton.heuristics({
    "EVEN_N": lambda args: args["m"] % args["BLOCK_N"] == 0,
    "EVEN_M": lambda args: args["n"] % args["BLOCK_M"] == 0,
})
@triton.jit
def _flashsinkhorn_symmetric_step_kernel(
    x_ptr,           # Source coordinates: [n, d] (NOT pre-scaled!)
    y_ptr,           # Target coordinates: [m, d] (NOT pre-scaled!)
    f_hat_ptr,       # Current shifted f potential: [n]
    g_hat_ptr,       # Current shifted g potential: [m]
    log_a_ptr,       # Log source weights: [n]
    log_b_ptr,       # Log target weights: [m]
    f_out_ptr,       # Output shifted f potential: [n]
    g_out_ptr,       # Output shifted g potential: [m]
    # OTDD label cost parameters (optional)
    label_x_ptr,     # int32 labels for x: [n] or dummy if not used
    label_y_ptr,     # int32 labels for y: [m] or dummy if not used
    W_ptr,           # Label cost matrix: [V, V] flattened or dummy
    n,
    m,
    V,               # Number of classes (1 if no label cost)
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    eps,             # Regularization parameter
    alpha,           # Symmetric averaging weight (0.5 for symmetric, 1.0 for full update)
    damping_f,       # Unbalanced OT damping for f (1.0 for balanced)
    damping_g,       # Unbalanced OT damping for g (1.0 for balanced)
    coord_scale,     # 2 * cost_scale (to scale x @ y.T to match Q @ K.T)
    half_cost_scale, # cost_scale (for label cost: cost_scale * W)
    lambda_x,        # Weight for Euclidean cost (1.0 default)
    lambda_y,        # Weight for label cost (0.0 if no labels)
    CACHE_KEY_N,     # Bucketed n for autotune cache (n // 32)
    CACHE_KEY_M,     # Bucketed m for autotune cache (m // 32)
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,  # For autotune cache consistency (harmonized with alternating)
    USE_EXP2: tl.constexpr,
    USE_LABEL_COST: tl.constexpr,  # Whether to use label cost (compile-time)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Fused symmetric Sinkhorn step: computes both f and g updates in one kernel.

    Grid structure: (blocks_f + blocks_g,) where blocks_f = cdiv(n, BLOCK_M)
    - Programs 0..blocks_f-1: compute f-update (LSE over m dimension)
    - Programs blocks_f..end: compute g-update (LSE over n dimension)

    This matches the GeomLoss kernel structure for maximum efficiency.

    KEY OPTIMIZATION: Uses x, y directly (no Q, K pre-allocation needed).
    The scale factor (2*cost_scale) is passed as coord_scale and applied in-kernel.

    Bias computation is done inline:
    - f-update: u = ĝ/ε + log(b)
    - g-update: v = f̂/ε + log(a)
    """
    pid = tl.program_id(0)
    inv_eps = 1.0 / eps
    blocks_f = tl.cdiv(n, BLOCK_M)

    # Constants for exp2/log2 optimization
    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453
    # Combined scale: coord_scale / eps (for xy @ xy.T -> QK^T / eps)
    scaled_inv_eps = coord_scale * inv_eps
    scaled_inv_eps_log2 = scaled_inv_eps * log2e

    # =========================================================================
    # F-UPDATE: first blocks_f programs
    # =========================================================================
    if pid < blocks_f:
        offs_i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_i = offs_i < n

        # Load current f_hat for symmetric averaging
        f_old = tl.load(f_hat_ptr + offs_i, mask=mask_i, other=0.0).to(tl.float32)

        # Load labels for this block if using label cost
        if USE_LABEL_COST:
            label_i = tl.load(label_x_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)

        # Online LSE accumulators
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        s_i = tl.zeros([BLOCK_M], tl.float32)

        # Iterate over all j (target points)
        for j0 in range(0, m, BLOCK_N):
            j0 = tl.multiple_of(j0, BLOCK_N)
            offs_j = j0 + tl.arange(0, BLOCK_N)
            mask_j = offs_j < m

            # Load ĝ and log(b) to compute bias: u = ĝ/ε + log(b)
            if EVEN_N:
                g_hat = tl.load(g_hat_ptr + offs_j, eviction_policy="evict_first").to(tl.float32)
                log_b = tl.load(log_b_ptr + offs_j, eviction_policy="evict_first").to(tl.float32)
            else:
                g_hat = tl.load(g_hat_ptr + offs_j, mask=mask_j, other=0.0, eviction_policy="evict_first").to(tl.float32)
                log_b = tl.load(log_b_ptr + offs_j, mask=mask_j, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

            # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
            dot = _tiled_dot(
                x_ptr, y_ptr, offs_i, offs_j,
                stride_x0, stride_x1, stride_y0, stride_y1,
                D, mask_i, mask_j,
                BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
            )

            # Compute label cost if enabled
            # The cost term 2*cs*x·y contributes POSITIVELY to the exponent (lower cost = higher prob)
            # Label cost must be SUBTRACTED to increase cost for mismatched labels
            if USE_LABEL_COST:
                if EVEN_N:
                    label_j = tl.load(label_y_ptr + offs_j, eviction_policy="evict_first").to(tl.int32)
                else:
                    label_j = tl.load(label_y_ptr + offs_j, mask=mask_j, other=0, eviction_policy="evict_first").to(tl.int32)
                # Compute flattened indices into W: W[label_i, label_j]
                w_idx = label_i[:, None] * V + label_j[None, :]
                # Gather from W (label cost matrix)
                w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                # Label cost is SUBTRACTED (divided by eps) from the shifted dot product
                # Combined: lambda_x * (2*cs*dot) - lambda_y * (cs * w_cost)
                # Scale factors: coord_scale = 2*cost_scale, half_cost_scale = cost_scale
                effective_dot = lambda_x * dot * coord_scale - lambda_y * half_cost_scale * w_cost
            else:
                effective_dot = dot * coord_scale

            # Form logits: effective_dot/ε + ĝ/ε + log(b)
            if USE_EXP2:
                inv_eps_log2 = inv_eps * log2e
                bias = g_hat * inv_eps_log2 + log_b * log2e
                vals = tl.fma(effective_dot, inv_eps * log2e, bias[None, :])
            else:
                bias = g_hat * inv_eps + log_b
                vals = effective_dot * inv_eps + bias[None, :]
            vals = tl.where(mask_j[None, :], vals, -float("inf"))

            # Online LSE update
            new_m, rescale, w = _online_softmax_rescale(vals, m_i, USE_EXP2)
            s_i = s_i * rescale + tl.sum(w, axis=1)
            m_i = new_m

        # Final LSE and output
        lse = _final_lse(m_i, s_i, USE_EXP2)
        f_cand = -eps * damping_f * lse
        f_new = (1.0 - alpha) * f_old + alpha * f_cand
        tl.store(f_out_ptr + offs_i, f_new, mask=mask_i)
        return

    # =========================================================================
    # G-UPDATE: remaining programs (pid >= blocks_f)
    # =========================================================================
    pid_g = pid - blocks_f
    offs_j = pid_g * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_j = offs_j < m

    # Load current g_hat for symmetric averaging
    g_old = tl.load(g_hat_ptr + offs_j, mask=mask_j, other=0.0).to(tl.float32)

    # Load labels for this block if using label cost
    if USE_LABEL_COST:
        label_j = tl.load(label_y_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)

    # Online LSE accumulators
    m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
    s_j = tl.zeros([BLOCK_N], tl.float32)

    # Iterate over all i (source points)
    for i0 in range(0, n, BLOCK_M):
        i0 = tl.multiple_of(i0, BLOCK_M)
        offs_i = i0 + tl.arange(0, BLOCK_M)
        mask_i = offs_i < n

        # Load f̂ and log(a) to compute bias: v = f̂/ε + log(a)
        if EVEN_M:
            f_hat = tl.load(f_hat_ptr + offs_i, eviction_policy="evict_first").to(tl.float32)
            log_a = tl.load(log_a_ptr + offs_i, eviction_policy="evict_first").to(tl.float32)
        else:
            f_hat = tl.load(f_hat_ptr + offs_i, mask=mask_i, other=0.0, eviction_policy="evict_first").to(tl.float32)
            log_a = tl.load(log_a_ptr + offs_i, mask=mask_i, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

        # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
        dot = _tiled_dot(
            x_ptr, y_ptr, offs_i, offs_j,
            stride_x0, stride_x1, stride_y0, stride_y1,
            D, mask_i, mask_j,
            BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
        )

        # Compute label cost if enabled (same formula as f-update)
        if USE_LABEL_COST:
            if EVEN_M:
                label_i = tl.load(label_x_ptr + offs_i, eviction_policy="evict_first").to(tl.int32)
            else:
                label_i = tl.load(label_x_ptr + offs_i, mask=mask_i, other=0, eviction_policy="evict_first").to(tl.int32)
            # Compute flattened indices into W: W[label_i, label_j]
            w_idx = label_i[:, None] * V + label_j[None, :]
            # Gather from W (label cost matrix)
            w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
            # Combined: lambda_x * (2*cs*dot) - lambda_y * (cs * w_cost)
            effective_dot = lambda_x * dot * coord_scale - lambda_y * half_cost_scale * w_cost
        else:
            effective_dot = dot * coord_scale

        # Form logits: effective_dot/ε + f̂/ε + log(a) - reduce over i dimension
        # vals[i, j] = effective_dot_ij/ε + f̂_i/ε + log(a_i)
        if USE_EXP2:
            inv_eps_log2 = inv_eps * log2e
            bias = f_hat * inv_eps_log2 + log_a * log2e
            vals = tl.fma(effective_dot, inv_eps * log2e, bias[:, None])
        else:
            bias = f_hat * inv_eps + log_a
            vals = effective_dot * inv_eps + bias[:, None]
        vals = tl.where(mask_i[:, None], vals, -float("inf"))

        # Online LSE update (reduce over axis=0, i.e., over i dimension)
        new_m, rescale, w = _online_softmax_rescale_axis0(vals, m_j, USE_EXP2)
        s_j = s_j * rescale + tl.sum(w, axis=0)
        m_j = new_m

    # Final LSE and output
    lse = _final_lse(m_j, s_j, USE_EXP2)
    g_cand = -eps * damping_g * lse
    g_new = (1.0 - alpha) * g_old + alpha * g_cand
    tl.store(g_out_ptr + offs_j, g_new, mask=mask_j)


def _flashsinkhorn_symmetric_autotune_configs() -> Sequence[triton.Config]:
    """Autotuning configs for FlashSinkhorn symmetric fused kernel.

    Key insight: Fused f+g kernel. Keep BLOCK_N moderate for g-update accumulators.
    Similar to OTT-style _update_potential_autotune_configs_axis0().

    The g-update reduces over axis=0 (rows), storing accumulators of length BLOCK_N.

    NOTE: num_stages=3 is critical for large d (512+). Pipeline staging helps
    hide memory latency during k-iterations. Without stages=3, large d shows
    ~35% regression vs the OLD GeomLoss kernel.

    IMPORTANT: Include BLOCK_N=128 for large n (50k+) to match OLD kernel's configs.
    At n=50k, larger BLOCK_N reduces tile count and improves tensor core utilization.

    HARMONIZED with alternating kernel:
    - BLOCK_M: (128, 64, 32) - includes 32 for small n
    - BLOCK_N: (64, 32) standard + 128 for large n
    - DTYPE_ID in autotune key for consistent cache behavior
    """
    configs = []
    # Standard configs with moderate BLOCK_N for register pressure
    # Include BLOCK_M=32 for small n (harmonized with alternating kernel)
    for block_m in (128, 64, 32):
        for block_n in (64, 32):
            for block_k in (32, 16):
                for num_warps in (4, 8):
                    for num_stages in (2, 3):
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )
    # Large-n configs with BLOCK_N=128 (matches OLD GeomLoss kernel)
    # Critical for n=50k+ where larger tiles reduce launch overhead
    for block_m in (128, 64):
        for block_k in (32, 16):
            for num_warps in (4, 8):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": 128, "BLOCK_K": block_k},
                        num_warps=num_warps,
                        num_stages=3,  # stages=3 for latency hiding
                    )
                )
    return configs


# Create autotuned version of symmetric step kernel
# NOTE: DTYPE_ID in key ensures separate autotune cache per dtype (harmonized with alternating)
_flashsinkhorn_symmetric_step_kernel_autotune = triton.autotune(
    configs=_flashsinkhorn_symmetric_autotune_configs(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_symmetric_step_kernel)


def _flashsinkhorn_symmetric_autotune_configs_label() -> Sequence[triton.Config]:
    """Autotune configs optimized for label cost workloads (OTDD).

    Label cost requires scatter-gather access to W[label_x, label_y] every iteration.
    Larger BLOCK_K reduces k-loop iterations -> better W matrix cache reuse.
    Only valid for d >= 128 (need BLOCK_K < d for correctness).
    """
    # Use the fastest label config (same as old kernel)
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        )
    ]


# Separate autotune kernel for label workloads (with BLOCK_K=64 only)
_flashsinkhorn_symmetric_step_kernel_autotune_label = triton.autotune(
    configs=_flashsinkhorn_symmetric_autotune_configs_label(),
    key=["CACHE_KEY_N", "CACHE_KEY_M", "D", "ALLOW_TF32", "DTYPE_ID", "USE_EXP2"],
)(_flashsinkhorn_symmetric_step_kernel)


def _default_block_sizes_alternating(n: int, m: int, d: int) -> Tuple[int, int, int, int]:
    """Default block sizes for alternating LSE kernel (like OTT-style).

    Adapts to problem dimensions. Single bias load per tile allows larger BLOCK_N.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    The Triton compiler has a bug where BLOCK_K >= D causes incorrect results.
    """
    # BLOCK_M adapts to n
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    else:
        block_m = 32

    # BLOCK_N adapts to m - favor larger BN for single bias load
    if m >= 128:
        block_n = 128
    elif m >= 64:
        block_n = 64
    else:
        block_n = 32

    # BLOCK_K based on d (BLOCK_K < D for correctness)
    if d >= 64:
        block_k = 32  # Forces at least 2 k iterations for d >= 64
    elif d >= 32:
        block_k = 16  # Forces at least 2 k iterations for d >= 32
    else:
        block_k = 16  # Minimum for tl.dot

    num_warps = 4
    return block_m, block_n, block_k, num_warps


def _default_block_sizes_symmetric(n: int, m: int, d: int) -> Tuple[int, int, int, int]:
    """Default block sizes for symmetric fused kernel (like OTT-style axis=0).

    Keeps BLOCK_N smaller because g-update stores accumulators of length BLOCK_N.

    CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
    """
    # BLOCK_M adapts to n
    if n >= 128:
        block_m = 128
    elif n >= 64:
        block_m = 64
    else:
        block_m = 32

    # BLOCK_N smaller (g-update stores accumulators of length BLOCK_N)
    if m >= 64:
        block_n = 64
    else:
        block_n = 32

    # BLOCK_K based on d (BLOCK_K < D for correctness)
    if d >= 64:
        block_k = 32
    elif d >= 32:
        block_k = 16
    else:
        block_k = 16

    num_warps = 4
    return block_m, block_n, block_k, num_warps


def flashsinkhorn_lse(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    cost_scale: float = 1.0,
    damping: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
) -> torch.Tensor:
    """Compute shifted potential using FlashSinkhorn kernel.

    This computes: out_i = -ε * damping * LSE_j[coord_scale * x_i·y_j / ε + bias_j]

    The kernel applies coord_scale = 2 * cost_scale inside the kernel to scale
    the dot product, ensuring consistent TF32 rounding between fused and separate
    kernel paths.

    Args:
        x: Source coordinates [n, d]
        y: Target coordinates [m, d]
        bias: Pre-scaled bias [m]
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full ||x-y||², 0.5 for half ||x-y||²/2)
        damping: Unbalanced OT damping (1.0 for balanced)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 for better numerical stability
        block_m, block_n, block_k: Manual block sizes (disables autotune)
        num_warps: Number of warps (disables autotune)
        num_stages: Number of pipeline stages
        autotune: Enable autotuning

    Returns:
        out: Shifted potential [n]
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

    # Dtype ID for autotuning (captured BEFORE fp32 cast)
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # Conditional FP32/contiguous: skip for already-FP32-contiguous tensors
    x = x if (x.dtype == torch.float32 and x.is_contiguous()) else x.contiguous().float()
    y = y if (y.dtype == torch.float32 and y.is_contiguous()) else y.contiguous().float()
    bias = bias if (bias.dtype == torch.float32 and bias.is_contiguous()) else bias.contiguous().float()

    # coord_scale = 2 * cost_scale (to match QK^T = 2*cs*xy^T)
    coord_scale = 2.0 * cost_scale
    out = torch.empty((n,), device=x.device, dtype=torch.float32)

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = autotune and not manual_blocks

    # Launcher helper functions
    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes_alternating(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        grid = (triton.cdiv(n, bm),)
        _flashsinkhorn_lse_raw_kernel_impl[grid](
            x, y, bias, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            bias.stride(0),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        _flashsinkhorn_lse_raw_kernel_autotune[grid](
            x, y, bias, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            bias.stride(0),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
        )

    # Select and execute launcher
    launch = _launch_autotune if use_autotune else _launch_manual
    launch()

    return out


def flashsinkhorn_lse_fused(
    x: torch.Tensor,
    y: torch.Tensor,
    g_hat: torch.Tensor,
    log_w: torch.Tensor,
    eps: float,
    *,
    cost_scale: float = 1.0,
    damping: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    autotune: bool = True,
) -> torch.Tensor:
    """Fused LSE kernel that computes bias in SRAM (matches symmetric kernel interface).

    This computes: out_i = -ε * damping * LSE_j[coord_scale * x_i·y_j / ε + ĝ_j/ε + log(w_j)]

    KEY OPTIMIZATION: Load ĝ and log(w) separately and compute bias = ĝ/ε + log(w) in SRAM.
    This matches the symmetric kernel interface and eliminates Python kernel launch overhead.

    Args:
        x: Source coordinates [n, d]
        y: Target coordinates [m, d]
        g_hat: Shifted potential [m] (ĝ = g - β for f-update)
        log_w: Log marginal [m] (log(b) for f-update, NOT scaled by eps!)
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        damping: Unbalanced OT damping (1.0 for balanced)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 for numerical stability
        block_m, block_n, block_k, num_warps: Manual block sizes
        num_stages: Pipeline stages
        autotune: Enable autotuning

    Returns:
        out: Shifted potential [n]
    """
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension")
    if g_hat.shape[0] != m or log_w.shape[0] != m:
        raise ValueError(f"g_hat and log_w must have length {m}")

    # Dtype ID for autotuning (captured BEFORE fp32 cast)
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # Conditional FP32/contiguous: skip for already-FP32-contiguous tensors
    x = x if (x.dtype == torch.float32 and x.is_contiguous()) else x.contiguous().float()
    y = y if (y.dtype == torch.float32 and y.is_contiguous()) else y.contiguous().float()
    g_hat = g_hat if (g_hat.dtype == torch.float32 and g_hat.is_contiguous()) else g_hat.contiguous().float()
    log_w = log_w if (log_w.dtype == torch.float32 and log_w.is_contiguous()) else log_w.contiguous().float()

    coord_scale = 2.0 * cost_scale
    out = torch.empty((n,), device=x.device, dtype=torch.float32)

    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune = autotune and not manual_blocks

    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes_alternating(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        grid = (triton.cdiv(n, bm),)
        _flashsinkhorn_lse_fused_kernel_impl[grid](
            x, y, g_hat, log_w, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        _flashsinkhorn_lse_fused_kernel_autotune[grid](
            x, y, g_hat, log_w, out,
            n, m,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            out.stride(0),
            float(coord_scale),
            float(eps),
            float(damping),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
        )

    launch = _launch_autotune if use_autotune else _launch_manual
    launch()

    return out


def flashsinkhorn_symmetric_step(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    *,
    cost_scale: float = 1.0,
    alpha: float = 0.5,
    damping_f: float = 1.0,
    damping_g: float = 1.0,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    # OTDD label cost parameters
    label_x: Optional[torch.Tensor] = None,
    label_y: Optional[torch.Tensor] = None,
    label_cost_matrix: Optional[torch.Tensor] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused symmetric Sinkhorn step: computes both f and g updates in ONE kernel.

    This is the key optimization over separate f/g kernel calls - reduces kernel
    launch overhead by 50% and improves GPU occupancy.

    KEY: Uses x, y directly (no Q, K pre-allocation). The coord_scale = 2*cost_scale
    is applied inside the kernel, avoiding memory allocation overhead.

    Args:
        x: Source coordinates [n, d] (NOT pre-scaled!)
        y: Target coordinates [m, d] (NOT pre-scaled!)
        f_hat: Current shifted f potential [n]
        g_hat: Current shifted g potential [m]
        log_a: Log source weights [n]
        log_b: Log target weights [m]
        eps: Regularization parameter
        cost_scale: Cost scaling (1.0 for full, 0.5 for half cost)
        alpha: Averaging weight (0.5 for symmetric, 1.0 for full update)
        damping_f: Unbalanced OT damping for f (1.0 for balanced)
        damping_g: Unbalanced OT damping for g (1.0 for balanced)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable Triton autotuning (recommended)
        block_m, block_n, block_k: Manual block sizes (disables autotune)
        num_warps: Number of warps (disables autotune)
        num_stages: Pipeline stages
        label_x: int32/int64 labels for x [n] (OTDD)
        label_y: int32/int64 labels for y [m] (OTDD)
        label_cost_matrix: W [V, V] label distance matrix (OTDD)
        lambda_x: Weight for Euclidean cost (default 1.0)
        lambda_y: Weight for label cost (default 0.0 = no label cost)

    Returns:
        f_hat_new, g_hat_new: Updated shifted potentials
    """
    n, d = x.shape
    m = y.shape[0]

    # Dtype ID for autotuning (captured BEFORE fp32 cast)
    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # Ensure contiguous and float32
    x = x.contiguous().float()
    y = y.contiguous().float()
    f_hat = f_hat.contiguous().float()
    g_hat = g_hat.contiguous().float()
    log_a = log_a.contiguous().float()
    log_b = log_b.contiguous().float()

    # Output tensors
    f_out = torch.empty((n,), device=x.device, dtype=torch.float32)
    g_out = torch.empty((m,), device=x.device, dtype=torch.float32)

    # coord_scale = 2 * cost_scale (to match QK^T = 2*cost_scale * x·y)
    coord_scale = 2.0 * cost_scale
    half_cost_scale = cost_scale  # For label cost term

    # OTDD label cost setup
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
        if label_cost_matrix.ndim != 2:
            raise ValueError("label_cost_matrix must be 2D (V, V)")
        if label_cost_matrix.shape[0] != label_cost_matrix.shape[1]:
            raise ValueError("label_cost_matrix must be square (V, V)")

        V = label_cost_matrix.shape[0]

        # Validate label ranges
        if label_x.max() >= V or label_x.min() < 0:
            raise ValueError(f"label_x values must be in [0, {V}), got range [{label_x.min()}, {label_x.max()}]")
        if label_y.max() >= V or label_y.min() < 0:
            raise ValueError(f"label_y values must be in [0, {V}), got range [{label_y.min()}, {label_y.max()}]")

        # Ensure contiguous and correct dtype
        label_x_t = label_x.to(dtype=torch.int32, device=x.device).contiguous()
        label_y_t = label_y.to(dtype=torch.int32, device=x.device).contiguous()
        W = label_cost_matrix.to(dtype=torch.float32, device=x.device).contiguous()
    else:
        # Provide dummy tensors (size 1) to satisfy Triton's pointer requirements
        V = 1
        label_x_t = torch.zeros(1, dtype=torch.int32, device=x.device)
        label_y_t = torch.zeros(1, dtype=torch.int32, device=x.device)
        W = torch.zeros(1, dtype=torch.float32, device=x.device)

    # Check if manual block sizes were specified
    manual_blocks = (
        block_m is not None
        or block_n is not None
        or block_k is not None
        or num_warps is not None
    )
    use_autotune_flag = autotune and not manual_blocks

    # Launcher helper functions (like GeomLoss-style)
    def _launch_manual():
        bm, bn, bk, nw = _default_block_sizes_symmetric(n, m, d)
        bm = block_m if block_m is not None else bm
        bn = block_n if block_n is not None else bn
        bk = block_k if block_k is not None else bk
        nw = num_warps if num_warps is not None else nw
        if bk < 16:
            bk = 16

        # Grid: blocks_f + blocks_g (fused kernel handles both)
        blocks_f = triton.cdiv(n, bm)
        blocks_g = triton.cdiv(m, bn)
        grid = (blocks_f + blocks_g,)

        _flashsinkhorn_symmetric_step_kernel[grid](
            x, y,
            f_hat, g_hat,
            log_a, log_b,
            f_out, g_out,
            # OTDD label cost pointers
            label_x_t, label_y_t, W,
            n, m, V,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            float(eps),
            float(alpha),
            float(damping_f),
            float(damping_g),
            float(coord_scale),
            float(half_cost_scale),
            float(lambda_x),
            float(lambda_y),
            _cache_key_bucket(n), _cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=nw,
            num_stages=num_stages,
        )

    def _launch_autotune():
        def grid(meta):
            blocks_f = triton.cdiv(n, meta["BLOCK_M"])
            blocks_g = triton.cdiv(m, meta["BLOCK_N"])
            return (blocks_f + blocks_g,)

        # Select autotune kernel based on label cost usage
        kernel = (_flashsinkhorn_symmetric_step_kernel_autotune_label if use_label_cost
                  else _flashsinkhorn_symmetric_step_kernel_autotune)

        kernel[grid](
            x, y,
            f_hat, g_hat,
            log_a, log_b,
            f_out, g_out,
            # OTDD label cost pointers
            label_x_t, label_y_t, W,
            n, m, V,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            float(eps),
            float(alpha),
            float(damping_f),
            float(damping_g),
            float(coord_scale),
            float(half_cost_scale),
            float(lambda_x),
            float(lambda_y),
            CACHE_KEY_N=_cache_key_bucket(n),
            CACHE_KEY_M=_cache_key_bucket(m),
            D=d,
            ALLOW_TF32=allow_tf32,
            DTYPE_ID=dtype_id,
            USE_EXP2=use_exp2,
            USE_LABEL_COST=use_label_cost,
        )

    # Select and execute launcher
    launch = _launch_autotune if use_autotune_flag else _launch_manual
    launch()

    return f_out, g_out


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def shifted_to_standard_potentials(
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert shifted potentials back to standard form.

    Args:
        f_hat: Shifted source potential [n]
        g_hat: Shifted target potential [m]
        alpha: Source squared norms [n] = cost_scale * ||x||²
        beta: Target squared norms [m] = cost_scale * ||y||²

    Returns:
        f: Standard source potential [n] = f_hat + alpha
        g: Standard target potential [m] = g_hat + beta
    """
    return f_hat + alpha, g_hat + beta


def standard_to_shifted_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert standard potentials to shifted form.

    Args:
        f: Standard source potential [n]
        g: Standard target potential [m]
        alpha: Source squared norms [n] = cost_scale * ||x||²
        beta: Target squared norms [m] = cost_scale * ||y||²

    Returns:
        f_hat: Shifted source potential [n] = f - alpha
        g_hat: Shifted target potential [m] = g - beta
    """
    return f - alpha, g - beta


# =============================================================================
# HIGH-LEVEL SOLVER FUNCTIONS — moved to flash_sinkhorn.sinkhorn_solvers
# =============================================================================
# Backwards-compatible re-exports: all existing import paths continue to work.
from flash_sinkhorn.sinkhorn_solvers import (  # noqa: E402, F401
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)
