"""Fused Schur complement matvec kernel for HVP CG acceleration.

This module implements a two-phase persistent kernel that fuses the axis1 and axis0
transport plan applications used in the HVP's CG linear operator, reducing kernel
launches from 2 to 1 per CG iteration.

The key innovation is using a spin-wait barrier with atomic counter for grid-wide
synchronization between phases, allowing both operations to run in a single kernel.

IMPORTANT: The spin barrier requires all blocks to be co-resident on the GPU.
If grid_size exceeds the maximum resident blocks (num_SMs * ~4), the kernel
will deadlock. This module automatically falls back to the two-kernel approach
when the grid size would be too large.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Import the non-fused implementation for fallback
from flash_sinkhorn.kernels._triton_helpers import _tiled_dot
from flash_sinkhorn.kernels.sinkhorn_triton_apply_sqeuclid import apply_plan_vec_flashstyle
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import standard_to_shifted_potentials


def _get_max_resident_blocks(device: torch.device) -> int:
    """Get conservative estimate of max co-resident blocks for spin barrier safety.

    For spin barriers to work, all blocks must be scheduled simultaneously.
    A100 has 108 SMs and can typically run 2-4 blocks per SM depending on
    register pressure and shared memory usage.

    We use a conservative estimate of 2 blocks/SM to avoid deadlocks.
    """
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    # Conservative: 2 blocks per SM (actual can be 4+ but depends on kernel resources)
    return num_sms * 2


@triton.jit
def _grid_barrier_phase1(
    barrier_ptr,
    num_phase1_blocks: tl.constexpr,
):
    """Increment barrier and spin-wait until all phase 1 blocks arrive.

    This implements a device-wide barrier using atomic operations and spin-waiting.
    All blocks must call this with the same num_phase1_blocks value.

    IMPORTANT: We use atomic_add(ptr, 0) for the spin-wait check instead of
    regular loads, as regular loads may be cached and not see updates from
    other blocks. Atomic operations ensure we get fresh values from global memory.
    """
    pid = tl.program_id(0)

    # Only phase 1 blocks increment the counter
    if pid < num_phase1_blocks:
        # Atomically increment after completing phase 1 work
        tl.atomic_add(barrier_ptr, 1)

    # All blocks spin until all phase 1 blocks have arrived
    # Use atomic_add with 0 to force a fresh read from global memory
    # (regular tl.load may be cached and miss updates from other blocks)
    while tl.atomic_add(barrier_ptr, 0) < num_phase1_blocks:
        pass  # Spin


@triton.jit
def _fused_schur_matvec_kernel(
    # Input point clouds
    x_ptr,
    y_ptr,
    # Potentials (OTT-style)
    f_ptr,
    g_ptr,
    # Precomputed squared norms
    x2_ptr,
    y2_ptr,
    # CG vector and diagonals
    z_ptr,       # Input vector z (m,)
    diag_x_ptr,  # Diagonal D_x = diag_factor_x * a_hat (n,)
    denom_ptr,   # Denominator D_y = diag_factor_y * b_hat + eps*tau2 (m,)
    # Intermediate buffer
    piz_ptr,     # Intermediate P @ z result (n,)
    # Output
    out_ptr,     # Output: denom * z - P^T @ (P @ z / D_x) (m,)
    # Barrier for grid-wide sync
    barrier_ptr,
    # Sizes
    n,
    m,
    # Strides
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_f,
    stride_g,
    stride_x2,
    stride_y2,
    stride_z,
    stride_diag_x,
    stride_denom,
    stride_piz,
    stride_out,
    # Regularization
    eps,
    # Constexpr
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Block size for n dimension
    BLOCK_N: tl.constexpr,  # Block size for m dimension
    BLOCK_K: tl.constexpr,  # Block size for d dimension
    NUM_PHASE1_BLOCKS: tl.constexpr,  # Number of blocks for phase 1
    NUM_PHASE2_BLOCKS: tl.constexpr,  # Number of blocks for phase 2
    USE_EXP2: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Two-phase fused kernel for Schur complement matvec.

    Computes: out = denom * z - P^T @ (P @ z / diag_x)

    Phase 1: Compute piz = P @ z (axis1 reduction)
    Barrier: Grid-wide sync using atomic counter
    Phase 2: Compute P^T @ (piz / diag_x) (axis0 reduction) and final output

    The grid size is max(NUM_PHASE1_BLOCKS, NUM_PHASE2_BLOCKS) blocks.
    Each block participates in both phases if its pid is within range.
    """
    pid = tl.program_id(0)

    inv_eps = 1.0 / eps
    log2e = 1.4426950408889634
    inv_eps_log2 = inv_eps * log2e

    # ===== PHASE 1: Compute piz = P @ z (axis1) =====
    # Each block handles BLOCK_M rows (source indices)
    if pid < NUM_PHASE1_BLOCKS:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n

        # Load source data for this block
        f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
        x2_local = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

        # Online softmax accumulator
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        s_i = tl.zeros([BLOCK_M], tl.float32)

        # Loop over target columns (j)
        for j0 in range(0, m, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < m

            # Load target data
            g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
            y2_local = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)
            z_local = tl.load(z_ptr + offs_n * stride_z, mask=mask_n, other=0.0).to(tl.float32)

            # Compute dot product x_i @ y_j
            dot = _tiled_dot(
                x_ptr, y_ptr, offs_m, offs_n,
                stride_x0, stride_x1, stride_y0, stride_y1,
                D, mask_m, mask_n,
                BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
            )

            # Compute log-kernel values
            cost = x2_local[:, None] + y2_local[None, :] - 2.0 * dot
            if USE_EXP2:
                logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
            else:
                logits = (f[:, None] + g[None, :] - cost) * inv_eps
            logits = tl.where(mask_n[None, :], logits, -float("inf"))

            # Online softmax update
            block_max = tl.max(logits, axis=1)
            new_m = tl.maximum(m_i, block_max)
            new_m_neg_inf = new_m == -float("inf")
            if USE_EXP2:
                alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_i - new_m))
                w = tl.where(new_m_neg_inf[:, None], 0.0, tl.exp2(logits - new_m[:, None]))
            else:
                alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_i - new_m))
                w = tl.where(new_m_neg_inf[:, None], 0.0, tl.exp(logits - new_m[:, None]))

            s_i = s_i * alpha + tl.sum(w * z_local[None, :], axis=1)
            m_i = new_m

        # Final result for phase 1
        if USE_EXP2:
            piz_result = tl.exp2(m_i) * s_i
        else:
            piz_result = tl.exp(m_i) * s_i

        # Store intermediate result
        tl.store(piz_ptr + offs_m * stride_piz, piz_result, mask=mask_m)

    # ===== BARRIER: Wait for all phase 1 blocks =====
    # Note: The atomic_add in the barrier provides a release-acquire semantic
    # that ensures the stores above are visible after the barrier completes.
    _grid_barrier_phase1(barrier_ptr, NUM_PHASE1_BLOCKS)

    # ===== PHASE 2: Compute out = denom * z - P^T @ (piz / diag_x) =====
    # Each block handles BLOCK_N columns (target indices)
    if pid < NUM_PHASE2_BLOCKS:
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m

        # Load target data for this block
        g = tl.load(g_ptr + offs_n * stride_g, mask=mask_n, other=0.0).to(tl.float32)
        y2_local = tl.load(y2_ptr + offs_n * stride_y2, mask=mask_n, other=0.0).to(tl.float32)
        z_local = tl.load(z_ptr + offs_n * stride_z, mask=mask_n, other=0.0).to(tl.float32)
        denom_local = tl.load(denom_ptr + offs_n * stride_denom, mask=mask_n, other=1.0).to(tl.float32)

        # Online softmax accumulator for P^T
        m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
        s_j = tl.zeros([BLOCK_N], tl.float32)

        # Loop over source rows (i)
        for i0 in range(0, n, BLOCK_M):
            offs_m = i0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < n

            # Load source data
            f = tl.load(f_ptr + offs_m * stride_f, mask=mask_m, other=0.0).to(tl.float32)
            x2_local = tl.load(x2_ptr + offs_m * stride_x2, mask=mask_m, other=0.0).to(tl.float32)

            # Load intermediate result from phase 1
            piz_local = tl.load(piz_ptr + offs_m * stride_piz, mask=mask_m, other=0.0).to(tl.float32)
            diag_x_local = tl.load(diag_x_ptr + offs_m * stride_diag_x, mask=mask_m, other=1.0).to(tl.float32)
            # Guard against division by zero when diag_x underflows to 0
            # Note: Cast to float32 to avoid Triton type mismatch
            diag_x_safe = tl.maximum(diag_x_local, tl.full([BLOCK_M], 1e-40, tl.float32))
            scaled_piz = piz_local / diag_x_safe

            # Compute dot product x_i @ y_j
            dot = _tiled_dot(
                x_ptr, y_ptr, offs_m, offs_n,
                stride_x0, stride_x1, stride_y0, stride_y1,
                D, mask_m, mask_n,
                BLOCK_M, BLOCK_N, BLOCK_K, ALLOW_TF32,
            )

            # Compute log-kernel values
            cost = x2_local[:, None] + y2_local[None, :] - 2.0 * dot
            if USE_EXP2:
                logits = tl.fma(f[:, None] + g[None, :] - cost, inv_eps_log2, 0.0)
            else:
                logits = (f[:, None] + g[None, :] - cost) * inv_eps
            logits = tl.where(mask_m[:, None], logits, -float("inf"))

            # Online softmax update for P^T (reduction along axis 0)
            block_max = tl.max(logits, axis=0)
            new_m = tl.maximum(m_j, block_max)
            new_m_neg_inf = new_m == -float("inf")
            if USE_EXP2:
                alpha = tl.where(new_m_neg_inf, 0.0, tl.exp2(m_j - new_m))
                w = tl.where(new_m_neg_inf[None, :], 0.0, tl.exp2(logits - new_m[None, :]))
            else:
                alpha = tl.where(new_m_neg_inf, 0.0, tl.exp(m_j - new_m))
                w = tl.where(new_m_neg_inf[None, :], 0.0, tl.exp(logits - new_m[None, :]))

            # Weighted sum: P^T @ (piz / diag_x)
            s_j = s_j * alpha + tl.sum(w * scaled_piz[:, None], axis=0)
            m_j = new_m

        # Final result: denom * z - P^T @ (piz / diag_x)
        if USE_EXP2:
            pt_piz_over_diag = tl.exp2(m_j) * s_j
        else:
            pt_piz_over_diag = tl.exp(m_j) * s_j

        out_result = denom_local * z_local - pt_piz_over_diag
        tl.store(out_ptr + offs_n * stride_out, out_result, mask=mask_n)


def fused_schur_matvec_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    z: torch.Tensor,
    diag_x: torch.Tensor,
    denom: torch.Tensor,
    *,
    eps: float,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    piz_buffer: Optional[torch.Tensor] = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    use_exp2: bool = True,
    allow_tf32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Schur complement matvec: out = denom * z - P^T @ (P @ z / diag_x).

    This is a single-kernel implementation that fuses two transport plan applications
    (axis1 and axis0) with a grid-wide barrier for synchronization.

    Args:
        x: Source points (n, d)
        y: Target points (m, d)
        f: Source potential (n,) - OTT-style: P = exp((f+g-C)/eps)
        g: Target potential (m,) - OTT-style: P = exp((f+g-C)/eps)
        z: Input CG vector (m,)
        diag_x: Source diagonal D_x = diag_factor_x * a_hat (n,)
        denom: Target denominator D_y = diag_factor_y * b_hat + eps*tau2 (m,)
        eps: Entropy regularization
        x2: Precomputed ||x||^2 (n,) [optional]
        y2: Precomputed ||y||^2 (m,) [optional]
        piz_buffer: Reusable buffer for P @ z intermediate (n,) [optional]

    Returns:
        out: Result denom * z - P^T @ (P @ z / diag_x), shape (m,)
        piz_buffer: The intermediate buffer (for reuse)

    Note:
        This kernel uses a spin-wait barrier for grid-wide synchronization.
        The grid size must not exceed the number of co-resident blocks on the GPU.
        If the grid would be too large, this function automatically falls back to
        a two-kernel approach using apply_plan_vec_flashstyle (axis=1, then axis=0).

        The fallback path converts OTT potentials to shifted form and uses FlashStyle apply kernels.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (n,d) and (m,d).")
    if not x.is_cuda:
        raise ValueError("CUDA required.")

    # Validate all tensors are on the same CUDA device
    ref_device = x.device
    for name, tensor in [("y", y), ("f", f), ("g", g), ("z", z),
                         ("diag_x", diag_x), ("denom", denom)]:
        if not tensor.is_cuda or tensor.device != ref_device:
            raise ValueError(f"{name} must be on the same CUDA device as x ({ref_device}).")
    if x2 is not None and (not x2.is_cuda or x2.device != ref_device):
        raise ValueError(f"x2 must be on the same CUDA device as x ({ref_device}).")
    if y2 is not None and (not y2.is_cuda or y2.device != ref_device):
        raise ValueError(f"y2 must be on the same CUDA device as x ({ref_device}).")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension.")
    if f.shape[0] != n or g.shape[0] != m:
        raise ValueError("f and g shapes must match x and y.")
    if z.shape[0] != m:
        raise ValueError("z must have shape (m,).")
    if diag_x.shape[0] != n:
        raise ValueError("diag_x must have shape (n,).")
    if denom.shape[0] != m:
        raise ValueError("denom must have shape (m,).")

    eps_f = float(eps)

    # Precompute squared norms if not provided
    if x2 is None:
        x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    if y2 is None:
        y2 = (y.float() * y.float()).sum(dim=1).contiguous()

    # Ensure contiguous and float32
    x = x.contiguous()
    y = y.contiguous()
    f = f.float().contiguous()
    g = g.float().contiguous()
    z = z.float().contiguous()
    diag_x = diag_x.float().contiguous()
    denom = denom.float().contiguous()
    x2 = x2.float().contiguous()
    y2 = y2.float().contiguous()

    # Allocate or reuse intermediate buffer
    if piz_buffer is None:
        piz_buffer = torch.empty((n,), device=x.device, dtype=torch.float32)
    else:
        if piz_buffer.shape[0] != n:
            raise ValueError("piz_buffer must have shape (n,).")
        piz_buffer = piz_buffer.float().contiguous()

    # Output buffer
    out = torch.empty((m,), device=x.device, dtype=torch.float32)

    # Barrier counter (reset to 0 before kernel launch)
    barrier = torch.zeros((1,), device=x.device, dtype=torch.int32)

    # Compute grid sizes for each phase
    num_phase1_blocks = triton.cdiv(n, block_m)
    num_phase2_blocks = triton.cdiv(m, block_n)

    # Total grid size is max of both phases
    # All blocks participate in both phases (with masking for out-of-range)
    grid_size = max(num_phase1_blocks, num_phase2_blocks)

    # Check if grid size exceeds safe limit for spin barrier
    # If too large, fall back to two-kernel approach to avoid deadlock
    max_resident = _get_max_resident_blocks(x.device)
    if grid_size > max_resident:
        # Fallback: use two separate FlashStyle kernels (no spin barrier deadlock risk)
        # f, g are OTT-style potentials: P = exp((f+g-C)/eps)
        # FlashStyle kernels expect shifted potentials + log_a, log_b
        # Convert: OTT → shifted by subtracting squared norms
        alpha = (x.float() ** 2).sum(dim=1)
        beta = (y.float() ** 2).sum(dim=1)
        f_shift = f - alpha
        g_shift = g - beta

        # OTT potentials already absorb log marginals (f_ott = f_geom + eps*log(a)),
        # so use zeros to avoid double-counting
        log_a = torch.zeros(n, device=x.device, dtype=torch.float32)
        log_b = torch.zeros(m, device=x.device, dtype=torch.float32)

        # Phase 1: piz = P @ z (axis=1)
        piz_result = apply_plan_vec_flashstyle(
            x, y, f_shift, g_shift, log_a, log_b, z,
            eps=eps_f,
            axis=1,
            cost_scale=1.0,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        # Phase 2: P^T @ (piz / diag_x) and compute final output
        # Guard against division by zero when diag_x underflows to 0
        diag_x_safe = diag_x.clamp(min=1e-40)
        scaled_piz = piz_result / diag_x_safe
        pt_scaled_piz = apply_plan_vec_flashstyle(
            x, y, f_shift, g_shift, log_a, log_b, scaled_piz,
            eps=eps_f,
            axis=0,
            cost_scale=1.0,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        out = denom * z - pt_scaled_piz
        # Update piz_buffer for reuse
        if piz_buffer is None:
            piz_buffer = piz_result
        else:
            piz_buffer.copy_(piz_result)
        return out, piz_buffer

    # Launch the fused kernel (grid size is safe for spin barrier)
    _fused_schur_matvec_kernel[(grid_size,)](
        x, y, f, g, x2, y2,
        z, diag_x, denom,
        piz_buffer, out, barrier,
        n, m,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        f.stride(0), g.stride(0),
        x2.stride(0), y2.stride(0),
        z.stride(0), diag_x.stride(0), denom.stride(0),
        piz_buffer.stride(0), out.stride(0),
        eps_f,
        D=d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_PHASE1_BLOCKS=num_phase1_blocks,
        NUM_PHASE2_BLOCKS=num_phase2_blocks,
        USE_EXP2=use_exp2,
        ALLOW_TF32=allow_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, piz_buffer
