"""High-level Sinkhorn solver loops using FlashSinkhorn kernels.

This module contains the pure-PyTorch solver orchestration code:
- Alternating (Gauss-Seidel) updates with fixed epsilon
- Symmetric (GeomLoss-style) updates with epsilon scheduling

Both solvers use the FlashSinkhorn shifted-potential kernels from
``flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid`` for the actual
GPU work.

Moved from ``kernels/sinkhorn_flashstyle_sqeuclid.py`` for cleaner
separation between Triton kernel code and solver logic.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from flash_sinkhorn.kernels._common import dampening, epsilon_schedule, log_weights, max_diameter
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    flashsinkhorn_lse,
    flashsinkhorn_lse_fused,
    flashsinkhorn_symmetric_step,
    shifted_to_standard_potentials,
)


def sinkhorn_flashstyle_alternating(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 0.1,
    n_iters: int = 100,
    cost_scale: float = 1.0,
    # Unbalanced OT parameters (supports semi-unbalanced with separate x/y)
    rho: Optional[float] = None,  # Deprecated: use rho_x/rho_y for semi-unbalanced
    reach: Optional[float] = None,  # Deprecated: use reach_x/reach_y
    rho_x: Optional[float] = None,  # Source marginal penalty (None = strict/balanced)
    rho_y: Optional[float] = None,  # Target marginal penalty (None = strict/balanced)
    reach_x: Optional[float] = None,  # Alternative: reach_x^2 = rho_x
    reach_y: Optional[float] = None,  # Alternative: reach_y^2 = rho_y
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    threshold: Optional[float] = None,
    check_every: int = 5,
    return_n_iters: bool = False,
    ott_convention: bool = False,
    # Adaptive padding: mask convergence check to unpadded slices
    n_orig: Optional[int] = None,
    m_orig: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, int]:
    """FlashSinkhorn with alternating (Gauss-Seidel) updates.

    This uses the shifted potential formulation for reduced memory traffic
    and better FlashAttention alignment.

    Alternating Sinkhorn uses Gauss-Seidel updates where the g-update uses
    the NEWLY computed f potential. This requires 2 kernel launches per
    iteration (unlike symmetric which can fuse both updates).

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        a: Source marginal weights [n]
        b: Target marginal weights [m]
        eps: Regularization parameter
        n_iters: Number of Sinkhorn iterations
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        rho: Unbalanced OT marginal penalty (None for balanced) - DEPRECATED
        reach: Alternative to rho (reach^2 = rho) - DEPRECATED
        rho_x: Source marginal KL penalty (None = balanced/strict)
        rho_y: Target marginal KL penalty (None = balanced/strict)
        reach_x: Alternative to rho_x (reach_x^2 = rho_x)
        reach_y: Alternative to rho_y (reach_y^2 = rho_y)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable kernel autotuning
        threshold: Early stopping threshold (None = no early stopping)
        check_every: Check convergence every N iterations
        return_n_iters: If True, also return number of iterations used
        ott_convention: If True, return potentials in OTT convention where
            log marginals are absorbed into potentials:
            - OTT: f = eps*log(a) - eps*LSE[(g-C)/eps], P = exp((f+g-C)/eps)
            If False (default), use GeomLoss convention:
            - GeomLoss: f = -eps*LSE[(g-C)/eps + log(b)], P = a*b*exp((f+g-C)/eps)

    Returns:
        f, g: Converged potentials (convention depends on ott_convention flag)
        n_iters_used: (optional) Number of iterations if return_n_iters=True

    Note:
        Both conventions produce EQUIVALENT transport plans. The only difference
        is whether log marginals are absorbed into the potentials or kept separate.

        OTT convention: P_ij = exp((f_i + g_j - C_ij) / eps)
        GeomLoss convention: P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)

        Semi-unbalanced OT:
        - Use rho_x, rho_y to set different marginal penalties for source/target
        - rho_x=None, rho_y=float gives strict source constraint, relaxed target
        - damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
    """
    n, d = x.shape
    m = y.shape[0]

    # Handle rho/reach for unbalanced OT (supports semi-unbalanced with separate x/y)
    # Priority: rho_x/rho_y > reach_x/reach_y > rho > reach
    if rho is not None and reach is not None:
        raise ValueError("Specify either rho or reach, not both.")
    if reach is not None:
        rho = reach ** 2  # GeomLoss convention: rho = reach^p where p=2

    # Handle legacy rho/reach: apply to both sides if new params not specified
    if rho is not None:
        if rho_x is None and reach_x is None:
            rho_x = rho
        if rho_y is None and reach_y is None:
            rho_y = rho

    # Handle reach_x/reach_y to rho_x/rho_y conversion
    if reach_x is not None:
        if rho_x is not None:
            raise ValueError("Specify either rho_x or reach_x, not both.")
        rho_x = reach_x ** 2
    if reach_y is not None:
        if rho_y is not None:
            raise ValueError("Specify either rho_y or reach_y, not both.")
        rho_y = reach_y ** 2

    # Semi-unbalanced OT: damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
    # CRITICAL FIX: rho_x controls SOURCE marginal → damps f potential
    #               rho_y controls TARGET marginal → damps g potential
    damp_f = dampening(eps, rho_x)
    damp_g = dampening(eps, rho_y)

    # Conditional FP32/contiguous cast: skip for already-FP32-contiguous
    x_f32 = x if (x.dtype == torch.float32 and x.is_contiguous()) else x.float().contiguous()
    y_f32 = y if (y.dtype == torch.float32 and y.is_contiguous()) else y.float().contiguous()

    # Precompute static bias components (NO Q, K allocation - use raw x, y!)
    # This ensures TF32 parity with the fused kernel
    alpha = cost_scale * (x_f32 ** 2).sum(dim=1)  # [n]
    beta = cost_scale * (y_f32 ** 2).sum(dim=1)   # [m]
    log_a = log_weights(a)  # [n], for g-update (NOT scaled by eps for fused kernel)
    log_b = log_weights(b)  # [m], for f-update (NOT scaled by eps for fused kernel)
    # For OTT convention (non-fused kernel), we need scaled versions
    gamma = eps * log_a  # [n], scaled for OTT
    delta = eps * log_b  # [m], scaled for OTT

    # Initialize shifted potentials: f̂ = f - α, ĝ = g - β
    # Standard init f=0, g=0 means f̂ = -α, ĝ = -β
    f_hat = -alpha.clone()
    g_hat = -beta.clone()

    prev_f_hat = f_hat.clone() if threshold is not None else None
    prev_g_hat = g_hat.clone() if threshold is not None else None

    n_iters_used = 0

    if ott_convention:
        # =================================================================
        # OTT CONVENTION: log marginals OUTSIDE the logsumexp
        # f = eps*log(a) - eps*LSE[(g-C)/eps]
        # g = eps*log(b) - eps*LSE[(f-C)/eps]
        #
        # Key difference from GeomLoss: potentials do NOT include ||x||², ||y||²
        # The squared norms only appear in the cost C, not in f or g.
        #
        # Math derivation (for f-update):
        #   (g - C)/eps = (g - alpha - beta + 2*cs*x·y)/eps
        #               = (g_hat + 2*cs*x·y)/eps - alpha/eps  [since g_hat = g - beta]
        #
        # Since alpha_i is constant w.r.t. j, it factors out of LSE:
        #   LSE_j[(g-C)/eps] = LSE_j[(g_hat + 2*cs*x·y)/eps] - alpha/eps
        #
        # Therefore:
        #   f = eps*log(a) - eps*LSE_j[(g-C)/eps]
        #     = gamma - eps*(LSE_j[...] - alpha/eps)
        #     = gamma - eps*LSE_j[...] + alpha
        #     = gamma + f_hat_lse + alpha
        #
        # But we want f (standard OTT potential), not f_hat. In OTT convention,
        # f does NOT include alpha, so we should NOT add alpha at the end.
        # This means f_ott = gamma + f_hat_lse (no alpha).
        # =================================================================
        for i in range(n_iters):
            # IMPORTANT: OTT-style does g-update FIRST, then f-update (Gauss-Seidel)
            # This matches sinkhorn_potentials_sqeuclid iteration order.

            # g-update FIRST: g = δ + (-ε * LSE_i[y·x^T * coord_scale/ε + f̂/ε]) + β
            # g_ott = delta + g_hat_lse + beta
            # We store g_hat = g_ott - beta, so g_hat = delta + g_hat_lse
            v = f_hat / eps  # Use current f_hat (shifted)
            g_hat_lse = flashsinkhorn_lse(
                y_f32, x_f32, v, eps, cost_scale=cost_scale,
                damping=damp_g,  # Semi-unbalanced: damp_g = 1/(1+eps/rho_x)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )
            g_hat = delta + g_hat_lse

            # f-update SECOND: f = γ + (-ε * LSE_j[x·y^T * coord_scale/ε + ĝ/ε]) + α
            # Uses the NEWLY computed g_hat (Gauss-Seidel!)
            u = g_hat / eps  # Bias is updated g_hat divided by eps
            f_hat_lse = flashsinkhorn_lse(
                x_f32, y_f32, u, eps, cost_scale=cost_scale,
                damping=damp_f,  # Semi-unbalanced: damp_f = 1/(1+eps/rho_y)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )
            # f_ott = gamma + f_hat_lse + alpha
            # We store f_hat = f_ott - alpha, so f_hat = gamma + f_hat_lse
            f_hat = gamma + f_hat_lse

            n_iters_used += 1

            # Early stopping check
            if threshold is not None and (i + 1) % check_every == 0:
                _no = n_orig if n_orig is not None else len(f_hat)
                _mo = m_orig if m_orig is not None else len(g_hat)
                f_change = (f_hat[:_no] - prev_f_hat[:_no]).abs().max().item()
                g_change = (g_hat[:_mo] - prev_g_hat[:_mo]).abs().max().item()
                if max(f_change, g_change) < threshold:
                    break
                prev_f_hat.copy_(f_hat)
                prev_g_hat.copy_(g_hat)

    else:
        # =================================================================
        # GEOMLOSS CONVENTION (default): log marginals INSIDE the logsumexp
        # f = -eps*LSE[(g-C)/eps + log(b)]
        # g = -eps*LSE[(f-C)/eps + log(a)]
        #
        # KEY: Uses FUSED kernel that computes bias in SRAM:
        #   bias = g_hat/eps + log_b  (same formula as symmetric kernel)
        # This eliminates Python kernel launch overhead.
        # =================================================================
        for i in range(n_iters):
            # f-update: f̂ = -ε * LSE_j[x·y^T * coord_scale/ε + ĝ/ε + log(b)]
            # FUSED: kernel computes bias = g_hat/eps + log_b in SRAM
            f_hat = flashsinkhorn_lse_fused(
                x_f32, y_f32, g_hat, log_b, eps, cost_scale=cost_scale,
                damping=damp_f,  # Semi-unbalanced: damp_f = 1/(1+eps/rho_y)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )

            # g-update: ĝ = -ε * LSE_i[y·x^T * coord_scale/ε + f̂/ε + log(a)]
            # FUSED: kernel computes bias = f_hat/eps + log_a in SRAM
            g_hat = flashsinkhorn_lse_fused(
                y_f32, x_f32, f_hat, log_a, eps, cost_scale=cost_scale,
                damping=damp_g,  # Semi-unbalanced: damp_g = 1/(1+eps/rho_x)
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
            )

            n_iters_used += 1

            # Early stopping check
            if threshold is not None and (i + 1) % check_every == 0:
                _no = n_orig if n_orig is not None else len(f_hat)
                _mo = m_orig if m_orig is not None else len(g_hat)
                f_change = (f_hat[:_no] - prev_f_hat[:_no]).abs().max().item()
                g_change = (g_hat[:_mo] - prev_g_hat[:_mo]).abs().max().item()
                if max(f_change, g_change) < threshold:
                    break
                prev_f_hat.copy_(f_hat)
                prev_g_hat.copy_(g_hat)

    # Convert back to standard potentials
    if ott_convention:
        # OTT convention: potentials do NOT include ||x||², ||y||²
        # f_hat already contains the correct value (gamma + lse_result)
        # g_hat already contains the correct value (delta + lse_result)
        # But we need to add alpha/beta to match the shifted_to_standard formula
        # Actually no - for OTT, f_ott = gamma + lse + alpha, and we stored f_hat = gamma + lse
        # So we DO need to add alpha and beta to get the final OTT potentials!
        f, g = f_hat + alpha, g_hat + beta
    else:
        # GeomLoss convention: potentials include ||x||², ||y||² via shifted formulation
        f, g = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

    if return_n_iters:
        return f, g, n_iters_used
    return f, g


def sinkhorn_flashstyle_symmetric(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    blur: float = 0.05,
    scaling: float = 0.5,
    use_epsilon_scaling: bool = True,
    last_extrapolation: bool = True,
    cost_scale: float = 1.0,
    eps: Optional[float] = None,
    n_iters: Optional[int] = None,
    diameter: Optional[float] = None,
    eps_list: Optional[Sequence[float]] = None,
    # Unbalanced OT parameters (supports semi-unbalanced with separate x/y)
    rho: Optional[float] = None,  # Deprecated: use rho_x/rho_y for semi-unbalanced
    reach: Optional[float] = None,  # Deprecated: use reach_x/reach_y
    rho_x: Optional[float] = None,  # Source marginal penalty (None = strict/balanced)
    rho_y: Optional[float] = None,  # Target marginal penalty (None = strict/balanced)
    reach_x: Optional[float] = None,  # Alternative: reach_x^2 = rho_x
    reach_y: Optional[float] = None,  # Alternative: reach_y^2 = rho_y
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    fused: Optional[bool] = None,
    threshold: Optional[float] = None,
    check_every: int = 5,
    return_n_iters: bool = False,
    return_prelast: bool = False,
    # Warm-start parameters (standard potentials, not shifted)
    f_init: Optional[torch.Tensor] = None,  # Initial f potential [n]
    g_init: Optional[torch.Tensor] = None,  # Initial g potential [m]
    # OTDD label-augmented cost parameters
    label_x: Optional[torch.Tensor] = None,  # int32/int64 labels for x: [n]
    label_y: Optional[torch.Tensor] = None,  # int32/int64 labels for y: [m]
    label_cost_matrix: Optional[torch.Tensor] = None,  # W: [V, V] label distances
    lambda_x: float = 1.0,  # Weight for Euclidean cost
    lambda_y: float = 0.0,  # Weight for label cost (0 = Euclidean only)
    # Adaptive padding: mask convergence check to unpadded slices
    n_orig: Optional[int] = None,
    m_orig: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, int]:
    """FlashSinkhorn with GeomLoss-style symmetric updates.

    This uses the shifted potential formulation for reduced memory traffic
    and epsilon scaling for numerical stability.

    Args:
        x: Source points [n, d]
        y: Target points [m, d]
        a: Source marginal weights [n]
        b: Target marginal weights [m]
        blur: Target blur (final eps = blur^2)
        scaling: Epsilon decay factor per iteration
        use_epsilon_scaling: If True, use exponential epsilon schedule
        last_extrapolation: If True, do final full update
        cost_scale: Cost scaling (1.0 for full, 0.5 for half)
        eps: Fixed regularization (if not using epsilon scaling)
        n_iters: Number of iterations (if not using epsilon scaling)
        diameter: Point cloud diameter (auto-computed if None)
        eps_list: Explicit epsilon schedule (overrides other params)
        rho: Unbalanced OT marginal penalty (None for balanced) - DEPRECATED
        reach: Alternative to rho (reach^2 = rho) - DEPRECATED
        rho_x: Source marginal KL penalty (None = balanced/strict)
        rho_y: Target marginal KL penalty (None = balanced/strict)
        reach_x: Alternative to rho_x (reach_x^2 = rho_x)
        reach_y: Alternative to rho_y (reach_y^2 = rho_y)
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable kernel autotuning
        fused: None (default) = auto-select based on n (fused for n < 30000).
               True = always fused (1 kernel launch per iteration).
               False = always separate (2 launches per iteration).
        threshold: Early stopping threshold
        check_every: Check convergence every N iterations
        return_n_iters: If True, also return number of iterations used
        return_prelast: If True, also return pre-extrapolation potentials
        f_init: Initial f potential for warm-start (standard form, not shifted)
        g_init: Initial g potential for warm-start (standard form, not shifted)

    Returns:
        f, g: Converged potentials in standard form
        f_prelast, g_prelast: (optional) Pre-extrapolation potentials
        n_iters_used: (optional) Number of iterations

    Note:
        Fused vs Separate kernels:
        - Fused: 1 kernel launch per iteration, both f and g computed in parallel
        - Separate: 2 kernel launches per iteration
        - Both produce identical results (symmetric averaging uses old potentials)
        - Fused has 50% fewer kernel launches (better for small n < 30000)
        - Separate has better memory patterns at large n (auto-switches at n >= 30000)

        Semi-unbalanced OT:
        - Use rho_x, rho_y to set different marginal penalties for source/target
        - rho_x=None, rho_y=float gives strict source constraint, relaxed target
        - damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)

        OTDD Label Cost (requires fused=True):
        - label_x: int32/int64 labels for source points [n]
        - label_y: int32/int64 labels for target points [m]
        - label_cost_matrix: W [V, V] matrix of label distances
        - lambda_x: weight for Euclidean cost (default 1.0)
        - lambda_y: weight for label cost (default 0.0)
        - Combined cost: C = lambda_x * ||x-y||² + lambda_y * W[label_x, label_y]
    """
    n, d = x.shape
    m = y.shape[0]

    # Handle rho/reach for unbalanced OT (supports semi-unbalanced with separate x/y)
    # Priority: rho_x/rho_y > reach_x/reach_y > rho > reach
    if rho is not None and reach is not None:
        raise ValueError("Specify either rho or reach, not both.")
    if reach is not None:
        rho = reach ** 2  # GeomLoss convention: rho = reach^p where p=2

    # Handle legacy rho/reach: apply to both sides if new params not specified
    if rho is not None:
        if rho_x is None and reach_x is None:
            rho_x = rho
        if rho_y is None and reach_y is None:
            rho_y = rho

    # Handle reach_x/reach_y to rho_x/rho_y conversion
    if reach_x is not None:
        if rho_x is not None:
            raise ValueError("Specify either rho_x or reach_x, not both.")
        rho_x = reach_x ** 2
    if reach_y is not None:
        if rho_y is not None:
            raise ValueError("Specify either rho_y or reach_y, not both.")
        rho_y = reach_y ** 2

    # Build epsilon schedule
    if eps_list is None:
        if use_epsilon_scaling:
            if diameter is None:
                diameter = max_diameter(x, y)
            eps_list = epsilon_schedule(diameter, blur, scaling, p=2.0)
        else:
            if eps is None or n_iters is None:
                raise ValueError("When use_epsilon_scaling=False, provide eps and n_iters")
            eps_list = [float(eps)] * int(n_iters)

    if len(eps_list) == 0:
        raise ValueError("eps_list must be non-empty")
    if n_iters is not None:
        eps_list = list(eps_list)[:int(n_iters)]

    # Early stopping state (will be set after first iteration)
    prev_f = None
    prev_g = None

    n_iters_used = 0

    # Precompute small vectors
    x_f32 = x.float().contiguous()
    y_f32 = y.float().contiguous()
    alpha = cost_scale * (x_f32 ** 2).sum(dim=1)
    beta = cost_scale * (y_f32 ** 2).sum(dim=1)
    log_a = log_weights(a)
    log_b = log_weights(b)

    # Work entirely in shifted potential space: f̂ = f - α, ĝ = g - β
    # Standard init is f=0, g=0, so f̂ = 0 - α = -α, ĝ = 0 - β = -β
    # With warm-start: f̂ = f_init - α, ĝ = g_init - β
    if f_init is not None:
        f_hat = f_init.float().contiguous() - alpha
    else:
        f_hat = -alpha.clone()
    if g_init is not None:
        g_hat = g_init.float().contiguous() - beta
    else:
        g_hat = -beta.clone()

    # Auto-select fused vs separate kernels when fused=None (default).
    # Benchmark on A100-80GB shows separate kernels ~10% faster at n >= 30000.
    # Explicit fused=True/False overrides the heuristic.
    _FUSE_THRESHOLD = 30000
    if fused is None:
        fused = n < _FUSE_THRESHOLD

    if fused:
        # =====================================================================
        # FUSED PATH: Single kernel launch per iteration
        # Both f and g computed in parallel using old potentials
        # =====================================================================

        # Initial step at eps_list[0] (alpha=1.0) - FUSED kernel
        eps_0 = eps_list[0]
        # Semi-unbalanced OT: damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
        # CRITICAL FIX: rho_x controls SOURCE marginal → damps f potential
        damp_f = dampening(eps_0, rho_x)
        damp_g = dampening(eps_0, rho_y)
        f_hat, g_hat = flashsinkhorn_symmetric_step(
            x_f32, y_f32, f_hat, g_hat, log_a, log_b, eps_0,
            cost_scale=cost_scale, alpha=1.0, damping_f=damp_f, damping_g=damp_g,
            allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            label_x=label_x, label_y=label_y, label_cost_matrix=label_cost_matrix,
            lambda_x=lambda_x, lambda_y=lambda_y,
        )
        # UNBALANCED OT CORRECTION for initial step (alpha=1.0)
        if damp_f < 1.0:
            f_hat = f_hat - 1.0 * alpha * (1.0 - damp_f)
        if damp_g < 1.0:
            g_hat = g_hat - 1.0 * beta * (1.0 - damp_g)
        n_iters_used += 1

        # Symmetric updates with alpha=0.5 - FUSED kernel
        for iter_idx, step_eps in enumerate(eps_list):
            damp_f = dampening(step_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(step_eps, rho_y)  # FIXED: rho_y → g

            # FUSED: both f and g updates in ONE kernel launch
            f_hat, g_hat = flashsinkhorn_symmetric_step(
                x_f32, y_f32, f_hat, g_hat, log_a, log_b, step_eps,
                cost_scale=cost_scale, alpha=0.5, damping_f=damp_f, damping_g=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
                label_x=label_x, label_y=label_y, label_cost_matrix=label_cost_matrix,
                lambda_x=lambda_x, lambda_y=lambda_y,
            )
            # UNBALANCED OT CORRECTION: The kernel incorrectly applies damping to the
            # shift term (α, β). The correct formula is: f̂ = damping*f_raw - α
            # but the kernel computes: f̂_bug = damping*f_raw - damping*α
            # Correction: subtract sym_alpha * α * (1 - damping) to fix the shift
            # (because f_correct - f_bug = -α*(1-damping))
            if damp_f < 1.0:
                f_hat = f_hat - 0.5 * alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_hat = g_hat - 0.5 * beta * (1.0 - damp_g)
            n_iters_used += 1

            # Early stopping check (in shifted space)
            if threshold is not None and (iter_idx + 1) % check_every == 0:
                if prev_f is None:
                    prev_f = f_hat.clone()
                    prev_g = g_hat.clone()
                else:
                    _no = n_orig if n_orig is not None else len(f_hat)
                    _mo = m_orig if m_orig is not None else len(g_hat)
                    f_change = (f_hat[:_no] - prev_f[:_no]).abs().max().item()
                    g_change = (g_hat[:_mo] - prev_g[:_mo]).abs().max().item()
                    if max(f_change, g_change) < threshold:
                        break
                    prev_f.copy_(f_hat)
                    prev_g.copy_(g_hat)

        # Final extrapolation (alpha=1.0) - FUSED kernel
        if last_extrapolation:
            f_prelast = f_hat + alpha
            g_prelast = g_hat + beta

            final_eps = eps_list[-1]
            damp_f = dampening(final_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(final_eps, rho_y)  # FIXED: rho_y → g

            f_hat, g_hat = flashsinkhorn_symmetric_step(
                x_f32, y_f32, f_hat, g_hat, log_a, log_b, final_eps,
                cost_scale=cost_scale, alpha=1.0, damping_f=damp_f, damping_g=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
                label_x=label_x, label_y=label_y, label_cost_matrix=label_cost_matrix,
                lambda_x=lambda_x, lambda_y=lambda_y,
            )
            # UNBALANCED OT CORRECTION: alpha=1.0 for final step
            if damp_f < 1.0:
                f_hat = f_hat - 1.0 * alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_hat = g_hat - 1.0 * beta * (1.0 - damp_g)
            n_iters_used += 1

    else:
        # Note: SEPARATE PATH does not currently support OTDD label cost
        # For label cost, use fused=True (default)
        if label_x is not None and label_y is not None and label_cost_matrix is not None and lambda_y != 0.0:
            raise ValueError("OTDD label cost requires fused=True (default). Set fused=True or omit label_cost_matrix.")
        # =====================================================================
        # SEPARATE PATH: Two kernel launches per iteration
        # Uses flashsinkhorn_lse for each update with manual averaging
        #
        # KEY: Uses raw x, y coordinates (NOT pre-scaled Q, K) to ensure
        # TF32 rounding matches the fused kernel. This is critical for parity.
        # =====================================================================

        # Initial step at eps_list[0] (alpha=1.0)
        # CRITICAL: For initialization, GeomLoss uses softmin(log_a) and softmin(log_b)
        # Initial step at eps_list[0] (alpha=1.0)
        eps_0 = eps_list[0]
        # Semi-unbalanced OT: damping_f = 1/(1+eps/rho_x), damping_g = 1/(1+eps/rho_y)
        # CRITICAL FIX: rho_x controls SOURCE marginal → damps f potential
        damp_f = dampening(eps_0, rho_x)
        damp_g = dampening(eps_0, rho_y)
        gamma = eps_0 * log_a
        delta = eps_0 * log_b

        # Store old potentials for symmetric computation
        f_old = f_hat.clone()  # = -alpha
        g_old = g_hat.clone()  # = -beta

        # f-update (full, alpha=1.0) - uses OLD g
        u = (g_old + delta) / eps_0
        f_cand = flashsinkhorn_lse(
            x_f32, y_f32, u, eps_0, cost_scale=cost_scale, damping=damp_f,
            allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
        )

        # g-update (full, alpha=1.0) - uses OLD f (symmetric!)
        v = (f_old + gamma) / eps_0
        g_cand = flashsinkhorn_lse(
            y_f32, x_f32, v, eps_0, cost_scale=cost_scale, damping=damp_g,
            allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
        )

        # UNBALANCED OT CORRECTION for initial step
        if damp_f < 1.0:
            f_cand = f_cand - alpha * (1.0 - damp_f)
        if damp_g < 1.0:
            g_cand = g_cand - beta * (1.0 - damp_g)

        # alpha=1.0 means no averaging: new = candidate
        f_hat = f_cand
        g_hat = g_cand
        n_iters_used += 1

        # Symmetric updates with alpha=0.5
        for iter_idx, step_eps in enumerate(eps_list):
            damp_f = dampening(step_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(step_eps, rho_y)  # FIXED: rho_y → g
            delta = step_eps * log_b
            gamma = step_eps * log_a

            # Store old potentials for symmetric averaging
            f_old = f_hat.clone()
            g_old = g_hat.clone()

            # f-update candidate (uses old g)
            u = (g_old + delta) / step_eps
            f_cand = flashsinkhorn_lse(
                x_f32, y_f32, u, step_eps, cost_scale=cost_scale, damping=damp_f,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )

            # g-update candidate (uses old f, NOT f_cand - this is symmetric!)
            v = (f_old + gamma) / step_eps
            g_cand = flashsinkhorn_lse(
                y_f32, x_f32, v, step_eps, cost_scale=cost_scale, damping=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )

            # UNBALANCED OT CORRECTION: Fix damping over-application to shift term
            # f_correct - f_bug = -α*(1-damping), so subtract to correct
            if damp_f < 1.0:
                f_cand = f_cand - alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_cand = g_cand - beta * (1.0 - damp_g)

            # Symmetric averaging: new = 0.5 * old + 0.5 * candidate
            f_hat = 0.5 * f_old + 0.5 * f_cand
            g_hat = 0.5 * g_old + 0.5 * g_cand
            n_iters_used += 1

            # Early stopping check
            if threshold is not None and (iter_idx + 1) % check_every == 0:
                if prev_f is None:
                    prev_f = f_hat.clone()
                    prev_g = g_hat.clone()
                else:
                    _no = n_orig if n_orig is not None else len(f_hat)
                    _mo = m_orig if m_orig is not None else len(g_hat)
                    f_change = (f_hat[:_no] - prev_f[:_no]).abs().max().item()
                    g_change = (g_hat[:_mo] - prev_g[:_mo]).abs().max().item()
                    if max(f_change, g_change) < threshold:
                        break
                    prev_f.copy_(f_hat)
                    prev_g.copy_(g_hat)

        # Final extrapolation (alpha=1.0)
        if last_extrapolation:
            f_prelast = f_hat + alpha
            g_prelast = g_hat + beta

            final_eps = eps_list[-1]
            damp_f = dampening(final_eps, rho_x)  # FIXED: rho_x → f
            damp_g = dampening(final_eps, rho_y)  # FIXED: rho_y → g
            gamma = final_eps * log_a
            delta = final_eps * log_b

            # Store old for Jacobi-style (both use old potentials)
            f_old = f_hat.clone()
            g_old = g_hat.clone()

            # Full f-update (uses old g)
            u = (g_old + delta) / final_eps
            f_cand = flashsinkhorn_lse(
                x_f32, y_f32, u, final_eps, cost_scale=cost_scale, damping=damp_f,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )
            # Full g-update (uses old f - Jacobi style!)
            v = (f_old + gamma) / final_eps
            g_cand = flashsinkhorn_lse(
                y_f32, x_f32, v, final_eps, cost_scale=cost_scale, damping=damp_g,
                allow_tf32=allow_tf32, use_exp2=use_exp2, autotune=autotune,
            )
            # UNBALANCED OT CORRECTION: alpha=1.0 for final step
            if damp_f < 1.0:
                f_cand = f_cand - alpha * (1.0 - damp_f)
            if damp_g < 1.0:
                g_cand = g_cand - beta * (1.0 - damp_g)
            # alpha=1.0: no averaging
            f_hat = f_cand
            g_hat = g_cand
            n_iters_used += 1

    # Convert to standard potentials at the end
    f = f_hat + alpha
    g = g_hat + beta

    if return_prelast and last_extrapolation:
        if return_n_iters:
            return f, g, f_prelast, g_prelast, n_iters_used
        return f, g, f_prelast, g_prelast

    if return_n_iters:
        return f, g, n_iters_used
    return f, g
