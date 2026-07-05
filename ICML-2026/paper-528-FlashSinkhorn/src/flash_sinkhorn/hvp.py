"""Hessian-vector product (HVP) utilities for optimal transport."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "HvpInfo",
    "geomloss_to_ott_potentials",
    "hvp_x_sqeuclid",
    "hvp_x_sqeuclid_from_potentials",
    "inverse_hvp_x_sqeuclid_from_potentials",
]

from flash_sinkhorn.cg import conjugate_gradient
from flash_sinkhorn.kernels.sinkhorn_triton_cg_python_batched import (
    python_batched_cg_solve,
    compiled_cg_solve,
)
from flash_sinkhorn.kernels.sinkhorn_triton_cg_dense import (
    hvp_dense_cg,
    DenseCgInfo,
)
# Import from _common for utility functions
from flash_sinkhorn.kernels._common import log_weights
# Import FlashSinkhorn for the solver (no longer using deprecated geomloss module)
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_symmetric,
)
from flash_sinkhorn.kernels.sinkhorn_triton_apply_sqeuclid import (
    apply_plan_vec_flashstyle,
    apply_plan_mat_flashstyle,
    mat5_sqeuclid,  # Keep OTT-style for now, converts internally
)
from flash_sinkhorn.kernels.sinkhorn_triton_apply_fused_sqeuclid import (
    fused_schur_matvec_sqeuclid,
)


@dataclass(frozen=True)
class HvpInfo:
    cg_converged: bool
    cg_iters: int
    cg_residual: float
    cg_initial_residual: float
    converged_lambda: float = 0.0  # LM lambda that achieved convergence (0 if not using LM)
    hit_neg_curv: bool = False  # True if CG stopped due to negative curvature (truncated Newton)
    diagnostic_msg: str = ""  # Deferred diagnostic message for logging after step line


def geomloss_to_ott_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert GeomLoss-style potentials to OTT-style potentials.

    GeomLoss convention corresponds to a plan:
      P = diag(a) * exp((f+g-C)/eps) * diag(b)
    OTT convention uses:
      P = exp((f_hat+g_hat-C)/eps)
    with:
      f_hat = f + eps*log(a), g_hat = g + eps*log(b).
    """

    loga = log_weights(a)
    logb = log_weights(b)
    eps_f = float(eps)
    return f.float() + eps_f * loga, g.float() + eps_f * logb


def ott_to_geomloss_potentials(
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert OTT-style potentials to GeomLoss-style potentials.

    OTT convention uses:
      P = exp((f_hat+g_hat-C)/eps)
    GeomLoss convention corresponds to a plan:
      P = diag(a) * exp((f+g-C)/eps) * diag(b)
    with:
      f = f_hat - eps*log(a), g = g_hat - eps*log(b).
    """
    loga = log_weights(a)
    logb = log_weights(b)
    eps_f = float(eps)
    return f_hat.float() - eps_f * loga, g_hat.float() - eps_f * logb


def sinkhorn_prelast_potentials_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps_list: Sequence[float],
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,  # Ignored, kept for backward compat
    block_n: Optional[int] = None,  # Ignored, kept for backward compat
    block_k: Optional[int] = None,  # Ignored, kept for backward compat
    num_warps: Optional[int] = None,  # Ignored, kept for backward compat
    num_stages: int = 2,  # Ignored, kept for backward compat
    use_flashstyle: Optional[bool] = None,  # Deprecated, FlashSinkhorn is the only backend
    cost_scale: float = 1.0,  # Cost scaling (1.0 for full, 0.5 for half)
    rho_x: Optional[float] = None,  # Unbalanced OT source penalty
    rho_y: Optional[float] = None,  # Unbalanced OT target penalty
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return prelast potentials (f_grad, g_grad) at final epsilon.

    Args:
        x, y: Point clouds [n, d] and [m, d]
        a, b: Marginal weights [n] and [m]
        eps_list: Epsilon schedule
        allow_tf32: Enable TF32 for matmul
        use_exp2: Use exp2/log2 optimization
        autotune: Enable kernel autotuning
        cost_scale: Cost scaling (1.0 for full ||x-y||², 0.5 for half)
        rho_x, rho_y: Unbalanced OT marginal penalties (None = balanced)

    Returns:
        f_grad, g_grad: Pre-extrapolation potentials at final epsilon
    """
    if use_flashstyle is not None and not use_flashstyle:
        import warnings
        warnings.warn(
            "use_flashstyle=False is deprecated. FlashSinkhorn is now the only backend.",
            DeprecationWarning,
            stacklevel=2,
        )
    f_cost, g_cost, f_grad, g_grad = sinkhorn_flashstyle_symmetric(
        x,
        y,
        a,
        b,
        use_epsilon_scaling=False,
        last_extrapolation=True,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        eps_list=eps_list,
        autotune=autotune,
        return_prelast=True,
        cost_scale=cost_scale,
        rho_x=rho_x,
        rho_y=rho_y,
    )
    del f_cost, g_cost
    return f_grad, g_grad


def hvp_x_sqeuclid_from_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    rho_x: Optional[float] = None,
    rho_y: Optional[float] = None,
    # Cost convention: 1.0 = full ||x-y||² (FlashSinkhorn default), 0.5 = half (GeomLoss default)
    cost_scale: float = 1.0,
    # Selective damping: 1.0=full Hessian, 0.0=PSD-only, 0.5=half indefinite contribution
    # Use <1.0 to improve conditioning when Hessian is indefinite
    lambda_indef: float = 1.0,
    # If True, return (PSD_part, indefinite_part, info) for efficient damping sweep
    return_components: bool = False,
    # Inner Hessian regularization (Schur complement). 1e-5 is empirically stable.
    # Too large (>1e-3) slows convergence, too small (<1e-8) causes numerical issues.
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    # CG tolerances: 1e-6 is standard. Looser (1e-4) for unbalanced OT outer solve.
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    use_fused_matvec: bool = False,
    use_batched_cg: bool = False,
    use_compiled_cg: bool = False,
    use_dense_cg: bool = False,
    autotune: bool = True,
    cg_x0: Optional[torch.Tensor] = None,  # Initial guess for CG (warm-start)
) -> Tuple[torch.Tensor, HvpInfo]:
    """OTT-Hessian-style HVP w.r.t x using streaming transport primitives.

    Parameters
    ----------
    x : torch.Tensor
        Source points (n, d)
    y : torch.Tensor
        Target points (m, d)
    f_hat : torch.Tensor
        Source OTT-style potential (n,)
    g_hat : torch.Tensor
        Target OTT-style potential (m,)
    A : torch.Tensor
        Input matrix for HVP (n, d)
    eps : float
        Entropy regularization
    rho_x : Optional[float]
        Source marginal KL penalty (None = strict constraint)
    rho_y : Optional[float]
        Target marginal KL penalty (None = strict constraint)
    tau2 : float
        Tikhonov regularization (only used for balanced OT)
    max_cg_iter : int
        Maximum CG iterations
    cg_rtol : float
        Relative CG tolerance
    cg_atol : float
        Absolute CG tolerance
    use_dense_cg : bool
        If True, materialize the transport plan P (O(nm) memory) and use
        dense matrix-vector products instead of streaming Triton kernels.
        This is 8x faster for small problems (n <= 8192) but uses O(nm) memory.
        Recommended for n*m <= 64M entries (256 MB for float32).

    Notes
    -----
    - This implementation is correctness-first and uses vector transport kernels
      for all plan applications. It is *not* optimized yet (matrix applications
      are performed by looping over feature dimensions).
    - No cost/plan materialization occurs (except implicit work inside kernels).
    - For semi-unbalanced OT (rho_x or rho_y != None), the H matrix is PD
      (positive definite) instead of PSD, which improves CG convergence.
      The tau2 regularization is automatically disabled in this case.
    """

    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")
    if not x.is_cuda or not y.is_cuda or not f_hat.is_cuda or not g_hat.is_cuda:
        raise ValueError("hvp_x_sqeuclid_from_potentials requires CUDA tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")

    eps_f = float(eps)

    # Fast path: Dense CG with cached transport plan
    # This is 8x faster for small problems (n <= 8192) by avoiding kernel launch overhead
    if use_dense_cg:
        hvp, dense_info = hvp_dense_cg(
            x, y, f_hat, g_hat, A,
            eps=eps_f,
            rho_x=rho_x,
            rho_y=rho_y,
            tau2=tau2,
            max_cg_iter=max_cg_iter,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
        )
        return hvp, HvpInfo(
            cg_converged=bool(dense_info.cg_converged),
            cg_iters=int(dense_info.cg_iters),
            cg_residual=float(dense_info.cg_residual),
            cg_initial_residual=float(dense_info.cg_initial_residual),
        )

    x_norm2 = (x.float() * x.float()).sum(dim=1).contiguous()
    y_norm2 = (y.float() * y.float()).sum(dim=1).contiguous()

    # =========================================================================
    # Convert OTT-style potentials to FlashStyle shifted potentials (once)
    # =========================================================================
    # OTT convention: P = exp((f_hat + g_hat - C) / eps)
    # FlashStyle convention: P = a * b * exp((f_shift + g_shift + 2*cs*xy) / eps)
    #
    # Conversion: f_shift = f_hat - eps*log_a - alpha (where alpha = cs*||x||²)
    # log_a, log_b are a gauge choice: they cancel in the conversion+application
    # roundtrip (f_shift absorbs -eps*log_a, kernel re-adds +log_a). Any constant
    # works; uniform -log(n) is conventional.
    log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
    log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)
    alpha = cost_scale * x_norm2
    beta = cost_scale * y_norm2
    f_shift = f_hat.float() - eps_f * log_a - alpha
    g_shift = g_hat.float() - eps_f * log_b - beta

    def apply_axis1(vec_m: torch.Tensor) -> torch.Tensor:
        return apply_plan_vec_flashstyle(
            x, y, f_shift, g_shift, log_a, log_b, vec_m,
            eps=eps_f,
            axis=1,
            cost_scale=cost_scale,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
        )

    def apply_axis0(vec_n: torch.Tensor) -> torch.Tensor:
        return apply_plan_vec_flashstyle(
            x, y, f_shift, g_shift, log_a, log_b, vec_n,
            eps=eps_f,
            axis=0,
            cost_scale=cost_scale,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
        )

    ones_m = torch.ones((m,), device=x.device, dtype=torch.float32)
    ones_n = torch.ones((n,), device=x.device, dtype=torch.float32)
    a_hat = apply_axis1(ones_m)
    b_hat = apply_axis0(ones_n)

    # Compute diagonal scaling factors for semi-unbalanced OT
    # In Séjourné et al. "Sinkhorn Divergences for Unbalanced OT":
    #   tau = rho / (rho + eps) is the damping factor
    #   When rho -> infinity (balanced), tau -> 1
    #   When rho -> 0 (fully unbalanced), tau -> 0
    # The diagonal scaling factor is 1/tau = (rho + eps) / rho
    #   When rho -> infinity, diag_factor -> 1 (balanced)
    #   When rho -> 0, diag_factor -> infinity
    is_balanced = rho_x is None and rho_y is None

    if rho_x is not None:
        diag_factor_x = (float(rho_x) + eps_f) / float(rho_x)  # = 1/tau_x = (rho+eps)/rho
    else:
        diag_factor_x = 1.0

    if rho_y is not None:
        diag_factor_y = (float(rho_y) + eps_f) / float(rho_y)  # = 1/tau_y = (rho+eps)/rho
    else:
        diag_factor_y = 1.0

    # CRITICAL: Clamp marginals to prevent division by near-zero values.
    # For unbalanced OT with sparse transport plans (e.g., clustered data),
    # some rows/columns can have near-zero mass, causing 1/diag_x → ∞ → NaN.
    # Use a small epsilon relative to the mean marginal value for stability.
    marginal_clamp = max(1e-12, eps_f * 1e-10)
    a_hat_clamped = a_hat.clamp(min=marginal_clamp)
    b_hat_clamped = b_hat.clamp(min=marginal_clamp)

    # Effective diagonal entries for the H matrix (use clamped marginals)
    diag_x = diag_factor_x * a_hat_clamped  # (n,)
    diag_y = diag_factor_y * b_hat_clamped  # (m,)

    vec1 = torch.sum(x * A, dim=1)  # (n,)
    Py = apply_plan_mat_flashstyle(
        x, y, f_shift, g_shift, log_a, log_b, y.float(),
        eps=eps_f,
        axis=1,
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
    )

    x1 = (2.0 * cost_scale) * (a_hat * vec1 - torch.sum(A * Py, dim=1))

    PT_A = apply_plan_mat_flashstyle(
        x, y, f_shift, g_shift, log_a, log_b, A.float(),
        eps=eps_f,
        axis=0,
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
    )
    x2 = (2.0 * cost_scale) * (apply_axis0(vec1) - torch.sum(y * PT_A, dim=1))

    # Use diag_x (scaled diagonal) for the Schur complement solve
    y1 = x1 / diag_x
    y2_raw = -apply_axis0(y1) + x2

    # Denominator for Schur complement:
    # - Balanced: denom = b_hat + eps * tau2 (regularization for PSD H)
    # - Semi/fully unbalanced: denom = diag_y (H is PD, no regularization needed)
    if is_balanced:
        denom = diag_y + eps_f * float(tau2)
    else:
        denom = diag_y

    def _apply_B(z_vec: torch.Tensor) -> torch.Tensor:
        """Apply B = denom^{-1} @ P^T @ diag_x^{-1} @ P."""
        piz = apply_axis1(z_vec)
        piT_over_diag_x_piz = apply_axis0(piz / diag_x)
        return piT_over_diag_x_piz / denom

    precond_mode = str(preconditioner).lower()
    if not use_preconditioner:
        precond_mode = "none"
    if precond_mode not in ("none", "neumann", "jacobi"):
        raise ValueError("preconditioner must be one of {'none','neumann','jacobi'}.")
    precond_terms_i = int(precond_terms)
    if precond_terms_i < 0:
        raise ValueError("precond_terms must be >= 0.")

    if precond_mode == "none":
        rhs = y2_raw

        if use_compiled_cg:
            # Use torch.compile-optimized CG with CUDA graph capture
            # Best for: small problems (n<=512) or when max_iter iterations are needed.
            # Note: Runs fixed max_iter iterations (no early exit) for graph compatibility.
            z, batched_info = compiled_cg_solve(
                x, y, f_hat, g_hat, diag_x, denom, rhs,
                eps=eps_f,
                x2=x_norm2,
                y2=y_norm2,
                max_iter=max_cg_iter,
                rtol=cg_rtol,
                atol=cg_atol,
                use_compile=True,
            )
            from flash_sinkhorn.cg import CGInfo
            cg_info = CGInfo(
                cg_converged=batched_info.cg_converged,
                cg_iters=batched_info.cg_iters,
                cg_residual=batched_info.cg_residual,
                cg_initial_residual=batched_info.cg_initial_residual,
            )
        elif use_batched_cg:
            # Use Python batched CG with external apply_plan_vec kernels
            # This passes Sinkhorn data directly to the CG solver, avoiding
            # repeated closure overhead and providing a clean batched interface.
            z, batched_info = python_batched_cg_solve(
                x, y, f_hat, g_hat, diag_x, denom, rhs,
                eps=eps_f,
                x2=x_norm2,
                y2=y_norm2,
                max_iter=max_cg_iter,
                rtol=cg_rtol,
                atol=cg_atol,
                autotune=autotune,
                x0=cg_x0,
            )
            # Convert PythonBatchedCgInfo to CGInfo-like object
            from flash_sinkhorn.cg import CGInfo
            cg_info = CGInfo(
                cg_converged=batched_info.cg_converged,
                cg_iters=batched_info.cg_iters,
                cg_residual=batched_info.cg_residual,
                cg_initial_residual=batched_info.cg_initial_residual,
            )
        elif use_fused_matvec:
            # Use fused Schur complement matvec kernel (single kernel per CG iteration)
            # Pre-allocate buffer for intermediate P @ z result
            piz_buffer = torch.empty((n,), device=x.device, dtype=torch.float32)

            # Block sizes for fused kernel (use defaults if not specified)
            # CRITICAL: BLOCK_K must be < D to ensure multiple k iterations.
            # The Triton compiler has a bug where BLOCK_K >= D causes incorrect results.
            fused_block_m = block_m if block_m is not None else 64
            fused_block_n = block_n if block_n is not None else 64
            if block_k is not None:
                fused_block_k = block_k
            elif d >= 64:
                fused_block_k = 32  # Forces at least 2 k iterations for d >= 64
            elif d >= 32:
                fused_block_k = 16  # Forces at least 2 k iterations for d >= 32
            else:
                fused_block_k = 16  # Minimum for tl.dot

            def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
                nonlocal piz_buffer
                result, piz_buffer = fused_schur_matvec_sqeuclid(
                    x, y, f_hat, g_hat, z_vec, diag_x, denom,
                    eps=eps_f,
                    x2=x_norm2,
                    y2=y_norm2,
                    piz_buffer=piz_buffer,
                    block_m=fused_block_m,
                    block_n=fused_block_n,
                    block_k=fused_block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    use_exp2=use_exp2,
                    allow_tf32=allow_tf32,
                )
                return result

            z, cg_info = conjugate_gradient(
                linear_op,
                rhs,
                x0=cg_x0,
                max_iter=max_cg_iter,
                rtol=cg_rtol,
                atol=cg_atol,
                preconditioner=None,
                stabilise_every=cg_stabilise_every,
            )
        else:
            # Original two-kernel approach (axis1 + axis0)
            def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
                piz = apply_axis1(z_vec)
                piT_over_diag_x_piz = apply_axis0(piz / diag_x)
                return denom * z_vec - piT_over_diag_x_piz

            z, cg_info = conjugate_gradient(
                linear_op,
                rhs,
                x0=cg_x0,
                max_iter=max_cg_iter,
                rtol=cg_rtol,
                atol=cg_atol,
                preconditioner=None,
                stabilise_every=cg_stabilise_every,
            )
    else:
        rhs = y2_raw / denom

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            return z_vec - _apply_B(z_vec)

        precond_fn = None
        if precond_mode == "neumann":
            def precond_fn(v: torch.Tensor) -> torch.Tensor:
                out = v
                cur = v
                for _ in range(precond_terms_i):
                    cur = _apply_B(cur)
                    out = out + cur
                return out

        z, cg_info = conjugate_gradient(
            linear_op,
            rhs,
            x0=cg_x0,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=precond_fn,
            stabilise_every=cg_stabilise_every,
        )

    # Use diag_x for the back-solve
    z1 = y1 - apply_axis1(z) / diag_x
    z2 = z

    vec2_z = apply_axis1(z2)  # (n,)
    Py_z2 = apply_plan_mat_flashstyle(
        x, y, f_shift, g_shift, log_a, log_b, y.float(),
        eps=eps_f,
        axis=1,
        cost_scale=cost_scale,
        scale=z2,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
    )
    RTz = (2.0 * cost_scale) * (
        x * (a_hat * z1)[:, None] - Py * z1[:, None] + x * vec2_z[:, None] - Py_z2
    )

    # Full Hessian: H = (1/ε) R^T (H*)^{-1} R + E
    # where E = Mat1 + Mat2 + Mat3 + Mat4 + Mat5
    # - Mat1 = 2μI @ A (positive diagonal, PSD)
    # - Mat2, Mat3, Mat4, Mat5 = covariance terms (can cause indefiniteness)
    Mat1 = (2.0 * cost_scale) * a_hat[:, None] * A
    Mat2 = (-4.0 * cost_scale / eps_f) * x * (vec1 * a_hat)[:, None]
    Mat3 = (4.0 * cost_scale / eps_f) * Py * vec1[:, None]
    vec2 = torch.sum(Py * A, dim=1)
    Mat4 = (4.0 * cost_scale / eps_f) * x * vec2[:, None]

    Mat5 = mat5_sqeuclid(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps_f,
        cost_scale=cost_scale,
        x2=x_norm2,
        y2=y_norm2,
        allow_tf32=allow_tf32,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=use_exp2,
    )

    # Selective damping: PSD_part + lambda_indef * indefinite_part
    # - RTz/eps + Mat1 is always PSD (implicit term + positive diagonal)
    # - Mat2+Mat3+Mat4+Mat5 contains negative covariance terms (indefinite)
    # - lambda_indef=1.0: full Hessian (may have negative eigenvalues)
    # - lambda_indef<1.0: reduce indefinite contribution to improve conditioning
    # - lambda_indef=0.0: PSD-only (Gauss-Newton-like, but poor approximation)
    PSD_part = RTz / eps_f + Mat1
    indefinite_part = Mat2 + Mat3 + Mat4 + Mat5

    info = HvpInfo(
        cg_converged=bool(cg_info.cg_converged),
        cg_iters=int(cg_info.cg_iters),
        cg_residual=float(cg_info.cg_residual),
        cg_initial_residual=float(cg_info.cg_initial_residual),
    )

    if return_components:
        # Return components separately for efficient damping sweep
        # Caller can combine as: hvp = PSD_part + lambda_indef * indefinite_part
        return PSD_part, indefinite_part, info

    # Standard return: combined HVP
    if lambda_indef == 1.0:
        hvp = PSD_part + indefinite_part
    else:
        hvp = PSD_part + lambda_indef * indefinite_part

    return hvp, info


def hvp_x_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A: torch.Tensor,
    *,
    eps_list: Sequence[float],
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, HvpInfo]:
    """End-to-end HVP: solve prelast potentials then compute H@A (w.r.t x)."""

    eps = float(eps_list[-1])
    f_grad, g_grad = sinkhorn_prelast_potentials_sqeuclid(
        x,
        y,
        a,
        b,
        eps_list=eps_list,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    f_hat, g_hat = geomloss_to_ott_potentials(f_grad, g_grad, a, b, eps=eps)

    return hvp_x_sqeuclid_from_potentials(
        x,
        y,
        f_hat,
        g_hat,
        A,
        eps=eps,
        tau2=tau2,
        max_cg_iter=max_cg_iter,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_stabilise_every=cg_stabilise_every,
        preconditioner=preconditioner,
        precond_terms=precond_terms,
        use_preconditioner=use_preconditioner,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=int(num_warps or 4),
        num_stages=num_stages,
    )


def inverse_hvp_x_sqeuclid_from_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    gr: torch.Tensor,
    *,
    eps: float,
    rho_x: Optional[float] = None,
    rho_y: Optional[float] = None,
    tau2: float = 1e-5,
    max_outer_iter: int = 100,
    outer_rtol: float = 1e-6,
    outer_atol: float = 1e-6,
    max_inner_iter: int = 100,
    inner_rtol: float = 1e-8,
    inner_atol: float = 1e-8,
    cg_stabilise_every: int = 0,
    preconditioner: str = "none",
    precond_terms: int = 3,
    use_preconditioner: bool = True,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, HvpInfo]:
    """Solve H @ z = gr for z, i.e., compute z = H⁻¹ @ gr (inverse HVP).

    This is equivalent to what OTT-JAX's ImplicitDiff computes for gradient
    backpropagation through Sinkhorn. Uses CG with our HVP as the matvec.

    Parameters
    ----------
    x : torch.Tensor
        Source points (n, d)
    y : torch.Tensor
        Target points (m, d)
    f_hat : torch.Tensor
        Source OTT-style potential (n,)
    g_hat : torch.Tensor
        Target OTT-style potential (m,)
    gr : torch.Tensor
        Input gradient/vector to solve for (n, d) - the RHS of H @ z = gr
    eps : float
        Entropy regularization
    rho_x : Optional[float]
        Source marginal KL penalty (None = strict constraint)
    rho_y : Optional[float]
        Target marginal KL penalty (None = strict constraint)
    tau2 : float
        Tikhonov regularization
    max_outer_iter : int
        Maximum CG iterations for the outer solve (H @ z = gr)
    outer_rtol : float
        Relative tolerance for outer CG
    outer_atol : float
        Absolute tolerance for outer CG
    max_inner_iter : int
        Maximum CG iterations for HVP's internal Schur complement solve
    inner_rtol : float
        Relative tolerance for inner CG
    inner_atol : float
        Absolute tolerance for inner CG

    Returns
    -------
    z : torch.Tensor
        Solution z = H⁻¹ @ gr, shape (n, d)
    info : HvpInfo
        Convergence information for the outer CG solve

    Notes
    -----
    This function solves H @ z = gr where H is the Hessian of the OT cost
    w.r.t. x at the converged potentials. The solution z = H⁻¹ @ gr is what
    OTT-JAX's ImplicitDiff computes for gradient backpropagation.

    The solve uses CG with hvp_x_sqeuclid_from_potentials as the matvec,
    which itself uses CG internally for the Schur complement. This results
    in a nested CG structure, but converges quickly in practice.
    """
    if x.ndim != 2 or y.ndim != 2 or gr.ndim != 2:
        raise ValueError("x, y, gr must be 2D tensors.")
    if not x.is_cuda or not y.is_cuda or not f_hat.is_cuda or not g_hat.is_cuda:
        raise ValueError("inverse_hvp_x_sqeuclid_from_potentials requires CUDA tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or gr.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), gr:(n,d).")

    # Flatten gr for CG (CG works with 1D vectors)
    gr_flat = gr.reshape(-1)

    def hvp_matvec(v_flat: torch.Tensor) -> torch.Tensor:
        """Compute H @ v using our HVP function."""
        v = v_flat.reshape(n, d)
        hvp_result, _ = hvp_x_sqeuclid_from_potentials(
            x, y, f_hat, g_hat, v,
            eps=eps,
            rho_x=rho_x,
            rho_y=rho_y,
            tau2=tau2,
            max_cg_iter=max_inner_iter,
            cg_rtol=inner_rtol,
            cg_atol=inner_atol,
            cg_stabilise_every=cg_stabilise_every,
            preconditioner=preconditioner,
            precond_terms=precond_terms,
            use_preconditioner=use_preconditioner,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return hvp_result.reshape(-1)

    # Solve H @ z = gr using CG
    z_flat, cg_info = conjugate_gradient(
        hvp_matvec,
        gr_flat,
        max_iter=max_outer_iter,
        rtol=outer_rtol,
        atol=outer_atol,
        preconditioner=None,
        stabilise_every=cg_stabilise_every,
    )

    z = z_flat.reshape(n, d)
    return z, HvpInfo(
        cg_converged=bool(cg_info.cg_converged),
        cg_iters=int(cg_info.cg_iters),
        cg_residual=float(cg_info.cg_residual),
        cg_initial_residual=float(cg_info.cg_initial_residual),
    )
