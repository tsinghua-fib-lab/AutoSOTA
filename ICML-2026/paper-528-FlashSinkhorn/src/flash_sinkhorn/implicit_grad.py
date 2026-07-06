"""Implicit gradient computation for differentiating through Sinkhorn.

This implements the same computation as OTT-JAX's ImplicitDiff, using
FlashSinkhorn's streaming kernels for O(nd) memory.

The implicit gradient uses the Implicit Function Theorem (IFT):
1. At convergence: F(f*, g*, x) = 0 (optimality conditions)
2. By IFT: ∂L/∂x = -(∂F/∂x)^T @ (∂F/∂z)^{-T} @ [∂L/∂f; ∂L/∂g]

This requires solving a linear system with the Jacobian of the optimality
conditions, which has the same Schur complement structure as the HVP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from flash_sinkhorn.cg import CGInfo, conjugate_gradient
from flash_sinkhorn.kernels.sinkhorn_triton_apply_sqeuclid import (
    apply_plan_vec_flashstyle,
    apply_plan_mat_flashstyle,
)


@dataclass(frozen=True)
class ImplicitGradInfo:
    cg_converged: bool
    cg_iters: int
    cg_residual: float


def implicit_grad_x_from_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hat: torch.Tensor,
    g_hat: torch.Tensor,
    grad_f: torch.Tensor,
    grad_g: torch.Tensor,
    *,
    eps: float,
    log_a: Optional[torch.Tensor] = None,
    log_b: Optional[torch.Tensor] = None,
    cost_scale: float = 1.0,
    tau2: float = 1e-5,
    max_cg_iter: int = 100,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    allow_tf32: bool = False,
    use_exp2: bool = True,
) -> Tuple[torch.Tensor, ImplicitGradInfo]:
    """Compute implicit gradient ∂L/∂x via IFT.

    This is equivalent to what OTT-JAX's ImplicitDiff computes.

    Parameters
    ----------
    x : torch.Tensor
        Source points (n, d)
    y : torch.Tensor
        Target points (m, d)
    f_hat : torch.Tensor
        Shifted source potential (n,). For FlashStyle: f_hat = f - cost_scale * ||x||^2
    g_hat : torch.Tensor
        Shifted target potential (m,). For FlashStyle: g_hat = g - cost_scale * ||y||^2
    grad_f : torch.Tensor
        Gradient of loss w.r.t. f potential (n,)
    grad_g : torch.Tensor
        Gradient of loss w.r.t. g potential (m,)
    eps : float
        Entropy regularization
    log_a : Optional[torch.Tensor]
        Log source weights log(a), shape (n,). If provided with log_b, uses FlashStyle kernels.
    log_b : Optional[torch.Tensor]
        Log target weights log(b), shape (m,). If provided with log_a, uses FlashStyle kernels.
    cost_scale : float
        Scaling for squared Euclidean cost. 1.0 = ||x-y||^2, 0.5 = ||x-y||^2/2 (GeomLoss).
    tau2 : float
        Tikhonov regularization for numerical stability

    Returns
    -------
    grad_x : torch.Tensor
        Implicit gradient ∂L/∂x, shape (n, d)
    info : ImplicitGradInfo
        Convergence information

    Notes
    -----
    The implicit gradient is computed via the Implicit Function Theorem.
    At convergence, the optimality conditions F(f, g, x) = 0 hold:
        F_f = r - a = 0  (row sum constraint, r = P @ 1)
        F_g = c - b = 0  (col sum constraint, c = P^T @ 1)

    The Jacobian of F w.r.t. z = (f, g) is:
        ∂F/∂z = [[∂r/∂f, ∂r/∂g], [∂c/∂f, ∂c/∂g]]
              = [[diag(r)/eps, P/eps], [P^T/eps, diag(c)/eps]]

    Note: All off-diagonal blocks are POSITIVE since ∂P/∂f > 0 and ∂P/∂g > 0.
    The matrix is symmetric, so (∂F/∂z)^T = ∂F/∂z.

    We solve ∂F/∂z @ λ = [grad_f; grad_g] for λ = [λ_f; λ_g].
    Then: ∂L/∂x = -(∂F/∂x)^T @ λ
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x, y must be 2D tensors.")
    if not x.is_cuda:
        raise ValueError("CUDA required.")

    n, d = x.shape
    m = y.shape[0]
    eps_f = float(eps)

    # Default to uniform marginals if not provided
    if log_a is None:
        log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
    if log_b is None:
        log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)

    # Use FlashStyle kernels (always - the new primary implementation)
    def apply_P(vec_m: torch.Tensor) -> torch.Tensor:
        """Compute P @ vec using FlashStyle (row reduction)."""
        return apply_plan_vec_flashstyle(
            x, y, f_hat, g_hat, log_a, log_b, vec_m,
            eps=eps_f, axis=1, cost_scale=cost_scale,
            allow_tf32=allow_tf32, use_exp2=use_exp2,
        )

    def apply_PT(vec_n: torch.Tensor) -> torch.Tensor:
        """Compute P^T @ vec using FlashStyle (column reduction)."""
        return apply_plan_vec_flashstyle(
            x, y, f_hat, g_hat, log_a, log_b, vec_n,
            eps=eps_f, axis=0, cost_scale=cost_scale,
            allow_tf32=allow_tf32, use_exp2=use_exp2,
        )

    def apply_P_mat(mat: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute P @ mat using FlashStyle."""
        return apply_plan_mat_flashstyle(
            x, y, f_hat, g_hat, log_a, log_b, mat,
            eps=eps_f, axis=1, cost_scale=cost_scale,
            scale=scale,
            allow_tf32=allow_tf32, use_exp2=use_exp2,
        )

    # Compute marginals (row and column sums of P)
    ones_m = torch.ones(m, device=x.device, dtype=torch.float32)
    ones_n = torch.ones(n, device=x.device, dtype=torch.float32)
    r = apply_P(ones_m)  # Row sums: r_i = sum_j P_ij
    c = apply_PT(ones_n)  # Col sums: c_j = sum_i P_ij

    # =========================================================================
    # Solve ∂F/∂z @ λ = [grad_f; grad_g] using Schur complement
    # =========================================================================
    # The Jacobian ∂F/∂z is symmetric with structure:
    #   [[diag(r)/eps, P/eps], [P^T/eps, diag(c)/eps]]
    #
    # Block system:
    #   [diag(r)/eps] @ λ_f + [P/eps] @ λ_g = grad_f   (1)
    #   [P^T/eps] @ λ_f + [diag(c)/eps] @ λ_g = grad_g   (2)
    #
    # From (1): λ_f = eps * grad_f / r - P @ λ_g / r
    #
    # Substitute into (2):
    #   P^T @ (eps * grad_f / r - P @ λ_g / r) / eps + diag(c) @ λ_g / eps = grad_g
    #   P^T @ (grad_f / r) - P^T @ P @ λ_g / (eps * r) + diag(c) @ λ_g / eps = grad_g
    #
    # Schur complement system for λ_g:
    #   [diag(c)/eps - P^T @ diag(1/r) @ P / eps] @ λ_g = grad_g - P^T @ (grad_f / r)
    #
    # With regularization:
    #   [diag(c + eps*tau2)/eps - P^T @ diag(1/r) @ P / eps] @ λ_g = rhs

    # Compute RHS for Schur complement
    rhs = grad_g - apply_PT(grad_f / r)  # grad_g - P^T @ (grad_f / r)

    # Diagonal for Schur complement
    diag_schur = (c + eps_f * tau2) / eps_f

    def schur_matvec(z: torch.Tensor) -> torch.Tensor:
        """Apply Schur complement: [diag(c+tau2)/eps - P^T @ diag(1/r) @ P / eps] @ z"""
        Pz = apply_P(z)  # P @ z, shape (n,)
        PT_invr_Pz = apply_PT(Pz / r)  # P^T @ diag(1/r) @ P @ z, shape (m,)
        return diag_schur * z - PT_invr_Pz / eps_f

    # Solve for λ_g
    lambda_g, cg_info = conjugate_gradient(
        schur_matvec,
        rhs,
        max_iter=max_cg_iter,
        rtol=cg_rtol,
        atol=cg_atol,
    )

    # Back-solve for λ_f
    # λ_f = eps * grad_f / r - P @ λ_g / r
    lambda_f = eps_f * grad_f / r - apply_P(lambda_g) / r

    # =========================================================================
    # Compute ∂L/∂x = -(∂F/∂x)^T @ λ
    # =========================================================================
    # F_f = r - a, so ∂r/∂x contributes to ∂L/∂x via λ_f
    # F_g = c - b, so ∂c/∂x contributes to ∂L/∂x via λ_g
    #
    # ∂r_i/∂x_i = (2/eps) * P_i,:  @ (x_i - y)  (derivative of row sum)
    # ∂c_j/∂x = (2/eps) * P_:,j @ (x - y_j)     (derivative of col sum)
    #
    # The full gradient is:
    # ∂L/∂x_i = sum_j ∂r_i/∂x_i @ λ_f_i + sum_j ∂c_j/∂x_i @ λ_g_j
    #
    # After simplification (similar to explicit gradient):
    # ∂L/∂x_i = (2/eps) * [λ_f_i * (x_i * r_i - (P @ y)_i) +
    #                       (P @ (λ_g * y))_i - (P @ λ_g)_i * x_i]

    # Compute P @ y and P @ (λ_g * y)
    Py = apply_P_mat(y.float())

    # P @ (λ_g[:, None] * y) - we need to weight y by λ_g before applying P
    Py_lambda = apply_P_mat(y.float(), scale=lambda_g)

    # P @ λ_g
    P_lambda_g = apply_P(lambda_g)

    # Implicit gradient
    # ∂L/∂x = (2/eps) * [λ_f * (x * r - P @ y) + (P @ λ_g) * x - P @ (λ_g * y)]
    #
    # Term 1: λ_f * (x * r - P @ y)
    term1 = lambda_f[:, None] * (x * r[:, None] - Py)

    # Term 2: (P @ λ_g) * x - P @ (λ_g * y)
    term2 = P_lambda_g[:, None] * x - Py_lambda

    grad_x = (2.0 / eps_f) * (term1 + term2)

    return grad_x, ImplicitGradInfo(
        cg_converged=bool(cg_info.cg_converged),
        cg_iters=int(cg_info.cg_iters),
        cg_residual=float(cg_info.cg_residual),
    )


def implicit_grad_x(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_f: torch.Tensor,
    grad_g: torch.Tensor,
    *,
    eps: float,
    cost_scale: float = 1.0,
    n_iters: int = 100,
    tau2: float = 1e-5,
    max_cg_iter: int = 100,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    use_flashstyle: Optional[bool] = None,  # Deprecated, FlashSinkhorn is the only backend
) -> Tuple[torch.Tensor, ImplicitGradInfo]:
    """End-to-end implicit gradient: solve Sinkhorn then compute ∂L/∂x.

    This is equivalent to calling jax.grad on a loss through OTT-JAX Sinkhorn
    with ImplicitDiff.
    """
    import warnings
    if use_flashstyle is not None and not use_flashstyle:
        warnings.warn(
            "use_flashstyle=False is deprecated. FlashSinkhorn is now the only backend.",
            DeprecationWarning,
            stacklevel=2,
        )

    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
        sinkhorn_flashstyle_symmetric,
        standard_to_shifted_potentials,
    )

    # Solve Sinkhorn to get potentials
    f, g = sinkhorn_flashstyle_symmetric(
        x, y, a, b,
        use_epsilon_scaling=False,
        eps=eps,
        cost_scale=cost_scale,
        n_iters=n_iters,
        last_extrapolation=True,
        autotune=autotune,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
    )

    # Convert to shifted potentials for FlashSinkhorn kernels
    alpha = cost_scale * (x.float() ** 2).sum(dim=1)
    beta = cost_scale * (y.float() ** 2).sum(dim=1)
    f_hat, g_hat = standard_to_shifted_potentials(f, g, alpha, beta)
    log_a = torch.log(a.clamp(min=1e-20))
    log_b = torch.log(b.clamp(min=1e-20))

    return implicit_grad_x_from_potentials(
        x, y, f_hat, g_hat, grad_f, grad_g,
        eps=eps,
        log_a=log_a,
        log_b=log_b,
        cost_scale=cost_scale,
        tau2=tau2,
        max_cg_iter=max_cg_iter,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
    )
