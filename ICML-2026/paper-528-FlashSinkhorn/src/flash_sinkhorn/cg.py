"""Conjugate gradient solvers for optimization.

Includes:
- Standard CG for symmetric positive definite systems
- Steihaug-CG for trust-region subproblems (handles indefinite Hessians)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

__all__ = ["CGInfo", "conjugate_gradient", "SteihaugCGInfo", "steihaug_cg"]


@dataclass(frozen=True)
class CGInfo:
    cg_converged: bool
    cg_iters: int
    cg_residual: float
    cg_initial_residual: float


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 300,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stabilise_every: int = 0,
) -> Tuple[torch.Tensor, CGInfo]:
    """Minimal CG for symmetric positive definite systems.

    Parameters
    ----------
    matvec : callable
        Matrix-vector product function A @ v
    b : torch.Tensor
        Right-hand side vector
    x0 : torch.Tensor, optional
        Initial guess. If provided, CG starts from x0 instead of zeros.
        This enables warm-starting for faster convergence when solving
        similar systems (e.g., consecutive Newton steps).
    max_iter : int
        Maximum number of CG iterations
    rtol, atol : float
        Relative and absolute tolerance for convergence
    preconditioner : callable, optional
        Preconditioner function M^{-1} @ v
    stabilise_every : int
        Recompute true residual every N iterations to reduce drift (0 = disabled)

    Returns
    -------
    x : torch.Tensor
        Solution vector
    info : CGInfo
        Convergence information
    """

    stabilise_every_i = int(stabilise_every)
    if stabilise_every_i < 0:
        raise ValueError("stabilise_every must be >= 0.")

    # Initialize from x0 if provided (warm-start), otherwise from zeros
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()  # r = b - A @ 0 = b
    else:
        x = x0.clone()
        r = b - matvec(x)  # Recompute residual from warm-start
    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = torch.dot(r, z)

    init_res = float(torch.linalg.norm(r).detach().cpu())
    tol = max(atol, rtol * init_res)

    cg_converged = False
    for it in range(int(max_iter)):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if denom.abs() == 0:
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if stabilise_every_i and (it + 1) % stabilise_every_i == 0:
            r = b - matvec(x)

        res = float(torch.linalg.norm(r).detach().cpu())
        if res <= tol:
            cg_converged = True
            iters = it + 1
            return x, CGInfo(
                cg_converged=True,
                cg_iters=iters,
                cg_residual=res,
                cg_initial_residual=init_res,
            )

        z = preconditioner(r) if preconditioner is not None else r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    res = float(torch.linalg.norm(r).detach().cpu())
    return x, CGInfo(
        cg_converged=cg_converged,
        cg_iters=int(max_iter),
        cg_residual=res,
        cg_initial_residual=init_res,
    )


# =============================================================================
# Steihaug-CG for Trust-Region Subproblems
# =============================================================================


@dataclass(frozen=True)
class SteihaugCGInfo:
    """Information about Steihaug-CG iteration.

    Attributes:
        converged: True if CG converged within trust region
        iters: Number of CG iterations
        residual: Final residual norm
        termination_reason: Why CG terminated
            - "converged": Normal CG convergence within trust region
            - "negative_curvature": Hit negative curvature, moved to boundary
            - "boundary": Step would exit trust region, truncated to boundary
            - "max_iter": Hit iteration limit
        hit_boundary: True if solution is on trust region boundary
        negative_curvature_detected: True if pAp < 0 was detected
        predicted_reduction: Predicted reduction in quadratic model
    """
    converged: bool
    iters: int
    residual: float
    termination_reason: str
    hit_boundary: bool
    negative_curvature_detected: bool
    predicted_reduction: float


def _solve_trust_region_boundary(
    x: torch.Tensor,
    p: torch.Tensor,
    delta: float,
) -> float:
    """Find τ > 0 such that ||x + τp|| = delta.

    Solves the quadratic equation:
        ||x + τp||² = δ²
        τ²||p||² + 2τ(x·p) + ||x||² = δ²

    Returns the positive root (or larger root if both positive).
    """
    xx = torch.dot(x, x).item()
    xp = torch.dot(x, p).item()
    pp = torch.dot(p, p).item()

    # Quadratic: pp * τ² + 2*xp * τ + (xx - δ²) = 0
    a = pp
    b = 2 * xp
    c = xx - delta * delta

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # Shouldn't happen if ||x|| < delta, but handle gracefully
        return delta / math.sqrt(pp + 1e-10)

    sqrt_disc = math.sqrt(discriminant)

    # Two roots: (-b ± sqrt_disc) / (2a)
    tau1 = (-b + sqrt_disc) / (2 * a + 1e-10)
    tau2 = (-b - sqrt_disc) / (2 * a + 1e-10)

    # We want the positive root (going forward along p)
    # If both positive, take the smaller one (first intersection)
    if tau1 > 0 and tau2 > 0:
        return min(tau1, tau2)
    elif tau1 > 0:
        return tau1
    elif tau2 > 0:
        return tau2
    else:
        # Both non-positive, shouldn't happen
        return abs(tau1)


def steihaug_cg(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    grad: torch.Tensor,
    delta: float,
    *,
    max_iter: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    neg_curv_tol: float = 1e-10,
) -> Tuple[torch.Tensor, SteihaugCGInfo]:
    """Steihaug-CG for trust-region subproblems.

    Solves the trust-region subproblem:
        min_p  g^T p + 0.5 p^T H p   subject to ||p|| ≤ δ

    where g = grad and H is accessed via hvp_fn.

    Key features:
    1. Handles indefinite Hessians (saddle points)
    2. When negative curvature detected (pAp < 0), moves to trust region
       boundary in that direction → enables saddle escape!
    3. Truncates to boundary if step would exit trust region

    Parameters
    ----------
    hvp_fn : callable
        Hessian-vector product function H @ v
    grad : torch.Tensor
        Gradient at current point (negative of RHS for CG)
    delta : float
        Trust region radius
    max_iter : int
        Maximum CG iterations
    rtol, atol : float
        Convergence tolerances
    neg_curv_tol : float
        Threshold for detecting negative curvature (pAp < neg_curv_tol)

    Returns
    -------
    p : torch.Tensor
        Step direction (solution to trust-region subproblem)
    info : SteihaugCGInfo
        Detailed iteration information

    References
    ----------
    Steihaug, T. (1983). "The conjugate gradient method and trust regions
    in large scale optimization." SIAM J. Numerical Analysis.

    Nocedal & Wright, "Numerical Optimization", Algorithm 7.2.
    """
    device = grad.device
    dtype = grad.dtype
    n = grad.numel()

    # Initialize
    p = torch.zeros_like(grad)  # Current solution
    r = -grad.clone()  # Residual = -g - H @ 0 = -g
    d = r.clone()  # Search direction

    r_norm = torch.linalg.norm(r).item()
    init_res = r_norm
    tol = max(atol, rtol * init_res)

    # Handle zero gradient (already at stationary point)
    if r_norm < tol:
        return p, SteihaugCGInfo(
            converged=True,
            iters=0,
            residual=r_norm,
            termination_reason="converged",
            hit_boundary=False,
            negative_curvature_detected=False,
            predicted_reduction=0.0,
        )

    neg_curv_detected = False

    for it in range(max_iter):
        # Compute H @ d
        Hd = hvp_fn(d)
        dHd = torch.dot(d, Hd).item()

        # Check for negative curvature
        if dHd <= neg_curv_tol:
            neg_curv_detected = True
            # Move to trust region boundary in direction d
            tau = _solve_trust_region_boundary(p, d, delta)
            p_final = p + tau * d

            # Compute predicted reduction: -g^T p - 0.5 p^T H p
            Hp = hvp_fn(p_final)
            pred_red = -torch.dot(grad, p_final).item() - 0.5 * torch.dot(p_final, Hp).item()

            return p_final, SteihaugCGInfo(
                converged=False,
                iters=it + 1,
                residual=r_norm,
                termination_reason="negative_curvature",
                hit_boundary=True,
                negative_curvature_detected=True,
                predicted_reduction=pred_red,
            )

        # Standard CG step
        rr = torch.dot(r, r).item()
        alpha = rr / dHd
        p_new = p + alpha * d

        # Check if step exits trust region
        p_new_norm = torch.linalg.norm(p_new).item()
        if p_new_norm >= delta:
            # Truncate to boundary
            tau = _solve_trust_region_boundary(p, d, delta)
            p_final = p + tau * d

            Hp = hvp_fn(p_final)
            pred_red = -torch.dot(grad, p_final).item() - 0.5 * torch.dot(p_final, Hp).item()

            return p_final, SteihaugCGInfo(
                converged=False,
                iters=it + 1,
                residual=r_norm,
                termination_reason="boundary",
                hit_boundary=True,
                negative_curvature_detected=neg_curv_detected,
                predicted_reduction=pred_red,
            )

        # Update
        p = p_new
        r = r - alpha * Hd
        r_norm = torch.linalg.norm(r).item()

        # Check convergence
        if r_norm < tol:
            Hp = hvp_fn(p)
            pred_red = -torch.dot(grad, p).item() - 0.5 * torch.dot(p, Hp).item()

            return p, SteihaugCGInfo(
                converged=True,
                iters=it + 1,
                residual=r_norm,
                termination_reason="converged",
                hit_boundary=False,
                negative_curvature_detected=neg_curv_detected,
                predicted_reduction=pred_red,
            )

        # Update search direction
        rr_new = torch.dot(r, r).item()
        beta = rr_new / rr
        d = r + beta * d

    # Max iterations reached
    Hp = hvp_fn(p)
    pred_red = -torch.dot(grad, p).item() - 0.5 * torch.dot(p, Hp).item()
    p_norm = torch.linalg.norm(p).item()

    return p, SteihaugCGInfo(
        converged=False,
        iters=max_iter,
        residual=r_norm,
        termination_reason="max_iter",
        hit_boundary=(p_norm >= delta * 0.99),
        negative_curvature_detected=neg_curv_detected,
        predicted_reduction=pred_red,
    )
