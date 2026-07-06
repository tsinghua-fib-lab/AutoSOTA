from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from flash_sinkhorn.cg import CGInfo, conjugate_gradient

def sqeuclid_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dense squared Euclidean cost (tests/reference only)."""

    x_f = x.float()
    y_f = y.float()
    x2 = (x_f * x_f).sum(dim=1)[:, None]
    y2 = (y_f * y_f).sum(dim=1)[None, :]
    return x2 + y2 - 2.0 * (x_f @ y_f.t())


def plan_from_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Dense plan P = exp((f+g-C)/eps) (tests/reference only)."""

    c = sqeuclid_cost_matrix(x, y)
    return torch.exp((f[:, None].float() + g[None, :].float() - c) / float(eps))


def hvp_x_dense_sqeuclid(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    use_preconditioner: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Dense OTT-Hessian-style HVP w.r.t. x (tests/reference only).

    Mirrors `3rd-party/OTT-Hessian/test.py` but uses dense PyTorch tensors.
    """

    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")

    eps_f = float(eps)
    P = plan_from_potentials(x, y, f, g, eps=eps_f)

    a_hat = P.sum(dim=1)  # (n,)
    b_hat = P.sum(dim=0)  # (m,)

    def apply_axis1(vec: torch.Tensor) -> torch.Tensor:
        return P @ vec

    def apply_axis0(vec: torch.Tensor) -> torch.Tensor:
        return P.t() @ vec

    vec1 = torch.sum(x * A, dim=1)  # (n,)
    Py = apply_axis1(y)  # (n,d)

    x1 = 2.0 * (a_hat * vec1 - torch.sum(A * Py, dim=1))  # (n,)
    PT_A = apply_axis0(A)  # (m,d)
    x2 = 2.0 * (apply_axis0(vec1) - torch.sum(y * PT_A, dim=1))  # (m,)

    y1 = x1 / a_hat
    y2_raw = -apply_axis0(y1) + x2
    denom = b_hat + eps_f * float(tau2)

    def _apply_B(z_vec: torch.Tensor) -> torch.Tensor:
        piz = apply_axis1(z_vec)
        piT_over_a_piz = apply_axis0(piz / a_hat)
        return piT_over_a_piz / denom

    if use_preconditioner:
        rhs = y2_raw / denom

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            return z_vec - _apply_B(z_vec)

        def preconditioner(v: torch.Tensor) -> torch.Tensor:
            b1 = _apply_B(v)
            b2 = _apply_B(b1)
            b3 = _apply_B(b2)
            return v + b1 + b2 + b3

        z, info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=preconditioner,
        )
    else:
        rhs = y2_raw

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            piz = apply_axis1(z_vec)
            piT_over_a_piz = apply_axis0(piz / a_hat)
            return denom * z_vec - piT_over_a_piz

        z, info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=None,
        )

    z1 = y1 - apply_axis1(z) / a_hat
    z2 = z

    vec2_z = apply_axis1(z2)  # (n,)
    Mat4 = apply_axis1(y * z2[:, None])  # (n,d)
    RTz = 2.0 * (
        x * (a_hat * z1)[:, None]
        - Py * z1[:, None]
        + x * vec2_z[:, None]
        - Mat4
    )

    Mat1 = 2.0 * a_hat[:, None] * A
    Mat2 = (-4.0 / eps_f) * x * (vec1 * a_hat)[:, None]
    Mat3 = (4.0 / eps_f) * Py * vec1[:, None]
    vec2 = torch.sum(Py * A, dim=1)
    Mat4_e = (4.0 / eps_f) * x * vec2[:, None]

    dot_Ay = A @ y.t()  # (n,m)
    Mat5 = (-4.0 / eps_f) * ((P * dot_Ay) @ y)

    EA = Mat1 + Mat2 + Mat3 + Mat4_e + Mat5
    hvp = RTz / eps_f + EA
    return hvp, {
        "cg_converged": float(info.cg_converged),
        "cg_iters": float(info.cg_iters),
        "cg_residual": float(info.cg_residual),
        "cg_initial_residual": float(info.cg_initial_residual),
    }


def hvp_x_dense_sqeuclid_semi_unbalanced(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float,
    rho_x: Optional[float] = None,
    rho_y: Optional[float] = None,
    tau2: float = 1e-5,
    max_cg_iter: int = 300,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    use_preconditioner: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Dense OTT-Hessian-style HVP for semi-unbalanced/unbalanced OT.

    Extends hvp_x_dense_sqeuclid to support semi-unbalanced optimal transport
    where marginal constraints can be relaxed independently via KL divergence penalties.

    Args:
        x: Source points (n, d)
        y: Target points (m, d)
        f: Source OTT-style potential (n,)
        g: Target OTT-style potential (m,)
        A: Input matrix for HVP (n, d)
        eps: Entropy regularization
        rho_x: Source marginal penalty (None = strict constraint)
        rho_y: Target marginal penalty (None = strict constraint)
        tau2: Tikhonov regularization (only used for balanced OT)
        max_cg_iter: Max CG iterations
        cg_rtol: Relative CG tolerance
        cg_atol: Absolute CG tolerance
        use_preconditioner: Whether to use Neumann preconditioner

    Returns:
        hvp: Hessian-vector product (n, d)
        info: CG convergence info dict

    Notes:
        The H matrix structure depends on which marginals are relaxed:

        Balanced (rho_x=None, rho_y=None):
          H = [[diag(a_hat),        P     ],
               [    P^T,       diag(b_hat)]]
          H is PSD (singular), needs tau2 regularization.

        Semi-unbalanced with relaxed source (rho_x != None, rho_y=None):
          H = [[(rho_x+eps)/rho_x * diag(a_hat),       P       ],
               [           P^T,               diag(b_hat)]]
          H is PD (invertible), no tau2 needed.

        Semi-unbalanced with relaxed target (rho_x=None, rho_y != None):
          H = [[      diag(a_hat),                    P       ],
               [         P^T,        (rho_y+eps)/rho_y * diag(b_hat)]]
          H is PD (invertible), no tau2 needed.

        Fully unbalanced (rho_x != None, rho_y != None):
          Both diagonals are scaled, H is PD.
    """
    if x.ndim != 2 or y.ndim != 2 or A.ndim != 2:
        raise ValueError("x,y,A must be 2D tensors.")
    n, d = x.shape
    m, d2 = y.shape
    if d != d2 or A.shape != (n, d):
        raise ValueError("Shapes must satisfy x:(n,d), y:(m,d), A:(n,d).")

    eps_f = float(eps)
    P = plan_from_potentials(x, y, f, g, eps=eps_f)

    a_hat = P.sum(dim=1)  # (n,)
    b_hat = P.sum(dim=0)  # (m,)

    # Compute diagonal scaling factors for semi-unbalanced OT
    # lambda = rho / (rho + eps) is the damping factor
    # diag_factor = 1 / lambda = (rho + eps) / rho
    is_balanced = rho_x is None and rho_y is None

    if rho_x is not None:
        lambda_x = float(rho_x) / (float(rho_x) + eps_f)
        diag_factor_x = 1.0 / lambda_x  # = (rho_x + eps) / rho_x
    else:
        diag_factor_x = 1.0

    if rho_y is not None:
        lambda_y = float(rho_y) / (float(rho_y) + eps_f)
        diag_factor_y = 1.0 / lambda_y  # = (rho_y + eps) / rho_y
    else:
        diag_factor_y = 1.0

    # Effective diagonal entries
    diag_x = diag_factor_x * a_hat  # (n,)
    diag_y = diag_factor_y * b_hat  # (m,)

    def apply_axis1(vec: torch.Tensor) -> torch.Tensor:
        return P @ vec

    def apply_axis0(vec: torch.Tensor) -> torch.Tensor:
        return P.t() @ vec

    vec1 = torch.sum(x * A, dim=1)  # (n,)
    Py = apply_axis1(y)  # (n,d)

    # R^T @ A part (same as balanced, but using scaled diagonals)
    x1 = 2.0 * (a_hat * vec1 - torch.sum(A * Py, dim=1))  # (n,)
    PT_A = apply_axis0(A)  # (m,d)
    x2 = 2.0 * (apply_axis0(vec1) - torch.sum(y * PT_A, dim=1))  # (m,)

    # Solve H @ z = [x1; x2] using Schur complement
    # z = [z1; z2] where z2 solves: (diag_y - P^T @ diag_x^{-1} @ P) @ z2 = y2_raw
    # and z1 = diag_x^{-1} @ (x1 - P @ z2)

    y1 = x1 / diag_x  # Note: using diag_x, not a_hat
    y2_raw = -apply_axis0(y1) + x2

    # Denominator for Schur complement
    # For balanced: denom = b_hat + eps * tau2 (regularization for PSD)
    # For semi/fully unbalanced: denom = diag_y (H is PD, no regularization)
    if is_balanced:
        denom = diag_y + eps_f * float(tau2)
    else:
        denom = diag_y

    def _apply_B(z_vec: torch.Tensor) -> torch.Tensor:
        """Apply B = diag_y^{-1} @ P^T @ diag_x^{-1} @ P."""
        piz = apply_axis1(z_vec)
        piT_over_diag_x_piz = apply_axis0(piz / diag_x)
        return piT_over_diag_x_piz / denom

    if use_preconditioner:
        rhs = y2_raw / denom

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            return z_vec - _apply_B(z_vec)

        def preconditioner(v: torch.Tensor) -> torch.Tensor:
            b1 = _apply_B(v)
            b2 = _apply_B(b1)
            b3 = _apply_B(b2)
            return v + b1 + b2 + b3

        z, info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=preconditioner,
        )
    else:
        rhs = y2_raw

        def linear_op(z_vec: torch.Tensor) -> torch.Tensor:
            piz = apply_axis1(z_vec)
            piT_over_diag_x_piz = apply_axis0(piz / diag_x)
            return denom * z_vec - piT_over_diag_x_piz

        z, info = conjugate_gradient(
            linear_op,
            rhs,
            max_iter=max_cg_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            preconditioner=None,
        )

    z1 = y1 - apply_axis1(z) / diag_x  # Note: using diag_x, not a_hat
    z2 = z

    # R^T @ z part (same as balanced, uses a_hat not diag_x)
    vec2_z = apply_axis1(z2)  # (n,)
    Mat4 = apply_axis1(y * z2[:, None])  # (n,d)
    RTz = 2.0 * (
        x * (a_hat * z1)[:, None]
        - Py * z1[:, None]
        + x * vec2_z[:, None]
        - Mat4
    )

    # E @ A part (same as balanced)
    Mat1 = 2.0 * a_hat[:, None] * A
    Mat2 = (-4.0 / eps_f) * x * (vec1 * a_hat)[:, None]
    Mat3 = (4.0 / eps_f) * Py * vec1[:, None]
    vec2 = torch.sum(Py * A, dim=1)
    Mat4_e = (4.0 / eps_f) * x * vec2[:, None]

    dot_Ay = A @ y.t()  # (n,m)
    Mat5 = (-4.0 / eps_f) * ((P * dot_Ay) @ y)

    EA = Mat1 + Mat2 + Mat3 + Mat4_e + Mat5
    hvp = RTz / eps_f + EA

    return hvp, {
        "cg_converged": float(info.cg_converged),
        "cg_iters": float(info.cg_iters),
        "cg_residual": float(info.cg_residual),
        "cg_initial_residual": float(info.cg_initial_residual),
        "rho_x": rho_x,
        "rho_y": rho_y,
        "is_balanced": is_balanced,
    }
