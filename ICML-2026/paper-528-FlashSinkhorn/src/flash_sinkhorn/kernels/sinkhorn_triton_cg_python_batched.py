"""Python-level batched CG for HVP acceleration.

This module implements a batched CG solver that uses the FlashStyle
apply_plan_vec_flashstyle kernels for improved performance and reduced memory usage.

This avoids the mysterious Triton compilation state issues observed with
the fully-inline batched CG kernel while still providing a clean interface
for batched CG solving.

Includes a torch.compile-compatible version for additional speedup.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch

from flash_sinkhorn.kernels.sinkhorn_triton_apply_sqeuclid import apply_plan_vec_flashstyle


@dataclass
class PythonBatchedCgInfo:
    """Information about Python batched CG execution."""
    cg_converged: bool
    cg_iters: int
    cg_residual: float
    cg_initial_residual: float


# Cache for compiled CG functions (keyed by max_iter)
_compiled_cg_cache: dict = {}


def python_batched_cg_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    diag_x: torch.Tensor,
    denom: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 50,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    autotune: bool = False,
) -> Tuple[torch.Tensor, PythonBatchedCgInfo]:
    """Solve H @ x = b using CG with external apply_plan_vec kernels.

    This is a simpler, more reliable implementation that uses the proven
    apply_plan_vec_sqeuclid kernels instead of inline Triton softmax.

    The linear operator is:
        H @ v = denom * v - P^T @ (P @ v / diag_x)

    where P is the (n, m) transport plan matrix.

    Args:
        x: Source points (n, d)
        y: Target points (m, d)
        f: Source potential (n,)
        g: Target potential (m,)
        diag_x: Diagonal D_x (n,) - row sums of P
        denom: Denominator D_y (m,) - column sums of P plus regularization
        rhs: Right-hand side b (m,)
        eps: Regularization parameter
        x2: Precomputed ||x||^2 (n,) [optional]
        y2: Precomputed ||y||^2 (m,) [optional]
        x0: Initial guess (m,) [optional]
        max_iter: Maximum CG iterations
        rtol, atol: Convergence tolerances
        autotune: Whether to use Triton autotuning for apply_plan_vec

    Returns:
        sol: Solution x (m,)
        info: Convergence information
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (n,d) and (m,d).")
    if not x.is_cuda:
        raise ValueError("CUDA required.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension.")

    eps_f = float(eps)

    # Precompute squared norms if not provided
    if x2 is None:
        x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    if y2 is None:
        y2 = (y.float() * y.float()).sum(dim=1).contiguous()

    # Ensure float32
    f = f.float().contiguous()
    g = g.float().contiguous()
    diag_x = diag_x.float().contiguous()
    denom = denom.float().contiguous()
    rhs = rhs.float().contiguous()
    x2 = x2.float().contiguous()
    y2 = y2.float().contiguous()

    # Convert OTT-style potentials (f, g) to FlashStyle shifted potentials
    # For balanced OT: log_a = log_b = -log(n) = -log(m) = const
    log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
    log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)

    # Compute squared norms for shifting
    alpha = (x.float() ** 2).sum(dim=1)  # [n] = ||x||^2
    beta = (y.float() ** 2).sum(dim=1)   # [m] = ||y||^2

    # Shifted potentials: f_hat = f - alpha, g_hat = g - beta
    # (using cost_scale=1.0 since we're working with full squared Euclidean cost)
    f_hat = f - alpha
    g_hat = g - beta

    # Define the linear operator
    def linear_op(z: torch.Tensor) -> torch.Tensor:
        """Compute H @ z = denom * z - P^T @ (P @ z / diag_x)"""
        # piz = P @ z (axis 1 reduction)
        piz = apply_plan_vec_flashstyle(
            x, y, f_hat, g_hat, log_a, log_b, z,
            eps=eps_f, axis=1, cost_scale=1.0, allow_tf32=False,
        )
        # P^T @ (piz / diag_x) (axis 0 reduction)
        pt_piz_over_diag = apply_plan_vec_flashstyle(
            x, y, f_hat, g_hat, log_a, log_b, piz / diag_x,
            eps=eps_f, axis=0, cost_scale=1.0, allow_tf32=False,
        )
        return denom * z - pt_piz_over_diag

    # Initialize CG state
    if x0 is None:
        sol = torch.zeros_like(rhs)
        r = rhs.clone()
    else:
        sol = x0.float().clone()
        r = rhs - linear_op(sol)

    p = r.clone()
    rz_old = torch.dot(r, r).item()
    r0_norm = float(torch.sqrt(torch.tensor(rz_old)).item())

    converged = False
    num_iters = 0

    for k in range(max_iter):
        # Compute Ap
        Ap = linear_op(p)

        # CG update
        pAp = torch.dot(p, Ap).item()

        # Avoid division by zero
        if abs(pAp) < 1e-15:
            break

        alpha = rz_old / pAp

        sol = sol + alpha * p
        r = r - alpha * Ap

        rz_new = torch.dot(r, r).item()
        r_norm = float(torch.sqrt(torch.tensor(rz_new)).item())

        num_iters = k + 1

        # Check convergence
        if r_norm < atol or r_norm < rtol * r0_norm:
            converged = True
            break

        beta = rz_new / rz_old

        p = r + beta * p
        rz_old = rz_new

    return sol, PythonBatchedCgInfo(
        cg_converged=converged,
        cg_iters=num_iters,
        cg_residual=float(torch.linalg.norm(r).item()),
        cg_initial_residual=r0_norm,
    )


def _create_compiled_cg_core(max_iter: int):
    """Create a torch.compile-optimized CG core function for a fixed iteration count.

    This function is designed to be torch.compile friendly:
    - Fixed iteration count (no early exit)
    - All tensor operations (no .item() calls)
    - No Python control flow based on tensor values
    """

    def cg_core(
        x: torch.Tensor,
        y: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        diag_x: torch.Tensor,
        denom: torch.Tensor,
        rhs: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core CG loop - runs exactly max_iter iterations.

        Returns:
            sol: Solution vector (m,)
            r: Final residual vector (m,)
            r0_norm_sq: Initial residual norm squared (scalar tensor)
        """
        # Convert OTT-style potentials (f, g) to FlashStyle shifted potentials
        n, d = x.shape
        m = y.shape[0]
        log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
        log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)

        # Compute squared norms for shifting
        alpha = (x.float() ** 2).sum(dim=1)  # [n]
        beta = (y.float() ** 2).sum(dim=1)   # [m]

        # Shifted potentials: f_hat = f - alpha, g_hat = g - beta
        f_hat = f - alpha
        g_hat = g - beta

        # Initialize CG state
        sol = torch.zeros_like(rhs)
        r = rhs.clone()
        p = r.clone()
        rz_old = torch.dot(r, r)
        r0_norm_sq = rz_old.clone()

        # Fixed iteration CG loop
        for _ in range(max_iter):
            # Compute Ap = H @ p = denom * p - P^T @ (P @ p / diag_x)
            piz = apply_plan_vec_flashstyle(
                x, y, f_hat, g_hat, log_a, log_b, p,
                eps=eps, axis=1, cost_scale=1.0, allow_tf32=False,
            )
            pt_piz_over_diag = apply_plan_vec_flashstyle(
                x, y, f_hat, g_hat, log_a, log_b, piz / diag_x,
                eps=eps, axis=0, cost_scale=1.0, allow_tf32=False,
            )
            Ap = denom * p - pt_piz_over_diag

            # CG update (all tensor operations)
            pAp = torch.dot(p, Ap)

            # Safe division (avoid NaN if pAp is tiny)
            alpha_step = torch.where(
                pAp.abs() > 1e-15,
                rz_old / pAp,
                torch.zeros_like(pAp)
            )

            sol = sol + alpha_step * p
            r = r - alpha_step * Ap

            rz_new = torch.dot(r, r)

            # Safe division for beta
            beta_step = torch.where(
                rz_old.abs() > 1e-15,
                rz_new / rz_old,
                torch.zeros_like(rz_old)
            )

            p = r + beta_step * p
            rz_old = rz_new

        return sol, r, r0_norm_sq

    return cg_core


def compiled_cg_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    diag_x: torch.Tensor,
    denom: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
    x2: Optional[torch.Tensor] = None,
    y2: Optional[torch.Tensor] = None,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 50,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    use_compile: bool = True,
) -> Tuple[torch.Tensor, PythonBatchedCgInfo]:
    """Solve H @ x = b using CG with torch.compile optimization.

    This version uses torch.compile for the core CG loop, providing
    CUDA graph capture and kernel fusion for additional speedup.

    Note: This runs a fixed number of iterations (no early exit) to be
    torch.compile friendly. The convergence info reflects the state
    after max_iter iterations.

    Args:
        x: Source points (n, d)
        y: Target points (m, d)
        f: Source potential (n,)
        g: Target potential (m,)
        diag_x: Diagonal D_x (n,) - row sums of P
        denom: Denominator D_y (m,) - column sums of P plus regularization
        rhs: Right-hand side b (m,)
        eps: Regularization parameter
        x2: Precomputed ||x||^2 (n,) [optional]
        y2: Precomputed ||y||^2 (m,) [optional]
        x0: Initial guess (m,) [optional, ignored in compiled version]
        max_iter: Number of CG iterations to run (fixed, no early exit)
        rtol, atol: Convergence tolerances (for info only, no early exit)
        use_compile: Whether to use torch.compile (default True)

    Returns:
        sol: Solution x (m,)
        info: Convergence information
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (n,d) and (m,d).")
    if not x.is_cuda:
        raise ValueError("CUDA required.")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension.")

    eps_f = float(eps)

    # Precompute squared norms if not provided
    if x2 is None:
        x2 = (x.float() * x.float()).sum(dim=1).contiguous()
    if y2 is None:
        y2 = (y.float() * y.float()).sum(dim=1).contiguous()

    # Ensure float32 and contiguous
    f = f.float().contiguous()
    g = g.float().contiguous()
    diag_x = diag_x.float().contiguous()
    denom = denom.float().contiguous()
    rhs = rhs.float().contiguous()
    x2 = x2.float().contiguous()
    y2 = y2.float().contiguous()

    # Get or create compiled CG core function
    if use_compile:
        cache_key = max_iter
        if cache_key not in _compiled_cg_cache:
            cg_core = _create_compiled_cg_core(max_iter)
            # Compile with reduce-overhead mode for CUDA graph capture
            _compiled_cg_cache[cache_key] = torch.compile(
                cg_core,
                mode="reduce-overhead",
                fullgraph=False,  # Allow graph breaks for Triton kernels
            )
        cg_fn = _compiled_cg_cache[cache_key]
    else:
        cg_fn = _create_compiled_cg_core(max_iter)

    # Mark step begin for CUDA graph (prevents tensor overwrite issues)
    if use_compile:
        torch.compiler.cudagraph_mark_step_begin()

    # Run CG
    sol, r, r0_norm_sq = cg_fn(x, y, f, g, diag_x, denom, rhs, x2, y2, eps_f)

    # Clone outputs to avoid CUDA graph memory reuse issues
    sol = sol.clone()
    r = r.clone()

    # Compute info (after synchronization)
    r_norm = torch.linalg.norm(r).item()
    r0_norm = torch.sqrt(r0_norm_sq).item()

    # Check convergence (for info only)
    converged = r_norm < atol or r_norm < rtol * r0_norm

    return sol, PythonBatchedCgInfo(
        cg_converged=converged,
        cg_iters=max_iter,
        cg_residual=r_norm,
        cg_initial_residual=r0_norm,
    )
