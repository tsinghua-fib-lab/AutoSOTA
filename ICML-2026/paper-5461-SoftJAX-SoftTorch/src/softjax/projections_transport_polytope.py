from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jax import Array
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import implicit_differentiation as idiff, sinkhorn

from softjax.utils import _validate_softness


def _proj_transport_polytope_entropic_sinkhorn(
    C: jax.Array,
    mu: jax.Array,
    nu: jax.Array,
    tol: float = 1e-6,
    max_iter: int = 1000,
    epsilon: float = 1.0,
    return_log_probs: bool = False,
) -> jax.Array:
    # Upcast to float64 for numerical stability of implicit differentiation.
    orig_dtype = C.dtype
    if C.dtype != jnp.float64:
        C = C.astype(jnp.float64)
        mu = mu.astype(jnp.float64)
        nu = nu.astype(jnp.float64)

    # Avoid exact zeros (helps implicit differentiation a lot)
    tiny = 1e-12
    mu = jnp.clip(mu, tiny)
    mu = mu / jnp.sum(mu)
    nu = jnp.clip(nu, tiny)
    nu = nu / jnp.sum(nu)

    geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=mu, b=nu)

    implicit = idiff.ImplicitDiff(
        solver_kwargs={"ridge_identity": 1e-6},
    )

    solver = sinkhorn.Sinkhorn(
        lse_mode=True,
        threshold=tol,
        max_iterations=max_iter,
        implicit_diff=implicit,
    )

    out = solver(prob)
    if return_log_probs:
        log_gamma = (out.f[:, None] + out.g[None, :] - C) / epsilon
        return log_gamma.astype(orig_dtype)
    return out.matrix.astype(orig_dtype)


def _proj_transport_polytope_entropic_lbfgs(
    C: jnp.ndarray,  # (n, m)
    mu: jnp.ndarray,  # (n,)
    nu: jnp.ndarray,  # (m,)
    epsilon: jnp.ndarray,  # scalar
    tol: float,
    max_steps: int,
    gauge_fix: bool = True,
    implicit_diff: bool = True,
    return_log_probs: bool = False,
):
    # Upcast to float64 to avoid dtype mismatch in optimistix L-BFGS
    # when jax_enable_x64=True (L-BFGS history becomes float64, causing scan dtype errors).
    orig_dtype = C.dtype
    if C.dtype != jnp.float64:
        C = C.astype(jnp.float64)
        mu = mu.astype(jnp.float64)
        nu = nu.astype(jnp.float64)
        epsilon = jnp.asarray(epsilon, dtype=jnp.float64)

    mu = jnp.clip(mu, 1e-12)
    nu = jnp.clip(nu, 1e-12)
    mu = mu / jnp.sum(mu)
    nu = nu / jnp.sum(nu)
    n, m = C.shape

    if gauge_fix:
        # Gauge fix: set g0 = 0, optimise f and g_rest to avoid singular system on implicit diff
        y0 = (jnp.zeros((n,), C.dtype), jnp.zeros((m - 1,), C.dtype))  # (f, g_rest)
    else:
        y0 = (jnp.zeros((n,), C.dtype), jnp.zeros((m,), C.dtype))  # (f, g)

    def neg_dual(y, args):
        C_, mu_, nu_, eps_, gauge_fix_ = args
        if gauge_fix_:
            f, g_rest = y
            g = jnp.concatenate([jnp.zeros((1,), C_.dtype), g_rest], axis=0)  # (m,)
        else:
            f, g = y
        Z = (f[:, None] + g[None, :] - C_) / eps_
        return -(jnp.dot(mu_, f) + jnp.dot(nu_, g) - eps_ * jnp.sum(jnp.exp(Z)))

    solver = optx.LBFGS(rtol=tol, atol=tol)
    if implicit_diff:
        adj = optx.ImplicitAdjoint(linear_solver=lx.AutoLinearSolver(well_posed=False))
    else:
        adj = optx.RecursiveCheckpointAdjoint()
    sol = optx.minimise(
        neg_dual,
        solver=solver,
        y0=y0,
        args=(C, mu, nu, epsilon, gauge_fix),
        max_steps=max_steps,
        adjoint=adj,
        throw=True,
    )

    if gauge_fix:
        f, g_rest = sol.value
        g = jnp.concatenate([jnp.zeros((1,), C.dtype), g_rest], axis=0)
    else:
        f, g = sol.value
    log_gamma = (f[:, None] + g[None, :] - C) / epsilon
    if return_log_probs:
        return log_gamma.astype(orig_dtype)
    Gamma = jnp.exp(log_gamma)
    return Gamma.astype(orig_dtype)


def _proj_transport_polytope_pnorm_lbfgs(
    C: jnp.ndarray,  # (n, m)
    mu: jnp.ndarray,  # (n,)
    nu: jnp.ndarray,  # (m,)
    lam: jnp.ndarray,  # scalar
    tol: float,
    max_steps: int,
    gauge_fix: bool = True,
    p: float = 6 / 5,  # 1 < p <= 2
    implicit_diff: bool = True,
):
    # Upcast to float64 to avoid dtype mismatch in optimistix L-BFGS
    # when jax_enable_x64=True (L-BFGS history becomes float64, causing scan dtype errors).
    orig_dtype = C.dtype
    if C.dtype != jnp.float64:
        C = C.astype(jnp.float64)
        mu = mu.astype(jnp.float64)
        nu = nu.astype(jnp.float64)
        lam = jnp.asarray(lam, dtype=jnp.float64)

    mu = jnp.clip(mu, 1e-12)
    nu = jnp.clip(nu, 1e-12)
    mu = mu / jnp.sum(mu)
    nu = nu / jnp.sum(nu)
    n, m = C.shape
    q = p / (p - 1.0)  # conjugate exponent
    lam_pow = lam ** (-(q - 1.0))  # lam^{-(q-1)}

    if gauge_fix:
        # Gauge fix: set g0 = 0, optimise f and g_rest to avoid singular system on implicit diff
        y0 = (jnp.zeros((n,), C.dtype), jnp.zeros((m - 1,), C.dtype))
    else:
        y0 = (jnp.zeros((n,), C.dtype), jnp.zeros((m,), C.dtype))

    def neg_dual(y, args):
        C_, mu_, nu_, lam_pow_, gauge_fix_ = args
        if gauge_fix_:
            f, g_rest = y
            g = jnp.concatenate([jnp.zeros((1,), C_.dtype), g_rest], axis=0)
        else:
            f, g = y
        S = f[:, None] + g[None, :] - C_
        P = jnp.maximum(S, 0.0)
        dual = jnp.dot(mu_, f) + jnp.dot(nu_, g) - (lam_pow_ / q) * jnp.sum(P**q)
        return -dual

    solver = optx.LBFGS(rtol=tol, atol=tol)
    if implicit_diff:
        adj = optx.ImplicitAdjoint(linear_solver=lx.AutoLinearSolver(well_posed=False))
    else:
        adj = optx.RecursiveCheckpointAdjoint()
    sol = optx.minimise(
        neg_dual,
        solver=solver,
        y0=y0,
        args=(C, mu, nu, lam_pow, gauge_fix),
        max_steps=max_steps,
        adjoint=adj,
        throw=True,
    )

    if gauge_fix:
        f, g_rest = sol.value
        g = jnp.concatenate([jnp.zeros((1,), C.dtype), g_rest], axis=0)
    else:
        f, g = sol.value
    S = f[:, None] + g[None, :] - C
    Gamma = lam_pow * jnp.maximum(S, 0.0) ** (q - 1.0)  # = λ^{-(q-1)}[S]_+^{q-1}
    return Gamma.astype(orig_dtype)


@eqx.filter_jit
def _proj_transport_polytope(
    cost: Array,  # (..., n, m)
    mu: Array,  # ([n],)
    nu: Array,  # ([m],)
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
    use_entropic_ot_sinkhorn_on_entropic: bool = True,
    sinkhorn_tol: float = 1e-5,
    sinkhorn_max_iter: int = 10000,
    lbfgs_tol: float = 1e-5,
    lbfgs_max_iter: int = 10000,
    implicit_diff: bool = True,
    return_log_probs: bool = False,
) -> Array:  # (..., [n], m)
    """Projects a cost matrix onto the transport polytope between `mu` and `nu`.

    Solves the optimization problem:
        min_G <C, G> + softness * R(G)
        s.t. G 1_m = mu, G^T 1_n = nu, G >= 0
    where R(G) is the regularizer determined by `mode`.

    **Arguments:**

    - `cost`: Input cost Array of shape (..., n, m).
    - `mu`: Source marginal distribution Array of shape ([n],).
    - `nu`: Target marginal distribution Array of shape ([m],).
    - `softness`: Controls the strength of the regularizer.
    - `mode`: Controls the type of regularizer:
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer). Solved via Sinkhorn (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)) or LBFGS.
        - `c0`: C0 continuous (based on euclidean/L2 regularizer). Solved via LBFGS, projecting onto Birkhoff polytope (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`: C1 differentiable (p=3/2 p-norm). P-norm regularizer is inspired by [Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective](https://arxiv.org/abs/2302.01425). Solved via LBFGS.
        - `c2`: C2 twice differentiable (p=4/3 p-norm). P-norm regularizer is inspired by [Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective](https://arxiv.org/abs/2302.01425). Solved via LBFGS.
    - `use_entropic_ot_sinkhorn_on_entropic`: If True (default), use Sinkhorn. If False, use LBFGS.
    - `sinkhorn_tol`: Tolerance for convergence of the Sinkhorn solver.
    - `sinkhorn_max_iter`: Maximum number of iterations for the Sinkhorn solver.
    - `lbfgs_tol`: Tolerance for convergence of the LBFGS solver.
    - `lbfgs_max_iter`: Maximum number of iterations for the LBFGS solver.
    - `implicit_diff`: If True (default), use implicit differentiation (ImplicitAdjoint) instead of recursive checkpointing for LBFGS backward pass. More numerically stable gradients, especially at low softness.
    - `return_log_probs`: If True, returns log probabilities instead of probabilities. Exact zero probabilities are represented as `-inf`.

    !!! note "Numerical precision"

        The internal solvers upcast to float64 when possible for numerical stability.
        This requires ``jax.config.update("jax_enable_x64", True)`` (or the ``JAX_ENABLE_X64=1`` environment variable).
        Without it, the upcast is silently ignored and the solver may produce non-finite gradients at larger problem sizes (typically n ≥ 2048).

    **Returns:**

    A positive Array of shape (..., [n], m), which sums to 1 over the second to last dimension, and approximately sums to 1 over the last dimension. Represents the transport plan between the marginals `mu` and `nu`.
    """

    _validate_softness(softness)
    *batch_sizes, n, m = cost.shape
    C = cost.reshape(-1, n, m)  # (B, n, m)

    if mode == "smooth":
        use_entropic_ot_sinkhorn = use_entropic_ot_sinkhorn_on_entropic

        if use_entropic_ot_sinkhorn:
            proj_fn = lambda c: _proj_transport_polytope_entropic_sinkhorn(
                c,
                mu=mu,
                nu=nu,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol,
                epsilon=softness,
                return_log_probs=return_log_probs,
            )
        else:
            proj_fn = lambda c: _proj_transport_polytope_entropic_lbfgs(
                c,
                mu=mu,
                nu=nu,
                epsilon=softness,
                tol=lbfgs_tol,
                max_steps=lbfgs_max_iter,
                implicit_diff=implicit_diff,
                return_log_probs=return_log_probs,
            )

    else:
        if mode == "c0":
            # Curvature of (1/2)||y||^2 at transport polytope center: R''=1
            p = 2
        elif mode == "c1":
            p = 3 / 2
        elif mode == "c2":
            p = 4 / 3
        else:
            raise ValueError(f"Invalid mode: {mode}")
        proj_fn = lambda c: _proj_transport_polytope_pnorm_lbfgs(
            c,
            mu=mu,
            nu=nu,
            lam=softness,
            tol=lbfgs_tol,
            max_steps=lbfgs_max_iter,
            p=p,
            implicit_diff=implicit_diff,
        )

    results = jax.vmap(proj_fn, in_axes=(0,))(C)  # (B, n, m)
    if return_log_probs:
        log_n = jnp.log(jnp.asarray(n, dtype=results.dtype))
        if mode == "smooth":
            y = results + log_n
        else:
            y = jnp.log(results) + log_n
    else:
        y = results * n
    y = y.reshape(*batch_sizes, n, m)  # (..., [n], m)
    return y
