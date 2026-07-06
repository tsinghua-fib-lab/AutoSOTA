from __future__ import annotations

from typing import Literal

import equinox as eqx
import equinox.internal as eqi
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array, lax
from jax.ops import segment_sum

from softjax.utils import _validate_softness


def _high_precision_dtype() -> jnp.dtype:
    """float64 when x64 is enabled, otherwise float32 (avoids UserWarning)."""
    return jnp.result_type(float)


def _inv_permutation(p: jax.Array) -> jax.Array:
    inv = jnp.empty_like(p)
    return inv.at[p].set(jnp.arange(p.shape[0], dtype=p.dtype))


# ── smooth: entropic LP-LBFGS ──────────────────────────────────────────


def _diff_T(beta: jax.Array) -> jax.Array:
    # D is (n-1)x n with (Dy)_i = y_i - y_{i+1}
    # D^T beta = [beta0, beta1-beta0, ..., beta_{n-2}-beta_{n-3}, -beta_{n-2}]
    left = jnp.concatenate([jnp.zeros((1,), dtype=beta.dtype), beta], axis=0)  # (n,)
    right = jnp.concatenate([beta, jnp.zeros((1,), dtype=beta.dtype)], axis=0)  # (n,)
    return right - left


def _alpha_nu_from_beta(z_s: jax.Array, beta: jax.Array) -> tuple[jax.Array, jax.Array]:
    # stationarity: z_s - A^T alpha + D^T beta - nu*1 = 0
    # => A^T alpha = z_s + D^T beta - nu*1, with (A^T alpha)_n = 0
    dtb = _diff_T(beta)  # (n,)
    nu = z_s[-1] + dtb[-1]  # scalar, enforces last component = 0
    u = z_s + dtb - nu  # (n,), u[-1] == 0
    alpha = u[:-1] - u[1:]  # (n-1,), since u_i - u_{i+1} = alpha_i
    return alpha, nu


def _reconstruct_y_from_slacks_and_gaps(
    b: jax.Array,  # (n-1,) prefix bounds
    b_n: jax.Array,  # scalar sum bound
    s: jax.Array,  # (n-1,) prefix slacks, s>0
    d: jax.Array,  # (n-1,) gaps, d>0
) -> jax.Array:
    # We want Ay = b - s  and  Dy = d  and  1^T y = b_n.
    #
    # Use d to parameterize y = y_n + tail_sums(d), then pick y_n to best match Ay=b-s
    # (and include the sum equation as k=n).
    n = d.shape[0] + 1
    dtype = d.dtype

    # tail sums t_i = sum_{j=i}^{n-1} d_j, with t_n = 0
    tail = jnp.flip(jnp.cumsum(jnp.flip(d, axis=0), axis=0), axis=0)  # (n-1,)
    t = jnp.concatenate([tail, jnp.zeros((1,), dtype=dtype)], axis=0)  # (n,)

    # prefix sums of t: T_k = sum_{i<=k} t_i, k=1..n-1
    T = jnp.cumsum(t, axis=0)[:-1]  # (n-1,)

    r = b - s  # target prefix sums of y, (n-1,)
    rhs = r - T  # (n-1,) rhs_k in k*y_n = rhs_k

    ks = jnp.arange(1, n, dtype=dtype)  # 1..n-1
    rhs_n = b_n - jnp.sum(t)  # n*y_n = rhs_n

    num = jnp.sum(ks * rhs) + (n * rhs_n)
    den = jnp.sum(ks * ks) + (n * n)
    y_n = num / den

    y = y_n + t
    return y


def _smooth_majorization_bounds(
    w: jax.Array, tau: jax.Array, checkpointed: bool = True
):
    """C∞ smooth majorization bounds via elementary symmetric polynomials.

    Computes  b_k = τ · log e_k(exp(w/τ))  for k = 1 … n-1,
    where e_k is the k-th elementary symmetric polynomial.

    This is the log-sum-exp over all size-k subsets:
        b_k = τ · log Σ_{|S|=k} exp(Σ_{i∈S} w_i / τ)
    and serves as a C∞ relaxation of the hard order-statistic partial sums  b_k = Σ_{j≤k} w_{(j)}.

    Uses the recurrence
        E[k][j] = E[k][j-1] + exp(w_j/τ) · E[k-1][j-1]
    in log-space (logaddexp) for numerical stability.  O(n²).

    When ``checkpointed=True`` (default), uses optimal online gradient checkpointing (Stumm & Walther 2010) to reduce memory from O(n²) to O(n√n) at the cost of ~2× forward compute during backward.
    """
    n = w.shape[0]
    x = w / tau  # (n,)

    # Use a large finite sentinel instead of -inf to avoid NaN gradients
    # from logaddexp(-inf, -inf) at impossible subset sizes (k > j+1).
    _FLOOR = jnp.array(-1e30, dtype=w.dtype)
    # log_E[k] = log(E[k][j]) after processing j elements; k = 0..n-1
    log_E = jnp.full((n,), _FLOOR, dtype=w.dtype)
    log_E = log_E.at[0].set(0.0)  # e_0 = 1

    def _scan_step(log_E, x_j):
        # log E[k][j] = logaddexp(log E[k][j-1],  x_j + log E[k-1][j-1])
        log_E_prev = jnp.concatenate(
            [jnp.array([_FLOOR], dtype=log_E.dtype), log_E[:-1]]
        )
        log_E = jnp.logaddexp(log_E, x_j + log_E_prev)
        return log_E, None

    if checkpointed:
        log_E, _ = eqi.scan(_scan_step, log_E, x, kind="checkpointed")
    else:
        log_E, _ = lax.scan(_scan_step, log_E, x)

    # b_k = τ · log(e_k) for k = 1..n-1
    b = tau * log_E[1:]  # (n-1,)
    b_n = jnp.sum(w)  # exact: τ · log(e_n) = Σ w_i
    return b, b_n


def _tridiag_L_matvec(r):
    """Compute L·r where L is the tridiagonal second-difference matrix.

    L[i,i] = 2, L[i,i-1] = L[i,i+1] = -1.
    Result: (L·r)_i = 2·r[i] - r[i-1] - r[i+1], with boundary handling.
    """
    result = 2.0 * r
    result = result.at[:-1].add(-r[1:])
    result = result.at[1:].add(-r[:-1])
    return result


def _proj_permutahedron_entropic_lp_lbfgs(
    z: jax.Array,  # (n,)
    w: jax.Array,  # (n,)
    softness: float = 1.0,  # \tau for majorization slacks
    softness_mono: float | None = None,  # \tau_m for monotonicity gaps; None => \tau
    max_iter: int = 200,
    tol: float = 1e-5,
    history_length: int = 5,
    l2_beta: float = 0.0,  # optional stabilizer on beta
    throw: bool = True,
) -> jax.Array:
    """Linear projection onto the permutahedron with entropic regularization on:
      - majorization slacks s = b - A y  (strictly positive)
      - monotonicity gaps d = D y        (strictly positive)

    Solves the eliminated dual over beta (unconstrained) via optimistix.LBFGS.
    """
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    if z.ndim != 1 or w.ndim != 1 or z.shape != w.shape:
        raise ValueError(f"z,w must be 1D and same shape; got {z.shape}, {w.shape}")
    n = z.shape[0]
    if n <= 1:
        return w

    # Upcast to highest available float for solver precision.
    orig_dtype = z.dtype
    _hp = _high_precision_dtype()
    if z.dtype != _hp:
        z = z.astype(_hp)
        w = w.astype(_hp)

    tau = jnp.asarray(softness, dtype=z.dtype)
    tau_m = tau if softness_mono is None else jnp.asarray(softness_mono, dtype=z.dtype)

    # choose chamber by sorting z
    pz = jnp.argsort(-z, stable=True)
    iz = _inv_permutation(pz)
    z_s = z[pz]

    # permutahedron bounds from sorted w
    pw = jnp.argsort(-w, stable=True)
    w_s = w[pw]
    b_full = jnp.cumsum(w_s)
    b = b_full[:-1]  # (n-1,)
    b_n = b_full[-1]  # scalar

    # dual after elimination: minimize
    #   g(beta) = <alpha(beta), b> + nu(beta)*b_n
    #           + tau * sum exp(-alpha/tau) + tau_m * sum exp(-beta/tau_m)
    # where alpha,nu satisfy stationarity.
    def dual_obj(beta: jax.Array, args) -> jax.Array:
        z_s, b, b_n, tau, tau_m = args
        alpha, nu = _alpha_nu_from_beta(z_s, beta)

        val = jnp.dot(alpha, b) + nu * b_n
        val = val + tau * jnp.sum(jnp.exp(-alpha / tau))
        val = val + tau_m * jnp.sum(jnp.exp(-beta / tau_m))
        if l2_beta:
            val = val + 0.5 * jnp.asarray(l2_beta, z.dtype) * jnp.sum(beta * beta)
        return val

    beta0 = jnp.zeros((n - 1,), dtype=z.dtype)
    solver = optx.LBFGS(rtol=tol, atol=tol, history_length=history_length)

    sol = optx.minimise(
        dual_obj,
        solver,
        beta0,
        args=(z_s, b, b_n, tau, tau_m),
        max_steps=max_iter,
        throw=throw,
    )
    beta_star = sol.value

    alpha_star, _ = _alpha_nu_from_beta(z_s, beta_star)

    # primal variables from KKT
    s = jnp.exp(-alpha_star / tau)  # (n-1,)
    d = jnp.exp(-beta_star / tau_m)  # (n-1,)

    y_s = _reconstruct_y_from_slacks_and_gaps(b, b_n, s, d)
    result = y_s[iz]
    if result.dtype != orig_dtype:
        result = result.astype(orig_dtype)
    return result


def _hessian_beta_matvec(v, w_alpha, w_beta):
    """Matvec with L·diag(w_α)·L·v + diag(w_β)·v.  L is symmetric."""
    Lv = _tridiag_L_matvec(v)
    return _tridiag_L_matvec(w_alpha * Lv) + w_beta * v


def _make_proj_permutahedron_entropic_lp(
    tol=1e-5,
    max_iter=200,
    softness=1.0,
    softness_mono=None,
    bounds_softness=1.0,
    checkpointed=True,
):
    """Return a ``(z, w) -> result`` function with C∞ smooth gradients.

    Uses smooth majorization bounds (elementary symmetric polynomials) for C∞ gradients w.r.t. ``w``.
    The ``custom_vjp`` wraps only the LBFGS solver ``(z, b, b_n) → y``, providing analytical gradients for ``z`` and pass-through gradients for ``b`` / ``b_n``.
    Gradients from ``b`` / ``b_n`` to ``w`` flow via standard JAX autodiff through :func:`_smooth_majorization_bounds`.
    """

    # ── inner solver with custom_vjp on (z, b, b_n) ──

    @jax.custom_vjp
    def _solver(z, b, b_n):
        orig_dtype = z.dtype
        _hp = _high_precision_dtype()
        if z.dtype != _hp:
            z = z.astype(_hp)
            b = b.astype(_hp)
            b_n = b_n.astype(_hp)

        tau = jnp.asarray(softness, dtype=z.dtype)
        tau_m = (
            tau if softness_mono is None else jnp.asarray(softness_mono, dtype=z.dtype)
        )

        perm_z = jnp.argsort(-z, stable=True)
        inv_perm_z = _inv_permutation(perm_z)
        z_s = z[perm_z]

        def dual_obj(beta, args):
            z_s_, b_, b_n_, tau_, tau_m_ = args
            alpha, nu = _alpha_nu_from_beta(z_s_, beta)
            val = jnp.dot(alpha, b_) + nu * b_n_
            val = val + tau_ * jnp.sum(jnp.exp(-alpha / tau_))
            val = val + tau_m_ * jnp.sum(jnp.exp(-beta / tau_m_))
            return val

        n = z.shape[0]
        beta0 = jnp.zeros((n - 1,), dtype=z.dtype)
        solver_lbfgs = optx.LBFGS(rtol=tol, atol=tol, history_length=5)
        sol = optx.minimise(
            dual_obj,
            solver_lbfgs,
            beta0,
            args=(z_s, b, b_n, tau, tau_m),
            max_steps=max_iter,
            throw=False,
        )
        beta_star = sol.value
        alpha_star, _ = _alpha_nu_from_beta(z_s, beta_star)

        s = jnp.exp(-alpha_star / tau)
        d = jnp.exp(-beta_star / tau_m)

        y_s = _reconstruct_y_from_slacks_and_gaps(b, b_n, s, d)
        result = y_s[inv_perm_z]
        if result.dtype != orig_dtype:
            result = result.astype(orig_dtype)
        return result

    def _solver_fwd(z, b, b_n):
        z = jnp.asarray(z)
        b = jnp.asarray(b)
        b_n = jnp.asarray(b_n)
        n = z.shape[0]

        orig_dtype = z.dtype
        _hp = _high_precision_dtype()
        if z.dtype != _hp:
            z = z.astype(_hp)
            b = b.astype(_hp)
            b_n = b_n.astype(_hp)

        tau = jnp.asarray(softness, dtype=z.dtype)
        tau_m = (
            tau if softness_mono is None else jnp.asarray(softness_mono, dtype=z.dtype)
        )

        perm_z = jnp.argsort(-z, stable=True)
        inv_perm_z = _inv_permutation(perm_z)
        z_s = z[perm_z]

        if n <= 1:
            result = jnp.full_like(z, b_n)
            if orig_dtype != _hp:
                result = result.astype(orig_dtype)
            s = jnp.empty((0,), dtype=z.dtype)
            d = jnp.empty((0,), dtype=z.dtype)
            return result, (perm_z, inv_perm_z, s, d, tau, tau_m)

        def dual_obj(beta, args):
            z_s_, b_, b_n_, tau_, tau_m_ = args
            alpha, nu = _alpha_nu_from_beta(z_s_, beta)
            val = jnp.dot(alpha, b_) + nu * b_n_
            val = val + tau_ * jnp.sum(jnp.exp(-alpha / tau_))
            val = val + tau_m_ * jnp.sum(jnp.exp(-beta / tau_m_))
            return val

        beta0 = jnp.zeros((n - 1,), dtype=z.dtype)
        solver_lbfgs = optx.LBFGS(rtol=tol, atol=tol, history_length=5)
        sol = optx.minimise(
            dual_obj,
            solver_lbfgs,
            beta0,
            args=(z_s, b, b_n, tau, tau_m),
            max_steps=max_iter,
            throw=False,
        )
        beta_star = sol.value
        alpha_star, _ = _alpha_nu_from_beta(z_s, beta_star)

        s = jnp.exp(-alpha_star / tau)
        d = jnp.exp(-beta_star / tau_m)

        y_s = _reconstruct_y_from_slacks_and_gaps(b, b_n, s, d)
        result = y_s[inv_perm_z]
        if orig_dtype != jnp.float64:
            result = result.astype(orig_dtype)

        return result, (perm_z, inv_perm_z, s, d, tau, tau_m)

    def _solver_bwd(residuals, g):
        perm_z, inv_perm_z, s, d, tau, tau_m = residuals
        orig_dtype = g.dtype
        g = jnp.asarray(g)
        _hp = _high_precision_dtype()
        if g.dtype != _hp:
            g = g.astype(_hp)

        g_s = g[perm_z]

        inv_s = 1.0 / s
        inv_d = 1.0 / d

        # Solve H_y·λ + μ·1 = g_s with 1^T·λ = 0.
        # Substitute λ = D^T·η (zero-mean since 1^T·D^T = 0) to get
        # M·η = D·g_s where M = diag(1/w_α) + L·diag(1/w_β)·L.
        w_alpha = s / tau
        w_beta = d / tau_m
        rhs = g_s[:-1] - g_s[1:]  # D·g_s

        def m_matvec(v):
            return _hessian_beta_matvec(v, 1.0 / w_beta, 1.0 / w_alpha)

        eta, _ = jax.scipy.sparse.linalg.cg(m_matvec, rhs, tol=tol, maxiter=50)
        lam = jnp.concatenate([eta[:1], eta[1:] - eta[:-1], -eta[-1:]])  # D^T·η

        def _reverse_cumsum(u):
            return jnp.flip(jnp.cumsum(jnp.flip(u)))

        def _at_matvec(u):
            """A^T u: (n-1,) -> (n,), where A is the prefix-sum matrix."""
            return jnp.concatenate([_reverse_cumsum(u), jnp.zeros((1,), dtype=u.dtype)])

        def h_y_matvec(v):
            """H_y v = tau * A^T diag(1/s) A v + tau_m * D^T diag(1/d) D v"""
            Av = jnp.cumsum(v)[:-1]
            Dv = v[:-1] - v[1:]
            return tau * _at_matvec(Av * inv_s) + tau_m * _diff_T(Dv * inv_d)

        h_lam = h_y_matvec(lam)
        mu = jnp.mean(g_s - h_lam)

        # Gradients for (z, b, b_n) — no chain-rule to w here;
        # autodiff through _smooth_majorization_bounds handles that.
        grad_z_s = lam
        grad_b = tau * jnp.cumsum(lam)[:-1] * inv_s
        grad_b_n = mu

        grad_z = grad_z_s[inv_perm_z]
        if orig_dtype != jnp.float64:
            grad_z = grad_z.astype(orig_dtype)
            grad_b = grad_b.astype(orig_dtype)
            grad_b_n = grad_b_n.astype(orig_dtype)

        return (grad_z, grad_b, grad_b_n)

    _solver.defvjp(_solver_fwd, _solver_bwd)

    # ── outer function: smooth bounds + solver ──

    def _proj_fn(z, w):
        n = z.shape[0]
        if n <= 1:
            return w
        tau_bounds = jnp.asarray(bounds_softness, dtype=w.dtype)
        b, b_n = _smooth_majorization_bounds(w, tau_bounds, checkpointed)
        return _solver(z, b, b_n)

    return _proj_fn


# ── c0: euclidean (q=2) ──────────────────────────────────────────────


def _pav_isotonic_decreasing_pnorm_q2(y: jax.Array):
    y = jnp.asarray(y)
    n = y.shape[0]
    dtype = y.dtype

    starts0 = jnp.full((n,), n, dtype=jnp.int32)
    sums0 = jnp.zeros((n,), dtype=dtype)
    lens0 = jnp.ones((n,), dtype=jnp.int32)
    m0 = jnp.int32(0)

    def merge_cond(state):
        starts, sums, lens, m = state
        return (m >= 2) & (
            (sums[m - 2] / lens[m - 2].astype(dtype))
            < (sums[m - 1] / lens[m - 1].astype(dtype))
        )

    def merge_body(state):
        starts, sums, lens, m = state
        i_prev = m - 2
        i_top = m - 1

        sums = sums.at[i_prev].set(sums[i_prev] + sums[i_top])
        lens = lens.at[i_prev].set(lens[i_prev] + lens[i_top])

        sums = sums.at[i_top].set(jnp.array(0, dtype=dtype))
        lens = lens.at[i_top].set(jnp.int32(1))
        starts = starts.at[i_top].set(jnp.int32(n))

        return (starts, sums, lens, m - 1)

    def for_body(i, state):
        starts, sums, lens, m = state
        starts = starts.at[m].set(jnp.int32(i))
        sums = sums.at[m].set(y[i])
        lens = lens.at[m].set(jnp.int32(1))
        m = m + 1
        return lax.while_loop(merge_cond, merge_body, (starts, sums, lens, m))

    starts, sums, lens, m = lax.fori_loop(0, n, for_body, (starts0, sums0, lens0, m0))

    idx = jnp.arange(n, dtype=jnp.int32)
    starts = jnp.where(idx < m, starts, jnp.int32(n))
    sums = jnp.where(idx < m, sums, jnp.array(0, dtype=dtype))
    lens = jnp.where(idx < m, lens, jnp.int32(1))

    avgs = sums / lens.astype(dtype)
    block_idx = jnp.searchsorted(starts, idx, side="right") - jnp.int32(1)
    v = avgs[block_idx]
    return v, block_idx, starts, lens


@jax.custom_vjp
def _proj_permutahedron_pnorm_q2(z: jax.Array, w: jax.Array) -> jax.Array:
    z = jnp.asarray(z)
    w = jnp.asarray(w)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]

    y = z_sorted - w_sorted
    v, _, _, _ = _pav_isotonic_decreasing_pnorm_q2(y)

    p_sorted = z_sorted - v
    return p_sorted[inv_perm_z]


def _proj_permutahedron_pnorm_q2_fwd(z: jax.Array, w: jax.Array):
    z = jnp.asarray(z)
    w = jnp.asarray(w)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]
    inv_perm_w = _inv_permutation(perm_w)

    y = z_sorted - w_sorted
    v, block_idx, _, lens = _pav_isotonic_decreasing_pnorm_q2(y)

    p_sorted = z_sorted - v
    p = p_sorted[inv_perm_z]

    aux = (perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens)
    return p, aux


def _proj_permutahedron_pnorm_q2_bwd(aux, g: jax.Array):
    perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens = aux
    g = jnp.asarray(g)
    dtype = g.dtype
    n = g.shape[0]

    g_sorted = g[perm_z]
    block_sum_g = segment_sum(g_sorted, block_idx, num_segments=n)
    Jt_g = block_sum_g[block_idx] / lens[block_idx].astype(
        dtype
    )  # symmetric => same as J g

    grad_z_sorted = g_sorted - Jt_g
    grad_w_sorted = Jt_g

    return (grad_z_sorted[inv_perm_z], grad_w_sorted[inv_perm_w])


_proj_permutahedron_pnorm_q2.defvjp(
    _proj_permutahedron_pnorm_q2_fwd, _proj_permutahedron_pnorm_q2_bwd
)


# ── c1: p-norm q=3 (p=3/2) ───────────────────────────────────────────


def _solve_block_gamma_q3(
    s_sorted, prefix_s, prefix_s2, start, length, sum_w, min_s, max_s
):
    r"""Analytical solver for \sum_block (\gamma - s)|\gamma - s| + sum_w = 0.

    The function g(\gamma) is piecewise quadratic with breakpoints at each s_i.
    For each possible split point k (separating elements >= \gamma from those < \gamma), g is a quadratic a*\gamma^2 + b*\gamma + c whose coefficients are computed from prefix sums.
    We solve all quadratics in parallel and select the root that falls in its valid interval.
    """
    dtype = s_sorted.dtype
    n = s_sorted.shape[0]
    end = start + length

    # Possible split points k in {0, 1, ..., n}.
    # Split k: elements [start, k) have s >= gamma, [k, end) have s < gamma.
    ks = jnp.arange(n + 1)

    # Quadratic coefficients g_k(gamma) = a*gamma^2 + b*gamma + c = 0
    n_hi = (ks - start).astype(dtype)
    n_lo = (end - ks).astype(dtype)
    S_hi = prefix_s[ks] - prefix_s[start]
    S_lo = prefix_s[end] - prefix_s[ks]
    M2_hi = prefix_s2[ks] - prefix_s2[start]
    M2_lo = prefix_s2[end] - prefix_s2[ks]

    a = n_lo - n_hi
    b = 2.0 * (S_hi - S_lo)
    c = (M2_lo - M2_hi) + sum_w

    # The ascending zero-crossing is always at (-b + sqrt(disc)) / (2a).
    disc = b * b - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    a_safe = jnp.where(a != 0.0, a, 1.0)
    b_safe = jnp.where(b != 0.0, b, 1.0)
    gamma_quad = (-b + sqrt_disc) / (2.0 * a_safe)
    gamma_lin = -c / b_safe
    gamma_k = jnp.where(a != 0.0, gamma_quad, gamma_lin)

    # Validity interval (s sorted descending):
    #   k = start:        gamma > s[start],   upper = +inf
    #   start < k < end:  s[k] < gamma <= s[k-1]
    #   k = end:          lower = -inf,       gamma <= s[end-1]
    s_at_k = s_sorted[jnp.clip(ks, 0, n - 1)]
    s_at_km1 = s_sorted[jnp.clip(ks - 1, 0, n - 1)]
    lo_bound = jnp.where(ks < end, s_at_k, -jnp.inf)
    hi_bound = jnp.where(ks > start, s_at_km1, jnp.inf)

    eps = jnp.finfo(dtype).eps ** 0.75
    valid = (
        (ks >= start)
        & (ks <= end)
        & (gamma_k > lo_bound - eps)
        & (gamma_k <= hi_bound + eps)
        & (disc >= -eps)
        & ((a != 0.0) | (b != 0.0))
    )

    best = jnp.argmax(valid)
    result = gamma_k[best]
    return jnp.where(jnp.any(valid), result, (min_s + max_s) * 0.5)


def _pav_isotonic_decreasing_pnorm_q3(s: jax.Array, w: jax.Array):
    s = jnp.asarray(s)
    w = jnp.asarray(w)
    n = s.shape[0]
    dtype = s.dtype

    _zero = jnp.zeros((1,), dtype=dtype)
    prefix_s = jnp.concatenate([_zero, jnp.cumsum(s)])
    prefix_s2 = jnp.concatenate([_zero, jnp.cumsum(s * s)])

    starts0 = jnp.full((n,), n, dtype=jnp.int32)
    lens0 = jnp.ones((n,), dtype=jnp.int32)
    sumw0 = jnp.zeros((n,), dtype=dtype)
    mins0 = jnp.full((n,), jnp.inf, dtype=dtype)
    maxs0 = jnp.full((n,), -jnp.inf, dtype=dtype)
    gam0 = jnp.zeros((n,), dtype=dtype)
    m0 = jnp.int32(0)

    def merge_cond(state):
        starts, lens, sumw, mins, maxs, gam, m = state
        return (m >= 2) & (gam[m - 2] < gam[m - 1])

    def merge_body(state):
        starts, lens, sumw, mins, maxs, gam, m = state
        i_prev = m - 2
        i_top = m - 1

        lens_new = lens[i_prev] + lens[i_top]
        sumw_new = sumw[i_prev] + sumw[i_top]
        mins_new = jnp.minimum(mins[i_prev], mins[i_top])
        maxs_new = jnp.maximum(maxs[i_prev], maxs[i_top])

        gam_new = _solve_block_gamma_q3(
            s,
            prefix_s,
            prefix_s2,
            starts[i_prev],
            lens_new,
            sumw_new,
            mins_new,
            maxs_new,
        )

        lens = lens.at[i_prev].set(lens_new)
        sumw = sumw.at[i_prev].set(sumw_new)
        mins = mins.at[i_prev].set(mins_new)
        maxs = maxs.at[i_prev].set(maxs_new)
        gam = gam.at[i_prev].set(gam_new)

        starts = starts.at[i_top].set(jnp.int32(n))
        lens = lens.at[i_top].set(jnp.int32(1))
        sumw = sumw.at[i_top].set(jnp.array(0, dtype=dtype))
        mins = mins.at[i_top].set(jnp.array(jnp.inf, dtype=dtype))
        maxs = maxs.at[i_top].set(jnp.array(-jnp.inf, dtype=dtype))
        gam = gam.at[i_top].set(jnp.array(0, dtype=dtype))

        return (starts, lens, sumw, mins, maxs, gam, m - 1)

    def for_body(i, state):
        starts, lens, sumw, mins, maxs, gam, m = state

        si = s[i]
        wi = w[i]
        starts = starts.at[m].set(jnp.int32(i))
        lens = lens.at[m].set(jnp.int32(1))
        sumw = sumw.at[m].set(wi)
        mins = mins.at[m].set(si)
        maxs = maxs.at[m].set(si)

        gam_i = _solve_block_gamma_q3(
            s, prefix_s, prefix_s2, i, jnp.int32(1), wi, si, si
        )
        gam = gam.at[m].set(gam_i)

        m = m + 1
        return lax.while_loop(
            merge_cond, merge_body, (starts, lens, sumw, mins, maxs, gam, m)
        )

    starts, lens, sumw, mins, maxs, gam, m = lax.fori_loop(
        0, n, for_body, (starts0, lens0, sumw0, mins0, maxs0, gam0, m0)
    )

    idx = jnp.arange(n, dtype=jnp.int32)
    starts = jnp.where(idx < m, starts, jnp.int32(n))
    lens = jnp.where(idx < m, lens, jnp.int32(1))
    gam = jnp.where(idx < m, gam, jnp.array(0, dtype=dtype))

    block_idx = jnp.searchsorted(starts, idx, side="right") - jnp.int32(1)
    v = gam[block_idx]
    return v, block_idx, lens


@jax.custom_vjp
def _proj_permutahedron_pnorm_q3(z: jax.Array, w: jax.Array) -> jax.Array:
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    orig_dtype = z.dtype
    _hp = _high_precision_dtype()
    z, w = z.astype(_hp), w.astype(_hp)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]

    v, block_idx, lens = _pav_isotonic_decreasing_pnorm_q3(z_sorted, w_sorted)

    t = z_sorted - v
    y_sorted = t * jnp.abs(t)  # q=3 => \nabla R^*(t) = t|t|^{q-2} = t|t|

    return y_sorted[inv_perm_z].astype(orig_dtype)


def _proj_permutahedron_pnorm_q3_fwd(z: jax.Array, w: jax.Array):
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    orig_dtype = z.dtype
    _hp = _high_precision_dtype()
    z, w = z.astype(_hp), w.astype(_hp)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]
    inv_perm_w = _inv_permutation(perm_w)

    v, block_idx, lens = _pav_isotonic_decreasing_pnorm_q3(z_sorted, w_sorted)

    t = z_sorted - v
    y_sorted = t * jnp.abs(t)
    y = y_sorted[inv_perm_z].astype(orig_dtype)

    aux = (perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens, t, y)
    return y, aux


def _proj_permutahedron_pnorm_q3_bwd(aux, g: jax.Array):
    perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens, t, y = aux
    orig_dtype = g.dtype
    _hp = _high_precision_dtype()
    g = jnp.asarray(g).astype(_hp)
    n = g.shape[0]
    dtype = _hp

    # upstream grad on y_sorted
    g_sorted_y = g[perm_z]

    # y = t|t|  => dy/dt = 2|t|
    g_t = g_sorted_y * (2.0 * jnp.abs(t))

    # weights for (\partial v / \partial s)^T g_t : \alpha_i \propto |t_i|^{q-2} = |t_i|
    weight = jnp.abs(t)
    denom_block = segment_sum(weight, block_idx, num_segments=n)
    denom = denom_block[block_idx]

    sumg_block = segment_sum(g_t, block_idx, num_segments=n)
    sumg = sumg_block[block_idx]

    lens_elem = lens[block_idx].astype(dtype)
    alpha = jnp.where(
        denom > 0, weight / denom, jnp.array(1.0, dtype=dtype) / lens_elem
    )

    jtg_s = alpha * sumg
    grad_z_sorted = g_t - jtg_s

    # d\gamma/dw_k = -1/(2 \sum |t|)  => dt/dw_k = +1/(2 \sum |t|)
    grad_w_sorted = jnp.where(
        denom > 0, sumg / (2.0 * denom), jnp.array(0.0, dtype=dtype)
    )

    return (
        grad_z_sorted[inv_perm_z].astype(orig_dtype),
        grad_w_sorted[inv_perm_w].astype(orig_dtype),
    )


_proj_permutahedron_pnorm_q3.defvjp(
    _proj_permutahedron_pnorm_q3_fwd, _proj_permutahedron_pnorm_q3_bwd
)


# ── c2: p-norm q=4 (p=4/3) ───────────────────────────────────────────


def _solve_block_gamma_q4(len_b, sum_w, m1, m2, m3, min_s, max_s):
    r"""Closed-form solver for the q=4 block gamma.

    Solves \sum_block (\gamma - s)^3 + sum_w = 0 via Cardano's hyperbolic method.
    In shifted form u = \gamma - c, this is the depressed cubic u^3 + pu + q = 0
    with p = 3\mu_2/n \geq 0, which always has exactly one real root.
    """
    dtype = m1.dtype
    len_f = len_b.astype(dtype)
    c = m1 / len_f  # mean of s values in block
    # Central moments: \mu_k = \sum (s_i - c)^k
    mu2 = m2 - 2.0 * c * m1 + len_f * c * c
    mu3 = m3 - 3.0 * c * m2 + 3.0 * c * c * m1 - len_f * c**3

    # Depressed cubic: u^3 + p*u + q = 0
    p = 3.0 * mu2 / len_f  # >= 0 (sum of squares)
    q = (sum_w - mu3) / len_f

    # Hyperbolic Cardano: u = -sign(q) * 2*sqrt(p/3) * sinh(arcsinh(A)/3)
    # where A = 3|q| / (2*p*sqrt(p/3))
    sp3 = jnp.sqrt(jnp.maximum(p / 3.0, 0.0))  # sqrt(p/3)
    denom = 2.0 * jnp.maximum(p, jnp.finfo(dtype).tiny) * sp3
    A = 3.0 * jnp.abs(q) / denom
    u_hyp = -jnp.sign(q) * 2.0 * sp3 * jnp.sinh(jnp.arcsinh(A) / 3.0)

    # When p ~ 0 (all s equal): u^3 + q = 0 => u = cbrt(-q)
    u_cbrt = -jnp.sign(q) * jnp.abs(q) ** (1.0 / 3.0)

    u = jnp.where(
        p > jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(q), 1.0), u_hyp, u_cbrt
    )
    return u + c


def _pav_isotonic_decreasing_pnorm_q4(s: jax.Array, w: jax.Array):
    s = jnp.asarray(s)
    w = jnp.asarray(w)
    n = s.shape[0]
    dtype = s.dtype

    starts0 = jnp.full((n,), n, dtype=jnp.int32)
    lens0 = jnp.ones((n,), dtype=jnp.int32)
    sumw0 = jnp.zeros((n,), dtype=dtype)
    m10 = jnp.zeros((n,), dtype=dtype)
    m20 = jnp.zeros((n,), dtype=dtype)
    m30 = jnp.zeros((n,), dtype=dtype)
    mins0 = jnp.full((n,), jnp.inf, dtype=dtype)
    maxs0 = jnp.full((n,), -jnp.inf, dtype=dtype)
    gam0 = jnp.zeros((n,), dtype=dtype)
    m0 = jnp.int32(0)

    def merge_cond(state):
        starts, lens, sumw, m1, m2, m3, mins, maxs, gam, m = state
        return (m >= 2) & (gam[m - 2] < gam[m - 1])

    def merge_body(state):
        starts, lens, sumw, m1, m2, m3, mins, maxs, gam, m = state
        i_prev = m - 2
        i_top = m - 1

        lens_new = lens[i_prev] + lens[i_top]
        sumw_new = sumw[i_prev] + sumw[i_top]
        m1_new = m1[i_prev] + m1[i_top]
        m2_new = m2[i_prev] + m2[i_top]
        m3_new = m3[i_prev] + m3[i_top]
        mins_new = jnp.minimum(mins[i_prev], mins[i_top])
        maxs_new = jnp.maximum(maxs[i_prev], maxs[i_top])

        gam_new = _solve_block_gamma_q4(
            lens_new, sumw_new, m1_new, m2_new, m3_new, mins_new, maxs_new
        )

        lens = lens.at[i_prev].set(lens_new)
        sumw = sumw.at[i_prev].set(sumw_new)
        m1 = m1.at[i_prev].set(m1_new)
        m2 = m2.at[i_prev].set(m2_new)
        m3 = m3.at[i_prev].set(m3_new)
        mins = mins.at[i_prev].set(mins_new)
        maxs = maxs.at[i_prev].set(maxs_new)
        gam = gam.at[i_prev].set(gam_new)

        starts = starts.at[i_top].set(jnp.int32(n))
        lens = lens.at[i_top].set(jnp.int32(1))
        sumw = sumw.at[i_top].set(jnp.array(0, dtype=dtype))
        m1 = m1.at[i_top].set(jnp.array(0, dtype=dtype))
        m2 = m2.at[i_top].set(jnp.array(0, dtype=dtype))
        m3 = m3.at[i_top].set(jnp.array(0, dtype=dtype))
        mins = mins.at[i_top].set(jnp.array(jnp.inf, dtype=dtype))
        maxs = maxs.at[i_top].set(jnp.array(-jnp.inf, dtype=dtype))
        gam = gam.at[i_top].set(jnp.array(0, dtype=dtype))

        return (starts, lens, sumw, m1, m2, m3, mins, maxs, gam, m - 1)

    def for_body(i, state):
        starts, lens, sumw, m1, m2, m3, mins, maxs, gam, m = state

        si = s[i]
        wi = w[i]
        starts = starts.at[m].set(jnp.int32(i))
        lens = lens.at[m].set(jnp.int32(1))
        sumw = sumw.at[m].set(wi)
        m1 = m1.at[m].set(si)
        m2 = m2.at[m].set(si * si)
        m3 = m3.at[m].set(si * si * si)
        mins = mins.at[m].set(si)
        maxs = maxs.at[m].set(si)

        gam_i = _solve_block_gamma_q4(
            jnp.int32(1), wi, si, si * si, si * si * si, si, si
        )
        gam = gam.at[m].set(gam_i)

        m = m + 1
        return lax.while_loop(
            merge_cond, merge_body, (starts, lens, sumw, m1, m2, m3, mins, maxs, gam, m)
        )

    starts, lens, sumw, m1, m2, m3, mins, maxs, gam, m = lax.fori_loop(
        0, n, for_body, (starts0, lens0, sumw0, m10, m20, m30, mins0, maxs0, gam0, m0)
    )

    idx = jnp.arange(n, dtype=jnp.int32)
    starts = jnp.where(idx < m, starts, jnp.int32(n))
    lens = jnp.where(idx < m, lens, jnp.int32(1))
    gam = jnp.where(idx < m, gam, jnp.array(0, dtype=dtype))

    block_idx = jnp.searchsorted(starts, idx, side="right") - jnp.int32(1)
    v = gam[block_idx]
    return v, block_idx, lens


@jax.custom_vjp
def _proj_permutahedron_pnorm_q4(z: jax.Array, w: jax.Array) -> jax.Array:
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    orig_dtype = z.dtype
    _hp = _high_precision_dtype()
    z, w = z.astype(_hp), w.astype(_hp)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]
    v, block_idx, lens = _pav_isotonic_decreasing_pnorm_q4(z_sorted, w_sorted)

    t = z_sorted - v  # (n,)
    y_sorted = t * (jnp.abs(t) ** 2)  # q=4 => \nabla R^*(t) = t|t|^{q-2} = t^3

    return y_sorted[inv_perm_z].astype(orig_dtype)


def _proj_permutahedron_pnorm_q4_fwd(z: jax.Array, w: jax.Array):
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    orig_dtype = z.dtype
    _hp = _high_precision_dtype()
    z, w = z.astype(_hp), w.astype(_hp)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]
    inv_perm_w = _inv_permutation(perm_w)

    v, block_idx, lens = _pav_isotonic_decreasing_pnorm_q4(z_sorted, w_sorted)

    t = z_sorted - v
    y_sorted = t * (jnp.abs(t) ** 2)
    y = y_sorted[inv_perm_z].astype(orig_dtype)

    aux = (perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens, t, y)
    return y, aux


def _proj_permutahedron_pnorm_q4_bwd(aux, g: jax.Array):
    perm_z, inv_perm_z, perm_w, inv_perm_w, block_idx, lens, t, y = aux
    orig_dtype = g.dtype
    _hp = _high_precision_dtype()
    g = jnp.asarray(g).astype(_hp)
    n = g.shape[0]
    dtype = _hp

    # upstream grad on y_sorted
    g_sorted_y = g[perm_z]

    # y = t^3  => dy/dt = 3 t^2
    g_t = g_sorted_y * (3.0 * (t * t))

    # weights for (\partial v / \partial s)^T g_t : \alpha_i \propto |t_i|^{q-2} = |t_i|^2
    weight = jnp.abs(t) ** 2  # (n,)
    denom_block = segment_sum(weight, block_idx, num_segments=n)
    denom = denom_block[block_idx]

    sumg_block = segment_sum(g_t, block_idx, num_segments=n)
    sumg = sumg_block[block_idx]

    lens_elem = lens[block_idx].astype(dtype)
    alpha = jnp.where(
        denom > 0, weight / denom, jnp.array(1.0, dtype=dtype) / lens_elem
    )

    jtg_s = alpha * sumg
    grad_z_sorted = g_t - jtg_s

    # d\gamma/dw_k = -1/(3 \sum |t|^2)  => dt/dw_k = +1/(3 \sum |t|^2)
    grad_w_sorted = jnp.where(
        denom > 0, sumg / (3.0 * denom), jnp.array(0.0, dtype=dtype)
    )

    return (
        grad_z_sorted[inv_perm_z].astype(orig_dtype),
        grad_w_sorted[inv_perm_w].astype(orig_dtype),
    )


_proj_permutahedron_pnorm_q4.defvjp(
    _proj_permutahedron_pnorm_q4_fwd, _proj_permutahedron_pnorm_q4_bwd
)


# ── entropic isotonic regression (smooth mode) ────────────────────────


def _pav_isotonic_decreasing_entropic(s: jax.Array, w: jax.Array):
    """
    Solve v_E(s,w) = argmin_{v1>=...>=vn} <exp(s - v), 1> + <exp(w), v>.
    Returns v and block structure.
    """
    s = jnp.asarray(s)
    w = jnp.asarray(w)
    n = s.shape[0]
    dtype = s.dtype
    neg_inf = jnp.array(-jnp.inf, dtype=dtype)

    starts0 = jnp.full((n,), n, dtype=jnp.int32)
    logS0 = jnp.full((n,), neg_inf, dtype=dtype)  # logsumexp over s in block
    logW0 = jnp.full((n,), neg_inf, dtype=dtype)  # logsumexp over w in block
    m0 = jnp.int32(0)

    def gamma(logS, logW):
        return logS - logW

    def merge_cond(state):
        starts, logS, logW, m = state
        return (m >= 2) & (
            gamma(logS[m - 2], logW[m - 2]) < gamma(logS[m - 1], logW[m - 1])
        )

    def merge_body(state):
        starts, logS, logW, m = state
        i_prev = m - 2
        i_top = m - 1

        logS = logS.at[i_prev].set(jnp.logaddexp(logS[i_prev], logS[i_top]))
        logW = logW.at[i_prev].set(jnp.logaddexp(logW[i_prev], logW[i_top]))

        logS = logS.at[i_top].set(neg_inf)
        logW = logW.at[i_top].set(neg_inf)
        starts = starts.at[i_top].set(jnp.int32(n))

        return (starts, logS, logW, m - 1)

    def for_body(i, state):
        starts, logS, logW, m = state
        starts = starts.at[m].set(jnp.int32(i))
        logS = logS.at[m].set(s[i])
        logW = logW.at[m].set(w[i])
        m = m + 1
        return lax.while_loop(merge_cond, merge_body, (starts, logS, logW, m))

    starts, logS, logW, m = lax.fori_loop(0, n, for_body, (starts0, logS0, logW0, m0))

    idx = jnp.arange(n, dtype=jnp.int32)
    starts = jnp.where(idx < m, starts, jnp.int32(n))
    logS = jnp.where(idx < m, logS, neg_inf)
    logW = jnp.where(idx < m, logW, neg_inf)

    block_idx = jnp.searchsorted(starts, idx, side="right") - jnp.int32(1)
    gammas = logS - logW
    v = gammas[block_idx]
    return v, block_idx, starts, logS, logW


@jax.custom_vjp
def _proj_permutahedron_entropic(z: jax.Array, w: jax.Array) -> jax.Array:
    z = jnp.asarray(z)
    w = jnp.asarray(w)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]

    v, _, _, _, _ = _pav_isotonic_decreasing_entropic(z_sorted, w_sorted)
    p_sorted = z_sorted - v
    return p_sorted[inv_perm_z]


def _proj_permutahedron_entropic_fwd(z: jax.Array, w: jax.Array):
    z = jnp.asarray(z)
    w = jnp.asarray(w)

    perm_z = jnp.argsort(-z, stable=True)
    z_sorted = z[perm_z]
    inv_perm_z = _inv_permutation(perm_z)

    perm_w = jnp.argsort(-w, stable=True)
    w_sorted = w[perm_w]
    inv_perm_w = _inv_permutation(perm_w)

    v, block_idx, _, logS, logW = _pav_isotonic_decreasing_entropic(z_sorted, w_sorted)
    p_sorted = z_sorted - v
    p = p_sorted[inv_perm_z]

    aux = (
        perm_z,
        inv_perm_z,
        perm_w,
        inv_perm_w,
        block_idx,
        z_sorted,
        w_sorted,
        logS,
        logW,
    )
    return p, aux


def _proj_permutahedron_entropic_bwd(aux, g: jax.Array):
    (
        perm_z,
        inv_perm_z,
        perm_w,
        inv_perm_w,
        block_idx,
        z_sorted,
        w_sorted,
        logS,
        logW,
    ) = aux
    g = jnp.asarray(g)
    n = g.shape[0]

    g_sorted = g[perm_z]

    # per-block sum of incoming gradients (this is what the transpose needs)
    block_sum_g = segment_sum(g_sorted, block_idx, num_segments=n)  # (n,)

    # softmax weights within each block
    logS_b = logS[block_idx]
    logW_b = logW[block_idx]
    p_s = jnp.exp(z_sorted - logS_b)  # softmax(z_sorted within block)
    q_w = jnp.exp(w_sorted - logW_b)  # softmax(w_sorted within block)

    sum_g = block_sum_g[block_idx]
    Jt_g_s = p_s * sum_g  # (\partial v / \partial s)^T g
    Jt_g_w = (-q_w) * sum_g  # (\partial v / \partial w)^T g

    grad_z_sorted = g_sorted - Jt_g_s
    grad_w_sorted = -Jt_g_w  # dp/dw = -(\partial v / \partial w) => vjp adds minus

    return (grad_z_sorted[inv_perm_z], grad_w_sorted[inv_perm_w])


_proj_permutahedron_entropic.defvjp(
    _proj_permutahedron_entropic_fwd, _proj_permutahedron_entropic_bwd
)


# ── dispatcher ────────────────────────────────────────────────────────


@eqx.filter_jit
def _proj_permutahedron(
    z: Array,  # (..., n)
    w: Array,  # (..., n)
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
) -> Array:  # (..., n)
    """Projects `z` onto the permutahedron of `w`.

    Solves the optimization problem:
        min_y <z, y> + softness * R(y)
        s.t. y in Perm(w)
    where R(y) is the regularizer determined by `mode`.

    **Arguments:**
    - `z`: Input Array of shape (..., n) to be projected.
    - `w`: Array of shape (..., n) defining the permutahedron.
    - `softness`: Controls the strength of the regularizer.
    - `mode`: Controls the type of regularizer:
        - `smooth`: Entropic (log-KL) projection onto the permutahedron. Solved via isotonic regression, as described in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871). Not fully C∞ due to argsort discontinuities at the boundary of sorting chambers.
        - `c0`: C0 continuous (based on euclidean/L2 regularizer). Euclidean projection onto the permutahedron. Solved via the algorithm in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871).
        - `c1`: C1 differentiable (p=3/2 p-norm). p-norm projection onto the permutahedron with p=3/2. Solved via a PAV algorithm with closed-form block solvers. Inspired by [Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective](https://arxiv.org/abs/2302.01425).
        - `c2`: C2 twice differentiable (p=4/3 p-norm). p-norm projection onto the permutahedron with p=4/3. Solved via a PAV algorithm with closed-form block solvers. Inspired by [Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective](https://arxiv.org/abs/2302.01425).
    **Returns:**
    An Array of shape (..., n) representing the projected values onto the permutahedron of `w`.
    """
    if z.shape != w.shape:
        raise ValueError(
            f"Shapes of z and w must match, but got z.shape={z.shape} and w.shape={w.shape}."
        )
    _validate_softness(softness)
    *batch_sizes, n = z.shape
    z_batched = z.reshape(-1, n)  # (B, n)
    w_batched = w.reshape(-1, n)  # (B, n)
    z_batched = z_batched / softness
    if mode == "smooth":
        proj_fn = _proj_permutahedron_entropic
    elif mode == "c0":
        # Curvature of (1/2)||y||^2: R''=1, no scaling needed
        proj_fn = _proj_permutahedron_pnorm_q2
    elif mode == "c1":
        proj_fn = _proj_permutahedron_pnorm_q3
    elif mode == "c2":
        proj_fn = _proj_permutahedron_pnorm_q4
    else:
        raise ValueError(f"Invalid mode: {mode}")
    soft_values = jax.vmap(proj_fn, in_axes=(0, 0))(z_batched, w_batched)  # (B, n)
    soft_values = soft_values.reshape(*batch_sizes, n)  # (..., n)
    return soft_values


@eqx.filter_jit
def _proj_permutahedron_smooth_sort(
    z: Array,  # (..., n)
    w: Array,  # (..., n)
    softness: float | Array = 0.1,
    lbfgs_tol: float = 1e-5,
    lbfgs_max_iter: int = 10000,
) -> Array:  # (..., n)
    """Projects `z` onto the permutahedron of `w` using the C∞ smooth ESP+LBFGS method.

    Uses smooth majorization bounds (elementary symmetric polynomials) for C∞ gradients w.r.t. ``w``, and an LBFGS dual solver for the entropic LP relaxation.
    This is the ``smooth_sort`` method projection.

    **Arguments:**
    - `z`: Input Array of shape (..., n) to be projected.
    - `w`: Array of shape (..., n) defining the permutahedron.
    - `softness`: Controls the strength of the regularizer.
    - `lbfgs_tol`: Tolerance for the LBFGS solver.
    - `lbfgs_max_iter`: Maximum iterations for the LBFGS solver.

    **Returns:**
    An Array of shape (..., n) representing the projected values.
    """
    if z.shape != w.shape:
        raise ValueError(
            f"Shapes of z and w must match, but got z.shape={z.shape} and w.shape={w.shape}."
        )
    _validate_softness(softness)
    *batch_sizes, n = z.shape
    z_batched = z.reshape(-1, n)  # (B, n)
    w_batched = w.reshape(-1, n)  # (B, n)
    z_batched = z_batched / softness
    proj_fn = _make_proj_permutahedron_entropic_lp(
        tol=lbfgs_tol,
        max_iter=lbfgs_max_iter,
        bounds_softness=jnp.minimum(softness, 1.0),
    )
    soft_values = jax.vmap(proj_fn, in_axes=(0, 0))(z_batched, w_batched)  # (B, n)
    soft_values = soft_values.reshape(*batch_sizes, n)  # (..., n)
    return soft_values
