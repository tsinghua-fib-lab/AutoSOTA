# credal_dro/lv_dro.py

from typing import Any, Callable, Dict, Optional
import warnings

import numpy as np
import numpy.linalg as la

try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover - cvxpy may not be installed in all envs
    cp = None

BulkSpec = Dict[str, Any]
FXSpec = Dict[str, Any]
Sampler = Callable[[int, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Bulk-set helpers
# ---------------------------------------------------------------------------

def make_bulk_set_spec(score_meta: Dict[str, Any], t: float) -> BulkSpec:
    """
    Build a bulk-set specification xi0_spec from the score metadata produced
    by credal_dro.lv_bulk_set.build_score(...) and a DKW-selected threshold t.

    Parameters
    ----------
    score_meta : dict
        For ellipsoid scores (type "ellipsoid"):
            {"type": "ellipsoid", "mu": mu_c, "Sigma": Sigma_c}
        For box scores (type "box"):
            {"type": "box", "mu": mu_c, "w": w}

        Here (mu_c, Sigma_c, w) come from the *calibration/selection* data
        and define the geometry of Xi_0. They do NOT have to match the
        posterior predictive centre used for expectations.

    t : float
        DKW-selected score threshold t_hat. Xi_0 is defined as
        {xi: s(xi) <= t} using the same geometry as the score.

    Returns
    -------
    xi0_spec : dict
        A dictionary describing Xi_0 in a geometry-agnostic way, used by
        the LV-BAS routines in this module.
    """
    geom = score_meta.get("type", None)
    if geom is None:
        raise KeyError("score_meta must contain a 'type' field.")

    mu = np.asarray(score_meta["mu"], dtype=float)

    if geom == "ellipsoid":
        Sigma = np.asarray(score_meta["Sigma"], dtype=float)
        return {
            "geometry": "ellipsoid",
            "mu": mu,
            "Sigma": Sigma,
            "t": float(t),
        }
    elif geom == "box":
        w = np.asarray(score_meta["w"], dtype=float)
        return {
            "geometry": "box",
            "mu": mu,
            "w": w,
            "t": float(t),
        }
    else:
        raise ValueError(
            f"Unsupported score_meta['type']={geom!r} for LV-BAS closed-form "
            "bulk sets. Currently supported: 'ellipsoid', 'box'."
        )


def _bulk_scores(xi: np.ndarray, xi0_spec: BulkSpec) -> np.ndarray:
    """
    Compute the score s(xi) such that Xi_0 = {xi: s(xi) <= t}.

    This mirrors the scoring functions used in lv_bulk_set for ellipsoid and
    box geometries, but operates directly from xi0_spec.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi[None, :]

    mu = np.asarray(xi0_spec["mu"], dtype=float)
    geom = xi0_spec["geometry"]

    if geom == "ellipsoid":
        Sigma = np.asarray(xi0_spec["Sigma"], dtype=float)
        Sig_inv = xi0_spec.get("_Sig_inv", None)
        if Sig_inv is None:
            Sig_inv = la.inv(Sigma)
            xi0_spec["_Sig_inv"] = Sig_inv
        D = xi - mu
        quad = np.einsum("...i,ij,...j->...", D, Sig_inv, D)
        s = np.sqrt(np.maximum(quad, 0.0))
    elif geom == "box":
        w = np.asarray(xi0_spec["w"], dtype=float)
        D = (xi - mu) / w
        s = np.max(np.abs(D), axis=1)
    else:
        raise ValueError(f"Unknown bulk geometry {geom!r} in xi0_spec.")
    return s


def _bulk_indicator(xi: np.ndarray, xi0_spec: BulkSpec, tol: float = 1e-10) -> np.ndarray:
    """
    Boolean indicator of membership in Xi_0:

        mask[i] = True  iff  xi[i] in Xi_0.
    """
    t = float(xi0_spec["t"])
    scores = _bulk_scores(xi, xi0_spec)
    return scores <= t + tol


# ---------------------------------------------------------------------------
# f_x evaluation helpers (vectorised)
# ---------------------------------------------------------------------------

def _evaluate_fx_samples(f_spec: FXSpec, x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Evaluate f_x(xi) for a batch of samples xi of shape (n, d).

    For "linear"/"relu"/"abs"/"piecewise_linear", closed-form expressions are
    used. For "general", an explicit 'eval' callable must be provided in f_spec.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi[None, :]

    f_type = f_spec.get("type", "general")
    x = np.asarray(x, dtype=float)

    # If a direct evaluator is provided, use it.
    eval_fn = f_spec.get("eval", None)
    if eval_fn is not None:
        vals = np.asarray(eval_fn(x, xi), dtype=float).ravel()
        if vals.shape[0] != xi.shape[0]:
            raise ValueError(
                "f_spec['eval'] must return an array of shape (n,) for xi of shape (n,d)."
            )
        return vals

    if f_type in {"linear", "relu", "abs"}:
        if "a_fn" not in f_spec:
            raise KeyError("f_spec['a_fn'] required for type 'linear'/'relu'/'abs'.")
        a = np.asarray(f_spec["a_fn"](x), dtype=float).ravel()
        b_fn = f_spec.get("b_fn", None)
        b = float(b_fn(x)) if b_fn is not None else 0.0

        vals = xi @ a + b
        if f_type == "relu":
            vals = np.maximum(vals, 0.0)
        elif f_type == "abs":
            vals = np.abs(vals)
        return vals

    if f_type == "piecewise_linear":
        if "A_fn" not in f_spec or "b_vec_fn" not in f_spec:
            raise KeyError("f_spec['A_fn'] and f_spec['b_vec_fn'] required for 'piecewise_linear'.")
        A = np.asarray(f_spec["A_fn"](x), dtype=float)  # (J, d)
        b_vec = np.asarray(f_spec["b_vec_fn"](x), dtype=float).ravel()  # (J,)
        if A.ndim != 2:
            raise ValueError("A_fn(x) must return a 2D array of shape (J,d).")
        if A.shape[0] != b_vec.shape[0]:
            raise ValueError("A_fn(x) and b_vec_fn(x) must have consistent J.")

        # scores (n, J) = xi (n,d) @ A^T (d,J) + b_vec (J,)
        scores = xi @ A.T + b_vec[None, :]
        return np.max(scores, axis=1)

    # General case without eval_fn is not supported.
    raise ValueError(
        "For f_spec['type'] == 'general', you must provide f_spec['eval'] "
        "to evaluate f_x on samples."
    )


# ---------------------------------------------------------------------------
# Closed-form sup_{xi in Xi_0} f_x(xi) (and solver fallback)
# ---------------------------------------------------------------------------

def sup_fx_over_bulk(f_spec: FXSpec, xi0_spec: BulkSpec, x: np.ndarray) -> float:
    """
    Compute sup_{xi in Xi_0} f_x(xi).

    For simple f_x and ellipsoid/box Xi_0 this uses the closed-form formulas
    given in the LV-BAS table:

      Ellipsoid: Xi_0 = {xi: ||Sigma_c^{-1/2}(xi - mu_c)||_2 <= t}
      Box:       Xi_0 = {xi: max_i |(xi_i - mu_{c,i}) / w_i| <= t}

    - type "linear":           f_x(xi) = a_x^T xi + b_x
    - type "relu":             f_x(xi) = max{0, a_x^T xi + b_x}
    - type "abs":              f_x(xi) = |a_x^T xi + b_x|
    - type "piecewise_linear": f_x(xi) = max_{j<=J} { a_{x,j}^T xi + b_{x,j} }

    with C_x = a_x^T mu_c + b_x, m_2(a_x) = sqrt(a_x^T Sigma_c a_x),
    m_1(a_x) = sum_i w_i |a_{x,i}|.

    For any other f_spec["type"], or for geometries not covered by the
    closed-form table, the function falls back to a CVXPY-based solver.
    In that case f_spec must provide:

        f_spec["cvxpy_expr_builder"](x, xi_var) -> cp.Expression

    where xi_var is a cvxpy Variable of length d.
    """
    geom = xi0_spec["geometry"]
    f_type = f_spec.get("type", "general")
    x = np.asarray(x, dtype=float)

    mu = np.asarray(xi0_spec["mu"], dtype=float)
    t = float(xi0_spec["t"])

    # ---- Closed-form: linear / relu / abs
    if f_type in {"linear", "relu", "abs"}:
        if "a_fn" not in f_spec:
            raise KeyError("f_spec['a_fn'] required for type 'linear'/'relu'/'abs'.")
        a_x = np.asarray(f_spec["a_fn"](x), dtype=float).ravel()
        b_fn = f_spec.get("b_fn", None)
        b_x = float(b_fn(x)) if b_fn is not None else 0.0

        if geom == "ellipsoid":
            Sigma = np.asarray(xi0_spec["Sigma"], dtype=float)
            quad = float(a_x @ (Sigma @ a_x))
            m = float(np.sqrt(max(quad, 0.0)))
        elif geom == "box":
            w = np.asarray(xi0_spec["w"], dtype=float)
            m = float(np.sum(np.abs(a_x) * w))
        else:
            raise ValueError(f"Unsupported geometry {geom!r} for closed-form sup.")

        C_x = float(a_x @ mu + b_x)

        if f_type == "linear":
            # C_x + t * m_p(a_x)
            return float(C_x + t * m)
        elif f_type == "relu":
            # max{0, C_x + t * m_p(a_x)}
            return float(max(0.0, C_x + t * m))
        else:  # "abs"
            # |C_x| + t * m_p(a_x)
            return float(abs(C_x) + t * m)

    # ---- Closed-form: piecewise linear
    if f_type == "piecewise_linear":
        if "A_fn" not in f_spec or "b_vec_fn" not in f_spec:
            raise KeyError(
                "f_spec['A_fn'] and f_spec['b_vec_fn'] required for 'piecewise_linear'."
            )

        A = np.asarray(f_spec["A_fn"](x), dtype=float)  # (J, d)
        b_vec = np.asarray(f_spec["b_vec_fn"](x), dtype=float).ravel()  # (J,)
        if A.ndim != 2:
            raise ValueError("A_fn(x) must return a 2D array of shape (J,d).")
        if A.shape[0] != b_vec.shape[0]:
            raise ValueError("A_fn(x) and b_vec_fn(x) must have consistent J.")

        if geom == "ellipsoid":
            Sigma = np.asarray(xi0_spec["Sigma"], dtype=float)
            # C_j = a_{x,j}^T mu + b_{x,j}
            C = A @ mu + b_vec  # (J,)
            # m_j = ||Sigma^{1/2} a_{x,j}||_2 = sqrt(a_j^T Sigma a_j)
            AS = A @ Sigma  # (J,d)
            quad = np.einsum("ij,ij->i", AS, A)
            m = np.sqrt(np.maximum(quad, 0.0))
        elif geom == "box":
            w = np.asarray(xi0_spec["w"], dtype=float)
            C = A @ mu + b_vec
            m = np.sum(np.abs(A) * w[None, :], axis=1)
        else:
            raise ValueError(f"Unsupported geometry {geom!r} for closed-form sup.")

        vals = C + t * m
        return float(np.max(vals))

    # ---- Fallback: solver
    return float(_sup_fx_over_bulk_solver(f_spec, xi0_spec, x))


def _sup_fx_over_bulk_solver(f_spec: FXSpec, xi0_spec: BulkSpec, x: np.ndarray) -> float:
    """
    CVXPY-based solver fallback for sup_{xi in Xi_0} f_x(xi).

    Requires:
      - cvxpy to be importable, and
      - f_spec["cvxpy_expr_builder"](x, xi_var) to be defined.
    """
    if cp is None:
        raise ImportError(
            "cvxpy is not available, but required for solver-based LV-BAS "
            "supremum computation."
        )

    if "cvxpy_expr_builder" not in f_spec:
        raise KeyError(
            "f_spec['cvxpy_expr_builder'] must be provided for solver-based "
            "sup_fx_over_bulk()."
        )

    x = np.asarray(x, dtype=float)
    geom = xi0_spec["geometry"]
    mu = np.asarray(xi0_spec["mu"], dtype=float)
    t = float(xi0_spec["t"])

    d = mu.size
    xi_var = cp.Variable(d)

    constraints = []
    if geom == "ellipsoid":
        Sigma = np.asarray(xi0_spec["Sigma"], dtype=float)
        Sig_inv = la.inv(Sigma)
        Sig_inv_sym = 0.5 * (Sig_inv + Sig_inv.T)
        diff = xi_var - mu
        constraints.append(cp.quad_form(diff, Sig_inv_sym) <= t**2)
    elif geom == "box":
        w = np.asarray(xi0_spec["w"], dtype=float)
        constraints.append(xi_var >= mu - t * w)
        constraints.append(xi_var <= mu + t * w)
    else:
        raise ValueError(f"Unsupported geometry {geom!r} in solver fallback.")

    expr_builder: Callable[[np.ndarray, Any], Any] = f_spec["cvxpy_expr_builder"]
    obj_expr = expr_builder(x, xi_var)

    prob = cp.Problem(cp.Maximize(obj_expr), constraints)

    # Prefer MOSEK or GUROBI if available, then fall back to any installed solver.
    installed = set(cp.installed_solvers())
    preferred_solvers = ["CLARABEL", "GUROBI", "ECOS", "OSQP", "SCS"]
    chosen = None
    for s in preferred_solvers:
        if s in installed:
            chosen = s
            break

    if chosen is not None:
        prob.solve(solver=chosen)
    else:
        prob.solve()

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"Solver failed to compute sup_fx_over_bulk: status={prob.status}"
        )
    return float(prob.value)


# ---------------------------------------------------------------------------
# Truncated expectation under P_{c, Xi_0}
# ---------------------------------------------------------------------------

def truncated_expectation(
    f_spec: FXSpec,
    sampler_or_density: Sampler,
    xi0_spec: BulkSpec,
    x: np.ndarray,
    n_saa: int,
    rng: Optional[np.random.Generator] = None,
    draw_cap_factor: int = 5000,
) -> float:
    """
    Monte Carlo estimate of E_{P_{c, Xi_0}}[f_x(xi)] via rejection sampling.

    - sampler_or_density is assumed to sample from the LV-BAS centre P_c, i.e.
      the posterior predictive.
    - xi0_spec encodes Xi_0 calibrated from a *separate* empirical / selection
      set via DKW.
    - We then form the restriction P_{c, Xi_0} by rejecting samples with
      s(xi) > t, and average f_x over the accepted samples.

    Parameters
    ----------
    f_spec : dict
        Specification of f_x; see module docstring. Must either provide
        (a_fn, b_fn, A_fn, b_vec_fn) for the closed-form types, or an
        explicit 'eval' callable for 'general'.

    sampler_or_density : callable
        A function sampler(n, rng) that returns an array of shape (n, d) of
        i.i.d. samples from the posterior predictive P_c. This is the main
        hook to reuse the existing posterior predictive machinery in the
        repository.

    xi0_spec : dict
        Bulk-set geometry spec (ellipsoid or box).

    x : array_like
        Decision vector.

    n_saa : int
        Desired number of accepted samples from P_{c, Xi_0}.

    rng : numpy.random.Generator, optional
        Random number generator. If None, a fresh default_rng() is used.

    draw_cap_factor : int, default 50
        Safety factor c: we attempt at most c * n_saa raw draws from P_c.
        If not enough points land in Xi_0, a RuntimeError is raised.

    Returns
    -------
    float
        Monte Carlo estimate of E_{P_{c, Xi_0}}[f_x(xi)].
    """
    if n_saa <= 0:
        raise ValueError("n_saa must be a positive integer.")

    if rng is None:
        rng = np.random.default_rng()

    accepted: list[np.ndarray] = []
    total_draws = 0
    max_draws = draw_cap_factor * n_saa

    while sum(a.shape[0] for a in accepted) < n_saa and total_draws < max_draws:
        remaining = n_saa - sum(a.shape[0] for a in accepted)
        # Over-sample to compensate for expected rejections
        batch_size = max(2 * remaining, 256)
        xi_batch = np.asarray(sampler_or_density(batch_size, rng), dtype=float)
        if xi_batch.ndim != 2:
            raise ValueError("sampler_or_density must return an array of shape (n, d).")

        mask = _bulk_indicator(xi_batch, xi0_spec)
        if np.any(mask):
            accepted.append(xi_batch[mask])
        total_draws += batch_size

    if sum(a.shape[0] for a in accepted) < n_saa:
        # raise RuntimeError(
        #     f"Insufficient accepted samples for truncated expectation: "
        #     f"needed {n_saa}, got {sum(a.shape[0] for a in accepted)} "
        #     f"from at most {max_draws} draws. Xi_0 may have tiny P_c-mass."
        # )
        warnings.warn(
            f"Insufficient accepted samples for truncated expectation: "
            f"needed {n_saa}, got {sum(a.shape[0] for a in accepted)} "
            f"from at most {max_draws} draws. Xi_0 may have tiny P_c-mass."
        )

    if len(accepted) == 0:
        vals = 0
    else:
        xi_acc = np.concatenate(accepted, axis=0)[:n_saa]
        vals = _evaluate_fx_samples(f_spec, x, xi_acc)
    return float(np.mean(vals))

def truncated_mean(
    sampler_or_density: Sampler,
    xi0_spec: BulkSpec,
    n_saa: int,
    rng: Optional[np.random.Generator] = None,
    draw_cap_factor: int = 5000,
) -> np.ndarray:
    """
    Monte Carlo estimate of mu_trunc = E_{P_{c, Xi_0}}[xi] via rejection sampling.

    This is analogous to `truncated_expectation`, but returns the (d,)-dimensional
    mean vector instead of E[f_x(xi)].
    """
    if n_saa <= 0:
        raise ValueError("n_saa must be a positive integer.")

    if rng is None:
        rng = np.random.default_rng()

    accepted: list[np.ndarray] = []
    total_draws = 0
    max_draws = draw_cap_factor * n_saa

    while sum(a.shape[0] for a in accepted) < n_saa and total_draws < max_draws:
        remaining = n_saa - sum(a.shape[0] for a in accepted)
        batch_size = max(2 * remaining, 256)
        xi_batch = np.asarray(sampler_or_density(batch_size, rng), dtype=float)
        if xi_batch.ndim != 2:
            raise ValueError("sampler_or_density must return an array of shape (n, d).")

        mask = _bulk_indicator(xi_batch, xi0_spec)
        if np.any(mask):
            accepted.append(xi_batch[mask])
        total_draws += batch_size

    if sum(a.shape[0] for a in accepted) < n_saa:
        warnings.warn(
            f"Insufficient accepted samples for truncated mean: "
            f"needed {n_saa}, got {sum(a.shape[0] for a in accepted)} "
            f"from at most {max_draws} draws. Xi_0 may have tiny P_c-mass."
        )

    if not accepted:
        # Degenerate fallback
        d = xi0_spec["mu"].shape[0]
        return np.zeros(d, dtype=float)

    xi_acc = np.concatenate(accepted, axis=0)[:n_saa]
    return xi_acc.mean(axis=0)


# ---------------------------------------------------------------------------
# Top-level LV-BAS objective
# ---------------------------------------------------------------------------

def lv_objective(
    x: np.ndarray,
    eps: float,
    xi0_spec: BulkSpec,
    f_spec: FXSpec,
    sampler_or_density: Sampler,
    n_saa: int,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Compute the LV-BAS objective

        J(x; eps, Xi_0)
          = (1 - eps) E_{P_{c, Xi_0}}[f_x(xi)]
            + eps * sup_{xi in Xi_0} f_x(xi).

    Here:
      - P_c is the posterior predictive (sampler_or_density),
      - Xi_0 is the bulk set calibrated from an empirical selection set.

    Parameters
    ----------
    x : array_like
        Decision vector.

    eps : float in [0,1]
        LV-BAS distortion parameter.

    xi0_spec : dict
        Bulk-set specification (ellipsoid or box).

    f_spec : dict
        Specification of f_x; see module docstring.

    sampler_or_density : callable
        Sampler for P_c, see truncated_expectation().

    n_saa : int
        Number of accepted samples for the truncated expectation.

    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Approximate LV-BAS objective J(x; eps, Xi_0).
    """
    eps = float(eps)
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps must lie in [0,1].")

    x = np.asarray(x, dtype=float)

    sup_term = sup_fx_over_bulk(f_spec, xi0_spec, x)
    if eps == 1.0:
        # Purely vacuous (no in-bulk expectation).
        return float(sup_term)

    mean_term = truncated_expectation(
        f_spec=f_spec,
        sampler_or_density=sampler_or_density,
        xi0_spec=xi0_spec,
        x=x,
        n_saa=n_saa,
        rng=rng,
    )
    return float((1.0 - eps) * mean_term + eps * sup_term)
