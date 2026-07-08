"""Functions for the Newsvendor Problem.

This module contains:
1) A CVXPY implementation of the (separable) newsvendor loss.
2) An LV-BAS CVXPY problem builder for the newsvendor dataset, using:
   - truncated posterior predictive SAA for the (1-ε) branch, and
   - a closed-form ellipsoidal worst-case term for the ε branch.

The LV-BAS objective implemented is:
    (1-ε) E_{P_{c, Xi_0}}[f_x(ξ)] + ε sup_{ξ in Xi_0} f_x(ξ),
where Xi_0 is an ellipsoid and P_{c, Xi_0} is the posterior predictive truncated to Xi_0.
"""

from __future__ import annotations

from typing import Final

import cvxpy as cp
import numpy as np

from bayesian_dro.Bayesian_DRO_continuous import SMALLEST_X

BACKORDER_COST: Final[float] = 8.0  # denoted b
HOLDING_COST: Final[float] = 3.0  # denoted h


def newsvendor_cost_cvxpy(x: cp.Expression, xi: cp.Expression) -> cp.Expression:
    """Vectorised separable newsvendor cost in CVXPY.

    For each coordinate i:
        f_i(x_i, ξ_i) = max(h (x_i - ξ_i), b (ξ_i - x_i))
    and f_x(ξ) = sum_i f_i(x_i, ξ_i).

    Parameters
    ----------
    x:
        Decision variable of shape (dim,).
    xi:
        Demand samples of shape (N, dim). Can be a numpy array or a CVXPY Parameter/Expression.

    Returns
    -------
    cp.Expression
        Cost vector of shape (N,), one cost per row of xi.
    """
    dim = int(x.shape[0])
    h = HOLDING_COST * np.ones(dim, dtype=float)
    b = BACKORDER_COST * np.ones(dim, dtype=float)

    # Broadcast x to (N, dim) by stacking
    n = int(xi.shape[0])
    X = cp.vstack([x for _ in range(n)])
    return cp.maximum(0, X - xi) @ h + cp.maximum(0, xi - X) @ b


def make_newsvendor_a_mat(
    dim: int,
    h: float = HOLDING_COST,
    b: float = BACKORDER_COST,
    *,
    max_dim: int = 15,
) -> np.ndarray:
    """Enumerate all sign patterns a ∈ {-h, b}^d.

    This is used for the identity (ellipsoidal Xi_0):
        sup_{ξ∈Xi_0} f_x(ξ) = max_{a∈{-h,b}^d} [ aᵀ(μ-x) + t ||Σ^{1/2} a||₂ ].

    Raises
    ------
    NotImplementedError
        If dim is too large (since K = 2^dim patterns are enumerated explicitly).
    """
    if dim < 1:
        raise ValueError("dim must be a positive integer.")
    if dim > max_dim:
        raise NotImplementedError(
            f"LV-BAS newsvendor worst-case enumerates 2^dim patterns; "
            f"dim={dim} exceeds max_dim={max_dim}. Increase max_dim only if you accept the 2^dim blow-up."
        )

    vals = [-float(h), float(b)]
    grids = np.meshgrid(*([vals] * dim), indexing="ij")
    a_mat = np.stack(grids, axis=-1).reshape(-1, dim).astype(float)
    return a_mat


def get_lv_newsvendor_problem(
    dim: int,
    n_trunc: int,
    h: float = HOLDING_COST,
    b: float = BACKORDER_COST,
    *,
    max_dim: int = 15,
) -> cp.Problem:
    """Build the LV-BAS newsvendor CVXPY problem (constructed once, per run()).

    Parameters (to be set per replication)
    -------------------------------------
    epsilon : scalar, nonnegative
        Weight on the worst-case term.
    one_minus_epsilon : scalar, nonnegative
        Weight on the truncated SAA term. We keep this separate from (1 - epsilon)
        to satisfy CVXPY's DCP rules (coefficient sign must be known at compile time).
    xi_trunc : (n_trunc, dim)
        Fixed accepted samples from the posterior predictive truncated to Xi_0.
    wcs_const : (K,)
        Precomputed constants wcs_const[k] = a_kᵀ μ + t ||chol(Σ)ᵀ a_k||₂,
        where K = 2^dim and a_k are the rows of a_mat.

    Returns
    -------
    cp.Problem
        Minimise: one_minus_epsilon * mean_j f_x(xi_trunc[j])
                 + epsilon * max_k (wcs_const[k] - a_kᵀ x)
    """
    if n_trunc <= 0:
        raise ValueError("n_trunc must be a positive integer.")

    a_mat = make_newsvendor_a_mat(dim, h=h, b=b, max_dim=max_dim)
    K = int(a_mat.shape[0])

    # Decision variable
    x = cp.Variable(dim, name="x")

    # Parameters set per replication
    epsilon = cp.Parameter(name="epsilon", nonneg=True)
    one_minus_epsilon = cp.Parameter(name="one_minus_epsilon", nonneg=True)
    xi_trunc = cp.Parameter((n_trunc, dim), name="xi_trunc")
    wcs_const = cp.Parameter(K, name="wcs_const")

    # Truncated expectation via SAA (inside the CVXPY objective)
    mean_term = cp.sum(newsvendor_cost_cvxpy(x, xi_trunc)) / float(n_trunc)

    # Worst-case term via max over affine functions in x
    worst_term = cp.max(wcs_const - a_mat @ x)

    obj = cp.Minimize(one_minus_epsilon * mean_term + epsilon * worst_term)
    constraints = [x >= SMALLEST_X]

    prob = cp.Problem(obj, constraints)

    # Convenience: keep the enumeration around for per-replication wcs_const construction.
    prob._lv_newsvendor_a_mat = a_mat  # type: ignore[attr-defined]

    return prob
