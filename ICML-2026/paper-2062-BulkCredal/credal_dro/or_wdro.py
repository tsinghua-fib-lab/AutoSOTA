"""
Outlier-Robust Wasserstein DRO (OR-WDRO) baseline.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import cvxpy as cp

from bayesian_dro.Bayesian_DRO_continuous import SMALLEST_X
from .newsvendor import make_newsvendor_a_mat, HOLDING_COST, BACKORDER_COST


def cheap_robust_mean_estimate(X: np.ndarray, eps: float) -> np.ndarray:
    """
    Direct port of cheap_robust_mean_estimate.m (coordinate-wise trimmed mean).

    MATLAB:
        trim_start = ceil(eps*n)
        trim_end   = floor((1-eps)*n)
        X_trimmed = X_sorted(trim_start:trim_end,:)

    Notes:
      - We mimic MATLAB's 1-based slicing via (start-1):end in Python.
      - eps must be in [0, 0.5) for this to make sense.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, d = X.shape
    if not (0.0 <= eps < 0.5):
        raise ValueError(f"eps must be in [0, 0.5), got {eps}")

    X_sorted = np.sort(X, axis=0)

    trim_start = int(math.ceil(eps * n))
    trim_end = int(math.floor((1.0 - eps) * n))

    # MATLAB is 1-based; if trim_start==0 we start at row 1.
    start_idx = max(trim_start, 1) - 1
    end_idx = min(trim_end, n)

    if end_idx <= start_idx:
        raise ValueError(
            f"Trimming removed all samples: n={n}, eps={eps}, "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

    X_trimmed = X_sorted[start_idx:end_idx, :]
    return np.mean(X_trimmed, axis=0)


def robust_sigma_sq_estimate(
    X: np.ndarray,
    z0: np.ndarray,
    eps: float,
    *,
    trim_both_tails: bool = True,
    min_sigma_sq: float = 1e-12,
) -> float:
    """
    Conservative, robust sigma^2 estimator for q=2.

    We estimate a second-moment scale around z0 using trimmed mean of
    squared Euclidean distances.
    """
    X = np.asarray(X, dtype=float)
    z0 = np.asarray(z0, dtype=float).reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if z0.shape[1] != X.shape[1]:
        raise ValueError(f"z0 dim mismatch: X has {X.shape[1]} cols but z0 has {z0.shape[1]}")
    if not (0.0 <= eps < 0.5):
        raise ValueError(f"eps must be in [0, 0.5), got {eps}")

    dist2 = np.sum((X - z0) ** 2, axis=1)
    dist2_sorted = np.sort(dist2)
    n = dist2_sorted.shape[0]

    if trim_both_tails:
        trim_start = int(math.ceil(eps * n))
        trim_end = int(math.floor((1.0 - eps) * n))
        start_idx = max(trim_start, 1) - 1
        end_idx = min(trim_end, n)
        if end_idx <= start_idx:
            trimmed = dist2_sorted
        else:
            trimmed = dist2_sorted[start_idx:end_idx]
    else:
        # Drop only upper tail.
        trim_end = int(math.floor((1.0 - eps) * n))
        end_idx = max(trim_end, 1)
        trimmed = dist2_sorted[:end_idx]

    sigma_sq = float(np.mean(trimmed)) if trimmed.size else float(np.mean(dist2_sorted))
    return max(sigma_sq, float(min_sigma_sq))


def _rotated_soc_lambda_tau(
    zeta_g: cp.Expression,
    lambda_1: cp.Expression,
    tau: cp.Expression,
) -> cp.Constraint:
    """
    Implements the MATLAB/YALMIP rotated cone constraint:
        rcone(zeta_G, lambda_1, 0.5*tau)
    which enforces:
        ||zeta_G||_2^2 <= lambda_1 * tau,  with lambda_1 >= 0, tau >= 0.

    Supports:
      - Single block: zeta_g has shape (d,), tau is scalar.
      - Vectorised blocks: zeta_g has shape (d, M), tau has shape (M,).

    Standard SOC encoding (equivalent to the rotated cone above):
        || [ sqrt(2)*zeta_G ; lambda_1 - 0.5*tau ] ||_2 <= lambda_1 + 0.5*tau
    """
    # Single (non-vectorised) constraint
    if len(zeta_g.shape) == 1:
        return cp.norm(cp.hstack([np.sqrt(2.0) * zeta_g, lambda_1 - 0.5 * tau])) <= (
            lambda_1 + 0.5 * tau
        )

    # Vectorised (column-wise) constraints
    if len(zeta_g.shape) != 2:
        raise ValueError(f"zeta_g must be a vector or a 2D matrix, got shape {zeta_g.shape}")

    M = int(zeta_g.shape[1])
    v = lambda_1 - 0.5 * tau
    t = lambda_1 + 0.5 * tau
    cone_mat = cp.vstack([np.sqrt(2.0) * zeta_g, cp.reshape(v, (1, M), order="F")])
    return cp.norm(cone_mat, axis=0) <= t


def get_or_wdro_newsvendor_problem(
    *,
    dim: int,
    num_observations: int,
    holding_cost: float = HOLDING_COST,
    backorder_cost: float = BACKORDER_COST,
    dual_norm: int | float = 2,
    max_dim: int = 15,
) -> cp.Problem:
    """
    Build the OR-WDRO conic programme for the newsvendor loss, following
    outlier_robust_WDRO.m as literally as possible.

    Parameters are set per replication:
      - Z (n x dim): training data (demand samples)
      - z0 (dim): robust mean estimate
      - sigma_sq: robust second-moment radius estimate (sigma^2)
      - rho: Wasserstein radius
      - inv_one_minus_vareps: 1/(1 - vareps) to avoid non-DCP expressions

    Decision variable:
      - x in R^dim (order quantities)

    Complexity warning:
      J = 2^dim pieces, and we instantiate (zeta_G, zeta_W, tau) for each (i, k).
      Total blocks = n * J.
    """
    n = int(num_observations)
    if n <= 0:
        raise ValueError(f"num_observations must be positive, got {n}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    a_mat = make_newsvendor_a_mat(dim=dim, h=holding_cost, b=backorder_cost, max_dim=max_dim)
    a_mat = np.asarray(a_mat, dtype=float)
    J = int(a_mat.shape[0])  # J = 2^dim

    # === Parameters (set in run_replication) ===
    Z = cp.Parameter((n, dim), name="Z")                       # data
    z0 = cp.Parameter(dim, name="z0")                          # robust mean
    sigma_sq = cp.Parameter(nonneg=True, name="sigma_sq")      # sigma^2 (q=2)
    rho = cp.Parameter(nonneg=True, name="rho")                # Wasserstein radius
    inv_one_minus_vareps = cp.Parameter(nonneg=True, name="inv_one_minus_vareps")  # 1/(1-vareps)

    # === Decision variable ===
    x = cp.Variable(dim, name="x")

    # === OR-WDRO variables (names match MATLAB) ===
    lambda_1 = cp.Variable(nonneg=True, name="lambda_1")
    lambda_2 = cp.Variable(nonneg=True, name="lambda_2")
    alpha = cp.Variable(name="alpha")
    s = cp.Variable(n, nonneg=True, name="s")

   # Flatten (i,k) into a single index m = i*J + k  (m = i*J + k)
    M = n * J
    tau = cp.Variable(M, nonneg=True, name="tau")
    zeta_W = cp.Variable((dim, M), name="zeta_W")

    # Objective: lambda_1*sigma^2 + lambda_2*rho + alpha + (1/(n(1-vareps))) sum s
    objective = cp.Minimize(
        lambda_1 * sigma_sq
        + lambda_2 * rho
        + (inv_one_minus_vareps / n) * cp.sum(s)
        + alpha
    )

    constraints: list[cp.Constraint] = []

    # Same lower bound used by other newsvendor solvers in this repo
    constraints.append(x >= SMALLEST_X)

    # === Vectorised model construction ===
    
    A_big = np.tile(a_mat, (n, 1))          # (M, dim)
    A_big_T = A_big.T                       # (dim, M)

    # Eliminate zeta_G:  zeta_G(:,m) = a_k - zeta_W(:,m)
    zeta_G_expr = A_big_T - zeta_W         # (dim, M) expression

    # Replicate Z to match (i,k) indexing: Z_rep[m,:] = Z[i,:]
    ones_1J = np.ones((1, J))
    Z_rep_T = cp.kron(Z.T, ones_1J)         # (dim, M)

    # Replicate s to match (i,k) indexing: s_rep[m] = s[i]
    # We form a 1×M row vector then reshape to (M,).
    s_rep = cp.reshape(cp.kron(cp.reshape(s, (1, n), order="F"), ones_1J), (M,), order="F")

    # Term-by-term assembly of the s-constraint RHS (vectorised over m=1..M)
    ax = A_big @ x                          # (M,)  with ax[m] = a_k^T x
    z0_dot_zetaG = cp.reshape(z0 @ zeta_G_expr, (M,), order="F")  # (M,)
    Zi_dot_zetaW = cp.sum(cp.multiply(Z_rep_T, zeta_W), axis=0)  # (M,)

    rhs = (-ax) + z0_dot_zetaG + tau + Zi_dot_zetaW - alpha


    constraints.append(s_rep >= rhs)

    # Dual norm constraints (vectorised over columns m):
    if dual_norm == np.inf:
        constraints.append(cp.abs(zeta_W) <= lambda_2)
    else:
        constraints.append(cp.norm(zeta_W, dual_norm, axis=0) <= lambda_2)

    # Rotated cone constraints (vectorised):
    # Enforces ||zeta_G||_2^2 <= lambda_1 * tau  (matches Theorem 2).
    constraints.append(_rotated_soc_lambda_tau(zeta_G_expr, lambda_1, tau))

    return cp.Problem(objective, constraints)
