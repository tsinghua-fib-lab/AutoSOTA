"""
FILE: nodewise.py
INPUT: Normalized sample matrices and nodewise regression tuning settings.
OUTPUT: Lasso and square-root Lasso self-regression coefficient matrices.
POS: Source-grounded nodewise-regression baselines for paper experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from dro_subspace.synthetic import normalize_design

FloatArray = NDArray[np.float64]

DEFAULT_LASSO_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def _require_cvxpy() -> ModuleType:
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        message = "Install optional baseline dependencies with `pip install -e '.[baselines]'`."
        raise ModuleNotFoundError(message) from exc
    return cp


def _penalty_vector(penalty: float | Sequence[float], dim: int) -> FloatArray:
    if isinstance(penalty, Real):
        return np.full(dim, float(penalty), dtype=float)
    vector = np.asarray(penalty, dtype=float)
    if vector.shape != (dim,):
        raise ValueError(f"penalty must be scalar or a length-{dim} vector.")
    return vector


def solve_nodewise_regression(
    samples: FloatArray,
    target_idx: int,
    regression_method: str,
    penalty_l1: float,
    penalty_l2: float = 0.0,
) -> FloatArray:
    """Solve one CVXPY nodewise regression copied from the old SubspaceClustering source."""
    if regression_method not in {"lasso", "sqrt", "L1"}:
        raise ValueError("regression_method must be one of: lasso, sqrt, L1.")
    cp = _require_cvxpy()
    normalized = normalize_design(samples)
    dim = normalized.shape[1]
    if target_idx < 0 or target_idx >= dim:
        raise ValueError("target_idx is out of bounds.")

    beta = cp.Variable(dim)
    constraints = [beta[target_idx] == 0.0]
    residual = normalized @ beta - normalized[:, target_idx]
    if regression_method == "sqrt":
        loss = cp.norm2(residual)
    elif regression_method == "lasso":
        loss = cp.sum_squares(residual)
    else:
        loss = cp.norm1(residual) / normalized.shape[0]
    loss += float(penalty_l1) * cp.norm1(beta)
    if penalty_l2 != 0.0:
        loss += float(penalty_l2) * cp.sum_squares(beta)
    problem = cp.Problem(cp.Minimize(loss), constraints)
    problem.solve()
    if beta.value is None:
        raise RuntimeError(f"CVXPY did not return a solution for target column {target_idx}.")
    return np.asarray(beta.value, dtype=float)


def compute_nodewise_coefficients(
    samples: FloatArray,
    regression_method: str,
    penalty_l1: float | Sequence[float],
    penalty_l2: float = 0.0,
) -> FloatArray:
    """Compute the full self-regression matrix with the old separate nodewise formulation."""
    dim = samples.shape[1]
    penalties = _penalty_vector(penalty_l1, dim)
    columns = [
        solve_nodewise_regression(samples, target_idx, regression_method, penalties[target_idx], penalty_l2)
        for target_idx in range(dim)
    ]
    return np.stack(columns, axis=1)


def lambda_sqrt_lasso(
    samples: FloatArray,
    target_idx: int,
    alpha: float = 0.05,
    n_sim: int = 1000,
    seed: int = 2020,
) -> float:
    """Compute the square-root Lasso lambda simulation from the old SubspaceClustering source."""
    normalized = samples - samples.mean(axis=0)
    normalized = normalized / (np.linalg.norm(normalized, ord=2, axis=0) / np.sqrt(samples.shape[0]))
    reduced = np.delete(normalized, target_idx, axis=1)
    rng = np.random.RandomState(seed)
    eta = rng.normal(size=(normalized.shape[0], n_sim))
    empirical_maxima = np.abs(reduced.T @ eta / normalized.shape[0]).max(axis=0)
    return float(np.quantile(empirical_maxima, 1.0 - alpha, method="higher"))


def calculate_sqrt_lasso_penalties(
    samples: FloatArray,
    alpha: float = 0.05,
    n_sim: int = 1000,
    seed: int = 2020,
    sep: bool = True,
    n_clusters: int = 1,
) -> FloatArray:
    """Calculate square-root Lasso penalties with the paper notebook's `sep=True` option by default."""
    n_samples, dim = samples.shape
    if sep:
        sigma = np.cov(samples.T)
        return np.array(
            [
                min(
                    np.sqrt(12.0 * np.log(dim) / (n_samples - 1)),
                    np.quantile(np.abs(sigma[target_idx]) * (n_samples - 1), 1.0 - 1.0 / n_clusters),
                )
                for target_idx in range(dim)
            ],
            dtype=float,
        )
    return np.array(
        [lambda_sqrt_lasso(samples, target_idx, alpha=alpha, n_sim=n_sim, seed=seed) for target_idx in range(dim)],
        dtype=float,
    )


class LassoNodewiseRegressor(BaseEstimator):
    """GridSearchCV-compatible estimator ported from the old portfolio_backtesting source."""

    def __init__(self, penalty_l1: float = 0.01) -> None:
        self.penalty_l1 = penalty_l1
        self.weights: FloatArray | None = None

    def fit(self, X: FloatArray, y: FloatArray) -> "LassoNodewiseRegressor":
        normalized = normalize_design(np.asarray(X, dtype=float))
        self.weights = compute_nodewise_coefficients(normalized, "lasso", self.penalty_l1, 0.0)
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        if self.weights is None:
            raise RuntimeError("The estimator must be fitted before prediction.")
        normalized = normalize_design(np.asarray(X, dtype=float))
        return normalized @ self.weights

    def score(self, X: FloatArray, y: FloatArray) -> float:
        normalized = normalize_design(np.asarray(X, dtype=float))
        predicted = self.predict(X)
        return -float(np.sum((predicted - normalized) ** 2))


def compute_lasso_coefficients(
    samples: FloatArray,
    penalty_l1_grid: Sequence[float] = DEFAULT_LASSO_GRID,
    cv: int = 3,
) -> FloatArray:
    """Run the old cross-validated Lasso nodewise baseline."""
    estimator = LassoNodewiseRegressor()
    grid = GridSearchCV(estimator, {"penalty_l1": list(penalty_l1_grid)}, refit=True, cv=cv, n_jobs=1)
    grid.fit(samples, samples)
    weights = grid.best_estimator_.weights
    if weights is None:
        raise RuntimeError("GridSearchCV did not produce fitted Lasso weights.")
    return np.asarray(weights, dtype=float)


def compute_sqrt_lasso_coefficients(
    samples: FloatArray,
    alpha: float = 0.05,
    n_sim: int = 1000,
    seed: int = 2020,
    l1_penalty_multiplier: float = 1.0,
    l2_penalty: float = 0.0,
    sep: bool = True,
    n_clusters: int = 1,
) -> FloatArray:
    """Run the old square-root Lasso nodewise baseline with separate CVXPY regressions."""
    penalties = calculate_sqrt_lasso_penalties(
        samples,
        alpha=alpha,
        n_sim=n_sim,
        seed=seed,
        sep=sep,
        n_clusters=n_clusters,
    )
    return compute_nodewise_coefficients(samples, "sqrt", l1_penalty_multiplier * penalties, l2_penalty)
