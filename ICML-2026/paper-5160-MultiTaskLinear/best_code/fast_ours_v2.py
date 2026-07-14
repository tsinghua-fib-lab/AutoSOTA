"""Fast matrix-weighted MTL estimator using BCD with scipy per-task optimization.

Key insight: the problem is separable in theta_j given bar.
Each per-task subproblem has only (d+1) = 562 parameters for HAR,
so L-BFGS-B converges quickly on each.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


def solve_theta_j_scipy(Xj, yj, theta_init, bar, Sigma_j, lam_j, maxiter=100):
    """Solve per-task subproblem using L-BFGS-B (only ~562 params)."""
    y_flat = yj.ravel()
    n = len(y_flat)

    def objective(theta):
        z = Xj @ theta
        p = expit(z)
        nll = np.sum(np.logaddexp(0.0, z) - y_flat * z)
        grad_nll = Xj.T @ (p - y_flat)

        diff = theta - bar
        sigma_diff = Sigma_j @ diff
        quad_form = float(diff @ sigma_diff)
        norm_val = math.sqrt(max(quad_form, 1e-12))
        reg_val = lam_j * norm_val
        grad_reg = (lam_j / norm_val) * sigma_diff if norm_val > 1e-12 else np.zeros_like(diff)

        return nll + reg_val, grad_nll + grad_reg

    result = minimize(
        objective,
        theta_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-8},
    )
    return result.x


def solve_bar_closed(thetas, sigmas, lam_list, bar_init, maxiter=50):
    """Closed-form bar update via fixed-point iteration."""
    bar = bar_init.copy().astype(np.float64)
    m = len(thetas)
    d = len(bar)

    for _ in range(maxiter):
        weighted_sum = np.zeros(d, dtype=np.float64)
        weighted_sigma_sum = np.zeros((d, d), dtype=np.float64)

        for j in range(m):
            diff = thetas[j] - bar
            sigma_diff = sigmas[j] @ diff
            norm_val = math.sqrt(max(float(diff @ sigma_diff), 1e-16))
            w = lam_list[j] / max(norm_val, 1e-12)
            weighted_sum += w * (sigmas[j] @ thetas[j])
            weighted_sigma_sum += w * sigmas[j]

        try:
            bar_new = np.linalg.solve(weighted_sigma_sum, weighted_sum)
        except np.linalg.LinAlgError:
            bar_new = bar + 0.1 * (weighted_sum - weighted_sigma_sum @ bar)

        change = np.max(np.abs(bar_new - bar))
        bar = bar_new
        if change < 1e-6:
            break

    return bar


def fit_ours_bcd_v2(data, q=1.0, max_outer=30, max_inner=100):
    """BCD with scipy per-task optimization."""
    X_list, y_list = data
    m = len(X_list)
    d = X_list[0].shape[1]
    n_list = [len(y) for y in y_list]
    sigmas = [(X.T @ X) / max(n, 1) for X, n in zip(X_list, n_list)]
    lam_list = [q * math.sqrt(d) * math.sqrt(max(n, 1)) for n in n_list]

    thetas = [np.zeros(d) for _ in range(m)]
    bar = np.zeros(d)

    for outer_iter in range(max_outer):
        # Step 1: Update each theta_j (can be parallelized)
        for j in range(m):
            thetas[j] = solve_theta_j_scipy(
                X_list[j], y_list[j], thetas[j], bar,
                sigmas[j], lam_list[j], maxiter=max_inner
            )

        # Step 2: Update bar
        bar_old = bar.copy()
        bar = solve_bar_closed(thetas, sigmas, lam_list, bar)

        if np.max(np.abs(bar - bar_old)) < 1e-6:
            break

    return [thetas[j] for j in range(m)]


def evaluate_ours(data_test, thetas):
    X_list, y_list = data_test
    errors = []
    for X, y, theta in zip(X_list, y_list, thetas):
        logits = X @ theta
        y_hat = (expit(logits) >= 0.5).astype(int)
        errors.append(np.mean(y_hat != y.ravel()))
    return float(np.mean(errors))
