"""Fast block coordinate descent version of the matrix-weighted MTL estimator.

Instead of using L-BFGS-B on the full (m*d + d)-dimensional space (17K+ params
for HAR), this alternates between:
  1. Per-task logistic regression with a matrix-weighted proximal term
  2. Closed-form or gradient-based bar update

The per-task subproblems are only (d+1)-dimensional and converge quickly.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


def nll_and_grad_theta(X, y, theta):
    """Logistic negative log-likelihood and gradient."""
    z = X @ theta
    p = expit(z)
    nll = np.mean(np.logaddexp(0.0, z) - y * z)
    grad = (X.T @ (p - y)) / X.shape[0]
    return nll, grad


def solve_theta_j(Xj, yj, theta_init, bar, Sigma_j, lam_j, n_j, maxiter=200, lr=0.1):
    """Solve for theta_j given bar using gradient descent with backtracking.

    min_theta  n_j * f_j(theta) + lam_j * ||theta - bar||_{Sigma_j}
    """
    theta = theta_init.copy()
    n = len(yj)
    best_loss = float("inf")
    best_theta = theta.copy()

    for it in range(maxiter):
        # Logistic loss and gradient
        z = Xj @ theta
        p = expit(z)
        loss_logistic = np.sum(np.logaddexp(0.0, z) - yj.ravel() * z)
        grad_logistic = Xj.T @ (p - yj.ravel())

        # Regularization: lam_j * ||theta - bar||_{Sigma_j}
        diff = theta - bar
        sigma_diff = Sigma_j @ diff
        quad_form = float(diff @ sigma_diff)
        norm_val = math.sqrt(max(quad_form, 1e-16))
        loss_reg = lam_j * norm_val
        grad_reg = (lam_j / max(norm_val, 1e-12)) * sigma_diff

        loss = loss_logistic + loss_reg
        grad = grad_logistic + grad_reg

        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()

        # Simple gradient descent with line search
        step = lr
        for _ in range(10):  # backtracking
            theta_new = theta - step * grad
            diff_new = theta_new - bar
            sigma_diff_new = Sigma_j @ diff_new
            quad_new = float(diff_new @ sigma_diff_new)
            norm_new = math.sqrt(max(quad_new, 1e-16))
            z_new = Xj @ theta_new
            loss_new = (np.sum(np.logaddexp(0.0, z_new) - yj.ravel() * z_new)
                        + lam_j * norm_new)
            if loss_new < loss:
                break
            step *= 0.5
        else:
            break  # line search failed

        theta = theta_new
        if np.max(np.abs(step * grad)) < 1e-8:
            break

    return best_theta


def solve_bar(thetas, sigmas, lam_list, bar_init, maxiter=50):
    """Solve for bar given all thetas using fixed-point iteration.

    min_bar  sum_j lam_j * ||theta_j - bar||_{Sigma_j}

    The gradient condition gives:
      sum_j lam_j * Sigma_j @ (bar - theta_j) / ||theta_j - bar||_{Sigma_j} = 0

    Fixed-point iteration:
      w_j = lam_j / ||theta_j - bar||_{Sigma_j}
      bar = (sum_j w_j * Sigma_j)^{-1} (sum_j w_j * Sigma_j @ theta_j)
    """
    bar = bar_init.copy()
    m = len(thetas)
    d = len(bar)

    for _ in range(maxiter):
        weights = []
        weighted_sum = np.zeros(d)
        weighted_sigma_sum = np.zeros((d, d))

        for j in range(m):
            diff = thetas[j] - bar
            sigma_diff = sigmas[j] @ diff
            norm_val = math.sqrt(max(float(diff @ sigma_diff), 1e-16))
            w = lam_list[j] / max(norm_val, 1e-12)
            weights.append(w)
            weighted_sum += w * (sigmas[j] @ thetas[j])
            weighted_sigma_sum += w * sigmas[j]

        try:
            bar_new = np.linalg.solve(weighted_sigma_sum, weighted_sum)
        except np.linalg.LinAlgError:
            bar_new = bar + 0.01 * (weighted_sum - weighted_sigma_sum @ bar)

        if np.max(np.abs(bar_new - bar)) < 1e-6:
            bar = bar_new
            break
        bar = bar_new

    return bar


def fit_ours_bcd(data, q=1.0, maxiter=50, inner_maxiter=200):
    """Fit the matrix-weighted logistic estimator using block coordinate descent.

    Much faster than L-BFGS-B on the full (m*d+d)-dimensional space.
    """
    X_list, y_list = data
    m = len(X_list)
    d = X_list[0].shape[1]
    n_list = [len(y) for y in y_list]
    sigmas = [(X.T @ X) / n for X, n in zip(X_list, n_list)]
    lam_list = [q * math.sqrt(d) * math.sqrt(n) for n in n_list]

    # Initialize
    thetas = [np.zeros(d) for _ in range(m)]
    bar = np.zeros(d)

    for outer_iter in range(maxiter):
        # Step 1: Update each theta_j
        for j in range(m):
            thetas[j] = solve_theta_j(
                X_list[j], y_list[j], thetas[j], bar,
                sigmas[j], lam_list[j], n_list[j],
                maxiter=inner_maxiter
            )

        # Step 2: Update bar
        bar_old = bar.copy()
        bar = solve_bar(thetas, sigmas, lam_list, bar)

        # Check convergence
        if np.max(np.abs(bar - bar_old)) < 1e-6:
            break

    return [thetas[j] for j in range(m)]


def evaluate_ours(data_test, thetas):
    """Evaluate classification error."""
    X_list, y_list = data_test
    errors = []
    for X, y, theta in zip(X_list, y_list, thetas):
        logits = X @ theta
        y_hat = (expit(logits) >= 0.5).astype(int)
        errors.append(np.mean(y_hat != y.ravel()))
    return float(np.mean(errors))
