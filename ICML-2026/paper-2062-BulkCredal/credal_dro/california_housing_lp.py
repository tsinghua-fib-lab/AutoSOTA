"""CVXPY formulations for the California Housing LV-BAS experiment."""

from __future__ import annotations

from typing import Dict

import cvxpy as cp
import numpy as np


def _attach_dicts(prob: cp.Problem, *, param_dict: Dict[str, cp.Parameter], var_dict: Dict[str, cp.Expression]) -> cp.Problem:
    """Attach param_dict and var_dict to a CVXPY Problem (repo convention)."""
    for param_name, value in param_dict.items():
        if param_name in prob.param_dict:
            # Assign the new value
            prob.param_dict[param_name].value = value
        else:
            print(f"Warning: Parameter '{param_name}' not found in problem.")
    for var_name, value in var_dict.items():
        if var_name in prob.var_dict:
            # Assign the new value
            prob.var_dict[var_name] = value
        else:
            print(f"Warning: Variable '{var_name}' not found in problem.")
    return prob


# ----------------------------------------------------------------------
# California Housing problems
# ----------------------------------------------------------------------

def get_lv_bas_ellipsoid_x_interval_y_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    LV-BAS with an ellipsoid in x and an independent interval in y (intersection set).

    Bulk set:
        Xi0 = { (x,y): ||L_x^{-1}(x - mu_x)||_2 <= t_x   and   |y - mu_y| <= r_y }.

    Worst-case absolute loss (closed form):
        sup_{Xi0} | y - (w^T x + b) |
        = |mu_y - mu_x^T w - b| + r_y + t_x * || L_x^T w ||_2.
    """
    d = int(dim)
    m = int(num_samples)
    if d <= 0 or m <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    # Parameters (SAA term)
    X_saa = cp.Parameter((m, d), name="X_saa")
    y_saa = cp.Parameter((m,), name="y_saa")

    epsilon = cp.Parameter(nonneg=True, name="epsilon")
    one_minus_epsilon = cp.Parameter(nonneg=True, name="one_minus_epsilon")

    # Parameters (bulk set)
    mu_x = cp.Parameter((d,), name="mu_x")
    mu_y = cp.Parameter(name="mu_y")
    sqrt_Sigma_x = cp.Parameter((d, d), name="sqrt_Sigma_x")  # Cholesky L_x of Sigma_x
    t_x = cp.Parameter(nonneg=True, name="t_x")
    r_y = cp.Parameter(nonneg=True, name="r_y")

    # Variables
    x = cp.Variable(d + 1, name="x")           # [w; b]
    w = x[:d]
    b = x[d]

    u = cp.Variable(m, nonneg=True, name="u")  # |residual_i|
    g = cp.Variable(nonneg=True, name="g")     # |mu_y - (mu_x^T w + b)|
    s = cp.Variable(nonneg=True, name="s")     # sup_{Xi0} |residual|

    residual = y_saa - (X_saa @ w + b)
    centre_residual = mu_y - (mu_x @ w + b)

    constraints = [
        u >= residual,
        u >= -residual,

        g >= centre_residual,
        g >= -centre_residual,

        s >= g + r_y + t_x * cp.norm(sqrt_Sigma_x.T @ w, 2),
    ]

    objective = cp.Minimize(one_minus_epsilon * (1.0 / m) * cp.sum(u) + epsilon * s)
    prob = cp.Problem(objective, constraints)
    return prob

def get_lv_bas_ch_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    LV-BAS (continuous centre, ellipsoidal bulk set) for California Housing.

    Only used as a demonstration of why ellipsoidal bulk set is not a good choice in this paper.

    Bulk set Xi0 is an ellipsoid in xi=(x,y) defined by a Mahalanobis score:
        Xi0 = { xi : ||Sigma^{-1/2}(xi - mu)||_2 <= t_hat }.

    Decision variable is x = [w; b] in R^{d+1}.
    """
    d = int(dim)
    m = int(num_samples)
    if d <= 0 or m <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    # Parameters
    X_saa = cp.Parameter((m, d), name="X_saa")
    y_saa = cp.Parameter((m,), name="y_saa")

    epsilon = cp.Parameter(nonneg=True, name="epsilon")
    one_minus_epsilon = cp.Parameter(nonneg=True, name="one_minus_epsilon")

    mu_x = cp.Parameter((d,), name="mu_x")
    mu_y = cp.Parameter(name="mu_y")

    # Cholesky factor L of Sigma for xi=(x,y), so Sigma = L L^T (L lower-triangular).
    sqrt_Sigma_xi = cp.Parameter((d + 1, d + 1), name="sqrt_Sigma_xi")
    t_hat = cp.Parameter(nonneg=True, name="t_hat")

    # Variables
    x = cp.Variable(d + 1, name="x")              # [w; b]
    w = x[:d]
    b = x[d]

    u = cp.Variable(m, nonneg=True, name="u")     # |residual_i|
    g = cp.Variable(nonneg=True, name="g")        # |mu_y - (mu_x^T w + b)|
    s = cp.Variable(nonneg=True, name="s")        # sup_{xi in Xi0} |y - (w^T x + b)|

    residual = y_saa - (X_saa @ w + b)
    centre_residual = mu_y - (mu_x @ w + b)

    # a = [-w; 1] so that residual(xi) = a^T xi - b when xi=(x,y)
    a = cp.hstack([-w, cp.Constant(np.array([1.0]))])

    constraints = [
        u >= residual,
        u >= -residual,

        g >= centre_residual,
        g >= -centre_residual,

        # Exact supremum over the ellipsoid:
        # sup_{||Sigma^{-1/2}(xi-mu)||<=t_hat} |a^T xi - b|
        #   = |a^T mu - b| + t_hat * ||Sigma^{1/2} a||_2
        s >= g + t_hat * cp.norm(sqrt_Sigma_xi.T @ a, 2),
    ]

    objective = cp.Minimize(one_minus_epsilon * (1.0 / m) * cp.sum(u) + epsilon * s)
    prob = cp.Problem(objective, constraints)

    return prob


def get_erm_lad_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    ERM baseline: minimise mean absolute deviation on TRAIN.
    """
    d = int(dim)
    n = int(num_samples)
    if d <= 0 or n <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    X_train = cp.Parameter((n, d), name="X_train")
    y_train = cp.Parameter((n,), name="y_train")

    x = cp.Variable(d + 1, name="x")
    w = x[:d]
    b = x[d]

    u = cp.Variable(n, nonneg=True, name="u")
    residual = y_train - (X_train @ w + b)

    constraints = [
        u >= residual,
        u >= -residual,
    ]

    objective = cp.Minimize((1.0 / n) * cp.sum(u))
    prob = cp.Problem(objective, constraints)

    return prob


def get_cvar_lad_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    CVaR baseline at tail-mass epsilon (provided via cvar_coeff = 1/(epsilon*n)):

      min_{w,b,eta,z,u}  eta + (1/(epsilon*n)) * sum_i z_i
      s.t. u_i >= |y_i - (w^T x_i + b)|
           z_i >= u_i - eta
           z_i >= 0
    """
    d = int(dim)
    n = int(num_samples)
    if d <= 0 or n <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    X_train = cp.Parameter((n, d), name="X_train")
    y_train = cp.Parameter((n,), name="y_train")
    cvar_coeff = cp.Parameter(nonneg=True, name="cvar_coeff")  # set to 1/(tail_mass*n)

    x = cp.Variable(d + 1, name="x")
    w = x[:d]
    b = x[d]

    u = cp.Variable(n, nonneg=True, name="u")
    eta = cp.Variable(name="eta")
    z = cp.Variable(n, nonneg=True, name="z")

    residual = y_train - (X_train @ w + b)

    constraints = [
        u >= residual,
        u >= -residual,
        z >= u - eta,
    ]

    objective = cp.Minimize(eta + cvar_coeff * cp.sum(z))
    prob = cp.Problem(objective, constraints)

    return prob


def get_max_lad_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    epsilon = 0 special case for CVaR: minimise the maximum absolute deviation (Chebyshev).

      min_{w,b,t,u} t
      s.t. u_i >= |residual_i|
           t >= u_i  for all i
    """
    d = int(dim)
    n = int(num_samples)
    if d <= 0 or n <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    X_train = cp.Parameter((n, d), name="X_train")
    y_train = cp.Parameter((n,), name="y_train")

    x = cp.Variable(d + 1, name="x")
    w = x[:d]
    b = x[d]

    u = cp.Variable(n, nonneg=True, name="u")
    t = cp.Variable(nonneg=True, name="t")

    residual = y_train - (X_train @ w + b)

    constraints = [
        u >= residual,
        u >= -residual,
        t >= u,
    ]

    objective = cp.Minimize(t)
    prob = cp.Problem(objective, constraints)

    return prob


def get_erm_ridge_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    ERM baseline: Ridge regression (squared loss + L2 penalty on w):

      min_{w,b}  (1/n) * sum_i (y_i - (w^T x_i + b))^2  +  ridge_lambda * ||w||_2^2

    Notes:
      * We do NOT regularise the intercept b.
      * No y-standardisation occurs here; y is used as provided.
    """
    d = int(dim)
    n = int(num_samples)
    if d <= 0 or n <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    X_train = cp.Parameter((n, d), name="X_train")
    y_train = cp.Parameter((n,), name="y_train")
    ridge_lambda = cp.Parameter(nonneg=True, name="ridge_lambda")

    x = cp.Variable(d + 1, name="x")
    w = x[:d]
    b = x[d]

    residual = y_train - (X_train @ w + b)

    objective = cp.Minimize(
        (1.0 / n) * cp.sum_squares(residual)
        + ridge_lambda * cp.sum_squares(w)
    )
    prob = cp.Problem(objective)

    return prob

def get_wass_lad_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    Wasserstein DRO baseline (1-Wasserstein ball) for LAD regression.
    """
    d = int(dim)
    n = int(num_samples)
    if d <= 0 or n <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    X_train = cp.Parameter((n, d), name="X_train")
    y_train = cp.Parameter((n,), name="y_train")
    wass_rho = cp.Parameter(nonneg=True, name="wass_rho")

    x = cp.Variable(d + 1, name="x")
    w = x[:d]
    b = x[d]

    u = cp.Variable(n, nonneg=True, name="u")
    residual = y_train - (X_train @ w + b)

    constraints = [
        u >= residual,
        u >= -residual,
    ]

    y_transport_coeff = cp.Parameter(pos=True, name="y_transport_coeff")
    objective = cp.Minimize((1.0 / n) * cp.sum(u) + wass_rho * cp.norm(cp.hstack([w, y_transport_coeff]), 2))
    prob = cp.Problem(objective, constraints)
    return prob


def get_lv_bas_ellipsoid_xi_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    Full ellipsoid in xi=(x,y).

    Alias for `get_lv_bas_ch_problem` (current LV-BAS-CH) for naming symmetry with the other geometries.
    """
    return get_lv_bas_ch_problem(dim=int(dim), num_samples=int(num_samples))

def get_lv_bas_box_xi_problem(dim: int, num_samples: int) -> cp.Problem:
    """
    LV-BAS with an axis-aligned box bulk set in xi=(x,y).

    Bulk set:
        Xi0 = { (x,y): |x_j - mu_{x,j}| <= r_{x,j} (j=1..d),  |y - mu_y| <= r_y }.

    DKW score on SELECT:
        s_i = ||diag(1/q)(xi_i - mu)||_inf,  with r = t_hat * q.

    Worst-case absolute loss (closed form):
        sup_{xi in Xi0} |y - (w^T x + b)|
        = |mu_y - mu_x^T w - b| + sum_j r_{x,j} |w_j| + r_y.
    """
    d = int(dim)
    m = int(num_samples)
    if d <= 0 or m <= 0:
        raise ValueError("dim and num_samples must be positive.")
    if d != 8:
        raise ValueError("California Housing expects dim=8 (number of features).")

    # Parameters (SAA term)
    X_saa = cp.Parameter((m, d), name="X_saa")
    y_saa = cp.Parameter((m,), name="y_saa")

    epsilon = cp.Parameter(nonneg=True, name="epsilon")
    one_minus_epsilon = cp.Parameter(nonneg=True, name="one_minus_epsilon")

    # Parameters (bulk set)
    mu_x = cp.Parameter((d,), name="mu_x")
    mu_y = cp.Parameter(name="mu_y")
    r_x = cp.Parameter((d,), nonneg=True, name="r_x")
    r_y = cp.Parameter(nonneg=True, name="r_y")

    # Variables
    x = cp.Variable(d + 1, name="x")          # [w; b]
    w = x[:d]
    b = x[d]

    u = cp.Variable(m, nonneg=True, name="u")  # |residual_i|
    g = cp.Variable(nonneg=True, name="g")     # |mu_y - (mu_x^T w + b)|
    s = cp.Variable(nonneg=True, name="s")     # sup_{xi in Xi0} |y - (w^T x + b)|

    residual = y_saa - (X_saa @ w + b)
    centre_residual = mu_y - (mu_x @ w + b)

    abs_w = cp.Variable(d, nonneg=True, name="abs_w")

    constraints = [
        # SAA absolute deviations
        u >= residual,
        u >= -residual,

        # |mu_y - (mu_x^T w + b)|
        g >= centre_residual,
        g >= -centre_residual,

        # abs(w)
        abs_w >= w,
        abs_w >= -w,

        # Worst-case over the box
        s >= g + cp.sum(cp.multiply(r_x, abs_w)) + r_y,
    ]

    objective = cp.Minimize(one_minus_epsilon * (1.0 / m) * cp.sum(u) + epsilon * s)
    prob = cp.Problem(objective, constraints)
    return prob