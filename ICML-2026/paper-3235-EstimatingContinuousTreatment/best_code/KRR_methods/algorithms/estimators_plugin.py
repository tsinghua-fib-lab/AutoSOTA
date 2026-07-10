# KRR_methods/algorithms/estimators_plugin.py
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
from scipy.optimize import minimize_scalar

from ..kernels import get_kernel_matrix, tensor_product_kernel
from ..synthetic_dgps import split_data


# ----------------------------------------------------------------------
# Plug-in baselines used in our experiments
# ----------------------------------------------------------------------
# This module contains two plug-in estimators that are actually used by the
# provided experiment notebooks/scripts:
#
#   (1) Nyström + LOOCV plug-in (tensor-product kernel on Z=(X,T))
#       - Tunes the ridge penalty via LOOCV using an efficient SVD-based formula.
#       - Returns h_hat(t) = E_X[ f_hat(X,t) ] evaluated on a user-specified grid.
#
#   (2) Joint ND-kernel plug-in (synthetic experiments)
#       - Fits f_hat(x,t) with an ND kernel on Z=(X,T) using exact KRR (no Nyström).
#       - Tunes the ridge penalty by validation on the second split.
#       - Returns h_hat(t) by averaging f_hat(X,t) over X from the training split.
#
# Note: The earlier validation-split Nyström plug-in variant was removed to keep
# the file minimal for the current notebooks.
# ----------------------------------------------------------------------


def mise_against(
    grid_t: np.ndarray,
    est_vals: np.ndarray,
    h_star_vals: np.ndarray,
) -> float:
    """
    Mean Integrated Squared Error (MISE) computed on a discrete t-grid.

    Note
    ----
    `grid_t` is accepted for API consistency; the computation uses only the
    pointwise squared errors averaged over the grid.
    """
    return float(
        np.mean((np.asarray(est_vals) - np.asarray(h_star_vals)) ** 2)
    )


# ----------------------------------------------------------------------
# Plug-in estimator with Nyström + LOOCV (tensor-product kernel)
# ----------------------------------------------------------------------


def estimate_h_grid_plugin_loocv(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    t_grid: np.ndarray,
    beta_min: float,
    beta_max: float,
    # Stage-1 tensor-product kernel
    kernel_type_f: str = "matern",
    ell_x: float = 8.0,
    nu_x: float = 1.5,
    ell_t: float = 2.0,
    nu_t: float = 1.5,
    m_f: int = 500,
    verbose: bool = False,
) -> np.ndarray:
    """
    Plug-in estimator with Nyström features and LOOCV-based tuning of the ridge
    parameter beta (tensor-product kernel on Z=(X,T)).

    Notes
    -----
    This cleaned version removes only timing/profiling code and verbose prints.
    All numerical computations (including deterministic landmark selection) are unchanged.

    Args:
        X: Covariates, shape (n, d).
        T: Treatment values, shape (n,).
        Y: Outcomes, shape (n,).
        t_grid: Evaluation grid for h_hat(t), shape (m,).
        beta_min, beta_max: Search bounds for the ridge parameter.
        kernel_type_f: Base kernel type used for both X and T (e.g., "matern").
        ell_x, nu_x: Length-scale and smoothness for the X kernel.
        ell_t, nu_t: Length-scale and smoothness for the T kernel.
        m_f: Number of Nystrom landmarks.
        verbose: Kept for API compatibility (no extra prints here).

    Returns:
        h_grid: Estimated h_hat(t) evaluated on t_grid.
    """
    # NOTE: `verbose` is kept for API compatibility with notebooks,
    # but this cleaned version does not print timing/profiling logs.

    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    Z = np.hstack([X, T.reshape(-1, 1)])
    n = len(Y)

    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_min <= 0:
        beta_min = 1e-8
    if beta_max <= beta_min:
        beta_max = beta_min * 10.0

    # Nyström landmarks (fixed RNG for determinism across runs)  <-- DO NOT CHANGE
    m = min(m_f, n)
    rng = np.random.RandomState(0)
    landmark_idx = rng.choice(n, size=m, replace=False)
    Z_landmarks = Z[landmark_idx]

    # Build K_mm and K_nm for Nyström feature construction
    K_mm = tensor_product_kernel(
        Z_landmarks,
        Z_landmarks,
        kernel_type_f,
        ell_x,
        nu_x,
        ell_t,
        nu_t,
    )
    K_mm = 0.5 * (K_mm + K_mm.T)
    K_mm += 1e-8 * np.eye(m)

    K_nm = tensor_product_kernel(
        Z,
        Z_landmarks,
        kernel_type_f,
        ell_x,
        nu_x,
        ell_t,
        nu_t,
    )

    # Eigendecomposition of K_mm
    eigvals_mm, eigvecs_mm = np.linalg.eigh(K_mm)
    eigvals_clamped = np.maximum(eigvals_mm, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(eigvals_clamped)

    # Standard Nyström feature construction (with sqrt(m/n) scaling)
    scale = np.sqrt(m / float(n))
    Phi = scale * (K_nm @ (eigvecs_mm * inv_sqrt))

    def phi_fn(Z_eval: np.ndarray) -> np.ndarray:
        K_eval_m = tensor_product_kernel(
            Z_eval,
            Z_landmarks,
            kernel_type_f,
            ell_x,
            nu_x,
            ell_t,
            nu_t,
        )
        return scale * (K_eval_m @ (eigvecs_mm * inv_sqrt))

    # SVD of Phi for efficient LOOCV evaluations
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)

    Uy = U.T @ Y
    U_sq = U ** 2
    s2 = s ** 2

    def loocv_loss(beta: float) -> float:
        if beta <= 0:
            return float("inf")

        s_beta = s2 / (s2 + beta)
        y_hat = U @ (s_beta * Uy)
        diagS = U_sq @ s_beta

        eps = 1e-12
        resid = Y - y_hat
        loo_resid = resid / (1.0 - diagS + eps)
        return float(np.mean(loo_resid ** 2))

    # Use the user-specified bounds directly as the search interval
    lower = max(beta_min, 1e-8)
    upper = max(beta_max, lower * 1.0001)

    res_opt = minimize_scalar(
        loocv_loss,
        bounds=(lower, upper),
        method="bounded",
    )
    beta_star = float(res_opt.x)

    # Closed-form ridge solution in SVD coordinates:
    # w = V diag(s/(s^2+beta)) U^T y
    coeff_factor = s / (s2 + beta_star)
    w_star = Vt.T @ (coeff_factor * Uy)

    t_grid = np.asarray(t_grid, dtype=float)

    # Evaluate h_hat(t) by averaging predictions over the observed X
    def mean_over_X_at_t(t_s: float) -> float:
        Z_eval = np.hstack([X, np.full((X.shape[0], 1), t_s)])
        Phi_eval = phi_fn(Z_eval)
        preds = Phi_eval @ w_star
        return float(np.mean(preds))

    h_grid = np.array([mean_over_X_at_t(t_s) for t_s in t_grid])
    return h_grid

# ----------------------------------------------------------------------
# Convenience wrapper (Nyström + LOOCV plug-in only)
# ----------------------------------------------------------------------


def run_plugin_loocv_on_original_grid(
    Xss,
    T,
    Y,
    t_grid_original,
    h_star_vals_original,
    beta_min: float,
    beta_max: float,
    kernel_type_f,
    ell_x,
    nu_x,
    ell_t,
    nu_t,
    m_f,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - Runs `estimate_h_grid_plugin_loocv` on the original t-grid
    - Returns both the estimated curve and its MISE against h*(t)

    The LOOCV ridge search interval is [beta_min, beta_max].

    Args:
        Xss, T, Y: Data for a single run (X can be a DataFrame or array).
        t_grid_original: Grid for evaluating h_hat(t).
        h_star_vals_original: Ground-truth h*(t) on the same grid.
        beta_min, beta_max: Ridge search bounds.
        kernel_type_f, ell_x, nu_x, ell_t, nu_t: Kernel hyperparameters.
        m_f: Number of Nystrom landmarks.
        verbose: Kept for API compatibility.

    Returns:
        dict with keys:
        - "h_plugin": estimated curve on t_grid_original
        - "mise_plugin": MISE against h_star_vals_original
    """
    X_arr = Xss.values if hasattr(Xss, "values") else np.asarray(Xss, float)
    T_arr = np.asarray(T, float).ravel()
    Y_arr = np.asarray(Y, float).ravel()

    h_plugin_loocv = estimate_h_grid_plugin_loocv(
        X_arr,
        T_arr,
        Y_arr,
        t_grid_original,
        beta_min=beta_min,
        beta_max=beta_max,
        kernel_type_f=kernel_type_f,
        ell_x=ell_x,
        nu_x=nu_x,
        ell_t=ell_t,
        nu_t=nu_t,
        m_f=m_f,
        verbose=verbose,
    )

    mise_plugin = mise_against(t_grid_original, h_plugin_loocv, h_star_vals_original)

    return {
        "h_plugin": h_plugin_loocv,
        "mise_plugin": mise_plugin,
    }


# ======================================================================
# Joint ND-kernel plug-in KRR for synthetic experiments (no Nyström)
# ======================================================================


def _solve_krr_eig_joint(
    K: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    jitter: float = 1e-8,
) -> List[np.ndarray]:
    """
    Solve (K + lambda I) alpha = y for multiple lambda values using a single
    eigendecomposition of K (joint-kernel plug-in baseline).
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float).ravel()

    evals, evecs = np.linalg.eigh(K)
    evals = np.maximum(evals, jitter)
    Uy = evecs.T @ y

    alphas_list: List[np.ndarray] = []
    for lam in lambdas:
        denom = evals + float(lam)
        alphas = evecs @ (Uy / denom)
        alphas_list.append(alphas)
    return alphas_list


def run_plugin_on_original_grid_joint(
    D1,
    D2,
    beta_grid_for_f,
    length_scale_nd: float = 3.0,
    nu_nd: float = 1.5,
    kernel_type_nd: str = "matern",
) -> Dict[str, Any]:
    """
    Baseline plug-in KRR with a joint kernel on Z = (X, T).

    Given two splits (D1, D2):
      - Fit KRR for f(x,t) on D1 across a ridge grid.
      - Select the ridge by validation MSE on D2.
      - Return h_hat(t) = E_X[f_hat(X,t)] computed by empirical averaging over X1.

    Args:
        D1, D2: Tuples (X, T, Y) for training and validation splits.
        beta_grid_for_f: Candidate ridge values.
        length_scale_nd, nu_nd, kernel_type_nd: Joint kernel hyperparameters.

    Returns:
        dict with:
        - h_hat_joint: callable for h_hat(t)
        - best_beta: selected ridge parameter
    """
    X1, T1, Y1 = D1
    X2, T2, Y2 = D2

    X1 = np.asarray(X1, float)
    T1 = np.asarray(T1, float).ravel()
    Y1 = np.asarray(Y1, float).ravel()

    X2 = np.asarray(X2, float)
    T2 = np.asarray(T2, float).ravel()
    Y2 = np.asarray(Y2, float).ravel()

    Z1 = np.hstack([X1, T1.reshape(-1, 1)])
    Z2 = np.hstack([X2, T2.reshape(-1, 1)])

    beta_grid = np.asarray(beta_grid_for_f, float).ravel()

    # Gram matrix on D1 and eigen-based solves for all ridge values
    K_11 = get_kernel_matrix(
        Z1,
        Z1,
        length_scale=length_scale_nd,
        kernel_type=kernel_type_nd,
        nu=nu_nd,
    )
    alphas_list = _solve_krr_eig_joint(K_11, Y1, beta_grid)

    # Validation kernel: D2 against D1
    K_21 = get_kernel_matrix(
        Z2,
        Z1,
        length_scale=length_scale_nd,
        kernel_type=kernel_type_nd,
        nu=nu_nd,
    )

    best_loss = np.inf
    best_alpha = None
    best_beta = None

    # Pick beta minimizing validation MSE on D2
    for alpha, beta_f in zip(alphas_list, beta_grid):
        preds = K_21 @ alpha
        loss = float(np.mean((preds - Y2) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_beta = float(beta_f)

    if best_alpha is None:
        raise RuntimeError("No valid alpha found in run_plugin_on_original_grid_joint.")

    # Prediction closure for the selected f_hat
    def _predict_f_joint(Z_new: np.ndarray) -> np.ndarray:
        Z_new = np.asarray(Z_new, float)
        K_new = get_kernel_matrix(
            Z_new,
            Z1,
            length_scale=length_scale_nd,
            kernel_type=kernel_type_nd,
            nu=nu_nd,
        )
        return K_new @ best_alpha

    # Final h_hat(t) = E_{X1}[ f_hat(X1, t) ] implemented by empirical averaging
    def h_hat_joint(t_array: np.ndarray) -> np.ndarray:
        t_array = np.asarray(t_array, float).ravel()
        out = []
        for t_val in t_array:
            ZT = np.hstack([X1, np.full((X1.shape[0], 1), t_val)])
            out.append(_predict_f_joint(ZT).mean())
        return np.asarray(out, dtype=float)

    return {
        "h_hat_joint": h_hat_joint,
        "best_beta": best_beta,
    }


# ----------------------------------------------------------------------
# Synthetic helper: single-run wrapper used by Demo_synthetic
# ----------------------------------------------------------------------


def run_single_plugin_synthetic(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    t_grid: np.ndarray,
    *,
    beta_grid_for_f: np.ndarray,
    nu_nd: float,
    length_scale_nd: float,
    kernel_type_nd: str,
) -> tuple[np.ndarray, float]:
    """
    Synthetic single-run wrapper for the joint-kernel plug-in estimator.
    Hyperparameters are passed in from the notebook for easy experimentation.

    Returns:
        (h_hat_grid, beta_selected)
    """
    D1, D2 = split_data(X, T, Y)
    out = run_plugin_on_original_grid_joint(
        D1=D1,
        D2=D2,
        beta_grid_for_f=beta_grid_for_f,
        nu_nd=nu_nd,
        length_scale_nd=length_scale_nd,
        kernel_type_nd=kernel_type_nd,
    )
    h_hat_vals = out["h_hat_joint"](t_grid)
    beta_sel = float(out["best_beta"])
    return h_hat_vals, beta_sel


# ----------------------------------------------------------------------
# Semi-real helper: single-run wrapper used by Demo_semi-real
# ----------------------------------------------------------------------


def run_single_plugin_semireal(
    Xss: np.ndarray,
    Ts: np.ndarray,
    Ys: np.ndarray,
    *,
    t_grid_original: np.ndarray,
    h_star_vals_original: np.ndarray,
    beta_min: float,
    beta_max: float,
    kernel_type_f: str,
    ell_x: float,
    nu_x: float,
    ell_t: float,
    nu_t: float,
    m_f: int,
    verbose: bool = False,
):
    """
    Semi-real single-run wrapper for the plug-in estimator (Nyström + LOOCV).
    Hyperparameters are passed in from the notebook for easy experimentation.

    Returns:
        dict from run_plugin_loocv_on_original_grid.
    """
    return run_plugin_loocv_on_original_grid(
        Xss,
        Ts,
        Ys,
        t_grid_original=t_grid_original,
        h_star_vals_original=h_star_vals_original,
        beta_min=beta_min,
        beta_max=beta_max,
        kernel_type_f=kernel_type_f,
        ell_x=ell_x,
        nu_x=nu_x,
        ell_t=ell_t,
        nu_t=nu_t,
        m_f=m_f,
        verbose=verbose,
    )
