# KRR_methods/algorithms/length_selection.py
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import minimize_scalar

from ..kernels import tensor_product_kernel  # type: ignore  # circular-safe import


# ----------------------------------------------------------------------
# Nyström + LOOCV utilities for tuning beta (and optionally length-scales)
#
# This slimmed file keeps only what the length-selection notebook uses:
#   - make_nystrom_features_tensor_product
#   - tune_beta_loocv_krr_nystrom
#   - tune_length2d_and_beta_loocv_krr_nystrom
# ----------------------------------------------------------------------


def make_nystrom_features_tensor_product(
    Xss: np.ndarray,
    Ts: np.ndarray,
    m: int = 700,
    kernel_type_f: str = "matern",
    ell_x: float = 8.0,
    nu_x: float = 1.5,
    ell_t: float = 2.0,
    nu_t: float = 1.5,
    random_state: int = 0,
):
    """
    Construct Nyström features Phi for the tensor-product kernel on Z=(Xss, Ts).

    Steps
    -----
    1) Form Z_full = [Xss, Ts].
    2) Sample m landmark points to build K_mm and K_nm.
    3) Compute eigendecomposition of K_mm and return the scaled Nyström features:

         Phi = sqrt(m/n) * K_nm * K_mm^{-1/2}

    Returns
    -------
    Phi : (n, m) array
        Nyström feature matrix.
    Z_landmarks : (m, d+1) array
        Landmark (inducing) points used in the approximation.
    eigvals_mm, eigvecs_mm :
        Eigenpairs of K_mm (returned for debugging / reuse).

    Args:
        Xss: Covariates, shape (n, d).
        Ts: Treatment values, shape (n,).
        m: Number of landmarks (capped at n).
        kernel_type_f: Base kernel type used for both X and T.
        ell_x, nu_x: Length-scale and smoothness for the X kernel.
        ell_t, nu_t: Length-scale and smoothness for the T kernel.
        random_state: RNG seed for landmark selection.
    """
    Xss = np.asarray(Xss, float)
    Ts = np.asarray(Ts, float).ravel()
    n = Xss.shape[0]

    Z_full = np.hstack([Xss, Ts.reshape(-1, 1)])

    m = min(m, n)
    rng = np.random.RandomState(random_state)
    landmark_idx = rng.choice(n, size=m, replace=False)
    Z_landmarks = Z_full[landmark_idx]

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
        Z_full,
        Z_landmarks,
        kernel_type_f,
        ell_x,
        nu_x,
        ell_t,
        nu_t,
    )

    eigvals_mm, eigvecs_mm = np.linalg.eigh(K_mm)

    eigvals_clamped = np.maximum(eigvals_mm, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(eigvals_clamped)

    scale = np.sqrt(m / float(n))
    Phi = scale * (K_nm @ (eigvecs_mm * inv_sqrt))

    return Phi, Z_landmarks, eigvals_mm, eigvecs_mm


def tune_beta_loocv_krr_nystrom(
    Xss,
    Ts,
    Ys,
    m: int = 700,
    kernel_type_f: str = "matern",
    ell_x: float = 8.0,
    nu_x: float = 1.5,
    ell_t: float = 2.0,
    nu_t: float = 1.5,
    beta_bounds=(1e-4, 1e2),
    random_state: int = 0,
):
    """
    Tune the ridge parameter beta for Nyström-approximated KRR using LOOCV.

    For fixed (ell_x, ell_t, nu_x, nu_t), this:
      1) Builds Nyström features Phi for the tensor-product kernel.
      2) Computes SVD(Phi) for efficient LOOCV evaluation.
      3) Minimizes LOOCV loss over beta in `beta_bounds` using bounded search.

    Returns
    -------
    beta_star : float
        LOOCV-optimal beta (feature-space ridge penalty).
    lambda_star : float
        Corresponding lambda = beta_star / n (often reported as KRR lambda).
    loocv_mse_star : float
        LOOCV objective value at beta_star.
    Phi, U, s, Vt :
        Feature matrix and its SVD components (returned for reuse/debugging).

    Args:
        Xss, Ts, Ys: Training data arrays.
        m: Number of Nystrom landmarks.
        kernel_type_f: Base kernel type used for both X and T.
        ell_x, nu_x: Length-scale and smoothness for the X kernel.
        ell_t, nu_t: Length-scale and smoothness for the T kernel.
        beta_bounds: (min, max) bounds for beta search.
        random_state: RNG seed for landmark selection.
    """
    Xss = np.asarray(Xss, float)
    Ts = np.asarray(Ts, float).ravel()
    Ys = np.asarray(Ys, float).ravel()
    n = Ys.shape[0]

    Phi, Z_landmarks, eigvals_mm, eigvecs_mm = make_nystrom_features_tensor_product(
        Xss,
        Ts,
        m=m,
        kernel_type_f=kernel_type_f,
        ell_x=ell_x,
        nu_x=nu_x,
        ell_t=ell_t,
        nu_t=nu_t,
        random_state=random_state,
    )

    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)

    Uy = U.T @ Ys
    U_sq = U ** 2
    s2 = s ** 2

    def loocv_loss(beta: float) -> float:
        """
        LOOCV loss for ridge regression in feature space, computed via SVD.

          y_hat(beta) = U diag(s^2/(s^2+beta)) U^T y
          diag(H(beta)) = row-sums of U^2 * (s^2/(s^2+beta))
        """
        if beta <= 0:
            return float("inf")

        s_beta = s2 / (s2 + beta)
        y_hat = U @ (s_beta * Uy)
        diagS = U_sq @ s_beta

        eps = 1e-12
        resid = Ys - y_hat
        loo_resid = resid / (1.0 - diagS + eps)
        return float(np.mean(loo_resid ** 2))

    res = minimize_scalar(
        loocv_loss,
        bounds=beta_bounds,
        method="bounded",
    )

    beta_star = float(res.x)
    lambda_star = beta_star / n
    loocv_mse_star = float(loocv_loss(beta_star))

    return beta_star, lambda_star, loocv_mse_star, Phi, U, s, Vt


def tune_length2d_and_beta_loocv_krr_nystrom(
    Xss,
    Ts,
    Ys,
    length_grid_x,
    length_grid_t,
    m: int = 700,
    kernel_type_f: str = "matern",
    nu_x: float = 1.5,
    nu_t: float = 1.5,
    beta_bounds=(1e-4, 1e2),
    random_state: int = 0,
) -> Dict[str, object]:
    """
    Grid search over (ell_x, ell_t) and LOOCV-tune beta for each pair, using
    Nyström features for scalability.

    Returns a dictionary containing:
      - best (ell_x, ell_t, beta, lambda, mse)
      - full grids of beta*, lambda*, and LOOCV MSE* values

    Args:
        Xss, Ts, Ys: Training data arrays.
        length_grid_x: Candidate ell_x values.
        length_grid_t: Candidate ell_t values.
        m: Number of Nystrom landmarks.
        kernel_type_f: Base kernel type used for both X and T.
        nu_x, nu_t: Matern smoothness parameters.
        beta_bounds: (min, max) bounds for beta search.
        random_state: RNG seed for landmark selection.
    """
    Xss = np.asarray(Xss, float)
    Ts = np.asarray(Ts, float).ravel()
    Ys = np.asarray(Ys, float).ravel()

    length_grid_x = np.asarray(length_grid_x, float)
    length_grid_t = np.asarray(length_grid_t, float)

    Lx = len(length_grid_x)
    Lt = len(length_grid_t)

    beta_stars = np.zeros((Lx, Lt), dtype=float)
    lambda_stars = np.zeros((Lx, Lt), dtype=float)
    mse_stars = np.zeros((Lx, Lt), dtype=float)

    for ix, ell_x in enumerate(length_grid_x):
        for it, ell_t in enumerate(length_grid_t):
            beta_star, lambda_star, loocv_mse_star, Phi, U, s, Vt = tune_beta_loocv_krr_nystrom(
                Xss,
                Ts,
                Ys,
                m=m,
                kernel_type_f=kernel_type_f,
                ell_x=float(ell_x),
                nu_x=nu_x,
                ell_t=float(ell_t),
                nu_t=nu_t,
                beta_bounds=beta_bounds,
                random_state=random_state,
            )

            beta_stars[ix, it] = beta_star
            lambda_stars[ix, it] = lambda_star
            mse_stars[ix, it] = loocv_mse_star

    # Select the best (ell_x, ell_t) pair by the minimum LOOCV MSE
    best_flat_idx = int(np.argmin(mse_stars))
    best_ix, best_it = np.unravel_index(best_flat_idx, mse_stars.shape)

    best_length_x = float(length_grid_x[best_ix])
    best_length_t = float(length_grid_t[best_it])
    best_beta = float(beta_stars[best_ix, best_it])
    best_lambda = float(lambda_stars[best_ix, best_it])
    best_mse = float(mse_stars[best_ix, best_it])

    print(
        f"Best ell_x = {best_length_x}, "
        f"ell_t = {best_length_t}, "
        f"beta* = {best_beta}, "
        f"lambda* = {best_lambda}, "
        f"LOOCV MSE = {best_mse}",
    )

    return {
        "best_length_x": best_length_x,
        "best_length_t": best_length_t,
        "best_beta": best_beta,
        "best_lambda": best_lambda,
        "best_mse": best_mse,
        "length_grid_x": length_grid_x,
        "length_grid_t": length_grid_t,
        "beta_stars": beta_stars,
        "lambda_stars": lambda_stars,
        "mse_stars": mse_stars,
    }



# ============================================================
# Joint-kernel (X,T) full KRR: LOOCV tuning via eigendecomposition
# ============================================================

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from math import sqrt
from scipy.optimize import minimize_scalar

try:
    import pandas as pd
    ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame"]
except Exception:
    pd = None
    ArrayLike = np.ndarray


def _pairwise_dist(Z: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distance matrix.

    Args:
        Z: Array of shape (n, d) or (n,).

    Returns:
        D: Distance matrix of shape (n, n).
    """
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    G = Z @ Z.T
    sq = np.sum(Z * Z, axis=1, keepdims=True)
    D2 = np.maximum(sq + sq.T - 2.0 * G, 0.0)
    return np.sqrt(D2, dtype=Z.dtype)


def _matern_c(nu: float) -> float:
    """
    Scaling constant in z = c(nu) * r / ell for half-integer Matérn kernels.
    """
    if abs(nu - 0.5) < 1e-8:
        return 1.0
    if abs(nu - 1.5) < 1e-8:
        return sqrt(3.0)
    if abs(nu - 2.5) < 1e-8:
        return sqrt(5.0)
    raise ValueError("nu must be 0.5, 1.5 or 2.5")


def _matern_from_dist(D: np.ndarray, nu: float, ell: float) -> np.ndarray:
    """
    Matérn kernel from pairwise distances (nu in {0.5, 1.5, 2.5}).
    """
    ell = float(ell) if ell is not None else 1.0
    if ell <= 0:
        ell = 1e-9

    c = _matern_c(nu)
    Z = (c * D) / ell

    if abs(nu - 0.5) < 1e-8:
        return np.exp(-Z)
    elif abs(nu - 1.5) < 1e-8:
        return (1.0 + Z) * np.exp(-Z)
    else:
        return (1.0 + Z + (Z**2) / 3.0) * np.exp(-Z)


def _gaussian_from_dist(D: np.ndarray, ell: float) -> np.ndarray:
    """
    Gaussian (RBF) kernel from pairwise distances: exp(-r^2 / (2 ell^2)).
    """
    ell = float(ell) if ell is not None else 1.0
    if ell <= 0:
        ell = 1e-9
    Z2 = (D / ell) ** 2
    return np.exp(-0.5 * Z2)


def _loocv_from_kernel_eig(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    y: np.ndarray,
    beta_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Given K = Q diag(e) Q^T and y,
    tune beta by LOOCV using closed-form LOO residual formula.

    Args:
        eigvals, eigvecs: Eigenpairs of K.
        y: Target vector.
        beta_bounds: (min, max) bounds for beta search.

    Returns:
        (beta_star, loocv_mse_star)
    """
    e = np.maximum(np.asarray(eigvals, float), 1e-12)
    Q = np.asarray(eigvecs, float)
    y = np.asarray(y, float).reshape(-1)

    Qt_y = Q.T @ y
    Q_sq = Q ** 2

    def loocv_loss(beta: float) -> float:
        if beta <= 0:
            return float("inf")

        s_beta = e / (e + beta)
        y_hat = Q @ (s_beta * Qt_y)
        diagS = Q_sq @ s_beta

        eps = 1e-12
        resid = y - y_hat
        loo_resid = resid / (1.0 - diagS + eps)
        return float(np.mean(loo_resid ** 2))

    res = minimize_scalar(loocv_loss, bounds=beta_bounds, method="bounded")
    beta_star = float(res.x)
    return beta_star, float(loocv_loss(beta_star))


def krr_length_selection_loocv_joint(
    Xs: ArrayLike,
    Ts: ArrayLike,
    Ys: ArrayLike,
    nu_list: List[float] = (1.5, 2.5),
    ell_list: List[float] = (2, 4, 6, 8, 10, 12, 14, 16, 18),
    beta_bounds: Tuple[float, float] = (1e-4, 1e2),
    kernel_type: str = "matern",  # "matern" or "gaussian"
) -> Dict[str, object]:
    """
    LOOCV length-scale + ridge selection for a *joint* kernel on (X, T).
    Full kernel (no Nyström): O(n^3) eigendecomposition per (nu, ell).

    Args:
        Xs, Ts, Ys: Training data arrays.
        nu_list: Candidate smoothness parameters (Matérn only).
        ell_list: Candidate length-scales.
        beta_bounds: (min, max) bounds for beta search.
        kernel_type: "matern" or "gaussian".

    Returns:
        Dictionary containing per-grid results and the best configuration.
    """
    kernel_type = kernel_type.lower()
    if kernel_type not in ("matern", "gaussian"):
        raise ValueError("kernel_type must be 'matern' or 'gaussian'")

    # cast inputs
    if pd is not None and isinstance(Xs, pd.DataFrame):
        Xv = Xs.to_numpy(dtype=float, copy=True)
    else:
        Xv = np.asarray(Xs, dtype=float)

    Tv = np.asarray(Ts, dtype=float).reshape(-1, 1)
    y = np.asarray(Ys, dtype=float).reshape(-1)

    n = Xv.shape[0]
    if Tv.shape[0] != n or y.shape[0] != n:
        raise ValueError("Xs, Ts, Ys must have the same length.")

    # joint feature + distances
    F = np.hstack([Xv, Tv])
    D_full = _pairwise_dist(F)

    rows = []
    winners_per_nu = {}
    per_nu_ellbest = {}

    nu_values: List[object] = list(nu_list) if kernel_type == "matern" else ["gaussian"]

    for nu_label in nu_values:
        ell_best_rows = []
        for ell in ell_list:
            if kernel_type == "matern":
                K_full = _matern_from_dist(D_full, float(nu_label), ell)
            else:
                K_full = _gaussian_from_dist(D_full, ell)

            K_full = 0.5 * (K_full + K_full.T)
            K_full += 1e-10 * np.eye(n)

            eigvals, eigvecs = np.linalg.eigh(K_full)
            beta_star, loocv_mse_star = _loocv_from_kernel_eig(eigvals, eigvecs, y, beta_bounds)

            row = {
                "kernel": kernel_type,
                "nu": nu_label,
                "ell": ell,
                "beta_star": beta_star,
                "loocv_mse": loocv_mse_star,
            }
            rows.append(row)
            ell_best_rows.append(row)

        # store per-nu best
        if pd is None:
            # fallback: return lists if pandas not available
            ell_best_rows_sorted = sorted(ell_best_rows, key=lambda r: r["loocv_mse"])
            winners_per_nu[nu_label] = {"per_ell_best": ell_best_rows_sorted, "best_overall": ell_best_rows_sorted[0]}
            per_nu_ellbest[nu_label] = ell_best_rows_sorted
        else:
            per_ell_best_df = pd.DataFrame(ell_best_rows).sort_values("loocv_mse").reset_index(drop=True)
            winners_per_nu[nu_label] = {"per_ell_best": per_ell_best_df, "best_overall": per_ell_best_df.iloc[0].to_dict()}
            per_nu_ellbest[nu_label] = per_ell_best_df

    # global best across nu
    if pd is None:
        best_candidates = [pack["best_overall"] for pack in winners_per_nu.values()]
        best_global = sorted(best_candidates, key=lambda r: r["loocv_mse"])[0]
        results_table = sorted(rows, key=lambda r: (r["kernel"], str(r["nu"]), r["ell"]))
    else:
        best_candidates = []
        for nu_label, pack in winners_per_nu.items():
            r = pack["best_overall"].copy()
            r["nu"] = nu_label
            best_candidates.append(r)
        best_global = pd.DataFrame(best_candidates).sort_values("loocv_mse").iloc[0].to_dict()
        results_table = pd.DataFrame(rows).sort_values(["kernel", "nu", "ell"]).reset_index(drop=True)

    return {
        "results_table": results_table,
        "per_nu_ellbest": per_nu_ellbest,
        "winners_per_nu": winners_per_nu,
        "best_global": best_global,
    }
