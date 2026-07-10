from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

from ..kernels import get_kernel_matrix
from ..synthetic_dgps import split_data

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
# - _solve_krr_eig: solves many KRR ridge values efficiently via eigendecomposition
# - mise_against: computes mean integrated squared error on a provided grid
# ----------------------------------------------------------------------


def _solve_krr_eig(
    K: np.ndarray,
    y: np.ndarray,
    lambdas: List[float],
    jitter: float = 1e-8,
) -> List[np.ndarray]:
    """
    Solve (K + lambda I) alpha = y for multiple lambda values using a single
    eigendecomposition of K.

    Parameters
    ----------
    K : (n, n) array
        Kernel (Gram) matrix.
    y : (n,) array
        Targets.
    lambdas : list of float
        Ridge parameters to solve for.
    jitter : float
        Floors eigenvalues at `jitter` for numerical stability.

    Returns
    -------
    alphas_list : list of (n,) arrays
        Solution vectors alpha for each lambda in `lambdas`.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    evals, evecs = np.linalg.eigh(K)
    evals = np.maximum(evals, jitter)
    Uy = evecs.T @ y

    alphas_list: List[np.ndarray] = []
    for lam in lambdas:
        lam = float(lam)
        denom = evals + lam
        alphas = evecs @ (Uy / denom)
        alphas_list.append(alphas)
    return alphas_list


def mise_against(
    grid_t: np.ndarray,
    est_vals: np.ndarray,
    h_star_vals: np.ndarray,
) -> float:
    """
    Mean Integrated Squared Error (MISE) computed on a discrete t-grid.

    Note: `grid_t` is accepted for API consistency; the computation uses only
    pointwise squared errors averaged over the grid.
    """
    return float(np.mean((np.asarray(est_vals) - np.asarray(h_star_vals)) ** 2))


# ----------------------------------------------------------------------
# Our estimator: two-stage KRR (no Nyström), tensor-product kernel
# ----------------------------------------------------------------------
# Stage 1: fit f̂(x, t) with a tensor-product kernel K_f = K_x * K_t
# Stage 2: estimate h(t) by smoothing the marginal mean curve m(t)=E_X[f̂(X,t)]
#          with a 1D KRR on t, selecting the ridge parameter via sample splitting
# ----------------------------------------------------------------------


def estimate_h_grid_tensor(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    t_grid: np.ndarray,
    beta0_f: float = 0.05,
    beta_grid: np.ndarray | None = None,
    beta0_prime_f: float = 0.05,
    # Stage-1 f(x,t) kernel
    kernel_type_f: str = "matern",
    ell_x: float = 8.0,
    nu_x: float = 1.5,
    ell_t: float = 2.0,
    nu_t: float = 1.5,
    # Stage-2 h(t) kernel
    kernel_type_H: str = "matern",
    l_H: float = 2.0,
    nu_H: float = 2.5,
    # Second-stage sampling
    second_stage_range: tuple[float, float] = (0.0, 3000.0),
    second_stage_n: int = 2000,
) -> np.ndarray:
    """
    Estimate h(t) on the original (unscaled) treatment grid `t_grid` using the
    proposed two-stage procedure.

    High-level algorithm
    --------------------
    1) Randomly split data into D1 and D2.
    2) On D1:
       - Fit f̂(x,t) via KRR with tensor-product kernel K_x(x,x') * K_t(t,t').
       - Sample t_samp1 and compute m_vals1(t) ≈ E_{X1}[ f̂(X1, t) ] efficiently.
       - Fit candidate curves {ĥ_λ} via 1D KRR on (t_samp1, m_vals1).
    3) On D2:
       - Fit f̃(x,t) similarly (with possibly different ridge).
       - Sample t_samp2 and compute m_vals2(t) ≈ E_{X2}[ f̃(X2, t) ].
       - Fit a fixed curve \tilde h via 1D KRR on (t_samp2, m_vals2).
    4) Model selection:
       - Sample fresh t_eval and choose λ minimizing || ĥ_λ(t_eval) - \tilde h(t_eval) ||_2^2.
    5) Return ĥ(t_grid) using the selected λ.

    Notes
    -----
    - T and t_grid are used on the original scale (no rescaling here).

    Args:
        X: Covariates, shape (n, d).
        T: Treatment values, shape (n,).
        Y: Outcomes, shape (n,).
        t_grid: Evaluation grid for h_hat(t).
        beta0_f: Ridge penalty for the stage-1 f_hat fit on D1.
        beta_grid: Candidate ridge values for the stage-2 h_hat fit.
        beta0_prime_f: Ridge penalty for the stage-1 f_tilde fit on D2.
        kernel_type_f, ell_x, nu_x, ell_t, nu_t: Kernel hyperparameters for f_hat.
        kernel_type_H, l_H, nu_H: Kernel hyperparameters for the stage-2 h(t) fit.
        second_stage_range: Range for sampling t values in stage 2.
        second_stage_n: Number of sampled t values in stage 2.

    Returns:
        h_vals: Estimated h_hat(t) evaluated on t_grid.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    if beta_grid is None:
        beta_grid = np.array([0.05 * (2**i) for i in range(1, 9)], dtype=float)
    else:
        beta_grid = np.asarray(beta_grid, dtype=float)

    n = len(Y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    mid = n // 2
    i1, i2 = idx[:mid], idx[mid:]

    X1, T1, Y1 = X[i1], T[i1], Y[i1]
    X2, T2, Y2 = X[i2], T[i2], Y[i2]

    n1 = len(Y1)
    n2 = len(Y2)

    low, high = second_stage_range

    # ----- D1: build candidate curves { ĥ_λ } -----
    # Tensor-product Gram matrix for f̂ on D1: Kf11 = Kx11 ⊙ Kt11
    Kx11 = get_kernel_matrix(X1, X1, ell_x, kernel_type_f, nu_x)
    Kt11 = get_kernel_matrix(
        T1.reshape(-1, 1),
        T1.reshape(-1, 1),
        ell_t,
        kernel_type_f,
        nu_t,
    )
    Kf11 = Kx11 * Kt11

    # Solve for f̂ coefficients at ridge beta0_f
    alpha1 = _solve_krr_eig(Kf11, Y1, [beta0_f])[0]

    # Sample t_samp1 for approximating m(t) = E_X[f̂(X,t)]
    t_samp1 = np.random.uniform(low, high, size=second_stage_n).astype(float)

    # Efficient computation of m_vals1:
    # f̂(x,t) = sum_i alpha1_i Kx(x, X1_i) Kt(t, T1_i)
    # E_{x~empirical(X1)}[f̂(x,t)] = (1/n1) * sum_i alpha1_i (sum_j Kx(X1_j, X1_i)) Kt(t, T1_i)
    s1 = Kx11.sum(axis=0)
    v1 = alpha1 * s1

    Kt_s1 = get_kernel_matrix(
        t_samp1.reshape(-1, 1),
        T1.reshape(-1, 1),
        ell_t,
        kernel_type_f,
        nu_t,
    )
    m_vals1 = (Kt_s1 @ v1) / float(n1)

    # Fit 1D KRR candidates ĥ_λ on (t_samp1, m_vals1)
    K_H_11 = get_kernel_matrix(
        t_samp1.reshape(-1, 1),
        t_samp1.reshape(-1, 1),
        l_H,
        kernel_type_H,
        nu_H,
    )

    alphas_H_list = _solve_krr_eig(K_H_11, m_vals1, beta_grid.tolist())

    # ----- D2: build a fixed curve \tilde h for model selection -----
    Kx22 = get_kernel_matrix(X2, X2, ell_x, kernel_type_f, nu_x)
    Kt22 = get_kernel_matrix(
        T2.reshape(-1, 1),
        T2.reshape(-1, 1),
        ell_t,
        kernel_type_f,
        nu_t,
    )
    Kf22 = Kx22 * Kt22

    alpha2 = _solve_krr_eig(Kf22, Y2, [beta0_prime_f])[0]

    t_samp2 = np.random.uniform(low, high, size=second_stage_n).astype(float)

    s2 = Kx22.sum(axis=0)
    v2 = alpha2 * s2

    Kt_s2 = get_kernel_matrix(
        t_samp2.reshape(-1, 1),
        T2.reshape(-1, 1),
        ell_t,
        kernel_type_f,
        nu_t,
    )
    m_vals2 = (Kt_s2 @ v2) / float(n2)

    K_H_22 = get_kernel_matrix(
        t_samp2.reshape(-1, 1),
        t_samp2.reshape(-1, 1),
        l_H,
        kernel_type_H,
        nu_H,
    )

    # Fit \tilde h using ridge beta0_prime_f (kept as-is)
    alpha_tilde = _solve_krr_eig(K_H_22, m_vals2, [beta0_prime_f])[0]

    # ----- Model selection with fresh evaluation points t_eval -----
    t_eval = np.random.uniform(low, high, size=second_stage_n).astype(float)
    t_eval_col = t_eval.reshape(-1, 1)

    K_H_eval1 = get_kernel_matrix(
        t_eval_col,
        t_samp1.reshape(-1, 1),
        l_H,
        kernel_type_H,
        nu_H,
    )
    K_H_eval2 = get_kernel_matrix(
        t_eval_col,
        t_samp2.reshape(-1, 1),
        l_H,
        kernel_type_H,
        nu_H,
    )

    tilde_vals_eval = K_H_eval2 @ alpha_tilde

    errs = []
    for alpha_H in alphas_H_list:
        pred_eval = K_H_eval1 @ alpha_H
        errs.append(np.mean((pred_eval - tilde_vals_eval) ** 2))

    best_j = int(np.argmin(errs))
    alpha_best = alphas_H_list[best_j]

    # ----- Evaluate the selected ĥ on the user-provided t-grid -----
    t_grid = np.asarray(t_grid, dtype=float)
    t_grid_col = t_grid.reshape(-1, 1)

    K_H_eval_grid = get_kernel_matrix(
        t_grid_col,
        t_samp1.reshape(-1, 1),
        l_H,
        kernel_type_H,
        nu_H,
    )

    h_vals = K_H_eval_grid @ alpha_best
    return h_vals


def run_ours_tensor_kernel(
    Xss,
    T,
    Y,
    t_grid_original,
    h_star_vals_original,
    beta_grid,
    kernel_type_f,
    ell_x,
    nu_x,
    ell_t,
    nu_t,
    beta0_f,
    beta0_prime_f,
    kernel_type_H,
    l_H,
    nu_H,
    second_stage_range,
    second_stage_n,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - Runs `estimate_h_grid_tensor` on the original t-grid
    - Returns both the estimated curve and its MISE against h*(t)

    Args:
        Xss, T, Y: Data for a single run (X can be a DataFrame or array).
        t_grid_original: Grid for evaluating h_hat(t).
        h_star_vals_original: Ground-truth h*(t) on the same grid.
        beta_grid: Candidate ridge values for the stage-2 h(t) fit.
        kernel_type_f, ell_x, nu_x, ell_t, nu_t: Kernel hyperparameters for f_hat.
        beta0_f, beta0_prime_f: Ridge penalties for the two stage-1 fits.
        kernel_type_H, l_H, nu_H: Kernel hyperparameters for the stage-2 fit.
        second_stage_range, second_stage_n: Sampling settings for stage 2.

    Returns:
        dict with keys:
        - "h_ours": estimated curve on t_grid_original
        - "mise_ours": MISE against h_star_vals_original
    """
    X_arr = Xss.values if hasattr(Xss, "values") else np.asarray(Xss, float)
    T_arr = np.asarray(T, float).ravel()
    Y_arr = np.asarray(Y, float).ravel()

    h_ours = estimate_h_grid_tensor(
        X_arr,
        T_arr,
        Y_arr,
        t_grid_original,
        beta0_f=beta0_f,
        beta_grid=beta_grid,
        beta0_prime_f=beta0_prime_f,
        kernel_type_f=kernel_type_f,
        ell_x=ell_x,
        nu_x=nu_x,
        ell_t=ell_t,
        nu_t=nu_t,
        kernel_type_H=kernel_type_H,
        l_H=l_H,
        nu_H=nu_H,
        second_stage_range=second_stage_range,
        second_stage_n=second_stage_n,
    )

    mise_ours = mise_against(t_grid_original, h_ours, h_star_vals_original)

    return {
        "h_ours": h_ours,
        "mise_ours": mise_ours,
    }


# ======================================================================
# Joint ND-kernel "ours" (synthetic experiments, no Nyström)
# ======================================================================
# Variant used in synthetic experiments:
# - Stage 1 fits f̂ on the joint input Z=(X,T) with an ND kernel.
# - The rest (m(t) estimation, 1D KRR for h, sample-splitting selection) mirrors
#   the tensor-product version, but uses explicit prediction calls.
# ======================================================================


def run_ours_joint_kernel(
    D1,
    D2,
    beta0_for_f_cand,
    beta_h_grid_for_h_cand,
    beta0_prime_for_f_tilde,
    nu_nd,
    nu_1d_h,
    length_scale_nd,
    length_scale_1d_h,
    second_stage_N_SIZE,
    kernel_type_nd: str = "matern",
    t_min: float = 0.0,
    t_max: float = 20.0,
    kernel_type_1d_h: str = "matern",
) -> Dict[str, Any]:
    """
    Proposed two-step KRR using a joint ND kernel on Z = (X, T).

    Procedure
    ---------
    1) On D1 = (X1, T1, Y1):
       - Fit f̂ via KRR on Z1 = [X1, T1] with ridge beta0_for_f_cand.
       - Sample t_samp1 in [t_min, t_max], compute m_vals1(t) = E_{X1}[f̂(X1,t)].
       - Fit candidate curves {ĥ_λ} via 1D KRR on (t_samp1, m_vals1).

    2) On D2 = (X2, T2, Y2):
       - Fit f̃ similarly with ridge beta0_prime_for_f_tilde.
       - Sample t_samp2, compute m_vals2, and fit a fixed curve \tilde h.

    3) Model selection:
       - Sample fresh t_eval and pick λ minimizing the L2 discrepancy between
         ĥ_λ(t_eval) and \tilde h(t_eval).

    Returns:
        dict with:
        - "h_hat_func": callable mapping an array of t values to selected ĥ(t)
        - "beta_selected": selected stage-2 ridge parameter

    Args:
        D1, D2: Tuples (X, T, Y) for training and validation splits.
        beta0_for_f_cand: Ridge for the first-stage f_hat fit on D1.
        beta_h_grid_for_h_cand: Candidate ridge values for h_hat.
        beta0_prime_for_f_tilde: Ridge for the first-stage f_tilde fit on D2.
        nu_nd, length_scale_nd, kernel_type_nd: Joint kernel hyperparameters.
        nu_1d_h, length_scale_1d_h, kernel_type_1d_h: 1D kernel hyperparameters.
        second_stage_N_SIZE: Number of sampled t values in stage 2.
        t_min, t_max: Range for sampling t values.
    """
    X1, T1, Y1 = D1
    X2, T2, Y2 = D2

    X1 = np.asarray(X1, float)
    T1 = np.asarray(T1, float).ravel()
    Y1 = np.asarray(Y1, float).ravel()

    X2 = np.asarray(X2, float)
    T2 = np.asarray(T2, float).ravel()
    Y2 = np.asarray(Y2, float).ravel()

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    beta_h_grid_for_h_cand = np.asarray(beta_h_grid_for_h_cand, float).ravel()

    # ---- Stage 1: fit f̂ on D1 and build candidate {ĥ_λ} ----
    Z1 = np.hstack([X1, T1.reshape(-1, 1)])
    K_11 = get_kernel_matrix(
        Z1,
        Z1,
        length_scale=length_scale_nd,
        kernel_type=kernel_type_nd,
        nu=nu_nd,
    )
    alpha1 = _solve_krr_eig(K_11, Y1, [beta0_for_f_cand])[0]

    def predict_f_cand(Z_new: np.ndarray) -> np.ndarray:
        Z_new = np.asarray(Z_new, float)
        K_new = get_kernel_matrix(
            Z_new,
            Z1,
            length_scale=length_scale_nd,
            kernel_type=kernel_type_nd,
            nu=nu_nd,
        )
        return K_new @ alpha1

    t_samp1 = np.random.uniform(t_min, t_max, size=second_stage_N_SIZE)
    m_vals1 = np.zeros_like(t_samp1, dtype=float)

    for j, tj in enumerate(t_samp1):
        ZT = np.hstack([X1, np.full((n1, 1), tj)])
        m_vals1[j] = float(predict_f_cand(ZT).mean())

    t_samp1_col = t_samp1.reshape(-1, 1)

    K_H_11 = get_kernel_matrix(
        t_samp1_col,
        t_samp1_col,
        length_scale=length_scale_1d_h,
        kernel_type=kernel_type_1d_h,
        nu=nu_1d_h,
    )

    alphas_H_list = _solve_krr_eig(
        K_H_11,
        m_vals1,
        beta_h_grid_for_h_cand.tolist(),
    )

    # ---- Stage 2: fit f̃ on D2 and build \tilde h ----
    Z2 = np.hstack([X2, T2.reshape(-1, 1)])
    K_22 = get_kernel_matrix(
        Z2,
        Z2,
        length_scale=length_scale_nd,
        kernel_type=kernel_type_nd,
        nu=nu_nd,
    )
    alpha2 = _solve_krr_eig(K_22, Y2, [beta0_prime_for_f_tilde])[0]

    def predict_f_tilde(Z_new: np.ndarray) -> np.ndarray:
        Z_new = np.asarray(Z_new, float)
        K_new = get_kernel_matrix(
            Z_new,
            Z2,
            length_scale=length_scale_nd,
            kernel_type=kernel_type_nd,
            nu=nu_nd,
        )
        return K_new @ alpha2

    t_samp2 = np.random.uniform(t_min, t_max, size=second_stage_N_SIZE)
    m_vals2 = np.zeros_like(t_samp2, dtype=float)

    for j, tj in enumerate(t_samp2):
        ZT = np.hstack([X2, np.full((n2, 1), tj)])
        m_vals2[j] = float(predict_f_tilde(ZT).mean())

    t_samp2_col = t_samp2.reshape(-1, 1)

    K_H_22 = get_kernel_matrix(
        t_samp2_col,
        t_samp2_col,
        length_scale=length_scale_1d_h,
        kernel_type=kernel_type_1d_h,
        nu=nu_1d_h,
    )

    alpha_tilde = _solve_krr_eig(
        K_H_22,
        m_vals2,
        [beta0_prime_for_f_tilde],
    )[0]

    # ---- Model selection with fresh t_eval ----
    t_eval = np.random.uniform(t_min, t_max, size=second_stage_N_SIZE)
    t_eval_col = t_eval.reshape(-1, 1)

    K_H_eval1 = get_kernel_matrix(
        t_eval_col,
        t_samp1_col,
        length_scale=length_scale_1d_h,
        kernel_type=kernel_type_1d_h,
        nu=nu_1d_h,
    )
    K_H_eval2 = get_kernel_matrix(
        t_eval_col,
        t_samp2_col,
        length_scale=length_scale_1d_h,
        kernel_type=kernel_type_1d_h,
        nu=nu_1d_h,
    )

    tilde_vals_eval = K_H_eval2 @ alpha_tilde

    # Track both the best alpha and the corresponding selected beta
    best_error = np.inf
    best_alpha_H: np.ndarray | None = None
    best_beta: float | None = None

    for beta, alpha_H in zip(beta_h_grid_for_h_cand, alphas_H_list):
        pred_eval = K_H_eval1 @ alpha_H
        err = float(np.mean((pred_eval - tilde_vals_eval) ** 2))
        if err < best_error:
            best_error = err
            best_alpha_H = alpha_H
            best_beta = float(beta)

    if best_alpha_H is None or best_beta is None:
        raise RuntimeError("Model selection failed: no beta was selected.")

    # ---- Return the selected ĥ as a callable function of t ----
    def best_h_func(t_array):
        t_array = np.asarray(t_array, float).ravel()
        t_col = t_array.reshape(-1, 1)
        K_eval = get_kernel_matrix(
            t_col,
            t_samp1_col,
            length_scale=length_scale_1d_h,
            kernel_type=kernel_type_1d_h,
            nu=nu_1d_h,
        )
        return K_eval @ best_alpha_H

    return {
        "h_hat_func": best_h_func,
        "beta_selected": float(best_beta),
    }


# ----------------------------------------------------------------------
# Synthetic helper: single-run wrapper used by Demo_synthetic
# ----------------------------------------------------------------------


def run_single_ours_synthetic(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    t_grid: np.ndarray,
    n_samples: int,
    *,
    beta0_for_f_cand: float,
    beta_h_grid_for_h_cand: np.ndarray,
    beta0_prime_for_f_tilde: float,
    nu_nd: float,
    nu_1d_h: float,
    length_scale_nd: float,
    length_scale_1d_h: float,
    kernel_type_nd: str,
    t_min: float,
    t_max: float,
) -> tuple[np.ndarray, float]:
    """
    Synthetic single-run wrapper for the joint-kernel two-stage estimator.
    Hyperparameters are passed in from the notebook for easy experimentation.

    Returns:
        (h_hat_grid, beta_selected)
    """
    second_stage_N_SIZE = n_samples // 2
    D1, D2 = split_data(X, T, Y)

    out = run_ours_joint_kernel(
        D1=D1,
        D2=D2,
        beta0_for_f_cand=beta0_for_f_cand,
        beta_h_grid_for_h_cand=beta_h_grid_for_h_cand,
        beta0_prime_for_f_tilde=beta0_prime_for_f_tilde,
        nu_nd=nu_nd,
        nu_1d_h=nu_1d_h,
        length_scale_nd=length_scale_nd,
        length_scale_1d_h=length_scale_1d_h,
        second_stage_N_SIZE=second_stage_N_SIZE,
        kernel_type_nd=kernel_type_nd,
        t_min=t_min,
        t_max=t_max,
    )
    h_hat = out["h_hat_func"](t_grid)
    return h_hat, float(out["beta_selected"])


# ----------------------------------------------------------------------
# Semi-real helper: single-run wrapper used by Demo_semi-real
# ----------------------------------------------------------------------


def run_single_ours_semireal(
    Xss: np.ndarray,
    Ts: np.ndarray,
    Ys: np.ndarray,
    *,
    t_grid_original: np.ndarray,
    h_star_vals_original: np.ndarray,
    beta_grid: np.ndarray,
    kernel_type_f: str,
    ell_x: float,
    nu_x: float,
    ell_t: float,
    nu_t: float,
    beta0_f: float,
    beta0_prime_f: float,
    kernel_type_H: str,
    l_H: float,
    nu_H: float,
    second_stage_range: tuple[float, float],
    second_stage_n: int,
):
    """
    Semi-real single-run wrapper for the tensor-product two-stage estimator.
    Hyperparameters are passed in from the notebook for easy experimentation.

    Returns:
        dict from run_ours_tensor_kernel.
    """
    return run_ours_tensor_kernel(
        Xss,
        Ts,
        Ys,
        t_grid_original=t_grid_original,
        h_star_vals_original=h_star_vals_original,
        beta_grid=beta_grid,
        kernel_type_f=kernel_type_f,
        ell_x=ell_x,
        nu_x=nu_x,
        ell_t=ell_t,
        nu_t=nu_t,
        beta0_f=beta0_f,
        beta0_prime_f=beta0_prime_f,
        kernel_type_H=kernel_type_H,
        l_H=l_H,
        nu_H=nu_H,
        second_stage_range=second_stage_range,
        second_stage_n=second_stage_n,
    )


# Backward-compatible aliases (deprecated names)
estimate_h_grid_ours = estimate_h_grid_tensor
run_ours_on_original_grid = run_ours_tensor_kernel
