from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ..synthetic_dgps import split_data

# Optional: SciPy for continuous 1D optimization of beta via bounded minimization.
try:  # pragma: no cover - optional dependency
    from scipy.optimize import minimize_scalar

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_SCIPY = False

# ----------------------------------------------------------------------
# Direct regression baseline (T-only 1D KRR; ignore X)
# ----------------------------------------------------------------------


def unpack_split(D):
    """Unpack the output of split_data into (X, T, Y).

    Accepts either:
    - A dict with keys like ("X", "T", "Y") or lowercase variants.
    - A tuple/list of length 3: (X, T, Y).
    """
    if isinstance(D, dict):
        for kx, kt, ky in [
            ("X", "T", "Y"),
            ("x", "t", "y"),
            ("Xs", "Ts", "Ys"),
            ("xs", "ts", "ys"),
        ]:
            if kx in D and kt in D and ky in D:
                return D[kx], D[kt], D[ky]
        raise KeyError(f"Cannot find X/T/Y keys in dict: {list(D.keys())}")

    if isinstance(D, (tuple, list)) and len(D) == 3:
        return D[0], D[1], D[2]

    raise TypeError(f"Unsupported split format: type={type(D)}, value={D}")


def matern32_kernel_1d(t1: np.ndarray, t2: np.ndarray, ell: float) -> np.ndarray:
    """Matérn 3/2 kernel for 1D inputs.

    Args:
        t1, t2: 1D arrays of inputs.
        ell: Length-scale parameter (positive).
    """
    t1 = np.asarray(t1, dtype=float).reshape(-1, 1)
    t2 = np.asarray(t2, dtype=float).reshape(-1, 1)
    r = np.abs(t1 - t2.T)
    s = (np.sqrt(3.0) * r) / float(ell)
    return (1.0 + s) * np.exp(-s)


def fit_krr_1d(T_train: np.ndarray, Y_train: np.ndarray, ell: float, beta: float):
    """Fit 1D KRR on (T_train, Y_train) and return a predictor function.

    Args:
        T_train: Treatment values, shape (n,).
        Y_train: Outcomes, shape (n,).
        ell: Length-scale for the kernel.
        beta: Ridge penalty (lambda) in KRR.

    Returns:
        Callable that maps t_query -> h_hat(t_query).
    """
    T_train = np.asarray(T_train, dtype=float).reshape(-1)
    Y_train = np.asarray(Y_train, dtype=float).reshape(-1)

    K = matern32_kernel_1d(T_train, T_train, ell=ell)
    n = K.shape[0]
    M = K + float(beta) * np.eye(n)

    L = np.linalg.cholesky(M)
    tmp = np.linalg.solve(L, Y_train)
    alpha = np.linalg.solve(L.T, tmp)

    def h_hat(t_query: np.ndarray) -> np.ndarray:
        t_query = np.asarray(t_query, dtype=float).reshape(-1)
        K_q = matern32_kernel_1d(t_query, T_train, ell=ell)
        return K_q @ alpha

    return h_hat


def select_beta_by_holdout(
    T_train,
    Y_train,
    T_valid,
    Y_valid,
    ell,
    beta_grid,
) -> Tuple[float, Callable[[np.ndarray], np.ndarray]]:
    """Select beta by validation MSE on a holdout split.

    Args:
        T_train, Y_train: Training split.
        T_valid, Y_valid: Validation split.
        ell: Length-scale for the kernel.
        beta_grid: Candidate ridge values to evaluate.

    Returns:
        (best_beta, best_model_callable)
    """
    best_beta, best_mse, best_model = None, np.inf, None
    for beta in beta_grid:
        h_hat = fit_krr_1d(T_train, Y_train, ell=float(ell), beta=float(beta))
        y_pred = h_hat(T_valid)
        mse = float(np.mean((y_pred - np.asarray(Y_valid).reshape(-1)) ** 2))
        if mse < best_mse:
            best_mse, best_beta, best_model = mse, float(beta), h_hat
    return best_beta, best_model


DIRECT_ELL_T = 2.0
DIRECT_C_VAL = 0.1
DIRECT_BETA_GRID = np.array([DIRECT_C_VAL * (2**i) for i in range(0, 9)], dtype=float)


def run_single_direct_synthetic(
    X,
    T,
    Y,
    t_grid,
    ell: float = DIRECT_ELL_T,
    beta_grid: np.ndarray = DIRECT_BETA_GRID,
):
    """Run a single synthetic T-only baseline with holdout beta selection.

    Args:
        X, T, Y: Full dataset. X is ignored (baseline uses T only).
        t_grid: Evaluation grid for h_hat(t).
        ell: Kernel length-scale.
        beta_grid: Candidate ridge values.

    Returns:
        (h_hat_grid, beta_selected)
    """
    D1, D2 = split_data(X, T, Y)
    _, T1, Y1 = unpack_split(D1)
    _, T2, Y2 = unpack_split(D2)

    T1, Y1 = np.asarray(T1).reshape(-1), np.asarray(Y1).reshape(-1)
    T2, Y2 = np.asarray(T2).reshape(-1), np.asarray(Y2).reshape(-1)

    beta_sel, h_hat = select_beta_by_holdout(
        T1,
        Y1,
        T2,
        Y2,
        ell=ell,
        beta_grid=beta_grid,
    )
    h_hat_vals = h_hat(t_grid)
    return h_hat_vals, beta_sel


# ----------------------------------------------------------------------
# Semi-real helper: T-only baseline (Laplace kernel + Nyström + LOOCV)
# ----------------------------------------------------------------------


def laplace_kernel_1d(t1: np.ndarray, t2: np.ndarray, ell: float) -> np.ndarray:
    """
    Laplace kernel in 1D:
      k(t, t') = exp(-|t - t'| / ell)
    """
    t1 = np.asarray(t1, dtype=float).reshape(-1, 1)
    t2 = np.asarray(t2, dtype=float).reshape(-1, 1)
    r = np.abs(t1 - t2.T)
    return np.exp(-r / float(ell))


def _build_nystrom_features_laplace(
    T_train: np.ndarray,
    ell_t: float,
    m: int,
    rng: np.random.RandomState,
    jitter: float = 1e-8,
):
    """
    Build Nystrom features Phi for a Laplace kernel using m landmarks.

    Args:
        T_train: Training treatments, shape (n,).
        ell_t: Length-scale for the Laplace kernel.
        m: Number of landmarks (capped at n).
        rng: RandomState for landmark selection.
        jitter: Diagonal jitter for numerical stability.

    Returns:
        (Phi, phi_eval_fn) where phi_eval_fn maps new T to Nystrom features.
    """
    T_train = np.asarray(T_train, dtype=float).ravel()
    n = len(T_train)
    m_eff = min(int(m), n)

    idx = rng.choice(n, size=m_eff, replace=False)
    Tm = T_train[idx]

    K_mm = laplace_kernel_1d(Tm, Tm, ell=ell_t)
    K_mm = 0.5 * (K_mm + K_mm.T)
    K_mm += float(jitter) * np.eye(m_eff)

    K_nm = laplace_kernel_1d(T_train, Tm, ell=ell_t)

    evals, evecs = np.linalg.eigh(K_mm)
    evals = np.maximum(evals, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(evals)

    # Standard Nyström features with sqrt(m/n) scaling
    scale = np.sqrt(m_eff / float(n))
    Phi = scale * (K_nm @ (evecs * inv_sqrt))

    def phi_eval(T_new: np.ndarray) -> np.ndarray:
        T_new = np.asarray(T_new, dtype=float).ravel()
        K_new_m = laplace_kernel_1d(T_new, Tm, ell=ell_t)
        return scale * (K_new_m @ (evecs * inv_sqrt))

    return Phi, phi_eval


def _loocv_beta_optimize_svd(
    Phi: np.ndarray,
    Y: np.ndarray,
    beta_min: float,
    beta_max: float,
) -> float:
    """
    Choose beta by minimizing LOOCV MSE using the SVD formula.

    Args:
        Phi: Nystrom feature matrix.
        Y: Training outcomes.
        beta_min, beta_max: Search bounds for beta.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    U, s, _ = np.linalg.svd(Phi, full_matrices=False)

    Uy = U.T @ Y
    U_sq = U ** 2
    s2 = s ** 2

    def loocv_mse(beta: float) -> float:
        if beta <= 0:
            return float("inf")
        shrink = s2 / (s2 + beta)
        y_hat = U @ (shrink * Uy)
        diagS = U_sq @ shrink
        eps = 1e-12
        resid = Y - y_hat
        loo_resid = resid / (1.0 - diagS + eps)
        return float(np.mean(loo_resid ** 2))

    lower = max(float(beta_min), 1e-8)
    upper = max(float(beta_max), lower * 1.0001)

    if _HAVE_SCIPY:
        res = minimize_scalar(loocv_mse, bounds=(lower, upper), method="bounded")
        return float(res.x)

    # Fallback: log-grid search
    grid = np.logspace(np.log10(lower), np.log10(upper), 30)
    vals = [loocv_mse(b) for b in grid]
    return float(grid[int(np.argmin(vals))])


def estimate_h_grid_t_only_laplace_fixed_ell(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    t_grid: np.ndarray,
    *,
    ell_t: float = 3000.0,
    m_t: int = 700,
    beta_min: float = 0.05,
    beta_max: float = 80.0,
    seed_landmarks: int = 0,
):
    """
    Fit T-only Nyström KRR with Laplace kernel (fixed ell_t), choose beta by LOOCV,
    then predict h_hat on t_grid.

    Args:
        T_train, Y_train: Training data for the T-only regression.
        t_grid: Evaluation grid for h_hat(t).
        ell_t: Laplace length-scale.
        m_t: Nystrom landmarks.
        beta_min, beta_max: Search bounds for ridge parameter.
        seed_landmarks: RNG seed for landmark selection.

    Returns:
        (h_hat_grid, beta_star)
    """
    T_train = np.asarray(T_train, dtype=float).ravel()
    Y_train = np.asarray(Y_train, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    rng = np.random.RandomState(seed_landmarks)
    Phi, phi_eval = _build_nystrom_features_laplace(T_train, ell_t=ell_t, m=m_t, rng=rng)

    beta_star = _loocv_beta_optimize_svd(Phi, Y_train, beta_min=beta_min, beta_max=beta_max)

    # Closed-form ridge in SVD coordinates
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    Uy = U.T @ Y_train
    s2 = s ** 2
    coeff = s / (s2 + beta_star)
    w_star = Vt.T @ (coeff * Uy)

    Phi_grid = phi_eval(t_grid)
    h_hat_grid = Phi_grid @ w_star
    return np.asarray(h_hat_grid, dtype=float), float(beta_star)


def mise_on_grid(h_hat_grid: np.ndarray, h_star_grid: np.ndarray) -> float:
    """Mean squared error between two curves evaluated on the same grid."""
    return float(np.mean((np.asarray(h_hat_grid) - np.asarray(h_star_grid)) ** 2))


def run_single_direct_semireal(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    t_grid: np.ndarray,
    h_star_vals: np.ndarray,
    *,
    ell_t: float = 3000.0,
    m_t: int = 700,
    beta_min: float = 0.05,
    beta_max: float = 80.0,
    seed_landmarks: int = 0,
):
    """
    Semi-real single-run wrapper for T-only Laplace KRR baseline.

    Args:
        T_train, Y_train: Training data for the semi-synthetic run.
        t_grid: Evaluation grid for h_hat(t).
        h_star_vals: Ground-truth curve on t_grid.
        ell_t, m_t: Laplace kernel length-scale and Nystrom landmarks.
        beta_min, beta_max: Search bounds for beta.
        seed_landmarks: RNG seed for landmark selection.

    Returns:
        (h_hat_grid, beta_star, mise)
    """
    h_hat_grid, beta_star = estimate_h_grid_t_only_laplace_fixed_ell(
        T_train=T_train,
        Y_train=Y_train,
        t_grid=t_grid,
        ell_t=ell_t,
        m_t=m_t,
        beta_min=beta_min,
        beta_max=beta_max,
        seed_landmarks=seed_landmarks,
    )
    mise = mise_on_grid(h_hat_grid, h_star_vals)
    return h_hat_grid, beta_star, mise
