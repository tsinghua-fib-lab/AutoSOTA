
from __future__ import annotations
import numpy as np

# ---------------------------
# Unscented Transform (UT)
# ---------------------------
def unscented_points(m: np.ndarray, S: np.ndarray, alpha: float = 1e-3, beta: float = 2.0, kappa: float | None = None):
    d = len(m)
    if kappa is None:
        kappa = 3 - d
    lam = alpha**2 * (d + kappa) - d

    Wm = np.full(2*d + 1, 1.0/(2*(d + lam)))
    Wc = Wm.copy()
    Wm[0] = lam/(d + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    Ssym = 0.5*(S + S.T)

    # Robust Cholesky with jitter
    jitter = 1e-9
    for _ in range(6):
        try:
            L = np.linalg.cholesky(Ssym + jitter*np.eye(d))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        # Fallback via eigen (SPD projection)
        w, U = np.linalg.eigh(Ssym)
        w = np.clip(w, 1e-12, None)
        L = U @ (np.sqrt(w)[:, None] * np.eye(d))

    pts = np.zeros((2*d + 1, d))
    pts[0] = m
    c = np.sqrt(d + lam)
    for i in range(d):
        e = c * L[:, i]
        pts[1 + i]     = m + e
        pts[1 + i + d] = m - e
    return pts, Wm, Wc

# ---------------------------
# Monte Carlo E_q[f(X)] helper (antithetic)
# ---------------------------
def _chol_spd(S: np.ndarray) -> np.ndarray:
    d = S.shape[0]
    Ssym = 0.5*(S + S.T)
    jitter = 1e-9
    for _ in range(6):
        try:
            return np.linalg.cholesky(Ssym + jitter*np.eye(d))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # Fallback
    w, U = np.linalg.eigh(Ssym)
    w = np.clip(w, 1e-12, None)
    return U @ (np.sqrt(w)[:, None] * np.eye(d))

def expect_scalar_mc(fn, m: np.ndarray, S: np.ndarray, N: int = 20000, antithetic: bool = True, seed: int | None = None) -> float:
    """
    Monte Carlo estimate of E_q[fn(X)] for q = N(m, S).
    Uses Cholesky, vectorized evaluation, optional antithetics.
    """
    rng = np.random.default_rng(seed)
    d = m.size
    L = _chol_spd(S)

    if antithetic:
        n = N // 2
        Z = rng.normal(size=(n, d))
        X1 = m + Z @ L.T
        X2 = m - Z @ L.T
        vals = np.concatenate([np.array([fn(x) for x in X1]),
                               np.array([fn(x) for x in X2])], axis=0)
        if 2*n < N:
            z = rng.normal(size=d)
            vals = np.concatenate([vals, [fn(m + L @ z)]])
    else:
        Z = rng.normal(size=(N, d))
        X = m + Z @ L.T
        vals = np.array([fn(x) for x in X])

    return float(vals.mean())

# ---------------------------
# Public evaluators
# ---------------------------
def expect_scalar(fn, m: np.ndarray, S: np.ndarray,
                  *, mc_threshold: int = 10,
                  alpha_small_dim: float = 1.0,
                  mc_N: int = 20000,
                  mc_antithetic: bool = True,
                  mc_seed: int | None = None) -> float:
    """
    E_q[fn(X)] with automatic method selection:
      - If d <= mc_threshold: Unscented Transform (UT) with alpha=alpha_small_dim
      - If d >  mc_threshold: Monte Carlo (antithetic), N=mc_N
    """
    d = len(m)
    if d > mc_threshold:
        return expect_scalar_mc(fn, m, S, N=mc_N, antithetic=mc_antithetic, seed=mc_seed)
    else:
        pts, Wm, _ = unscented_points(m, S, alpha=alpha_small_dim)
        vals = np.array([fn(x) for x in pts])
        return float((Wm * vals).sum())

def expect_grad_hess(grad_fn, hess_fn, m: np.ndarray, S: np.ndarray):
    pts, Wm, _ = unscented_points(m, S)  # keep UT here (you asked MC only for scalar expectations / KL)
    G = np.tensordot(Wm, np.stack([grad_fn(x) for x in pts], axis=0), axes=(0, 0))
    H = np.tensordot(Wm, np.stack([hess_fn(x) for x in pts], axis=0), axes=(0, 0))
    return G, H
