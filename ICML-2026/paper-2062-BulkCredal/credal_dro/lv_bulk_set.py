from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.linalg as la



def safe_cov(X: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """
    Sample covariance with a tiny ridge to ensure positive definiteness.

    Parameters
    ----------
    X : (n, d) array_like
        Data matrix.
    ridge : float, optional
        Ridge added to the diagonal for numerical stability.

    Returns
    -------
    Sigma : (d, d) ndarray
        Regularised sample covariance matrix.

    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    return S + ridge * np.eye(S.shape[0])


@dataclass
class GaussianCenter:
    """Gaussian centre with basic score functions.
    """

    mu: np.ndarray
    Sigma: np.ndarray

    @property
    def Sig_inv(self) -> np.ndarray:
        return la.inv(self.Sigma)

    def score_ellipsoid(self, X: np.ndarray) -> np.ndarray:
        """Elliptical (Mahalanobis) score.

        s(x) = ||Sigma^{-1/2} (x - mu)||_2.
        """
        X = np.asarray(X, dtype=float)
        D = X - self.mu
        return np.sqrt(np.einsum("...i,ij,...j->...", D, self.Sig_inv, D))

    def score_box(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Axis-aligned box score.

        w : feature scales (positive); score is max_i |(x_i - mu_i)/w_i|.
        """
        X = np.asarray(X, dtype=float)
        w = np.asarray(w, dtype=float)
        D = (X - self.mu) / w
        return np.max(np.abs(D), axis=1)

    def score_directional(self, X: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Directional polytope score.

        U : (J, d) unit directions, b_j > 0 scales.
        """
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        b = np.asarray(b, dtype=float)
        D = X - self.mu
        proj = (U @ D.T).T  # (n, J)
        s = np.max(np.abs(proj) / b, axis=1)
        return s


def fit_pc_gaussian(X: np.ndarray, ridge: float = 1e-6) -> GaussianCenter:
    """Fit the Gaussian centre PC from a reference dataset.

    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    Sigma = safe_cov(X, ridge=ridge)
    return GaussianCenter(mu=mu, Sigma=Sigma)

def make_score_function(
    score_type: str = "ellipsoid",
    center: Optional[GaussianCenter] = None,
    X_ref: Optional[np.ndarray] = None,
    J: int = 8,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:

    if center is None and X_ref is None:
        raise ValueError("Provide either a centre or reference data to define the score.")

    if score_type == "ellipsoid":
        if center is None:
            center = fit_pc_gaussian(X_ref)

        def s(X: np.ndarray) -> np.ndarray:
            return center.score_ellipsoid(X)

        meta = {"type": "ellipsoid", "mu": center.mu, "Sigma": center.Sigma}
        return s, meta

    elif score_type == "box":
        if center is None:
            X_ref = np.asarray(X_ref, dtype=float)
            mu = X_ref.mean(axis=0)
            w = X_ref.std(axis=0, ddof=1)
            w[w == 0.0] = 1.0
            center = GaussianCenter(mu=mu, Sigma=np.diag(w ** 2))
        else:
            # use marginal scales from Sigma
            w = np.sqrt(np.diag(center.Sigma))
            w[w == 0.0] = 1.0

        def s(X: np.ndarray) -> np.ndarray:
            return center.score_box(X, w=w)

        meta = {"type": "box", "mu": center.mu, "w": w}
        return s, meta

    elif score_type == "directional":
        if center is None:
            center = fit_pc_gaussian(X_ref)
        d = center.mu.size
        rng = np.random.default_rng(123)
        U = rng.normal(size=(J, d))
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        # scale b_j from center covariance
        b = np.sqrt(np.sum(U @ center.Sigma * U, axis=1))
        b[b == 0.0] = 1.0

        def s(X: np.ndarray) -> np.ndarray:
            return center.score_directional(X, U=U, b=b)

        meta = {"type": "directional", "mu": center.mu, "U": U, "b": b}
        return s, meta

    else:
        raise ValueError("Unknown score_type.")


def build_score(
    points: np.ndarray,
    score_type: str = "ellipsoid",
    score_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:

    if score_params is None:
        score_params = {}
    center = score_params.get("center", None)
    J = int(score_params.get("J", 8))
    return make_score_function(score_type=score_type, center=center, X_ref=points, J=J)


def dkw_select_threshold(audit_scores, gamma, delta):

    try:
        gamma = float(gamma); delta = float(delta)
    except Exception:
        raise ValueError("gamma and delta must be floats in (0,1).")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must lie in (0,1).")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0,1).")

    scores = np.asarray(audit_scores, dtype=float).ravel()
    m = scores.size
    if m < 1:
        raise ValueError("selection scores must be a non-empty 1-D array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("selection scores contains NaN/Inf; please clean or filter before calling.")

    # ---- DKW radius
    r = float(np.sqrt(np.log(2.0 / delta) / (2.0 * m)))
    certifiable_max_coverage = 1.0 - r

    # Existence condition: we need 1 - gamma <= 1 - r  <=>  gamma >= r
    exists = gamma >= r
    if not exists:
        return {
            "t_hat": np.nan,
            "j_star": None,
            "r": r,
            "certifiable_max_coverage": certifiable_max_coverage,
            "exists": False,
            "Fm_at_t_hat": np.nan,
            "L_at_t_hat": np.nan,
        }
    # Compute j*
    j_star = int(np.clip(np.ceil(m * (1.0 - gamma + r)), 1, m))
    sorted_scores = np.sort(scores)
    t_hat = float(sorted_scores[j_star - 1])  

    count_le = int(np.searchsorted(sorted_scores, t_hat, side="right"))
    Fm_at_t_hat = count_le / m
    L_at_t_hat = max(Fm_at_t_hat - r, 0.0)

    return {
        "t_hat": t_hat,
        "j_star": j_star,
        "r": r,
        "certifiable_max_coverage": certifiable_max_coverage,
        "exists": True,
        "Fm_at_t_hat": Fm_at_t_hat,
        "L_at_t_hat": L_at_t_hat,
    }


def dkw_certificate(audit_scores, gamma, delta):
    """One-step wrapper around :func:`dkw_select_threshold`.

    Prints a concise textual certificate and returns the result dict.
    """
    res = dkw_select_threshold(audit_scores, gamma, delta)
    m = np.asarray(audit_scores).size
    print("=== DKW bulk-mass certificate ===")
    print(f"m = {m}, gamma = {gamma:.6f}, delta = {delta:.6f}")
    print(f"r = sqrt(log(2/delta)/(2m)) = {res['r']:.6f}")
    print(f"certifiable_max_coverage = 1 - r = {res['certifiable_max_coverage']:.6f}")

    if not res["exists"]:
        shortfall = res["r"] - gamma
        print(f"\nNo solution exists because gamma < r by {shortfall:.6f}.")
        print("Action: increase m (more audit data), use a larger delta, or relax gamma.")
        return res

    print("\nSolution exists (gamma >= r).")
    print(f"j_star = ceil( m * (1 - gamma + r) ) = {res['j_star']}")
    print(f"t_hat  = order-statistic value = {res['t_hat']:.6f}")
    print(f"F_m(t_hat) = {res['Fm_at_t_hat']:.6f}")
    print(f"L^{'{'}DKW{'}'}(t_hat) = max(F_m - r, 0) = {res['L_at_t_hat']:.6f}")
    print(f"Certificate: P^*(Xi_0(t_hat)) >= {1.0 - gamma:.6f} with prob >= {1.0 - delta:.6f}.")
    return res
