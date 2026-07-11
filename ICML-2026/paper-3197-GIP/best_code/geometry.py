
# geometry.py  — BW (Takatsu/Otto-consistent) + LBW helpers
from __future__ import annotations
import numpy as np
from numpy.linalg import eigh

# ---------- basic utilities ----------

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _sqrtm_spd(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Principal matrix square root of SPD (via eigh), symmetrized."""
    S = _sym(S)
    w, U = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return U @ (np.diag(np.sqrt(w))) @ U.T

def _invsqrtm_spd(S: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Inverse square root of SPD (via eigh), symmetrized."""
    S = _sym(S)
    w, U = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return U @ (np.diag(1.0 / np.sqrt(w))) @ U.T

def _clip_eigs(S: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    S = _sym(S)
    w, U = eigh(S)
    if lo is None: lo = -np.inf
    if hi is None: hi =  np.inf
    w = np.clip(w, lo, hi)
    return _sym(U @ np.diag(w) @ U.T)

def bw_log(S_base: np.ndarray, S_bar: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Logarithmic map at S_base pointing to S_bar under the Bures–Wasserstein metric:
        V = S_base^{-1/2} ( S_base^{1/2} S_bar S_base^{1/2} )^{1/2} S_base^{-1/2} - I
    """
    S = _sym(S_base)
    Sbar = _sym(S_bar)
    S12   = _sqrtm_spd(S, eps)
    Sinv2 = _invsqrtm_spd(S, eps)
    M     = _sym(S12 @ Sbar @ S12)             # SPD by congruence
    Msq   = _sqrtm_spd(M, eps)                 # principal sqrt
    A     = Sinv2 @ Msq @ Sinv2
    return _sym(A - np.eye(S.shape[0]))

def bw_exp(S_base: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Exponential map at S_base applied to tangent V (symmetric):
        Exp_S(V) = (I + V) S (I + V)
    """
    S = _sym(S_base)
    V = _sym(V)
    A = np.eye(S.shape[0]) + V
    # A is (approximately) symmetric; symmetry of the result is guaranteed by construction
    return _sym(A @ S @ A)

# ---------- LBW (Linearized BW) at a reference Gaussian ----------


def lbw_weighted_average(lbw_states, weights):
    """
    Weighted average in LBW (linear) coords:
        \bar v_m = sum w_i v_m^i
        \bar V   = sum w_i V^i
    """
    vm = np.stack([vm for vm, _ in lbw_states], axis=0)
    VV = np.stack([V  for _,  V in lbw_states], axis=0)
    w = np.asarray(weights)
    w = w / w.sum()
    v_bar = (w[:, None]       * vm).sum(axis=0)
    V_bar = (w[:, None, None] * VV).sum(axis=0)
    return v_bar, _sym(V_bar)
