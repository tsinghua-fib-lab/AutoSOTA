"""
utils.py
--------
Shared utilities: Clayton copula sampler and transform families
used across CoBET, dCoBET, wa-dCoBET, HSIC, and dCov simulations.
"""

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────
# Clayton copula sampler (d-dimensional)
# ─────────────────────────────────────────────

def clayton_copula_sample_nd(n, theta, d, rng=None):
    """
    Sample n observations from a d-dimensional Clayton(theta) copula.

    Parameters
    ----------
    n     : int   – number of observations
    theta : float – Clayton dependence parameter (0 = independence)
    d     : int   – dimension
    rng   : np.random.RandomState or None  (uses global RNG if None)

    Returns
    -------
    U : ndarray, shape (n, d), values in (0, 1)
    """
    if rng is None:
        rng = np.random.RandomState()
    if theta == 0:
        return rng.uniform(0, 1, size=(n, d))
    # Gamma–Exponential construction
    S = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)       # (n,)
    E = rng.exponential(scale=1.0, size=(n, d))
    U = (1.0 + E / S[:, None]) ** (-1.0 / theta)
    return U


# ─────────────────────────────────────────────
# Transform families (coordinate-wise, nd)
# ─────────────────────────────────────────────

def _broadcast_b(b, d):
    """Broadcast scalar or length-d b to a length-d array."""
    b = np.asarray(b, dtype=float)
    if b.ndim == 0:
        return np.full(d, float(b))
    if b.shape != (d,):
        raise ValueError(f"b must be a scalar or shape ({d},), got {b.shape}")
    return b


def transform_trig_uniform_nd(u, v, b):
    """
    Trigonometric transform (coordinate-wise).

    x_j = sin(Φ⁻¹(u_j))
    y_j = cos(b_j · x_j + v_j)
    """
    n, d = u.shape
    b = _broadcast_b(b, d)
    x = np.sin(norm.ppf(u))
    y = np.cos(x * b[None, :] + v)
    return x, y


def transform_expquad_nd(u, v, b):
    """
    Exponential-quadratic transform (coordinate-wise).

    x_j = exp(−(Φ⁻¹(u_j))²)
    y_j = exp(−b_j · (x_j − 1)² + v_j)
    """
    n, d = u.shape
    b = _broadcast_b(b, d)
    z = norm.ppf(u)
    x = np.exp(-(z ** 2))
    y = np.exp(-b[None, :] * (x - 1.0) ** 2 + v)
    return x, y


def transform_linear_nd(u, v, b):
    """
    Linear transform (coordinate-wise).

    x_j = u_j
    y_j = b_j · x_j + v_j
    """
    n, d = u.shape
    b = _broadcast_b(b, d)
    x = u.copy()
    y = b[None, :] * x + v
    return x, y


def transform_logquad_nd(u, v, b):
    """
    Log-quadratic phase + amplitude modulation (coordinate-wise).

    Z_j  = Φ⁻¹(u_j)
    X_j  = log1p(Z_j²) / (1 + log1p(Z_j²))
    Y_j  = cos(b_j · X_j + v_j) · exp(−b_j · (X_j − 0.7)²)
    """
    n, d = u.shape
    b = _broadcast_b(b, d)
    Z = norm.ppf(u)
    X_base = np.log1p(Z ** 2)
    X = X_base / (1.0 + X_base)
    Y = np.cos(b[None, :] * X + v) * np.exp(-b[None, :] * (X - 0.7) ** 2)
    return X, Y


#: Registry of all transform families
TRANSFORM_MAP_ND = {
    "trigU":   transform_trig_uniform_nd,
    "expquad": transform_expquad_nd,
    "linear":  transform_linear_nd,
    "logquad": transform_logquad_nd,
}


def generate_XY(n, theta, D, transform_key, b, rng=None):
    """
    Generate (X, Y) pair of shape (n, D) from a Clayton copula
    with the chosen transform applied coordinate-wise.

    Parameters
    ----------
    n             : int
    theta         : float  – Clayton parameter
    D             : int    – dimension
    transform_key : str    – one of {'trigU', 'expquad', 'linear', 'logquad'}
    b             : float or array-like of shape (D,)
    rng           : np.random.RandomState or None

    Returns
    -------
    X, Y : ndarray, shape (n, D)
    """
    if transform_key not in TRANSFORM_MAP_ND:
        raise ValueError(f"Unknown transform '{transform_key}'. "
                         f"Choose from {list(TRANSFORM_MAP_ND)}")
    if rng is None:
        rng = np.random.RandomState()
    u = clayton_copula_sample_nd(n, theta, D, rng=rng)
    v = clayton_copula_sample_nd(n, theta, D, rng=rng)
    x, y = TRANSFORM_MAP_ND[transform_key](u, v, b=b)
    return x, y
