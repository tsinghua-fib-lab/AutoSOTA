# KRR_methods/kernels.py
from __future__ import annotations

import numpy as np


def get_kernel_matrix(
    X,
    Y,
    length_scale: float,
    kernel_type: str = "matern",
    nu: float = 1.5,
) -> np.ndarray:
    """
    General kernel function for 1D or multi-dimensional inputs.
    Supports Matérn (nu in {0.5, 1.5, 2.5, 3.5}) and Gaussian kernels.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    dist_sq = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)

    ell = float(length_scale)
    if ell == 0:
        ell = 1e-9

    if kernel_type == "gaussian":
        return np.exp(-dist_sq / (2.0 * ell**2))

    if kernel_type == "matern":
        if nu == 0.5:
            return np.exp(-dist / ell)
        if nu == 1.5:
            z = np.sqrt(3.0) * dist / ell
            return (1.0 + z) * np.exp(-z)
        if nu == 2.5:
            z = np.sqrt(5.0) * dist / ell
            return (1.0 + z + (z**2) / 3.0) * np.exp(-z)
        if nu == 3.5:
            z = np.sqrt(7.0) * dist / ell
            return (1.0 + z + 2.0 * (z**2) / 5.0 + (z**3) / 15.0) * np.exp(-z)
        raise ValueError("Unsupported Matérn nu (choose 0.5, 1.5, 2.5, 3.5).")

    raise ValueError("Unsupported kernel_type (choose 'matern' or 'gaussian').")


def tensor_product_kernel(
    Z1,
    Z2,
    kernel_type: str,
    ell_x: float,
    nu_x: float,
    ell_t: float,
    nu_t: float,
) -> np.ndarray:
    """
    Tensor product kernel:
      K((x, t), (x', t')) = K_x(x, x') * K_t(t, t').
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    X1, T1 = Z1[:, :-1], Z1[:, -1:]
    X2, T2 = Z2[:, :-1], Z2[:, -1:]

    K_x = get_kernel_matrix(X1, X2, ell_x, kernel_type, nu_x)
    K_t = get_kernel_matrix(T1, T2, ell_t, kernel_type, nu_t)

    return K_x * K_t
