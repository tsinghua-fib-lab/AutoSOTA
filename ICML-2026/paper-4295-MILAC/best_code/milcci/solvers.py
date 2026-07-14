# -*- coding: utf-8 -*-
"""
Solvers for MILCCI: least-squares, NNLS, and Lasso-style updates.
"""
import numpy as np
from scipy import linalg
from scipy.optimize import nnls
import warnings


def solve_least_squares(A, b):
    """
    Solve  min ||A x - b||^2  via pseudoinverse.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    b : np.ndarray, shape (m,) or (m, k)

    Returns
    -------
    x : np.ndarray, shape (n,) or (n, k)
    """
    if b.ndim == 1 or (b.ndim == 2 and b.shape[1] == 1):
        return linalg.pinv(A) @ b.reshape(-1, 1)
    return linalg.pinv(A) @ b


def solve_nnls_multi(A, B):
    """
    Solve  min ||A x - b||^2  s.t. x >= 0  for each column of B.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    B : np.ndarray, shape (m,) or (m, k)

    Returns
    -------
    X : np.ndarray, shape (n,) or (n, k)
    """
    if B.ndim == 1 or len(B.flatten()) == max(B.shape):
        x, _ = nnls(A, B.flatten())
        return x
    # multi-column
    assert B.shape[0] == A.shape[0], (
        'dimension mismatch: A %s vs B %s' % (str(A.shape), str(B.shape))
    )
    X = np.hstack([
        nnls(A, B[:, j].flatten())[0].reshape(-1, 1)
        for j in range(B.shape[1])
    ])
    assert X.shape[0] == A.shape[1], (
        'output shape[0] %d != A.shape[1] %d' % (X.shape[0], A.shape[1])
    )
    return X


def solve_regularized(A, b, solver='inv', l1=0.0, seed=0):
    """
    Solve  min (1/2)||Ax - b||^2 + l1*||x||_1.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray
    solver : str
        'inv' for pseudoinverse (ignores l1), 'nnls' for non-negative LS.
    l1 : float
        L1 penalty (only used with 'lasso' solver, currently unused).
    seed : int

    Returns
    -------
    x : np.ndarray
    """
    if np.isnan(A).any():
        warnings.warn('NaN detected in A matrix for solve_regularized')

    if len(b.flatten()) == max(b.shape):
        b = b.reshape(-1, 1)

    if solver == 'inv' or l1 == 0:
        return solve_least_squares(A, b)
    elif solver == 'nnls':
        return solve_nnls_multi(A, b)
    else:
        raise ValueError('Unknown solver: %s. Supported: inv, nnls' % solver)


def solve_ls_for_phi(data, A, lambda_l2=0.1):
    """
    Solve  min ||data - A @ phi||^2 + lambda_l2 * ||phi||^2.

    Parameters
    ----------
    data : np.ndarray, shape (N, T_total)
    A : np.ndarray, shape (N, P)  -- the spatial components
    lambda_l2 : float

    Returns
    -------
    phi : np.ndarray, shape (P, T_total)
    """
    P = A.shape[1]
    T = data.shape[1]
    left = np.vstack([data, np.zeros((P, T))])
    right = np.vstack([A, lambda_l2 * np.eye(P)])
    phi = linalg.pinv(right) @ left
    assert phi.shape == (P, T), (
        'phi shape %s != expected (%d, %d)' % (str(phi.shape), P, T)
    )
    return phi
