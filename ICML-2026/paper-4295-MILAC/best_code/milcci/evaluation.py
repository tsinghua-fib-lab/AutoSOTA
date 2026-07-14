# -*- coding: utf-8 -*-
"""
Evaluation utilities for MILCCI results.
"""
import numpy as np
from .utils import spec_corr
from .core import reconstruct


def per_trial_r2(data, A_full, Phi):
    """
    Compute per-trial R^2.

    Parameters
    ----------
    data : np.ndarray, shape (N, T, M)
    A_full : np.ndarray, shape (N, P, M)
    Phi : np.ndarray, shape (T, P, M)

    Returns
    -------
    r2_vec : np.ndarray, shape (M,)
    """
    M = data.shape[2]
    assert A_full.shape[2] == M, 'A_full trials %d != data trials %d' % (A_full.shape[2], M)
    assert Phi.shape[2] == M, 'Phi trials %d != data trials %d' % (Phi.shape[2], M)

    r2_vec = np.zeros(M)
    for m in range(M):
        y = data[:, :, m]
        y_hat = A_full[:, :, m] @ Phi[:, :, m].T
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_vec[m] = 1 - ss_res / (ss_tot + 1e-18)
    return r2_vec


def global_r2(data, A_full, Phi):
    """
    Compute global R^2 across all trials.

    Returns
    -------
    r2 : float
    """
    Y_hat = reconstruct(A_full, Phi)
    ss_res = np.sum((data - Y_hat) ** 2)
    ss_tot = np.sum((data - data.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-18)


def reconstruction_correlation(data, A_full, Phi):
    """
    Pearson correlation between data and reconstruction (flattened).

    Returns
    -------
    rho : float
    """
    Y_hat = reconstruct(A_full, Phi)
    return spec_corr(data.flatten(), Y_hat.flatten(), to_abs=False)
