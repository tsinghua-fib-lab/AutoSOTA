# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact.peren@finances.gouv.fr>
#
# SPDX-License-Identifier: MIT

import numpy as np
from reproducibility.icml2026.baselines.rcit.internal_momentchi2 import hbe, sw, lpb4

from reproducibility.icml2026.baselines.rcit.rcit import ApproxMethod


def compute_p_value_from_covariance(
    covariance_matrix: np.ndarray, quantile_values: float, approx_method: ApproxMethod
) -> float:
    """
    Helper function to compute the p-value based on the test statistic and
    the covariance matrix depending on the chosen approximation
    """
    eig_d = np.linalg.eigvals(covariance_matrix)
    
    # Debug: Check for complex eigenvalues
    if np.iscomplexobj(eig_d):
        # print(f"Warning: Got complex eigenvalues: {eig_d}")
        # print(f"Imaginary parts: {eig_d.imag}")
        # print(f"Max imaginary part: {np.max(np.abs(eig_d.imag))}")
        # Take only real parts if imaginary parts are tiny (numerical errors)
        if np.allclose(eig_d.imag, 0, atol=1e-12):
            eig_d = eig_d.real
            # print("Converting to real eigenvalues (imaginary parts were negligible)")
    
    eig_d = eig_d[eig_d > 0]

    approx_to_fun = {"gamma": sw, "hbe": hbe, "lpd4": lpb4}

    assert approx_method in approx_to_fun.keys()

    return 1 - approx_to_fun[approx_method](eig_d, quantile_values)


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalizes each column of a matrix
    See: https://github.com/ericstrobl/RCIT/blob/master/R/normalize.R
    """
    assert len(x.shape) == 2

    zero_mean_x = x - np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    std_x[std_x == 0] = 1

    return zero_mean_x / std_x


# def compute_p_value_from_covariance(residual_covariance, quantile_values, approx_method):
#     eig_d = np.linalg.eigvalsh(residual_covariance)

#     # Force real float inputs for moment-chi2 approximations
#     eig_d = np.real_if_close(eig_d, tol=1000)
#     eig_d = np.asarray(eig_d, dtype=np.float64)
#     eig_d = np.clip(eig_d, 0.0, None)  # tiny negative eigs from numerics -> 0

#     quantile_values = np.real_if_close(quantile_values, tol=1000)
#     quantile_values = np.asarray(quantile_values, dtype=np.float64)

#     return 1 - approx_to_fun[approx_method](eig_d, quantile_values)
