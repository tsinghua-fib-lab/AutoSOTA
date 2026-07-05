# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact.peren@finances.gouv.fr>
#
# SPDX-License-Identifier: MIT

import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances

from typing import Optional, TypedDict


from . import ApproxMethod
from reproducibility.icml2026.baselines.rcit.rcit.utils import compute_p_value_from_covariance
from reproducibility.icml2026.baselines.rcit.rcit.rff import compute_normalized_rff


class TestResult(TypedDict):
    p: float
    Sta: float


def RCoT(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    approx: ApproxMethod = "lpd4",
    num_f: int = 100,
    num_f2: int = 5,
    seed: Optional[int] = None,
) -> TestResult:
    """
    Tests whether x and y are conditionally independent given z.
    See: https://github.com/ericstrobl/RCIT/blob/master/R/RCoT.R
    """

    n_data = x.shape[0]

    if z is not None and np.all(np.std(z, axis=0) == 0):
        z = None

    if np.std(x) == 0 or np.std(y) == 0:
        return TestResult(p=1, Sta=0)

    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=-1)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=-1)

    small_data_sample = min(n_data, 500)
    lower_tri_indices = np.triu_indices(small_data_sample, k=1)

    if z is not None:
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=-1)
        y = np.hstack([y, z])
        f_z = compute_normalized_rff(
            z, num_f=num_f, seed=seed, sigma=np.median(pairwise_distances(z[:small_data_sample])[lower_tri_indices])
        )

    f_x = compute_normalized_rff(
        x, num_f=num_f2, seed=seed, sigma=np.median(pairwise_distances(x[:small_data_sample])[lower_tri_indices])
    )
    f_y = compute_normalized_rff(
        y, num_f=num_f2, seed=seed, sigma=np.median(pairwise_distances(y[:small_data_sample])[lower_tri_indices])
    )

    cov_x_y = get_covariance(f_x, f_y)
    if z is not None:
        cov_z_z = get_covariance(f_z, f_z)
        cov_x_z = get_covariance(f_x, f_z)
        cov_z_y = get_covariance(f_z, f_y)

        cov_z_z_inverse = np.linalg.pinv(cov_z_z + np.eye(num_f) * 1e-10)  # Generalized inverse

        z_cov_z_z_inverse = f_z @ cov_z_z_inverse
        residual_x = f_x - z_cov_z_z_inverse @ cov_x_z.T
        residual_y = f_y - z_cov_z_z_inverse @ cov_z_y
    else:
        residual_x = f_x
        residual_y = f_y

    if num_f2 == 1:
        approx = "hbe"

    if approx == "perm":
        return get_permutation_statistics_results(residual_x, residual_y, n_data)

    if z is None:
        residual_x = f_x - np.mean(f_x, axis=0)
        residual_y = f_y - np.mean(f_y, axis=0)
    else:
        cov_x_y_by_z = cov_x_y - cov_x_z @ cov_z_z_inverse @ cov_z_y
        quantile_value = n_data * np.sum(cov_x_y_by_z * cov_x_y_by_z)

    d = np.tile(range(f_x.shape[1]), f_y.shape[1]), np.repeat(range(f_y.shape[1]), f_x.shape[1])

    res = residual_x[:, d[0]] * residual_y[:, d[1]]
    residual_covariance = 1 / n_data * res.T @ res

    if approx == "chi2":
        residual_covariance_inverse = np.linalg.pinv(residual_covariance)
        partial_covariance = cov_x_y if z is None else cov_x_y_by_z
        quantile_value = n_data * partial_covariance @ residual_covariance_inverse @ partial_covariance
        p_value = 1 - sp.stats.chi2.cdf(quantile_value, ddof=len(partial_covariance))
    else:
        if z is None:
            quantile_value, cov_x_y = permutation_statistic(f_x=f_x, f_y=f_y, data_size=n_data)
        p_value = compute_p_value_from_covariance(residual_covariance, quantile_value, approx)

    return TestResult(p=max(p_value, 0), Sta=quantile_value)


def RIT(
    x: np.ndarray,
    y: np.ndarray,
    num_f: int = 5,
    approx: ApproxMethod = "lpd4",
    seed: Optional[int] = None,
) -> TestResult:
    """
    Tests whether x and y are unconditionally independent
    See: https://github.com/ericstrobl/RCIT/blob/master/R/RIT.R
    """
    return RCoT(x, y, z=None, approx=approx, num_f2=num_f, seed=seed)


def get_permutation_statistics_results(residual_x, residual_y, n_data):
    quantile_value, _cov_x_y = permutation_statistic(f_x=residual_x, f_y=residual_y, data_size=n_data)
    number_of_permutations = 1000
    quantile_values = np.array(
        [
            permutation_statistic(residual_x[np.random.randint(low=0, high=n_data, size=n_data)], residual_y, n_data)[0]
            for _ in range(number_of_permutations)
        ]
    )
    p_value = sum(quantile_value < quantile_values) / number_of_permutations
    return TestResult(p=max(p_value, 0), Sta=quantile_value)


def get_covariance(X: np.ndarray, Y: np.ndarray):
    return 1 / (len(X) - 1) * np.einsum("nf, nh -> fh", X, Y)


def permutation_statistic(f_x: np.ndarray, f_y: np.ndarray, data_size: int):
    """
    Permutation statistic
    See: https://github.com/ericstrobl/RCIT/blob/master/R/Sta_perm.R
    """
    cov_x_y = get_covariance(f_x, f_y)
    quantile_value = data_size * np.sum(cov_x_y * cov_x_y)
    return quantile_value, cov_x_y
