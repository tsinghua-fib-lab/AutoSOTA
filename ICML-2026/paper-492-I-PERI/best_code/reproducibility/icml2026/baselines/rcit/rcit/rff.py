# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact.peren@finances.gouv.fr>
#
# SPDX-License-Identifier: MIT

import math
import scipy as sp
import numpy as np
from typing import Optional, TypedDict

from reproducibility.icml2026.baselines.rcit.rcit.utils import normalize


class RandomFourierFeatures(TypedDict):
    feat: np.ndarray
    w: np.ndarray
    b: np.ndarray


def random_fourier_features(
    x: np.ndarray,
    omega: Optional[np.ndarray] = None,
    num_f: int = 25,
    sigma: Optional[float] = None,
    seed: Optional[int] = None,
) -> RandomFourierFeatures:
    """
    Generate Random Fourier features
    See: https://github.com/ericstrobl/RCIT/blob/master/R/random_fourier_features.R
    """

    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=-1)

    size, dimensions = x.shape

    if omega is None:
        if sigma is None or (math.isinf(sigma) or sigma == 0):
            sigma = 1
        np.random.seed(seed)
        omega = 1 / sigma * sp.stats.norm.rvs(size=(num_f, dimensions))
        np.random.seed(seed)
        b = np.repeat(2 * np.pi * sp.stats.uniform.rvs(size=num_f)[:, None], size, axis=1)

    features = np.sqrt(2) * np.cos(np.einsum("fd, nd -> fn", omega, x) + b).T

    return RandomFourierFeatures(feat=features, w=omega, b=b)


def compute_normalized_rff(data: np.ndarray, num_f: int, seed, sigma: Optional[float]):
    data = normalize(data)

    fourier_features = random_fourier_features(
        data,
        num_f=num_f,
        sigma=sigma,
        seed=seed,
    )["feat"]

    normalized_fourier_features = normalize(fourier_features)

    return normalized_fourier_features
