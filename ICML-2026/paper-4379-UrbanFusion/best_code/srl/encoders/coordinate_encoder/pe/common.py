#!/usr/bin/env python3
"""
Description: Implementation of calculation of frequency list for positional
encoding. Implementation by the paper "GEOGRAPHIC LOCATION ENCODING
WITH SPHERICAL SHARMONICS AND SINUSOIDAL REPRESENTATION NETWORKS"
"""

import math

import numpy as np


def _cal_freq_list(
    freq_init: str,
    frequency_num: int,
    max_radius: float,
    min_radius: float,
) -> np.ndarray:
    """
    Calculate the frequency list for positional encoding.

    Parameters:
    ----------
    freq_init : str
        Initialization method for frequency list. "random" or "geometric".
    frequency_num : int
        Number of frequencies.
    max_radius : float
        Maximum radius for frequency list.
    min_radius : float
        Minimum radius for frequency list.

    Returns:
    -------
    freq_list : np.ndarray
        Frequency list for positional encoding
    """
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = math.log(
            float(max_radius) / float(min_radius)
        ) / (frequency_num * 1.0 - 1)
        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment
        )
        freq_list = 1.0 / timescales
    return freq_list
