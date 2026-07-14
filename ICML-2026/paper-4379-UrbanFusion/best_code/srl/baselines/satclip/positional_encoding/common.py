"""
Original code from: https://github.com/microsoft/satclip

Original source:
Klemmer, Konstantin; Rolf, Esther; Robinson, Caleb; Mackey, Lester; Rußwurm, Marc.
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."
arXiv preprint published November 28, 2023. Later published as conference paper at AAAI 2025.
"""

import math

import numpy as np


def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
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
