"""
Base-10-hierarchical multiclass inference for models with a per-head class cap (e.g. TabPFN ≤10).

Decomposes encoded class indices 0..C-1 into decimal digits; trains one head per digit
position; fuses digit predict_proba in log space into a full C-way distribution.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Base10Decomposer:
    """Base-10 digit decomposition over encoded class indices 0..C-1."""

    def __init__(self, C: int):
        self.C = int(C)
        self.D = int(np.ceil(np.log10(self.C)))

    def decompose_class(self, c: int) -> List[int]:
        """Digits for class index c, least significant digit first."""
        c = int(c)
        return [int((c // (10**i)) % 10) for i in range(self.D)]

    def digits_for_array(self, arr: np.ndarray) -> List[np.ndarray]:
        arr = np.asarray(arr, dtype=int)
        return [
            ((arr // (10**i)) % 10).astype(int) for i in range(self.D)
        ]


def reconstruct_class_proba_from_digit_probs(
    decomposer: Base10Decomposer,
    digit_probs: List[np.ndarray],
    C: int,
) -> np.ndarray:
    """
    Combine per-digit predict_proba outputs into (n_test, C) class probabilities.

    digit_probs[i] has shape (n_test, k_i) with k_i <= 10.
    """
    n_test = digit_probs[0].shape[0]
    log_p = np.zeros((n_test, C), dtype=np.float64)
    eps = 1e-12
    for c in range(C):
        class_digits = decomposer.decompose_class(c)
        log_p_c = np.zeros(n_test, dtype=np.float64)
        for i, d_i in enumerate(class_digits):
            probs_i = digit_probs[i]
            d_i = int(d_i)
            if d_i < 0 or d_i >= probs_i.shape[1]:
                log_p_c += np.log(eps)
            else:
                log_p_c += np.log(probs_i[:, d_i] + eps)
        log_p[:, c] = log_p_c
    log_p_max = np.max(log_p, axis=1, keepdims=True)
    p = np.exp(log_p - log_p_max)
    p /= np.sum(p, axis=1, keepdims=True)
    return p
