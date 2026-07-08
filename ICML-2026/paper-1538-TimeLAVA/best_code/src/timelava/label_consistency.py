"""Label-consistency term of the WSW cost matrix (Eq. 6).

For label pair (y_i, y'_j), this measures the (balanced) Wasserstein distance
between the class-conditional distributions of wavelet features, using the
wavelet distance itself as the ground metric. Because the value depends only
on the label pair, it is computed once per distinct pair and broadcast back
to the full (n, m) matrix -- exactly the O(N M K^2) shortcut of App. C.1.3.
"""

from __future__ import annotations

import numpy as np
import ot

from .wavelet import wavelet_cost_matrix

__all__ = ["conditional_wasserstein_matrix"]


def conditional_wasserstein_matrix(
    feats_eval: np.ndarray,
    y_eval: np.ndarray,
    feats_ref: np.ndarray,
    y_ref: np.ndarray,
) -> np.ndarray:
    """W_{d_wav}( mu_eval(.|y_i), mu_ref(.|y'_j) ) for every (i, j) pair."""
    n, m = feats_eval.shape[0], feats_ref.shape[0]
    W = np.zeros((n, m), dtype=np.float64)

    eval_labels = np.unique(y_eval)
    ref_labels = np.unique(y_ref)

    eval_buckets = {lab: feats_eval[y_eval == lab] for lab in eval_labels}
    ref_buckets = {lab: feats_ref[y_ref == lab] for lab in ref_labels}

    cache: dict[tuple, float] = {}
    for la in eval_labels:
        Fa = eval_buckets[la]
        for lb in ref_labels:
            Fb = ref_buckets[lb]
            if Fa.shape[0] == 0 or Fb.shape[0] == 0:
                cache[(la, lb)] = 0.0
                continue
            Cab = wavelet_cost_matrix(Fa, Fb)  # ground metric = d_wav
            wa = np.full(Fa.shape[0], 1.0 / Fa.shape[0])
            wb = np.full(Fb.shape[0], 1.0 / Fb.shape[0])
            cache[(la, lb)] = float(ot.emd2(wa, wb, Cab))

    for i in range(n):
        row_lab = y_eval[i]
        for j in range(m):
            W[i, j] = cache[(row_lab, y_ref[j])]
    return W
