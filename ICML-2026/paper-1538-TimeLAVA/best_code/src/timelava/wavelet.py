"""Multi-scale wavelet representation and the wavelet ground metric.

Implements Definition 4.1 (Discrete Wavelet Transform), Definition 4.2 /
Eq. 4 (wavelet distance), and the pairwise cost matrix of Algorithm 1
lines 2-12.
"""

from __future__ import annotations

import numpy as np
import pywt

__all__ = ["wavelet_features", "wavelet_cost_matrix"]


def _dwt_coeffs_1d(signal: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Flattened DWT coefficients of a 1-D signal.

    Uses Mallat's fast pyramidal algorithm (``pywt.wavedec``), giving the
    O(L) cost per segment claimed in Appendix C.1.3. The approximation and
    all detail coefficients are concatenated into a single vector so that
    the L1 norm of the difference is exactly the wavelet distance of Eq. 4.
    """
    coeffs = pywt.wavedec(
        signal, wavelet=wavelet, level=level, mode="periodization"
    )
    return np.concatenate([c.ravel() for c in coeffs])


def wavelet_features(
    segments: np.ndarray, wavelet: str = "db4", level: int = 2
) -> np.ndarray:
    """Compute Psi(x) for a batch of (n, L, d) segments.

    The DWT is applied independently along each feature dimension and the
    per-channel coefficient vectors are concatenated, as specified for the
    multivariate case in Definition 4.1.

    Returns
    -------
    feats : (n, P) array of stacked wavelet coefficients.
    """
    segments = np.asarray(segments, dtype=np.float64)
    n = segments.shape[0]
    d = segments.shape[2]
    feats = []
    for i in range(n):
        per_channel = [
            _dwt_coeffs_1d(segments[i, :, ch], wavelet, level)
            for ch in range(d)
        ]
        feats.append(np.concatenate(per_channel))
    return np.stack(feats, axis=0)


def wavelet_cost_matrix(
    feats_a: np.ndarray, feats_b: np.ndarray
) -> np.ndarray:
    """Pairwise wavelet distance  D_{ij} = || Psi(x_i) - Psi(x'_j) ||_1.

    This is the ground metric of Eq. 4 / Algorithm 1 line 4.
    """
    diff = np.abs(feats_a[:, None, :] - feats_b[None, :, :])
    return diff.sum(axis=2)
