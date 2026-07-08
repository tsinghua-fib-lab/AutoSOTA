"""TimeLAVA: Learning-Agnostic Valuation for Time Series Data.

A faithful reference implementation of the ICML 2026 submission
"TimeLAVA: Learning-Agnostic Valuation for Time Series Data".

Public API
----------
    TimeLAVA, TimeLAVAResult, TimeLAVAConfig
    sliding_window_segments
    wavelet_features, wavelet_cost_matrix
    conditional_wasserstein_matrix
    unbalanced_sinkhorn_dual, psi_kappa
"""

from __future__ import annotations

from . import datasets
from .config import TimeLAVAConfig
from .core import TimeLAVA, TimeLAVAResult
from .label_consistency import conditional_wasserstein_matrix
from .segmentation import sliding_window_segments
from .uot import psi_kappa, unbalanced_sinkhorn_dual
from .wavelet import wavelet_cost_matrix, wavelet_features

__version__ = "0.1.0"

__all__ = [
    "TimeLAVA",
    "TimeLAVAResult",
    "TimeLAVAConfig",
    "sliding_window_segments",
    "wavelet_features",
    "wavelet_cost_matrix",
    "conditional_wasserstein_matrix",
    "unbalanced_sinkhorn_dual",
    "psi_kappa",
    "datasets",
    "__version__",
]
