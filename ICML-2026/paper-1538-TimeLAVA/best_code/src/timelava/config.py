"""Hyper-parameter configuration for TimeLAVA.

Defaults follow Appendix C.1.3 of the paper:
    kappa = 2.0, epsilon (entropy reg.) = 0.01, wavelet = 'db4', level = 2.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TimeLAVAConfig"]


@dataclass
class TimeLAVAConfig:
    """Hyper-parameters for TimeLAVA.

    Attributes
    ----------
    L : int
        Segment (sliding-window) length.
    S : int
        Sliding-window stride. ``S == L`` gives non-overlapping segments,
        as used in the data-pruning and label-noise experiments; ``S < L``
        gives the overlapping windows required for point-wise valuation
        (anomaly detection).
    kappa : float
        UOT marginal-relaxation parameter ``kappa > 0`` (Eq. 5). Paper
        default 2.0; the method is stable for kappa in [1, 10].
    reg : float
        Entropy-regularisation strength ``epsilon`` for the Sinkhorn solver
        (Algorithm 1 line 13). Paper default 0.01.
    c : float
        Weight ``c >= 0`` of the label-consistency term in the cost matrix
        (Eq. 6). ``c = 0`` for purely unsupervised valuation (anomaly
        detection, forecasting pruning); ``c = 1`` for label-noise detection.
    wavelet : str
        Mother wavelet. Paper default ``"db4"`` (Daubechies-4).
    level : int
        DWT decomposition level. Paper default 2.
    numItermax : int
        Maximum Sinkhorn iterations.
    stopThr : float
        Sinkhorn convergence threshold.
    normalize_segments : bool
        If True, z-normalise every segment (per channel) before the wavelet
        transform. This is standard pre-processing for time-series valuation
        and matches the heterogeneous-scale benchmarks; disable it to operate
        on raw amplitudes.
    """

    L: int = 64
    S: int = 1
    kappa: float = 2.0
    reg: float = 0.01
    c: float = 0.0
    wavelet: str = "db4"
    level: int = 2
    numItermax: int = 2000
    stopThr: float = 1e-9
    normalize_segments: bool = True
