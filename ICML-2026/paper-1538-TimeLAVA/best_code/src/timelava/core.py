"""The TimeLAVA estimator and its result container.

Ties together segmentation, wavelet features, the WSW cost, the unbalanced
OT solve and the value transform -- i.e. Algorithm 1 and Algorithm 2 of the
paper, end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .config import TimeLAVAConfig
from .label_consistency import conditional_wasserstein_matrix
from .segmentation import sliding_window_segments
from .uot import psi_kappa, unbalanced_sinkhorn_dual
from .wavelet import wavelet_cost_matrix, wavelet_features

__all__ = ["TimeLAVA", "TimeLAVAResult"]


@dataclass
class TimeLAVA:
    """Learning-agnostic time-series data valuation (TimeLAVA).

    Examples
    --------
    Unsupervised (anomaly detection / forecasting pruning), ``c = 0``::

        tl = TimeLAVA(TimeLAVAConfig(L=64, S=8, kappa=2.0, reg=0.01))
        result = tl.fit(X_eval, X_ref)
        result.segment_values   # one value per segment
        result.point_values     # one value per time step

    Supervised (label-noise detection), ``c = 1``::

        cfg = TimeLAVAConfig(L=64, S=64, c=1.0)
        result = TimeLAVA(cfg).fit(X_eval, X_ref, y_eval=ye, y_ref=yr)
    """

    config: TimeLAVAConfig = field(default_factory=TimeLAVAConfig)

    # ---- public API ------------------------------------------------------ #
    def fit(
        self,
        X_eval: np.ndarray,
        X_ref: np.ndarray,
        y_eval: Optional[Sequence] = None,
        y_ref: Optional[Sequence] = None,
    ) -> "TimeLAVAResult":
        """Run the full TimeLAVA pipeline (Algorithm 1 + Algorithm 2)."""
        cfg = self.config

        # --- segmentation (Section 3.2) -------------------------------- #
        seg_eval, starts_eval = sliding_window_segments(X_eval, cfg.L, cfg.S)
        seg_ref, starts_ref = sliding_window_segments(X_ref, cfg.L, cfg.S)
        n, m = seg_eval.shape[0], seg_ref.shape[0]

        seg_eval_p = self._maybe_normalize(seg_eval)
        seg_ref_p = self._maybe_normalize(seg_ref)

        # --- Algorithm 1, line 1: wavelet representations -------------- #
        feats_eval = wavelet_features(seg_eval_p, cfg.wavelet, cfg.level)
        feats_ref = wavelet_features(seg_ref_p, cfg.wavelet, cfg.level)

        # --- Algorithm 1, lines 2-12: pairwise cost matrix D^(W) ------ #
        D = wavelet_cost_matrix(feats_eval, feats_ref)

        if cfg.c > 0.0:
            if y_eval is None or y_ref is None:
                raise ValueError("c > 0 requires y_eval and y_ref (Eq. 6).")
            ye = self._segment_labels(y_eval, starts_eval, cfg.L, n)
            yr = self._segment_labels(y_ref, starts_ref, cfg.L, m)
            W_lab = conditional_wasserstein_matrix(
                feats_eval, ye, feats_ref, yr
            )
            D = D + cfg.c * W_lab

        # --- Algorithm 1, line 13: entropy-regularised UOT ------------ #
        a = np.full(n, 1.0 / n)  # Delta_n = 1_n / n
        b = np.full(m, 1.0 / m)  # Delta_m = 1_m / m
        f_star, g_star, T = unbalanced_sinkhorn_dual(
            a,
            b,
            D,
            reg=cfg.reg,
            kappa=cfg.kappa,
            numItermax=cfg.numItermax,
            stopThr=cfg.stopThr,
        )
        T = np.asarray(T, dtype=np.float64)

        # --- Algorithm 1, lines 14-20: data values -------------------- #
        phi = psi_kappa(f_star, cfg.kappa)  # phi_i = psi_kappa(f*_i)
        S_sum = phi.sum()  # S = sum_j phi_j
        if n > 1:
            # v_eps(x_i) = -( phi_i - (S - phi_i) / (n - 1) )
            seg_values = -(phi - (S_sum - phi) / (n - 1))
        else:
            seg_values = -phi.copy()

        # --- Algorithm 2: point-wise aggregation ---------------------- #
        T_eval = np.asarray(X_eval).shape[0]
        point_values = self._pointwise(
            seg_values, starts_eval, cfg.L, T_eval
        )

        return TimeLAVAResult(
            segment_values=seg_values,
            point_values=point_values,
            segment_starts=starts_eval,
            transport_plan=T,
            dual_potential_f=f_star,
            cost_matrix=D,
            config=cfg,
            n_eval_segments=n,
            n_ref_segments=m,
        )

    # ---- helpers --------------------------------------------------------- #
    def _maybe_normalize(self, segments: np.ndarray) -> np.ndarray:
        if not self.config.normalize_segments:
            return segments
        mu = segments.mean(axis=1, keepdims=True)
        sd = segments.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return (segments - mu) / sd

    @staticmethod
    def _segment_labels(
        y: Sequence, starts: np.ndarray, L: int, n_seg: int
    ) -> np.ndarray:
        """Resolve a per-segment label vector.

        Accepts labels given per time step (majority vote over the window)
        or already given per segment.
        """
        y = np.asarray(y)
        if y.shape[0] == n_seg:
            return y
        out = np.empty(n_seg, dtype=y.dtype)
        for k, s in enumerate(starts):
            window = y[s : s + L]
            vals, counts = np.unique(window, return_counts=True)
            out[k] = vals[np.argmax(counts)]
        return out

    @staticmethod
    def _pointwise(
        seg_values: np.ndarray, starts: np.ndarray, L: int, T: int
    ) -> np.ndarray:
        """Algorithm 2: v(t) = mean of v(x_i) over segments covering t."""
        sum_t = np.zeros(T, dtype=np.float64)
        count_t = np.zeros(T, dtype=np.float64)
        for val, s in zip(seg_values, starts):
            sum_t[s : s + L] += val
            count_t[s : s + L] += 1.0
        out = np.zeros(T, dtype=np.float64)
        covered = count_t > 0
        out[covered] = sum_t[covered] / count_t[covered]
        return out


@dataclass
class TimeLAVAResult:
    """Container for everything produced by :meth:`TimeLAVA.fit`."""

    segment_values: np.ndarray  # v_eps(x_i), shape (n,)
    point_values: np.ndarray  # v(t),       shape (T_eval,)
    segment_starts: np.ndarray  # start index of each segment, shape (n,)
    transport_plan: np.ndarray  # optimal T*, shape (n, m)
    dual_potential_f: np.ndarray  # f*_i,       shape (n,)
    cost_matrix: np.ndarray  # D^(W),      shape (n, m)
    config: TimeLAVAConfig
    n_eval_segments: int
    n_ref_segments: int

    # Convenience views ---------------------------------------------------- #
    def anomaly_scores(self) -> np.ndarray:
        """Point-wise anomaly score: higher = more anomalous.

        Per Section 6.1 / Theorem 4.5, anomalies *lower* a point's value
        (they increase distributional discrepancy). We therefore negate the
        point-wise values so larger scores indicate anomalies -- the
        convention used by the AUC / F1 evaluation in the paper.
        """
        return -self.point_values

    def corruption_scores(self) -> np.ndarray:
        """Segment-level corruption score: higher = more likely corrupted.

        Used for data-pruning noise detection (Section 6.2) and label-noise
        detection (Section 6.3): low-value segments are the suspect ones.
        """
        return -self.segment_values

    def rank_segments(self, ascending: bool = False) -> np.ndarray:
        """Indices of segments sorted by value (descending = best first)."""
        order = np.argsort(self.segment_values)
        return order if ascending else order[::-1]
