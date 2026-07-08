"""Synthetic data generators and lightweight metrics.

Kept dependency-free (numpy only) so the notebook, the example script and
the tests can all reuse them. The generators mirror the synthetic protocols
described in Section 6.1 and Appendices B.2 / B.5 of the paper.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_reference",
    "inject_point_anomalies",
    "make_regime_shift",
    "roc_auc",
    "js_divergence",
]


def make_reference(T: int, rng: np.random.Generator, noise: float = 0.05):
    """Clean reference series  x(t) = sin(2 pi t/100) + 0.5 sin(2 pi t/25) + e.

    This is exactly the clean signal used in the Theorem 5.1 validation
    (Appendix B.2).
    """
    t = np.arange(T)
    base = np.sin(2 * np.pi * t / 100) + 0.5 * np.sin(2 * np.pi * t / 25)
    return base + rng.normal(0, noise, T), base


def inject_point_anomalies(
    base: np.ndarray,
    rng: np.random.Generator,
    frac: float = 0.02,
    lo: float = 4.0,
    hi: float = 7.0,
    noise: float = 0.05,
):
    """Add isolated spike anomalies to a copy of ``base``.

    Returns
    -------
    X : contaminated series
    is_anom : boolean mask of anomalous time points
    """
    T = base.shape[0]
    X = base + rng.normal(0, noise, T)
    k = int(frac * T)
    idx = rng.choice(T, size=k, replace=False)
    X[idx] += rng.choice([-1, 1], size=k) * rng.uniform(lo, hi, k)
    mask = np.zeros(T, dtype=bool)
    mask[idx] = True
    return X, mask


def make_regime_shift(
    base: np.ndarray, rng: np.random.Generator, noise: float = 0.05
):
    """Systematic frequency/amplitude change at the midpoint (Dataset B)."""
    T = base.shape[0]
    t = np.arange(T)
    B = base.copy()
    half = T // 2
    B[half:] = 1.8 * np.sin(2 * np.pi * t[half:] / 40) + 0.5 * np.sin(
        2 * np.pi * t[half:] / 15
    )
    return B + rng.normal(0, noise, T)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (no sklearn dependency)."""
    y_true = np.asarray(y_true).astype(bool)
    pos = score[y_true]
    neg = score[~y_true]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """Jensen-Shannon divergence between two empirical value distributions."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa / pa.sum() + 1e-12
    pb = pb / pb.sum() + 1e-12
    mix = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log2(p / q))
    return float(0.5 * kl(pa, mix) + 0.5 * kl(pb, mix))
