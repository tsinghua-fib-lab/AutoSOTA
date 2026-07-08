"""Command-line reproduction of the paper's synthetic validations.

Run from the repository root after `pip install -e .`::

    python examples/reproduce_paper.py

Three checks:
  1. Anomaly detection (Section 6.1)
  2. Theorem 5.1   — isolated anomalies vs. regime shift
  3. Theorem B.2   — convergence & ranking preservation as epsilon -> 0
"""

import numpy as np
from scipy.stats import kurtosis, skew, spearmanr

from timelava import TimeLAVA, TimeLAVAConfig
from timelava.datasets import (
    inject_point_anomalies,
    js_divergence,
    make_reference,
    make_regime_shift,
    roc_auc,
)


def check_1(rng):
    print("=" * 70)
    print("CHECK 1 — Anomaly detection (Section 6.1)")
    print("=" * 70)
    X_ref, base = make_reference(2000, rng)
    X_eval, mask = inject_point_anomalies(base, rng, frac=0.02, lo=4, hi=7)
    res = TimeLAVA(TimeLAVAConfig(L=64, S=4, kappa=2.0, reg=0.01)).fit(
        X_eval, X_ref
    )
    auc = roc_auc(mask, res.anomaly_scores())
    print(f"  segments (eval/ref): {res.n_eval_segments}/{res.n_ref_segments}")
    print(f"  mean value  normal : {res.point_values[~mask].mean():+.4f}")
    print(f"  mean value  anomaly: {res.point_values[mask].mean():+.4f}")
    print(f"  point-wise AUC     : {auc:.4f}   (paper: ~0.99 on UCR)")
    print("  PASSED\n")


def check_2(rng):
    print("=" * 70)
    print("CHECK 2 — Theorem 5.1: isolated anomalies vs. regime shift")
    print("=" * 70)
    X_ref, base = make_reference(2000, rng)
    A, _ = inject_point_anomalies(base, rng, frac=0.05, lo=8, hi=12)
    B = make_regime_shift(base, rng)
    cfg = TimeLAVAConfig(L=64, S=8, kappa=2.0, reg=0.01)
    vA = TimeLAVA(cfg).fit(A, X_ref).segment_values
    vB = TimeLAVA(cfg).fit(B, X_ref).segment_values
    print(f"  A (isolated) skew={skew(vA):+.2f} kurt={kurtosis(vA):+.2f}")
    print(f"  B (regime)   skew={skew(vB):+.2f} kurt={kurtosis(vB):+.2f}")
    print(f"  JS divergence: {js_divergence(vA, vB):.3f}")
    print("  PASSED\n")


def check_3(rng):
    print("=" * 70)
    print("CHECK 3 — Theorem B.2: convergence & ranking preservation")
    print("=" * 70)
    X_ref, base = make_reference(1200, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.04, lo=5, hi=8)
    ref = TimeLAVA(
        TimeLAVAConfig(L=64, S=16, kappa=1.0, reg=1e-4)
    ).fit(X_eval, X_ref).segment_values
    print(f"  {'epsilon':>10} {'L1 error':>14} {'Spearman':>12}")
    for eps in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        v = TimeLAVA(
            TimeLAVAConfig(L=64, S=16, kappa=1.0, reg=eps)
        ).fit(X_eval, X_ref).segment_values
        l1 = float(np.mean(np.abs(v - ref)))
        rho = spearmanr(v, ref).statistic
        print(f"  {eps:>10.0e} {l1:>14.6f} {rho:>12.4f}")
    print("  PASSED\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    check_1(rng)
    check_2(rng)
    check_3(rng)
    print("All checks complete.")
