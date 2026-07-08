"""Test suite for the TimeLAVA reference implementation.

These tests check the structural invariants of the algorithm and the two
qualitative claims the paper makes on synthetic data (Theorem 5.1 score-shape
distinction and Theorem B.2 ranking preservation).
"""

import numpy as np
import pytest
from scipy.stats import kurtosis, spearmanr

from timelava import TimeLAVA, TimeLAVAConfig
from timelava.datasets import (
    inject_point_anomalies,
    js_divergence,
    make_reference,
    make_regime_shift,
    roc_auc,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Shape / API invariants
# --------------------------------------------------------------------------- #
def test_output_shapes(rng):
    X_ref, base = make_reference(800, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.03)
    res = TimeLAVA(TimeLAVAConfig(L=64, S=8)).fit(X_eval, X_ref)

    assert res.segment_values.shape == (res.n_eval_segments,)
    assert res.point_values.shape == (800,)
    assert res.transport_plan.shape == (
        res.n_eval_segments,
        res.n_ref_segments,
    )
    assert res.dual_potential_f.shape == (res.n_eval_segments,)
    assert np.all(np.isfinite(res.segment_values))
    assert np.all(np.isfinite(res.point_values))


def test_centering_property(rng):
    """v(x_i) is a centred quantity: it should sum to (approximately) zero."""
    X_ref, base = make_reference(800, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.03)
    res = TimeLAVA(TimeLAVAConfig(L=64, S=16)).fit(X_eval, X_ref)
    # v_i = -(phi_i - (S - phi_i)/(n-1))  =>  sum_i v_i = 0  exactly.
    assert abs(res.segment_values.sum()) < 1e-8


def test_rank_segments_consistency(rng):
    X_ref, base = make_reference(600, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.03)
    res = TimeLAVA(TimeLAVAConfig(L=48, S=24)).fit(X_eval, X_ref)
    order = res.rank_segments()
    vals = res.segment_values[order]
    assert np.all(np.diff(vals) <= 1e-12)  # descending


# --------------------------------------------------------------------------- #
# Behavioural / paper claims
# --------------------------------------------------------------------------- #
def test_anomalies_get_lower_values(rng):
    X_ref, base = make_reference(2000, rng)
    X_eval, mask = inject_point_anomalies(base, rng, frac=0.02, lo=4, hi=7)
    res = TimeLAVA(TimeLAVAConfig(L=64, S=4)).fit(X_eval, X_ref)

    anom = res.point_values[mask].mean()
    norm = res.point_values[~mask].mean()
    assert anom < norm
    assert roc_auc(mask, res.anomaly_scores()) > 0.80


def test_theorem_5_1_score_shape(rng):
    """Isolated anomalies -> heavy-tailed; regime shift -> spread/symmetric."""
    X_ref, base = make_reference(2000, rng)
    A, _ = inject_point_anomalies(base, rng, frac=0.05, lo=8, hi=12)
    B = make_regime_shift(base, rng)

    cfg = TimeLAVAConfig(L=64, S=8, kappa=2.0, reg=0.01)
    vA = TimeLAVA(cfg).fit(A, X_ref).segment_values
    vB = TimeLAVA(cfg).fit(B, X_ref).segment_values

    # Isolated anomalies produce a far heavier tail than the regime shift.
    assert kurtosis(vA) > kurtosis(vB) + 10
    # The two regimes are statistically distinguishable.
    assert js_divergence(vA, vB) > 0.2


def test_theorem_b_2_ranking_preserved(rng):
    """Ranking is preserved as epsilon shrinks (Spearman stays high)."""
    X_ref, base = make_reference(1200, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.04, lo=5, hi=8)

    ref = TimeLAVA(
        TimeLAVAConfig(L=64, S=16, kappa=1.0, reg=1e-4)
    ).fit(X_eval, X_ref).segment_values

    for eps in (1.0, 1e-1, 1e-2):
        v = TimeLAVA(
            TimeLAVAConfig(L=64, S=16, kappa=1.0, reg=eps)
        ).fit(X_eval, X_ref).segment_values
        assert spearmanr(v, ref).statistic > 0.90


def test_theorem_b_2_monotone_convergence(rng):
    """v_eps -> v monotonically as eps -> 0 (Theorem B.2 / Fig. 8).

    Regression test: a too-loose Sinkhorn convergence criterion returns
    *unconverged* potentials at small eps, producing a non-monotone L1
    error curve. The paper requires monotone O(eps)-like decay.
    """
    X_ref, base = make_reference(1200, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.04, lo=5, hi=8)

    ref = TimeLAVA(
        TimeLAVAConfig(L=64, S=16, kappa=1.0, reg=1e-4)
    ).fit(X_eval, X_ref).segment_values

    eps_grid = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    l1 = []
    for eps in eps_grid:
        v = TimeLAVA(
            TimeLAVAConfig(L=64, S=16, kappa=1.0, reg=eps)
        ).fit(X_eval, X_ref).segment_values
        l1.append(float(np.mean(np.abs(v - ref))))

    # Strictly decreasing as eps shrinks (allow tiny numerical slack).
    for earlier, later in zip(l1, l1[1:]):
        assert later <= earlier + 1e-7, f"non-monotone L1: {l1}"

    # Empirical convergence rate close to the paper's ~O(eps^0.88).
    slope = np.polyfit(np.log(eps_grid), np.log(l1), 1)[0]
    assert 0.7 < slope < 1.2, f"convergence rate O(eps^{slope:.2f}) off"


def test_label_consistency_path_runs(rng):
    """c > 0 path executes and consumes labels (Eq. 6)."""
    X_ref, base = make_reference(900, rng)
    X_eval, _ = inject_point_anomalies(base, rng, frac=0.03)
    t = np.arange(900)
    yr = (np.sin(2 * np.pi * t / 60) > 0).astype(int)
    ye = yr.copy()
    res = TimeLAVA(TimeLAVAConfig(L=60, S=60, c=1.0)).fit(
        X_eval, X_ref, y_eval=ye, y_ref=yr
    )
    assert res.segment_values.shape == (res.n_eval_segments,)
    with pytest.raises(ValueError):
        TimeLAVA(TimeLAVAConfig(L=60, S=60, c=1.0)).fit(X_eval, X_ref)
