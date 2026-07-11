"""Smoke + interface tests for DVM-AD."""

import importlib.util

import numpy as np
import pytest

from dvmad.core import DVMADCore

_HAS_PYOD = importlib.util.find_spec("pyod") is not None
pyod_missing = pytest.mark.skipif(not _HAS_PYOD, reason="pyod not installed")


def _toy_one_class(n_inliers=200, n_outliers=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X_in = rng.normal(0.0, 1.0, size=(n_inliers, d))
    X_out = rng.normal(6.0, 1.0, size=(n_outliers, d))
    X = np.vstack([X_in, X_out])
    y = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])
    return X_in, X, y


# ---------------------------------------------------------------------------
# Core (NumPy) interface
# ---------------------------------------------------------------------------
def test_core_fit_predict_shapes():
    X_train, X_test, _ = _toy_one_class()
    clf = DVMADCore(use_faiss=False).fit(X_train)
    scores = clf.predict(X_test)
    assert scores.shape == (X_test.shape[0],)
    assert np.all(np.isfinite(scores))


def test_core_separates_inliers_outliers():
    X_train, X_test, y_test = _toy_one_class()
    clf = DVMADCore(use_faiss=False).fit(X_train)
    scores = clf.predict(X_test)
    # Mean outlier score should be larger than mean inlier score.
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()


@pytest.mark.parametrize("mode", ["min", "max", "both"])
@pytest.mark.parametrize("artificial_mode", ["max", "min", "farthest"])
def test_core_modes_combinations(mode, artificial_mode):
    X_train, X_test, _ = _toy_one_class()
    clf = DVMADCore(
        mode=mode, artificial_mode=artificial_mode, use_faiss=False
    ).fit(X_train)
    scores = clf.predict(X_test)
    assert scores.shape[0] == X_test.shape[0]


def test_core_eps_validation():
    with pytest.raises(ValueError):
        DVMADCore(eps=0.0)
    with pytest.raises(ValueError):
        DVMADCore(eps=0.5)


def test_core_unfitted_raises():
    clf = DVMADCore(use_faiss=False)
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros((3, 4)))
    with pytest.raises(RuntimeError):
        clf.transform(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# PyOD wrapper (skipped if pyod unavailable)
# ---------------------------------------------------------------------------
@pyod_missing
def test_pyod_wrapper_basic():
    from dvmad import DVMAD

    X_train, X_test, y_test = _toy_one_class()
    clf = DVMAD(contamination=0.1, use_faiss=False).fit(X_train)

    # Required PyOD attributes
    assert hasattr(clf, "decision_scores_")
    assert hasattr(clf, "labels_")
    assert hasattr(clf, "threshold_")
    assert clf.decision_scores_.shape == (X_train.shape[0],)
    assert clf.labels_.shape == (X_train.shape[0],)

    # decision_function on test data
    scores = clf.decision_function(X_test)
    assert scores.shape == (X_test.shape[0],)

    # predict returns 0/1
    labels = clf.predict(X_test)
    assert set(np.unique(labels)).issubset({0, 1})

    # ROC AUC sanity check
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y_test, scores) > 0.9


@pyod_missing
def test_pyod_wrapper_predict_proba():
    from dvmad import DVMAD

    X_train, X_test, _ = _toy_one_class()
    clf = DVMAD(contamination=0.1, use_faiss=False).fit(X_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
