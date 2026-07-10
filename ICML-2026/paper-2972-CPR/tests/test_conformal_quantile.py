"""Unit tests for split conformal quantile (path CP)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr.conformal.path_cp import (
    conformal_quantile,
    nonconformity_score,
    fit_path_threshold,
    predict_answer_set,
)


def test_conformal_quantile_valid():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    t_lo = conformal_quantile(scores, alpha=0.2)  # stricter coverage -> higher quantile
    t_hi = conformal_quantile(scores, alpha=0.5)  # looser -> lower quantile
    assert min(scores) <= t_hi <= max(scores)
    assert min(scores) <= t_lo <= max(scores)
    assert t_lo >= t_hi  # higher (1-alpha) quantile at alpha=0.2


def test_nonconformity_min_correct_path():
    path_conf = {
        "p1": {"scores": [0.9], "tail": "m.0answer"},
        "p2": {"scores": [0.2], "tail": "m.0wrong"},
        "p3": {"scores": [0.1], "tail": "m.0answer"},
    }
    s = nonconformity_score(path_conf, {"m.0answer"})
    assert abs(s - 0.1) < 1e-6


def test_nonconformity_miss():
    path_conf = {"p1": {"scores": [0.1], "tail": "m.0wrong"}}
    s = nonconformity_score(path_conf, {"m.0gold"})
    assert s == float("inf")


def test_predict_answer_set():
    path_conf = {
        "p1": {"scores": [0.5], "tail": "a"},
        "p2": {"scores": [0.2], "tail": "b"},
        "p3": {"scores": [0.8], "tail": "c"},
    }
    ans = predict_answer_set(path_conf, tau_hat=0.6)
    assert ans == {"a", "b"}


def test_fit_path_threshold():
    cal = [0.1, 0.2, 0.3, 0.4]
    tau, stats = fit_path_threshold(cal, alpha=0.25)
    assert "tau_hat" in stats
    assert stats["used"] == 4


if __name__ == "__main__":
    test_conformal_quantile_valid()
    test_nonconformity_min_correct_path()
    test_nonconformity_miss()
    test_predict_answer_set()
    test_fit_path_threshold()
    print("All tests passed.")
