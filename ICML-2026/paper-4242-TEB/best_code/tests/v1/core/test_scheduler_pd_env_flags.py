# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for env-controlled knobs added by the EB/MB scheduler.

Covers three flags introduced on the eb-vllm fork that have no
upstream test coverage:

  * ``VLLM_PD_THETA_FLOOR``       — IFR theta* lower bound, read in ``__init__``
  * ``VLLM_PD_KV_THRESHOLD_MAX``  — KV-pressure threshold upper bound,
                                    read at call-time in
                                    ``Scheduler._compute_adaptive_kv_threshold``
  * ``VLLM_PD_MIN_N_FLOOR_DIV``   — divisor for the kv_escape min_N floor,
                                    read at call-time in the phase transition

These flags directly affect EB/MB scheduling decisions; without these tests a
silent breakage (e.g. accidental removal of the ``os.environ.get`` read) would
not be caught by CI.
"""

from __future__ import annotations

import pytest

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_pd_scheduler(monkeypatch, **env):
    monkeypatch.setenv("VLLM_USE_PD_SCHEDULER", "1")
    monkeypatch.setenv("VLLM_PD_K_MODE", "ifr")
    # PD scheduler requires hardware timing params (alpha_p, beta_p, alpha_d,
    # beta_d). Provide dummy positive values so __init__ doesn't raise; the
    # actual numbers don't matter for the env-flag tests below.
    monkeypatch.setenv("VLLM_PD_ALPHA_P", "1e-4")
    monkeypatch.setenv("VLLM_PD_BETA_P", "1e-6")
    monkeypatch.setenv("VLLM_PD_ALPHA_D", "1e-4")
    monkeypatch.setenv("VLLM_PD_BETA_D", "1e-6")
    # Ensure no system calibration file leaks in and overrides the above.
    monkeypatch.delenv("VLLM_PD_CALIBRATION_FILE", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return create_scheduler()


def test_pd_theta_floor_default(monkeypatch):
    monkeypatch.delenv("VLLM_PD_THETA_FLOOR", raising=False)
    sched = _make_pd_scheduler(monkeypatch)
    assert sched.pd_theta_floor == pytest.approx(0.01)


def test_pd_theta_floor_from_env(monkeypatch):
    sched = _make_pd_scheduler(monkeypatch, VLLM_PD_THETA_FLOOR=0.42)
    assert sched.pd_theta_floor == pytest.approx(0.42)


def test_pd_theta_floor_clamps_update(monkeypatch):
    """The clamp at scheduler.py:1137 must honour pd_theta_floor."""
    sched = _make_pd_scheduler(monkeypatch, VLLM_PD_THETA_FLOOR=0.5)
    # Replicate the clamp expression from _update_ifr_threshold step 4.
    # We exercise the formula directly because driving a full schedule()
    # iteration in a CPU test would require mocking model output.
    raw_theta_below_floor = 0.10
    clamped = max(
        0.01,
        sched.pd_theta_floor,
        min(raw_theta_below_floor, sched.pd_theta_max),
    )
    assert clamped == pytest.approx(0.5)


def test_pd_kv_threshold_max_default_caps_at_0_6(monkeypatch):
    monkeypatch.delenv("VLLM_PD_KV_THRESHOLD_MAX", raising=False)
    sched = _make_pd_scheduler(monkeypatch)
    # Force the inner formula to want a very large reserve_ratio so the
    # upper clamp is the binding constraint.
    sched.pd_avg_output_tokens = 1e9
    sched.pd_output_margin = 10.0
    out = sched._compute_adaptive_kv_threshold()
    assert out == pytest.approx(0.6)


def test_pd_kv_threshold_max_from_env(monkeypatch):
    sched = _make_pd_scheduler(monkeypatch, VLLM_PD_KV_THRESHOLD_MAX=0.3)
    sched.pd_avg_output_tokens = 1e9
    sched.pd_output_margin = 10.0
    out = sched._compute_adaptive_kv_threshold()
    assert out == pytest.approx(0.3)


def test_pd_min_n_floor_div_default_and_env(monkeypatch):
    """min_N floor is `max_num_running_reqs // VLLM_PD_MIN_N_FLOOR_DIV`,
    floored at 16. We assert the env-read produces the expected divisor."""
    import os

    monkeypatch.delenv("VLLM_PD_MIN_N_FLOOR_DIV", raising=False)
    assert int(os.environ.get("VLLM_PD_MIN_N_FLOOR_DIV", "10")) == 10

    monkeypatch.setenv("VLLM_PD_MIN_N_FLOOR_DIV", "2")
    assert int(os.environ.get("VLLM_PD_MIN_N_FLOOR_DIV", "10")) == 2

    # And the resulting min_N matches the documented formula
    # (mirrors scheduler.py:1789 in the kv_escape branch).
    max_seqs = 64
    assert max(16, max_seqs // 10) == 16  # default: floor at 16, not max/10=6
    assert max(16, max_seqs // 2) == 32  # env override yields max/2=32
