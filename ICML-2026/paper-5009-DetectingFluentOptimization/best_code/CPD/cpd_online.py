"""Online two-sided CUSUM detector on standardized entropy streams."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cpd_types import (
    EntropyLike,
    OnlineCUSUMConfig,
    OnlineCUSUMEvent,
    OnlineCUSUMState,
)


LOGGER = logging.getLogger(__name__)
MAD_SCALE = 1.4826
EPS = 1e-6


def _robust_location_scale(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional")
    if arr.size == 0:
        raise ValueError("Cannot compute baseline from empty sequence")
    if np.isnan(arr).any():
        raise ValueError("Baseline sequence contains NaNs")
    med = float(np.median(arr))
    mad = float(MAD_SCALE * np.median(np.abs(arr - med)))
    return med, max(mad, EPS)


def initialize_online(
    H0: Optional[Sequence[float]], cfg: OnlineCUSUMConfig
) -> OnlineCUSUMState:
    """Initialise a ``OnlineCUSUMState`` optionally using a warm-up sequence."""

    state = OnlineCUSUMState(
        k=cfg.k,
        h=cfg.h,
        baseline_window=cfg.baseline_window,
        m0=cfg.m0,
        s0=cfg.s0,
        reset_after_alarm=cfg.reset_after_alarm,
    )
    if state.m0 is not None and state.s0 is not None:
        state.baseline_ready = True

    if H0 is not None:
        arr = np.asarray(H0, dtype=float)
        for val in arr:
            state.baseline_buffer.append(float(val))
        if not state.baseline_ready and len(state.baseline_buffer) >= state.baseline_window:
            window = state.baseline_buffer[: state.baseline_window]
            state.m0, state.s0 = _robust_location_scale(window)
            state.baseline_ready = True

    # Adaptive k: set k = -k_scale * s0, bounded by k_min
    if cfg.adaptive_k and state.baseline_ready and state.s0 is not None:
        adaptive_k_val = max(-cfg.k_scale * state.s0, cfg.k_min)
        state.k = adaptive_k_val
        LOGGER.debug("Adaptive k set to %.4f (k_scale=%.2f, s0=%.4f, k_min=%.2f)",
                      adaptive_k_val, cfg.k_scale, state.s0, cfg.k_min)
    return state


def update(state: OnlineCUSUMState, H_t: float) -> Tuple[OnlineCUSUMState, OnlineCUSUMEvent]:
    """Advance the online detector by one observation."""

    state.t += 1
    entropy = float(H_t)
    state.baseline_buffer.append(entropy)
    if len(state.baseline_buffer) > max(state.baseline_window, 1):
        del state.baseline_buffer[:-state.baseline_window]

    if state.baseline_buffer:
        window_vals = np.asarray(state.baseline_buffer, dtype=float)
        window_mean = float(np.mean(window_vals))
        window_variance = float(np.var(window_vals))
        window_median = float(np.median(window_vals))
        mad = float(np.median(np.abs(window_vals - window_median)))
        window_scale = float(MAD_SCALE * mad)
    else:
        window_mean = float("nan")
        window_variance = float("nan")
        window_median = float("nan")
        window_scale = float("nan")

    z_t = 0.0
    if not state.baseline_ready and len(state.baseline_buffer) >= state.baseline_window:
        window = state.baseline_buffer[: state.baseline_window]
        state.m0, state.s0 = _robust_location_scale(window)
        state.baseline_ready = True

    if state.baseline_ready and state.s0 is not None:
        z_t = (entropy - state.m0) / max(state.s0, EPS)

        state.W_plus = max(0.0, state.W_plus + (z_t - state.k))
        state.W_minus = max(0.0, state.W_minus - (z_t + state.k))

        alarm_triggered = max(state.W_plus, state.W_minus) >= state.h
        if alarm_triggered:
            if not state.alarm:
                state.alarm = True
                state.T_alarm = state.t
        if alarm_triggered and state.reset_after_alarm:
            state.W_plus = 0.0
            state.W_minus = 0.0
    else:
        alarm_triggered = False

    event = OnlineCUSUMEvent(
        t=state.t,
        entropy=entropy,
        baseline_mean=window_mean,
        baseline_variance=window_variance,
        baseline_median=window_median,
        baseline_scale=window_scale,
        z_t=z_t,
        W_plus=state.W_plus,
        W_minus=state.W_minus,
        alarm=alarm_triggered,
    )
    return state, event


def run_full(
    H: Sequence[float],
    cfg: OnlineCUSUMConfig,
    *,
    baseline: Optional[Sequence[float]] = None,
) -> Tuple[OnlineCUSUMState, List[OnlineCUSUMEvent]]:
    state = initialize_online(baseline, cfg)
    events: List[OnlineCUSUMEvent] = []
    for h in H:
        state, event = update(state, float(h))
        events.append(event)
    return state, events


def calibrate_online_h(
    H_benign_list: Iterable[EntropyLike],
    cfg: OnlineCUSUMConfig,
    target_false_alarm_per_token: float = 1 / 1000,
    max_iter: int = 20,
    h_min: float = 0.5,
    h_max: float = 64.0,
) -> float:
    """Calibrate ``h`` to achieve the desired false-alarm rate on benign data."""

    benign = [np.asarray(H, dtype=float) for H in H_benign_list]
    if not benign:
        raise ValueError("No benign sequences provided for calibration")
    if target_false_alarm_per_token <= 0:
        raise ValueError("target_false_alarm_per_token must be positive")

    def false_alarm_rate(threshold: float) -> float:
        cfg_local = replace(cfg, h=threshold)
        total_tokens = 0
        alarms = 0
        for seq in benign:
            _, events = run_full(seq, cfg_local)
            total_tokens += len(events)
            alarms += sum(1 for ev in events if ev.alarm)
        return alarms / max(total_tokens, 1)

    # Expand search interval if needed.
    h_low = h_min
    h_high = max(h_max, cfg.h)
    rate = false_alarm_rate(h_high)
    iter_guard = 0
    while rate > target_false_alarm_per_token and iter_guard < max_iter:
        h_high *= 2.0
        rate = false_alarm_rate(h_high)
        iter_guard += 1

    for _ in range(max_iter):
        mid = 0.5 * (h_low + h_high)
        rate = false_alarm_rate(mid)
        if rate > target_false_alarm_per_token:
            h_low = mid
        else:
            h_high = mid
        if abs(h_high - h_low) < 1e-3:
            break

    return h_high


def save_online_state(path: str, state: OnlineCUSUMState) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
