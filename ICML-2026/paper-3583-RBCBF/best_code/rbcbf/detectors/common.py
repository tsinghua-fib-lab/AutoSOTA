"""Shared utilities for Stage-1.1 detectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

EPS = 1e-8


def compute_weight(value: float, tau: float, slope: float, family: str) -> float:
    """Compute soft activation weight for a single constraint."""
    if family == "sigmoid":
        s = slope if slope > EPS else 1.0
        z = (tau - value) / s
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)
    if family == "softplus":
        s = slope if slope > EPS else 1.0
        z = (tau - value) / s
        if z > 20:
            return z
        return math.log1p(math.exp(z))
    # relu / default
    return max(tau - value, 0.0)


def compute_active_indices(values: Sequence[float], tau_act: float) -> List[int]:
    return [idx for idx, val in enumerate(values) if val < tau_act]


@dataclass
class ContributionSnapshot:
    c_values: List[float]
    active_indices: List[int]
    sum_active: float
    joint_value: float
    weights: List[float]
    h_min: float


def compute_contributions(
    h_values: Sequence[float],
    delta_minus: Sequence[float],
    betas: Sequence[float],
    tau: float,
    tau_act: float,
    slope: float,
    family: str,
) -> ContributionSnapshot:
    weights = [
        compute_weight(h, tau=tau, slope=slope, family=family) for h in h_values
    ]
    c_vals = [w * float(dm) for w, dm in zip(weights, delta_minus)]
    active_indices = compute_active_indices(h_values, tau_act=tau_act)
    sum_active = float(sum(c_vals[idx] for idx in active_indices)) if active_indices else 0.0
    joint_value = float(
        sum(
            (c_vals[idx] * betas[idx] if idx < len(betas) else c_vals[idx])
            for idx in active_indices
        )
    )
    h_min = float(min(h_values)) if h_values else 0.0
    return ContributionSnapshot(
        c_values=c_vals,
        active_indices=active_indices,
        sum_active=sum_active,
        joint_value=joint_value,
        weights=weights,
        h_min=h_min,
    )


class HysteresisTracker:
    """Tracks unsafe state with hysteresis/debounce."""

    def __init__(self, eps_on: float, eps_off: float, enter_steps: int, exit_steps: int) -> None:
        self.eps_on = float(eps_on)
        self.eps_off = float(eps_off)
        self.enter_steps = max(1, int(enter_steps))
        self.exit_steps = max(1, int(exit_steps))
        self.state = False
        self._enter_count = 0
        self._exit_count = 0

    def reset(self) -> None:
        self.state = False
        self._enter_count = 0
        self._exit_count = 0

    def update(self, h_min: float) -> Tuple[bool, bool, bool]:
        enter_event = False
        exit_event = False
        if self.state:
            if h_min > self.eps_off:
                self._exit_count += 1
                if self._exit_count >= self.exit_steps:
                    self.state = False
                    exit_event = True
                    self._enter_count = 0
                    self._exit_count = 0
            else:
                self._exit_count = 0
        else:
            if h_min < -self.eps_on:
                self._enter_count += 1
                if self._enter_count >= self.enter_steps:
                    self.state = True
                    enter_event = True
                    self._exit_count = 0
                    self._enter_count = 0
            else:
                self._enter_count = 0
        return self.state, enter_event, exit_event

    def params_snapshot(self) -> dict:
        return {
            "eps_on": self.eps_on,
            "eps_off": self.eps_off,
            "enter_steps": self.enter_steps,
            "exit_steps": self.exit_steps,
        }


@dataclass
class SeriesSelection:
    candidates: List[int]
    selected_mass: float
    total_mass: float

    @property
    def coverage(self) -> float:
        if self.total_mass <= 0:
            return 0.0
        return self.selected_mass / self.total_mass


class SeriesSelector:
    """Maintains causal coverage selection with either alpha cumulative or sliding window."""

    def __init__(self, method: str, alpha: float, window: int | None = None) -> None:
        self.method = method
        self.alpha = float(alpha)
        self.window = max(1, int(window)) if window else 1
        self.series: List[float] = []
        self.step_indices: List[int] = []
        self.total_mass = 0.0
        self.window_buffer: List[float] = []
        self.window_total = 0.0

    def update(self, value: float, step_index: int | None = None) -> SeriesSelection:
        if step_index is None:
            step_index = len(self.series)
        self.series.append(float(value))
        self.step_indices.append(int(step_index))
        if self.method == "window":
            self._append_window(value)
            candidates, selected = self._select_window()
            total = self.window_total
        else:
            self.total_mass += float(value)
            candidates, selected = self._select_alpha()
            total = self.total_mass
        return SeriesSelection(candidates=candidates, selected_mass=selected, total_mass=total)

    def _append_window(self, value: float) -> None:
        self.window_buffer.append(float(value))
        self.window_total += float(value)
        if len(self.window_buffer) > self.window:
            removed = self.window_buffer.pop(0)
            self.window_total -= removed

    def _select_alpha(self) -> Tuple[List[int], float]:
        if self.total_mass <= 0:
            return [], 0.0
        target = self.alpha * self.total_mass
        selected_mass = 0.0
        candidates: List[int] = []
        for idx, contrib in enumerate(self.series):
            if contrib <= 0:
                continue
            candidates.append(self.step_indices[idx])
            selected_mass += contrib
            if selected_mass >= target:
                break
        return candidates, selected_mass

    def _select_window(self) -> Tuple[List[int], float]:
        if self.window_total <= 0:
            return [], 0.0
        target = self.alpha * self.window_total
        selected_mass = 0.0
        candidates: List[int] = []
        start_index = len(self.series) - len(self.window_buffer)
        for offset, contrib in enumerate(self.window_buffer):
            if contrib <= 0:
                continue
            idx = start_index + offset
            candidates.append(self.step_indices[idx])
            selected_mass += contrib
            if selected_mass >= target:
                break
        return candidates, selected_mass


__all__ = [
    "compute_weight",
    "compute_active_indices",
    "compute_contributions",
    "ContributionSnapshot",
    "HysteresisTracker",
    "SeriesSelector",
    "SeriesSelection",
]
