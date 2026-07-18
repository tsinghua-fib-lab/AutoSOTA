"""Peak-based detector enforcing alpha coverage within a critical window."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

from .base import Detector, DetectorOutput, StepObservation
from .common import HysteresisTracker


def _find_local_peaks(
    series: List[Tuple[int, float]],
    max_peaks: int,
    min_prominence: float,
) -> List[Tuple[int, float]]:
    """Return (step, value) for prominent local peaks sorted by value."""
    peaks: List[Tuple[int, float]] = []
    if not series:
        return peaks
    for idx, (step, value) in enumerate(series):
        if value <= 0:
            continue
        left = series[idx - 1][1] if idx > 0 else None
        right = series[idx + 1][1] if idx + 1 < len(series) else None
        is_peak = True
        if left is not None and value < left:
            is_peak = False
        if right is not None and value <= right:
            is_peak = False
        if not is_peak:
            continue
        if min_prominence > 0:
            left_drop = value - (left if left is not None else 0.0)
            right_drop = value - (right if right is not None else 0.0)
            if max(left_drop, right_drop) < min_prominence:
                continue
        peaks.append((step, value))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks[:max_peaks]


class PeakAlphaDetector(Detector):
    """Detects sparse high-risk steps via local peaks + alpha coverage."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
        beta_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(name, labels, config=config)
        cfg = self.config
        self.alpha = float(cfg.get("alpha", 0.6))
        self.window_pre = int(cfg.get("window_pre", 16))
        self.window_post = int(cfg.get("window_post", 0))
        self.neighbor_radius = int(cfg.get("neighbor_radius", 1))
        self.max_peaks = int(cfg.get("max_peaks", 5))
        self.min_prominence = float(cfg.get("min_prominence", 0.0))
        self.relative_prominence = float(cfg.get("relative_prominence", 0.0))
        self.trim_margin = int(cfg.get("trim_margin", self.window_pre + self.window_post + 8))
        self.beta_weights = list(beta_weights or [1.0] * len(self.labels))
        self.safe_threshold = float(cfg.get("safe_threshold", 0.0))
        self.hold_last = bool(cfg.get("hold_last", False))
        self.max_supplement = int(cfg.get("max_supplement", 10))
        hyst_cfg = cfg.get("hysteresis", {})
        self.hysteresis = HysteresisTracker(
            eps_on=float(hyst_cfg.get("eps_on", 0.015)),
            eps_off=float(hyst_cfg.get("eps_off", 0.015)),
            enter_steps=int(hyst_cfg.get("enter_steps", 2)),
            exit_steps=int(hyst_cfg.get("exit_steps", 2)),
        )
        self._buffer: deque[Tuple[int, float]] = deque()
        self._values: Dict[int, float] = {}
        self._hmins: Dict[int, float] = {}
        self._unsafe_start: int | None = None
        self._unsafe_end: int | None = None
        self._prev_candidates: set[int] = set()
        self._last_output: DetectorOutput | None = None

    def _reset_state(self) -> None:
        self.hysteresis.reset()
        self._buffer.clear()
        self._values.clear()
        self._hmins.clear()
        self._unsafe_start = None
        self._unsafe_end = None
        self._prev_candidates = set()
        self._last_output = None

    def _compute_contribution(self, obs: StepObservation) -> float:
        return float(sum(w * float(dm) for w, dm in zip(self.beta_weights, obs.delta_minus)))

    def _append_value(self, step: int, value: float, h_min: float) -> None:
        self._buffer.append((step, value))
        self._values[step] = value
        self._hmins[step] = h_min
        trim_threshold = step - self.trim_margin
        while self._buffer and self._buffer[0][0] < trim_threshold:
            old_step, _ = self._buffer.popleft()
            self._values.pop(old_step, None)
            self._hmins.pop(old_step, None)

    def _collect_window(self, current_step: int) -> Tuple[List[Tuple[int, float]], bool]:
        if self._unsafe_start is None:
            return [], False
        window_start = self._unsafe_start - self.window_pre
        window_end = min(
            self._unsafe_end if self._unsafe_end is not None else current_step,
            current_step,
        )
        series: List[Tuple[int, float]] = []
        has_negative = False
        for step, value in self._buffer:
            if window_start <= step <= window_end and value > 0:
                series.append((step, value))
                h_min = self._hmins.get(step, 0.0)
                if h_min < self.safe_threshold:
                    has_negative = True
        return series, has_negative

    def _build_candidates(
        self,
        series: List[Tuple[int, float]],
    ) -> Tuple[List[int], float, float, float, Dict[str, object]]:
        if not series:
            return [], 0.0, 0.0, 0.0, {"peaks": []}
        series_map = {step: value for step, value in series}
        total_mass = float(sum(series_map.values()))
        if total_mass <= 0:
            return [], 0.0, 0.0, 0.0, {"peaks": []}
        abs_min_prom = self.min_prominence
        if self.relative_prominence > 0:
            abs_min_prom = max(
                abs_min_prom,
                self.relative_prominence * max(series_map.values()),
            )
        peaks = _find_local_peaks(series, self.max_peaks, abs_min_prom)
        target = self.alpha * total_mass
        candidates: Dict[int, float] = {}
        selected_mass = 0.0
        for peak_step, _ in peaks:
            for neighbor in range(peak_step - self.neighbor_radius, peak_step + self.neighbor_radius + 1):
                if neighbor in candidates:
                    continue
                if neighbor not in series_map:
                    continue
                candidates[neighbor] = series_map[neighbor]
                selected_mass += series_map[neighbor]
            if selected_mass >= target:
                break
        supplement_count = 0
        if selected_mass < target:
            for step, value in sorted(series_map.items(), key=lambda item: item[1], reverse=True):
                if step in candidates:
                    continue
                candidates[step] = value
                selected_mass += value
                supplement_count += 1
                if selected_mass >= target or supplement_count >= self.max_supplement:
                    break
        candidate_steps = sorted(candidates.keys())
        coverage = selected_mass / total_mass if total_mass > 0 else 0.0
        meta = {
            "peaks": peaks,
            "target_mass": target,
            "selected_mass": selected_mass,
            "supplement_count": supplement_count,
        }
        return candidate_steps, coverage, total_mass, total_mass, meta

    def update(self, obs: StepObservation) -> DetectorOutput:
        contribution = self._compute_contribution(obs)
        h_min = float(min(obs.h)) if obs.h else 0.0
        self._append_value(obs.step_index, contribution, h_min)
        unsafe_state, enter_evt, exit_evt = self.hysteresis.update(h_min)
        if enter_evt:
            self._unsafe_start = obs.step_index
            self._unsafe_end = self._unsafe_start + self.window_post
        if exit_evt:
            self._unsafe_start = None
            self._unsafe_end = None
        if self._unsafe_end is not None and obs.step_index > self._unsafe_end:
            unsafe_state = False
        window_series, has_negative = self._collect_window(obs.step_index) if unsafe_state else ([], False)
        if unsafe_state and not has_negative:
            window_series = []
        candidates, coverage, total_mass, window_total, peak_meta = self._build_candidates(window_series)
        candidate_set = set(candidates)
        new_candidates = sorted(candidate_set - self._prev_candidates)
        self._prev_candidates = candidate_set
        meta = {
            "unsafe_state": unsafe_state,
            "window_series_len": len(window_series),
            "window_start": (
                self._unsafe_start - self.window_pre if self._unsafe_start is not None else None
            ),
            "window_end": (
                min(self._unsafe_end or obs.step_index, obs.step_index)
                if self._unsafe_start is not None
                else None
            ),
            "contribution": contribution,
            "window_total": window_total,
            "window_coverage": coverage,
            "window_has_negative": has_negative,
        }
        meta.update(peak_meta)
        output = DetectorOutput(
            step_index=obs.step_index,
            candidates=candidates,
            score=contribution,
            coverage=coverage,
            total_mass=total_mass,
            triggered=obs.step_index in candidate_set,
            meta={
                **meta,
                "new_candidates": new_candidates,
            },
        )
        if unsafe_state:
            self._last_output = output
            return output
        if self.hold_last and self._last_output is not None:
            return self._last_output
        return output


__all__ = ["PeakAlphaDetector", "_find_local_peaks"]
