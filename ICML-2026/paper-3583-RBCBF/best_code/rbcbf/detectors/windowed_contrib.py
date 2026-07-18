"""Sliding-window contribution detector."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence, Tuple

from .base import Detector, DetectorOutput, StepObservation


class WindowedContributionDetector(Detector):
    """Detector that focuses on a recent window of contributions."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        config: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(name, labels, config)
        cfg = self.config
        self.alpha = float(cfg.get("alpha", 0.6))
        self.window = max(1, int(cfg.get("window", 8)))
        self.weights = self._build_label_weights(cfg.get("label_weights"))
        self._buffer: Deque[Tuple[int, float]] = deque()
        self._buffer_mass = 0.0
        self._seen_candidates: set[int] = set()
        self._last_new: List[int] = []

    def _reset_state(self) -> None:
        self._buffer.clear()
        self._buffer_mass = 0.0
        self._seen_candidates = set()
        self._last_new = []

    def update(self, obs: StepObservation) -> DetectorOutput:
        contrib = self._compute_contribution(obs)
        self._buffer.append((obs.step_index, contrib))
        self._buffer_mass += contrib
        if len(self._buffer) > self.window:
            _, removed = self._buffer.popleft()
            self._buffer_mass -= removed
        candidates, selected_mass = self._select_candidates()
        coverage = selected_mass / self._buffer_mass if self._buffer_mass > 0 else 0.0
        triggered = self._update_new_candidates(candidates, obs.step_index)
        meta = {
            "alpha": self.alpha,
            "window": self.window,
            "selected_mass": selected_mass,
            "window_mass": self._buffer_mass,
            "new_candidates": list(self._last_new),
        }
        return DetectorOutput(
            step_index=obs.step_index,
            candidates=candidates,
            score=contrib,
            coverage=coverage,
            total_mass=self._buffer_mass,
            triggered=triggered,
            meta=meta,
        )

    def _compute_contribution(self, obs: StepObservation) -> float:
        return float(
            sum(weight * float(delta) for weight, delta in zip(self.weights, obs.delta_minus))
        )

    def _select_candidates(self) -> Tuple[List[int], float]:
        if self._buffer_mass <= 0:
            return [], 0.0
        target = self.alpha * self._buffer_mass
        selected: List[int] = []
        selected_mass = 0.0
        for idx, contrib in self._buffer:
            if contrib <= 0:
                continue
            selected.append(idx)
            selected_mass += contrib
            if selected_mass >= target:
                break
        return selected, selected_mass

    def _update_new_candidates(self, candidates: List[int], current_step: int) -> bool:
        new_steps = [idx for idx in candidates if idx not in self._seen_candidates]
        self._last_new = new_steps
        for idx in new_steps:
            self._seen_candidates.add(idx)
        return current_step in new_steps


__all__ = ["WindowedContributionDetector"]
