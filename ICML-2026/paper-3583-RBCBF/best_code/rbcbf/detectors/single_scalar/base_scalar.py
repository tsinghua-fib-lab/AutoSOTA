"""Base utilities for single-scalar detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from rbcbf.detectors.base import Detector, DetectorOutput, StepObservation
from rbcbf.detectors.common import SeriesSelector


class ScalarBuilder(ABC):
    """Builds a scalar value from StepObservation."""

    def __init__(self, labels: Sequence[str], config: Dict[str, Any] | None = None) -> None:
        self.labels = list(labels)
        self.config = dict(config or {})
        self.last_meta: Dict[str, Any] = {}

    def reset(self) -> None:
        self.last_meta = {}

    @abstractmethod
    def build(self, obs: StepObservation) -> float:
        """Return scalar value for this step."""


class ScalarSeriesDetector(Detector):
    """Detector that consumes a scalar time series x_t and applies causal selection."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        builder: ScalarBuilder,
        selector_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(name, labels, config=selector_cfg)
        self.builder = builder
        self.selector_cfg = selector_cfg
        self.selector: SeriesSelector | None = None
        self._prev_candidates: set[int] = set()

    def _reset_state(self) -> None:
        self.builder.reset()
        method = str(self.selector_cfg.get("selector", "alpha"))
        alpha = float(self.selector_cfg.get("alpha", 0.6))
        window = int(self.selector_cfg.get("window", 8))
        self.selector = SeriesSelector(
            method=method if method in {"alpha", "window"} else "alpha",
            alpha=alpha,
            window=window,
        )
        self._prev_candidates = set()

    def update(self, obs: StepObservation) -> DetectorOutput:
        assert self.selector is not None, "Detector not reset before use"
        scalar_value = float(self.builder.build(obs))
        selection = self.selector.update(scalar_value, step_index=obs.step_index)
        candidates = selection.candidates
        coverage = selection.coverage
        triggered = obs.step_index in candidates
        current_candidates = set(candidates)
        new_candidates = sorted(current_candidates - self._prev_candidates)
        self._prev_candidates = current_candidates
        meta = {
            "scalar_value": scalar_value,
            "coverage": coverage,
            "selected_mass": selection.selected_mass,
            "total_mass": selection.total_mass,
            "builder_meta": self.builder.last_meta,
            "new_candidates": new_candidates,
        }
        return DetectorOutput(
            step_index=obs.step_index,
            candidates=candidates,
            score=scalar_value,
            coverage=coverage,
            total_mass=selection.total_mass,
            triggered=triggered,
            meta=meta,
        )


__all__ = ["ScalarSeriesDetector", "ScalarBuilder"]
