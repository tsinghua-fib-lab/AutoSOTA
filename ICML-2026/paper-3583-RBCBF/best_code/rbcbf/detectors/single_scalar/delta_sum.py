"""Scalar builder: sum over delta_minus with optional beta weights."""

from __future__ import annotations

from typing import Dict, Sequence

from rbcbf.detectors.base import StepObservation

from .base_scalar import ScalarBuilder


class DeltaSumBuilder(ScalarBuilder):
    def __init__(
        self,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
        beta_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(labels, config)
        self.beta_mode = str(self.config.get("beta_mode", "uniform"))
        self.weights = list(beta_weights or [1.0] * len(self.labels))

    def build(self, obs: StepObservation) -> float:
        value = float(sum(w * float(dm) for w, dm in zip(self.weights, obs.delta_minus)))
        self.last_meta = {
            "beta_mode": self.beta_mode,
        }
        return value


__all__ = ["DeltaSumBuilder"]
