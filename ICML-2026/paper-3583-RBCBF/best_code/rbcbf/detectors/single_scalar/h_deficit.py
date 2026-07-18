"""Scalar builder: deficit from h_min relative to eps."""

from __future__ import annotations

from typing import Dict, Sequence

from rbcbf.detectors.base import StepObservation

from .base_scalar import ScalarBuilder


class HDeficitBuilder(ScalarBuilder):
    def __init__(
        self,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(labels, config)
        self.eps = float(self.config.get("eps", 0.0))

    def build(self, obs: StepObservation) -> float:
        h_min = float(min(obs.h)) if obs.h else 0.0
        deficit = max(self.eps - h_min, 0.0)
        self.last_meta = {
            "eps": self.eps,
            "h_min": h_min,
        }
        return float(deficit)


__all__ = ["HDeficitBuilder"]
