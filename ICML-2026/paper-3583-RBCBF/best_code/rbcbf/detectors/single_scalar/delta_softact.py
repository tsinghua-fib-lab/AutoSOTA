"""Scalar builder: soft-activated delta_minus sum."""

from __future__ import annotations

from typing import Dict, Sequence

from rbcbf.detectors.base import StepObservation
from rbcbf.detectors.common import compute_weight

from .base_scalar import ScalarBuilder


class DeltaSoftActBuilder(ScalarBuilder):
    def __init__(
        self,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(labels, config)
        self.weight_family = str(self.config.get("weight_family", "sigmoid"))
        self.tau = float(self.config.get("tau", 0.0))
        self.slope = float(self.config.get("slope", 0.5))

    def build(self, obs: StepObservation) -> float:
        weights = [
            compute_weight(h, tau=self.tau, slope=self.slope, family=self.weight_family)
            for h in obs.h
        ]
        value = float(sum(w * float(dm) for w, dm in zip(weights, obs.delta_minus)))
        self.last_meta = {
            "weight_family": self.weight_family,
            "tau": self.tau,
            "slope": self.slope,
        }
        return value


__all__ = ["DeltaSoftActBuilder"]
