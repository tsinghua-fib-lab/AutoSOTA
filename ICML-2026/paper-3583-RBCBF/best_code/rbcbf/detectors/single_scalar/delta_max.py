"""Scalar builder: maximum delta_minus across constraints."""

from __future__ import annotations

from typing import Dict, Sequence

from rbcbf.detectors.base import StepObservation

from .base_scalar import ScalarBuilder


class DeltaMaxBuilder(ScalarBuilder):
    def __init__(
        self,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(labels, config)

    def build(self, obs: StepObservation) -> float:
        value = max((float(dm) for dm in obs.delta_minus), default=0.0)
        self.last_meta = {"op": "max"}
        return value


__all__ = ["DeltaMaxBuilder"]
