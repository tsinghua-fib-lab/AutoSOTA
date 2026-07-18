"""Scalar builder: contributions only from active set."""

from __future__ import annotations

from typing import Dict, Sequence

from rbcbf.detectors.base import StepObservation
from rbcbf.detectors.common import compute_active_indices

from .base_scalar import ScalarBuilder


class DeltaActSetBuilder(ScalarBuilder):
    def __init__(
        self,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(labels, config)
        self.tau_act = float(self.config.get("tau_act", 0.0))

    def build(self, obs: StepObservation) -> float:
        active = compute_active_indices(obs.h, tau_act=self.tau_act)
        value = float(sum(float(obs.delta_minus[idx]) for idx in active))
        self.last_meta = {
            "tau_act": self.tau_act,
            "active_count": len(active),
        }
        return value


__all__ = ["DeltaActSetBuilder"]
