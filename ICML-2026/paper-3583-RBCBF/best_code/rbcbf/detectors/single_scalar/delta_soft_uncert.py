"""Scalar builder: soft activation times uncertainty gate."""

from __future__ import annotations

import math
from typing import Dict, Sequence

from rbcbf.detectors.base import StepObservation
from rbcbf.detectors.common import compute_weight

from .base_scalar import ScalarBuilder


class DeltaSoftUncertaintyBuilder(ScalarBuilder):
    def __init__(
        self,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(labels, config)
        self.weight_family = str(self.config.get("weight_family", "sigmoid"))
        self.tau = float(self.config.get("tau", 0.0))
        self.slope = float(self.config.get("slope", 0.5))
        self.entropy_center = float(self.config.get("entropy_center", 3.5))
        self.entropy_scale = float(self.config.get("entropy_scale", 1.0))
        self.margin_eps = float(self.config.get("margin_eps", 0.05))

    def build(self, obs: StepObservation) -> float:
        weights = [
            compute_weight(h, tau=self.tau, slope=self.slope, family=self.weight_family)
            for h in obs.h
        ]
        base_value = sum(w * float(dm) for w, dm in zip(weights, obs.delta_minus))

        entropy_gate = self._sigmoid((obs.entropy - self.entropy_center) / max(self.entropy_scale, 1e-3))
        margin_gate = 1.0 / (abs(obs.top1_top2_margin) + self.margin_eps)
        gate = entropy_gate * margin_gate
        value = float(base_value * gate)
        self.last_meta = {
            "gate_entropy": entropy_gate,
            "gate_margin": margin_gate,
            "weight_family": self.weight_family,
        }
        return value

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


__all__ = ["DeltaSoftUncertaintyBuilder"]
