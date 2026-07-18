"""Route-B joint scalar detector."""

from __future__ import annotations

from typing import Dict, Sequence

from .base import Detector, DetectorOutput, StepObservation
from .common import SeriesSelector, compute_contributions


class RouteBJointDetector(Detector):
    """Aggregates contributions into a scalar and runs a single selector."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        config: Dict[str, object] | None = None,
        beta_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(name, labels, config)
        cfg = self.config
        self.weight_family = str(cfg.get("weight_family", "sigmoid"))
        self.tau = float(cfg.get("tau", 0.0))
        self.tau_act = float(cfg.get("tau_act", 0.0))
        self.slope = float(cfg.get("slope", 0.5))
        self.method = str(cfg.get("selector", "alpha"))
        self.alpha = float(cfg.get("alpha", 0.6))
        self.window = int(cfg.get("window", 8))
        self.beta_weights = list(beta_weights or [1.0] * len(self.labels))
        self.selector: SeriesSelector | None = None

    def _reset_state(self) -> None:
        self.selector = SeriesSelector(
            method=self.method,
            alpha=self.alpha,
            window=self.window,
        )

    def update(self, obs: StepObservation) -> DetectorOutput:
        assert self.selector is not None
        snapshot = compute_contributions(
            h_values=obs.h,
            delta_minus=obs.delta_minus,
            betas=self.beta_weights,
            tau=self.tau,
            tau_act=self.tau_act,
            slope=self.slope,
            family=self.weight_family,
        )
        selection = self.selector.update(snapshot.joint_value)
        candidates = selection.candidates
        coverage = selection.coverage
        triggered = obs.step_index in candidates
        meta = {
            "route": "B_joint",
            "weight_family": self.weight_family,
            "tau": self.tau,
            "tau_act": self.tau_act,
            "selector_method": self.method,
            "window": self.window,
            "active_indices": snapshot.active_indices,
            "contrib": {
                "sum_c_t": snapshot.sum_active,
                "u_t": snapshot.joint_value,
            },
            "h_min": snapshot.h_min,
        }
        return DetectorOutput(
            step_index=obs.step_index,
            candidates=candidates,
            score=snapshot.joint_value,
            coverage=coverage,
            total_mass=selection.total_mass,
            triggered=triggered,
            meta=meta,
        )


__all__ = ["RouteBJointDetector"]
