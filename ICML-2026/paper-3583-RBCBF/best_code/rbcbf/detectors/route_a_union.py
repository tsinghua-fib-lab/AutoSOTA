"""Route-A union detector (per-constraint selectors + union)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set

from .base import Detector, DetectorOutput, StepObservation
from .common import ContributionSnapshot, SeriesSelector, compute_contributions


class RouteAUnionDetector(Detector):
    """Maintains per-constraint selectors and unions their candidate sets."""

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
        self.selectors: List[SeriesSelector] = []
        self.history: List[List[float]] = []
        self.total_mass = 0.0

    def _reset_state(self) -> None:
        self.selectors = [
            SeriesSelector(
                method=self.method,
                alpha=self.alpha,
                window=self.window,
            )
            for _ in self.labels
        ]
        self.history = [[] for _ in self.labels]
        self.total_mass = 0.0

    def update(self, obs: StepObservation) -> DetectorOutput:
        snapshot = compute_contributions(
            h_values=obs.h,
            delta_minus=obs.delta_minus,
            betas=self.beta_weights,
            tau=self.tau,
            tau_act=self.tau_act,
            slope=self.slope,
            family=self.weight_family,
        )
        union_candidates: Set[int] = set()
        for idx, selector in enumerate(self.selectors):
            selection = selector.update(snapshot.c_values[idx])
            self.history[idx].append(snapshot.c_values[idx])
            union_candidates.update(selection.candidates)
        union_list = sorted(union_candidates)
        selected_mass = self._compute_union_mass(union_list)
        self.total_mass += float(sum(snapshot.c_values))
        coverage = (
            selected_mass / self.total_mass if self.total_mass > 0 else 0.0
        )
        triggered = obs.step_index in union_candidates
        meta = {
            "route": "A_union",
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
            candidates=union_list,
            score=snapshot.sum_active,
            coverage=coverage,
            total_mass=self.total_mass,
            triggered=triggered,
            meta=meta,
        )

    def _compute_union_mass(self, union_indices: List[int]) -> float:
        total = 0.0
        for idx in union_indices:
            for label_history in self.history:
                if idx < len(label_history):
                    total += label_history[idx]
        return total


__all__ = ["RouteAUnionDetector"]
