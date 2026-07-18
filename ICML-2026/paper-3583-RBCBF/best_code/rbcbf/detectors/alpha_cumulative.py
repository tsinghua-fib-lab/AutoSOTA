"""Alpha cumulative detector family."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .base import Detector, DetectorOutput, StepObservation


class AlphaCumulativeDetector(Detector):
    """Causal detector that accumulates contributions and selects a minimal prefix covering alpha."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        config: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(name, labels, config)
        cfg = self.config
        self.alpha = float(cfg.get("alpha", 0.6))
        self.weights = self._build_label_weights(cfg.get("label_weights"))
        self.history: List[float] = []
        self.total_mass: float = 0.0
        self._seen_candidates: set[int] = set()
        self._last_new: List[int] = []

    def _reset_state(self) -> None:
        self.history = []
        self.total_mass = 0.0
        self._seen_candidates = set()
        self._last_new = []
        self._extra_reset()

    def _extra_reset(self) -> None:
        """Hook for subclasses."""

    def update(self, obs: StepObservation) -> DetectorOutput:
        contrib = self._compute_contribution(obs)
        self.history.append(contrib)
        self.total_mass += contrib
        self._after_append(obs, contrib)
        candidates, selected_mass = self._select_candidates()
        coverage = selected_mass / self.total_mass if self.total_mass > 0 else 0.0
        triggered = self._update_new_candidates(candidates, obs.step_index)
        meta = {
            "alpha": self.alpha,
            "selected_mass": selected_mass,
            "total_mass": self.total_mass,
            "new_candidates": list(self._last_new),
        }
        return DetectorOutput(
            step_index=obs.step_index,
            candidates=candidates,
            score=contrib,
            coverage=coverage,
            total_mass=self.total_mass,
            triggered=triggered,
            meta=meta,
        )

    def _compute_contribution(self, obs: StepObservation) -> float:
        return float(
            sum(weight * float(delta) for weight, delta in zip(self.weights, obs.delta_minus))
        )

    def _after_append(self, obs: StepObservation, contrib: float) -> None:
        """Subclass hook."""

    def _iter_candidate_steps(self) -> Iterable[Tuple[int, float]]:
        return enumerate(self.history)

    def _select_candidates(self) -> Tuple[List[int], float]:
        if self.total_mass <= 0:
            return [], 0.0
        target = self.alpha * self.total_mass
        selected: List[int] = []
        selected_mass = 0.0
        for idx, contrib in self._iter_candidate_steps():
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


__all__ = ["AlphaCumulativeDetector"]
