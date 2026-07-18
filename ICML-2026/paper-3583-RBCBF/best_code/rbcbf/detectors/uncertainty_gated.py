"""Uncertainty-gated variant of alpha cumulative detector."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

from .alpha_cumulative import AlphaCumulativeDetector
from .base import StepObservation


class UncertaintyGatedDetector(AlphaCumulativeDetector):
    """Alpha cumulative detector that drops steps with high entropy or low margin."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        config: Dict[str, float] | None = None,
    ) -> None:
        super().__init__(name, labels, config)
        cfg = self.config
        self.max_entropy = float(cfg.get("max_entropy", 6.0))
        self.min_margin = float(cfg.get("min_margin", 0.05))
        self._eligibility: list[bool] = []

    def _extra_reset(self) -> None:
        self._eligibility = []

    def _after_append(self, obs: StepObservation, contrib: float) -> None:
        eligible = True
        if self.max_entropy is not None and obs.entropy > self.max_entropy:
            eligible = False
        if self.min_margin is not None and obs.top1_top2_margin < self.min_margin:
            eligible = False
        self._eligibility.append(eligible)

    def _iter_candidate_steps(self) -> Iterable[Tuple[int, float]]:
        for idx, contrib in enumerate(self.history):
            if not self._eligibility[idx]:
                continue
            yield idx, contrib


__all__ = ["UncertaintyGatedDetector"]
