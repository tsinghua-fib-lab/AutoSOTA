"""Two-phase acquisition for IDEA-06: early exploration, later velocity-weighted refinement."""
from __future__ import annotations

from typing import Iterable

import numpy as np


class TwoPhaseAcquisition:
    """Switches between two acquisition strategies at a configurable step.

    Phase 1 (steps < switch_step): pure uncertainty exploration.
    Phase 2 (steps >= switch_step): velocity-weighted refinement.
    """

    def __init__(
        self,
        phase1_fn,
        phase2_fn,
        switch_step: int = 14,
    ) -> None:
        self.phase1_fn = phase1_fn
        self.phase2_fn = phase2_fn
        self.switch_step = int(switch_step)
        self._step_count = 0

    def optimize(
        self, posterior: object, candidates: Iterable[float]
    ) -> tuple[float, np.ndarray]:
        self._step_count += 1
        if self._step_count <= self.switch_step:
            return self.phase1_fn.optimize(posterior, candidates)
        else:
            return self.phase2_fn.optimize(posterior, candidates)
