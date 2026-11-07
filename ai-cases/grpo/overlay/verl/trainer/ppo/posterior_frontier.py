"""Cross-fitted historical advantages for homogeneous GRPO groups."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PosteriorFrontierConfig:
    coefficient: float = 0.5
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    min_observations: int = 1


class PosteriorFrontierController:
    """Use a problem's past success posterior when the current group collapses.

    Ordinary GRPO has zero correctness advantage for 0/4 and 4/4 groups.  For
    repeated problems, the Beta posterior *before the current group* supplies
    a cross-fitted reference: an all-correct group gets ``1 - p`` and an
    all-wrong group gets ``-p``.  Mixed groups keep ordinary GRPO untouched.
    """

    def __init__(self, config: PosteriorFrontierConfig):
        if config.coefficient < 0:
            raise ValueError("coefficient must be non-negative")
        if config.prior_alpha <= 0 or config.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if config.min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        self.config = config
        self._states: dict[object, list[float]] = {}

    def observe(self, sample_indices, correctness: torch.Tensor):
        residual = torch.zeros_like(correctness, dtype=torch.float32)
        groups = defaultdict(list)
        for position, sample_id in enumerate(sample_indices):
            groups[int(sample_id)].append(position)

        active = 0
        all_correct = 0
        all_wrong = 0
        posterior_means = []
        residual_magnitudes = []
        for sample_id, positions in groups.items():
            values = correctness[positions].to(torch.float32)
            successes = float(values.sum().detach().cpu().item())
            count = len(positions)
            state = self._states.get(sample_id)
            if state is None:
                state = [self.config.prior_alpha, self.config.prior_beta, 0.0]

            alpha, beta, observations = state
            posterior_mean = alpha / (alpha + beta)
            posterior_means.append(posterior_mean)
            homogeneous = successes == 0.0 or successes == float(count)
            if observations >= self.config.min_observations and homogeneous:
                value = (1.0 - posterior_mean) if successes == count else -posterior_mean
                residual[positions] = value
                residual_magnitudes.append(abs(value))
                active += 1
                all_correct += int(successes == count)
                all_wrong += int(successes == 0.0)

            # Update only after producing the residual: the current group is
            # excluded from its own baseline by construction.
            self._states[sample_id] = [
                alpha + successes,
                beta + count - successes,
                observations + 1.0,
            ]

        metrics = {
            "posterior_frontier/active_group_fraction": float(active / max(len(groups), 1)),
            "posterior_frontier/all_correct_group_fraction": float(
                all_correct / max(len(groups), 1)
            ),
            "posterior_frontier/all_wrong_group_fraction": float(
                all_wrong / max(len(groups), 1)
            ),
            "posterior_frontier/posterior_mean": float(
                sum(posterior_means) / max(len(posterior_means), 1)
            ),
            "posterior_frontier/residual_abs_mean": float(
                sum(residual_magnitudes) / max(len(residual_magnitudes), 1)
            ) if residual_magnitudes else 0.0,
            "posterior_frontier/covered_problem_count": float(len(self._states)),
        }
        return residual, metrics
