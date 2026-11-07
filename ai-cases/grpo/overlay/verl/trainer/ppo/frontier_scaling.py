"""Cross-fitted frontier weighting without inventing signed advantages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrontierScalingConfig:
    min_weight: float = 0.25
    max_weight: float = 1.75
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    min_observations: int = 1


class FrontierScalingController:
    """Scale genuine GRPO contrasts by historical probability of a mixed group.

    PFA added the same signed residual to every response in homogeneous groups,
    breaking GRPO's zero-mean control variate.  This controller never creates a
    new sign: it estimates, before observing the current group, how likely the
    problem is to yield a mixed group and scales the ordinary GRPO advantage.
    Homogeneous groups therefore remain exactly zero.
    """

    def __init__(self, config: FrontierScalingConfig):
        if not 0.0 <= config.min_weight <= config.max_weight:
            raise ValueError("weights must satisfy 0 <= min_weight <= max_weight")
        if config.prior_alpha <= 0 or config.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if config.min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        self.config = config
        self._states: dict[int, list[float]] = {}

    def observe(self, sample_indices, correctness: torch.Tensor):
        weights = torch.ones_like(correctness, dtype=torch.float32)
        groups = defaultdict(list)
        for position, sample_id in enumerate(sample_indices):
            groups[int(sample_id)].append(position)

        active = 0
        predicted_mixed = []
        applied_weights = []
        for sample_id, positions in groups.items():
            values = correctness[positions].to(torch.float32)
            count = len(positions)
            alpha, beta, observations = self._states.get(
                sample_id,
                [self.config.prior_alpha, self.config.prior_beta, 0.0],
            )
            posterior = alpha / (alpha + beta)
            mixed_probability = 1.0 - posterior ** count - (1.0 - posterior) ** count
            predicted_mixed.append(mixed_probability)

            if observations >= self.config.min_observations:
                # The maximum mixed probability is attained at p=0.5.  Mapping
                # by this quantity keeps the scale interpretable and bounded.
                maximum = 1.0 - 2.0 * (0.5 ** count)
                frontier_score = mixed_probability / max(maximum, 1e-12)
                weight = self.config.min_weight + (
                    self.config.max_weight - self.config.min_weight
                ) * frontier_score
                weight = min(self.config.max_weight, max(self.config.min_weight, weight))
                weights[positions] = weight
                applied_weights.append(weight)
                active += 1

            successes = float(values.sum().detach().cpu().item())
            self._states[sample_id] = [
                alpha + successes,
                beta + count - successes,
                observations + 1.0,
            ]

        metrics = {
            "frontier_scaling/active_group_fraction": float(active / max(len(groups), 1)),
            "frontier_scaling/predicted_mixed_probability_mean": float(
                sum(predicted_mixed) / max(len(predicted_mixed), 1)
            ),
            "frontier_scaling/weight_mean": float(
                sum(applied_weights) / max(len(applied_weights), 1)
            ) if applied_weights else 1.0,
            "frontier_scaling/downweighted_group_fraction": float(
                sum(weight < 1.0 for weight in applied_weights) / max(len(groups), 1)
            ),
            "frontier_scaling/upweighted_group_fraction": float(
                sum(weight > 1.0 for weight in applied_weights) / max(len(groups), 1)
            ),
            "frontier_scaling/covered_problem_count": float(len(self._states)),
        }
        return weights, metrics
