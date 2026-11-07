"""Budget-matched contextual bandit for adaptive UFT hint lengths.

This module is intentionally NumPy-only so the scheduling logic can be tested
without importing the distributed training stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MasteryConfig:
    ema_alpha: float = 0.30
    hint_tolerance: float = 0.05
    hard_threshold: float = 0.25
    easy_threshold: float = 0.75
    frontier_uncertainty_threshold: float = 0.50
    easy_max_hint_fraction: float = 0.25


class LeastAssistedMasteryTracker:
    """Estimate intrinsic mastery from the least-assisted evidence seen so far.

    Hint length is an intervention that directly changes success.  Mixing all
    observations in one EMA therefore confounds intrinsic sample difficulty
    with the scheduler's previous actions.  This tracker resets its estimate
    whenever a less-assisted observation arrives and otherwise updates only
    observations close to the current minimum hint fraction.
    """

    def __init__(self, num_samples: int, config: MasteryConfig):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not 0 < config.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 <= config.hint_tolerance <= 1:
            raise ValueError("hint_tolerance must be in [0, 1]")
        if not 0 <= config.hard_threshold < config.easy_threshold <= 1:
            raise ValueError("mastery thresholds must satisfy 0 <= hard < easy <= 1")
        if not 0 <= config.frontier_uncertainty_threshold <= 1:
            raise ValueError("frontier_uncertainty_threshold must be in [0, 1]")
        if not 0 <= config.easy_max_hint_fraction <= 1:
            raise ValueError("easy_max_hint_fraction must be in [0, 1]")
        self.config = config
        self.success = np.full(num_samples, np.nan, dtype=np.float64)
        self.uncertainty = np.full(num_samples, np.nan, dtype=np.float64)
        self.hint_floor = np.full(num_samples, np.inf, dtype=np.float64)
        self.update_counts = np.zeros(num_samples, dtype=np.int64)

    def observe(self, item: int, success_rate: float, hint_fraction: float) -> bool:
        success = float(np.clip(success_rate, 0.0, 1.0))
        hint = float(np.clip(hint_fraction, 0.0, 1.0))
        uncertainty = 4.0 * success * (1.0 - success)
        floor = self.hint_floor[item]
        tolerance = self.config.hint_tolerance
        if hint < floor - tolerance:
            self.hint_floor[item] = hint
            self.success[item] = success
            self.uncertainty[item] = uncertainty
            self.update_counts[item] = 1
            return True
        if hint <= floor + tolerance:
            alpha = self.config.ema_alpha
            self.hint_floor[item] = min(floor, hint)
            self.success[item] = (1.0 - alpha) * self.success[item] + alpha * success
            self.uncertainty[item] = (
                (1.0 - alpha) * self.uncertainty[item] + alpha * uncertainty
            )
            self.update_counts[item] += 1
            return True
        return False

    def context(self, item: int) -> int:
        if self.update_counts[item] == 0:
            return 1
        success = self.success[item]
        if self.uncertainty[item] >= self.config.frontier_uncertainty_threshold:
            return 1
        if success <= self.config.hard_threshold:
            return 0
        if (
            success >= self.config.easy_threshold
            and self.hint_floor[item] <= self.config.easy_max_hint_fraction
        ):
            return 2
        return 1

    def low_assistance_coverage(self) -> float:
        observed = self.update_counts > 0
        low_assistance = self.hint_floor <= self.config.easy_max_hint_fraction
        return float(np.mean(observed & low_assistance))


def group_learning_utility(success_rate: float, success_bonus: float = 0.25) -> float:
    """Score non-degenerate GRPO feedback, with a small preference for success."""
    if success_bonus < 0:
        raise ValueError("success_bonus must be non-negative")
    success = float(np.clip(success_rate, 0.0, 1.0))
    disagreement = 4.0 * success * (1.0 - success)
    return (disagreement + success_bonus * success) / (1.0 + success_bonus)


@dataclass(frozen=True)
class BanditConfig:
    reward_ema_alpha: float = 0.25
    exploration_coef: float = 0.10
    temperature: float = 0.15
    prior_utility: float = 0.50
    dual_bound: float = 100.0
    dual_iterations: int = 60


class BudgetedHintBandit:
    """Learn hint-length utility while matching a target mean hint budget.

    A separate table is kept for each number of available hint chunks.  One
    shared cost multiplier per table is solved by bisection, so the
    context-weighted expected fraction of revealed chunks equals ``target``.
    """

    def __init__(self, config: BanditConfig, num_contexts: int = 3):
        if num_contexts < 1:
            raise ValueError("num_contexts must be positive")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < config.reward_ema_alpha <= 1:
            raise ValueError("reward_ema_alpha must be in (0, 1]")
        self.config = config
        self.num_contexts = num_contexts
        self._values: dict[int, np.ndarray] = {}
        self._counts: dict[int, np.ndarray] = {}
        self._version = 0
        self._cache: dict[tuple, tuple[np.ndarray, float, float]] = {}

    def _ensure_table(self, max_chunks: int) -> None:
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        if max_chunks not in self._values:
            shape = (self.num_contexts, max_chunks + 1)
            self._values[max_chunks] = np.full(
                shape, self.config.prior_utility, dtype=np.float64
            )
            self._counts[max_chunks] = np.zeros(shape, dtype=np.int64)

    def observe(
        self,
        max_chunks: int,
        context: int,
        arm: int,
        group_utility: float,
    ) -> None:
        self._ensure_table(max_chunks)
        if not 0 <= context < self.num_contexts:
            raise ValueError(f"invalid context: {context}")
        if not 0 <= arm <= max_chunks:
            raise ValueError(f"invalid arm {arm} for {max_chunks} chunks")
        reward = float(np.clip(group_utility, 0.0, 1.0))
        count = self._counts[max_chunks][context, arm]
        if count == 0:
            value = reward
        else:
            alpha = self.config.reward_ema_alpha
            old = self._values[max_chunks][context, arm]
            value = (1.0 - alpha) * old + alpha * reward
        self._values[max_chunks][context, arm] = value
        self._counts[max_chunks][context, arm] = count + 1
        self._version += 1
        self._cache.clear()

    @staticmethod
    def _normalize_context_weights(weights: np.ndarray, size: int) -> np.ndarray:
        result = np.asarray(weights, dtype=np.float64)
        if result.shape != (size,):
            raise ValueError(f"context_weights must have shape ({size},)")
        if np.any(result < 0):
            raise ValueError("context_weights must be non-negative")
        total = result.sum()
        if total <= 0:
            return np.full(size, 1.0 / size, dtype=np.float64)
        return result / total

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        stable = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(stable)
        return exp / exp.sum(axis=-1, keepdims=True)

    def policy(
        self,
        max_chunks: int,
        target_fraction: float,
        context_weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return policies, cost multiplier, and achieved expected fraction."""
        self._ensure_table(max_chunks)
        target = float(np.clip(target_fraction, 0.0, 1.0))
        weights = self._normalize_context_weights(context_weights, self.num_contexts)
        cache_key = (
            max_chunks,
            round(target, 12),
            tuple(np.round(weights, 12)),
            self._version,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        counts = self._counts[max_chunks]
        estimates = self._values[max_chunks] + self.config.exploration_coef / np.sqrt(
            counts + 1.0
        )
        costs = np.arange(max_chunks + 1, dtype=np.float64) / max_chunks

        def distribution(multiplier: float) -> np.ndarray:
            logits = (estimates - multiplier * costs[None, :]) / self.config.temperature
            return self._softmax(logits)

        def expected_cost(multiplier: float) -> float:
            probs = distribution(multiplier)
            per_context = (probs * costs[None, :]).sum(axis=1)
            return float(np.dot(weights, per_context))

        lower = -self.config.dual_bound
        upper = self.config.dual_bound
        for _ in range(self.config.dual_iterations):
            middle = (lower + upper) / 2.0
            if expected_cost(middle) > target:
                lower = middle
            else:
                upper = middle
        multiplier = (lower + upper) / 2.0
        policies = distribution(multiplier)
        achieved = expected_cost(multiplier)
        result = (policies, multiplier, achieved)
        self._cache[cache_key] = result
        return result

    def sample(
        self,
        max_chunks: int,
        context: int,
        target_fraction: float,
        context_weights: np.ndarray,
        rng=np.random,
    ) -> tuple[int, float, float, float]:
        policies, multiplier, achieved = self.policy(
            max_chunks=max_chunks,
            target_fraction=target_fraction,
            context_weights=context_weights,
        )
        probabilities = policies[context]
        arm = int(rng.choice(max_chunks + 1, p=probabilities))
        return arm, float(probabilities[arm]), multiplier, achieved

    def snapshot(self) -> dict:
        return {
            "version": self._version,
            "values": {key: value.copy() for key, value in self._values.items()},
            "counts": {key: value.copy() for key, value in self._counts.items()},
        }
