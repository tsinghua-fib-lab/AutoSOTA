"""Correctness-preserving response-budget control for Bi-Frontier UFT.

The hint scheduler repairs the all-wrong end of GRPO.  This module repairs the
opposite end: groups containing several correct responses have useful
within-correct length variation that ordinary outcome-only GRPO discards.

The controller is deliberately stateful but framework-light so its update rule
can be unit-tested without starting Ray or a GPU worker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseBudgetConfig:
    target_tokens: float = 192.0
    dual_lr: float = 0.05
    max_coef: float = 0.35
    ema_alpha: float = 0.10
    warmup_steps: int = 40


class ResponseBudgetController:
    """Online dual controller for a mean-correct-response token constraint."""

    def __init__(self, config: ResponseBudgetConfig):
        if config.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if config.dual_lr < 0:
            raise ValueError("dual_lr must be non-negative")
        if config.max_coef < 0:
            raise ValueError("max_coef must be non-negative")
        if not 0 < config.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if config.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.config = config
        self.coef = 0.0
        self.correct_length_ema: float | None = None

    def observe(self, mean_correct_tokens: float | None, step: int) -> dict[str, float]:
        """Update the Lagrange multiplier and return logging metrics.

        Missing correct samples leave the state unchanged.  The instantaneous
        batch constraint drives the multiplier, while an EMA is exposed only
        as a lower-variance diagnostic.
        """
        if mean_correct_tokens is not None:
            current = float(mean_correct_tokens)
            if current < 0:
                raise ValueError("mean_correct_tokens must be non-negative")
            if self.correct_length_ema is None:
                self.correct_length_ema = current
            else:
                alpha = self.config.ema_alpha
                self.correct_length_ema = (
                    (1.0 - alpha) * self.correct_length_ema + alpha * current
                )
            if step > self.config.warmup_steps:
                violation = (current - self.config.target_tokens) / self.config.target_tokens
                self.coef = min(
                    self.config.max_coef,
                    max(0.0, self.coef + self.config.dual_lr * violation),
                )

        return {
            "bifrontier/length_coef": float(self.coef),
            "bifrontier/target_correct_tokens": float(self.config.target_tokens),
            "bifrontier/correct_length_ema": float(self.correct_length_ema or 0.0),
            "bifrontier/constraint_violation": float(
                ((mean_correct_tokens or 0.0) - self.config.target_tokens)
                / self.config.target_tokens
            ) if mean_correct_tokens is not None else 0.0,
        }
