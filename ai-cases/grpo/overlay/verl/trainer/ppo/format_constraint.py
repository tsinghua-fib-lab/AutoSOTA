"""Adaptive format-preservation constraint for GRPO training.

The controller treats answer-format validity as a soft safety constraint.  It
adds a bounded reward only to formatted-but-incorrect samples and therefore
never changes the verifier reward of a correct response.
"""

from dataclasses import dataclass

import torch


@dataclass
class FormatConstraintConfig:
    target_rate: float = 0.995
    dual_lr: float = 2.0
    max_bonus: float = 0.20
    min_bonus: float = 0.0
    release_lr_scale: float = 1.0
    lcb_z: float = 0.0
    ema_alpha: float = 0.10
    warmup_steps: int = 40
    feedback_source: str = 'train'
    apply_scope: str = 'all'
    apply_hint_threshold: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.target_rate <= 1.0:
            raise ValueError('format_constraint.target_rate must be in [0, 1]')
        if self.dual_lr < 0.0:
            raise ValueError('format_constraint.dual_lr must be non-negative')
        if self.max_bonus < 0.0:
            raise ValueError('format_constraint.max_bonus must be non-negative')
        if not 0.0 <= self.min_bonus <= self.max_bonus:
            raise ValueError(
                'format_constraint.min_bonus must be in [0, max_bonus]'
            )
        if not 0.0 < self.release_lr_scale <= 1.0:
            raise ValueError(
                'format_constraint.release_lr_scale must be in (0, 1]'
            )
        if self.lcb_z < 0.0:
            raise ValueError('format_constraint.lcb_z must be non-negative')
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError('format_constraint.ema_alpha must be in (0, 1]')
        if self.warmup_steps < 0:
            raise ValueError('format_constraint.warmup_steps must be non-negative')
        if self.feedback_source not in ('train', 'no_hint', 'validation', 'anchor'):
            raise ValueError(
                'format_constraint.feedback_source must be train, no_hint, '
                'validation, or anchor'
            )
        if self.apply_scope not in ('all', 'no_hint', 'low_hint'):
            raise ValueError(
                'format_constraint.apply_scope must be all, no_hint, or low_hint'
            )
        if not 0.0 <= self.apply_hint_threshold <= 1.0:
            raise ValueError(
                'format_constraint.apply_hint_threshold must be in [0, 1]'
            )


class FormatConstraintController:
    """Projected dual controller for a minimum format-validity rate."""

    def __init__(self, config: FormatConstraintConfig):
        self.config = config
        self.coef = 0.0
        self.format_rate_ema = None

    def _lower_confidence_rate(self, rate: float, sample_size=None) -> float:
        """Wilson lower bound used by fixed deployment-style anchor probes."""
        if self.config.lcb_z <= 0.0 or sample_size is None:
            return rate
        count = int(sample_size)
        if count <= 0:
            raise ValueError('format constraint sample_size must be positive')
        z = float(self.config.lcb_z)
        z2_over_n = z * z / count
        center = rate + z2_over_n / 2.0
        radius = z * (
            (rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
            ** 0.5
        )
        return max(0.0, (center - radius) / (1.0 + z2_over_n))

    def observe(
        self,
        format_rate: float,
        step: int,
        update: bool = True,
        sample_size=None,
    ) -> dict:
        rate = float(format_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError('format_rate must be in [0, 1]')
        control_rate = self._lower_confidence_rate(rate, sample_size)
        if update and step > self.config.warmup_steps:
            if self.format_rate_ema is None:
                self.format_rate_ema = control_rate
            else:
                alpha = self.config.ema_alpha
                self.format_rate_ema = (
                    alpha * control_rate
                    + (1.0 - alpha) * self.format_rate_ema
                )
            violation = self.config.target_rate - self.format_rate_ema
            step_size = self.config.dual_lr * (
                1.0 if violation >= 0.0 else self.config.release_lr_scale
            )
            self.coef = min(
                self.config.max_bonus,
                max(
                    self.config.min_bonus,
                    self.coef + step_size * violation,
                ),
            )
        elif update and step <= self.config.warmup_steps:
            self.coef = 0.0
        reference_rate = (
            self.format_rate_ema if self.format_rate_ema is not None else rate
        )
        violation = self.config.target_rate - reference_rate

        return {
            'format_constraint/batch_rate': rate,
            'format_constraint/control_rate': control_rate,
            'format_constraint/sample_size': float(sample_size or 0),
            # Tracking backends require numeric values even during warm-up,
            # before the first EMA update has been admitted.
            'format_constraint/rate_ema': reference_rate,
            'format_constraint/violation': violation,
            'format_constraint/bonus_coef': self.coef,
            'format_constraint/controller_updated': float(
                update and step > self.config.warmup_steps
            ),
        }


def build_format_application_mask(
    hint_props: torch.Tensor,
    apply_scope: str,
    low_hint_threshold: float = 0.0,
) -> torch.Tensor:
    """Select trajectories whose assistance level resembles deployment."""
    values = hint_props.to(dtype=torch.float32)
    if apply_scope == 'all':
        return torch.ones_like(values, dtype=torch.bool)
    if apply_scope == 'no_hint':
        return values <= 1e-9
    if apply_scope == 'low_hint':
        if not 0.0 <= float(low_hint_threshold) <= 1.0:
            raise ValueError('low_hint_threshold must be in [0, 1]')
        return values <= float(low_hint_threshold) + 1e-9
    raise ValueError('apply_scope must be all, no_hint, or low_hint')


def apply_format_constraint(
    token_level_rewards: torch.Tensor,
    token_level_scores: torch.Tensor,
    response_mask: torch.Tensor,
    controller: FormatConstraintController,
    correctness_score: float,
    format_score: float,
    global_step: int,
    observation_mask: torch.Tensor = None,
    apply_mask: torch.Tensor = None,
    update_controller: bool = True,
):
    """Add dual-controlled credit to formatted, verifier-incorrect responses."""
    sequence_scores = token_level_scores.sum(dim=-1)
    correctness = torch.isclose(
        sequence_scores,
        torch.as_tensor(
            correctness_score,
            device=sequence_scores.device,
            dtype=sequence_scores.dtype,
        ),
        atol=1e-6,
    )
    format_valid = sequence_scores >= float(format_score) - 1e-6
    if observation_mask is None:
        observation_mask = torch.ones_like(format_valid, dtype=torch.bool)
    else:
        observation_mask = observation_mask.to(
            device=format_valid.device, dtype=torch.bool
        )
    if not bool(observation_mask.any().item()):
        observation_mask = torch.ones_like(format_valid, dtype=torch.bool)
        update_controller = False
    observed_rate = float(
        format_valid[observation_mask].to(torch.float32).mean().detach().cpu().item()
    )
    metrics = controller.observe(
        format_rate=observed_rate,
        step=global_step,
        update=update_controller,
    )
    eligible = format_valid & ~correctness
    if apply_mask is not None:
        eligible = eligible & apply_mask.to(
            device=eligible.device, dtype=torch.bool
        )
    updated = token_level_rewards.clone()
    if controller.coef and bool(eligible.any().item()):
        rows = torch.nonzero(eligible, as_tuple=False).flatten()
        final_tokens = response_mask[rows].sum(dim=-1).to(torch.long) - 1
        updated[rows, final_tokens] += float(controller.coef)
    metrics['format_constraint/eligible_fraction'] = float(
        eligible.to(torch.float32).mean().detach().cpu().item()
    )
    metrics['format_constraint/applied_bonus_mean'] = float(
        controller.coef * metrics['format_constraint/eligible_fraction']
    )
    return updated, metrics
