"""Validation-probe feedback for deployment-aligned augmentation."""

from dataclasses import dataclass


@dataclass
class DeploymentProbeConfig:
    target_format_rate: float = 0.98
    breach_patience: int = 2
    decay_factor: float = 0.25
    recovery_step: float = 0.20
    min_scale: float = 0.10
    warmup_steps: int = 50
    feedback_source: str = 'no_hint'

    def __post_init__(self):
        if not 0.0 <= self.target_format_rate <= 1.0:
            raise ValueError('deployment_probe.target_format_rate must be in [0, 1]')
        if self.breach_patience < 1:
            raise ValueError('deployment_probe.breach_patience must be positive')
        if not 0.0 < self.decay_factor <= 1.0:
            raise ValueError('deployment_probe.decay_factor must be in (0, 1]')
        if self.recovery_step < 0.0:
            raise ValueError('deployment_probe.recovery_step must be non-negative')
        if not 0.0 <= self.min_scale <= 1.0:
            raise ValueError('deployment_probe.min_scale must be in [0, 1]')
        if self.warmup_steps < 0:
            raise ValueError('deployment_probe.warmup_steps must be non-negative')
        if self.feedback_source not in ('no_hint', 'validation'):
            raise ValueError(
                'deployment_probe.feedback_source must be no_hint or validation'
            )


class DeploymentProbeController:
    """Throttle semantic augmentation after repeated deployment-format breaches."""

    def __init__(self, config: DeploymentProbeConfig):
        self.config = config
        self.augmentation_scale = 1.0
        self.consecutive_breaches = 0

    def observe(self, format_rate: float, step: int) -> dict:
        rate = float(format_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError('format_rate must be in [0, 1]')
        updated = False
        if step >= self.config.warmup_steps:
            if rate < self.config.target_format_rate:
                self.consecutive_breaches += 1
                if self.consecutive_breaches >= self.config.breach_patience:
                    self.augmentation_scale = max(
                        self.config.min_scale,
                        self.augmentation_scale * self.config.decay_factor,
                    )
                    updated = True
            else:
                self.consecutive_breaches = 0
                old_scale = self.augmentation_scale
                self.augmentation_scale = min(
                    1.0, self.augmentation_scale + self.config.recovery_step
                )
                updated = self.augmentation_scale != old_scale
        return {
            'deployment_probe/format_rate': rate,
            'deployment_probe/consecutive_breaches': float(
                self.consecutive_breaches
            ),
            'deployment_probe/augmentation_scale': self.augmentation_scale,
            'deployment_probe/scale_updated': float(updated),
        }
