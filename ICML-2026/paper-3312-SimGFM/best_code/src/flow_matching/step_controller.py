import torch


class StepController:
    """Controls step size per sample for time advancement.

    Supports fixed and adaptive step sizes. Returns per-sample dt shaped (B, 1).
    """

    def __init__(self, sample_steps: int, adaptive: bool = False):
        self.sample_steps = sample_steps
        self.adaptive = adaptive

    def propose(self, current_t: torch.Tensor, kappa_t: torch.Tensor, kappa_dot_t: torch.Tensor) -> torch.Tensor:
        base_h = 1.0 / float(self.sample_steps)
        proposed_h = base_h * torch.ones_like(current_t)
        if self.adaptive:
            eps_dt = 1e-8
            safe_h = (1.0 - kappa_t).clamp(min=0.0) / torch.clamp(kappa_dot_t, min=eps_dt)
            return torch.minimum(proposed_h, safe_h)
        else:
            return proposed_h


