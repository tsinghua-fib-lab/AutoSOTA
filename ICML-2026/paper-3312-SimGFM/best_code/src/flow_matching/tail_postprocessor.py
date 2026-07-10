import torch
import torch.nn.functional as F


class TailPostProcessor:
    """Apply Endpoint Safety Projection when near t=1 (simple pass-through to pred)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def apply(self, current_t: torch.Tensor, kappa_t: torch.Tensor, h: torch.Tensor, prob_X: torch.Tensor, prob_E: torch.Tensor, pred_X: torch.Tensor, pred_E: torch.Tensor):
        near_one = torch.all((kappa_t + h) >= 1.0).item()
        if near_one and self.enabled:
            return pred_X, pred_E
        return prob_X, prob_E


