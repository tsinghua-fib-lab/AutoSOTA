"""Custom quantile loss with crossing penalty and smoothness regularization."""
import torch
import torch.nn as nn


class SmoothQuantileLoss(nn.Module):
    """Pinball loss with crossing penalty and smoothness regularization.

    Shapes:
    - y_hat_q: (Q, B, H, N)
    - y: (B, H, N)
    - mask: (B, H, N) or None
    """

    def __init__(self, quantiles, crossing_penalty_weight=0.0, smoothness_weight=0.0):
        super().__init__()
        q_tensor = torch.tensor(list(quantiles), dtype=torch.float32)
        self.register_buffer("quantiles", q_tensor)
        self.crossing_penalty = float(crossing_penalty_weight)
        self.smoothness_weight = float(smoothness_weight)
        sorted_idx = torch.argsort(self.quantiles)
        self.register_buffer("sorted_idx", sorted_idx)

    def forward(self, y_hat_q, y, mask=None):
        q = self.quantiles.to(y_hat_q.device).view(-1, 1, 1, 1)
        err = y.unsqueeze(0) - y_hat_q
        pinball = torch.maximum((q - 1) * err, q * err)

        if mask is not None:
            mask_exp = mask.unsqueeze(0)
            pinball = pinball * mask_exp
            loss = pinball.sum() / mask_exp.sum().clamp(min=1)
        else:
            loss = pinball.mean()

        if self.crossing_penalty > 0 and len(self.quantiles) > 1:
            pred_sorted = y_hat_q[self.sorted_idx]
            diffs = pred_sorted[1:] - pred_sorted[:-1]
            crossing = torch.relu(-diffs)
            loss = loss + self.crossing_penalty * crossing.mean()

        if self.smoothness_weight > 0 and len(self.quantiles) > 2:
            pred_sorted = y_hat_q[self.sorted_idx]
            second_diff = pred_sorted[2:] - 2 * pred_sorted[1:-1] + pred_sorted[:-2]
            smoothness = (second_diff ** 2).mean()
            loss = loss + self.smoothness_weight * smoothness

        return loss
