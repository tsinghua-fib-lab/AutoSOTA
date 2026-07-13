"""Losses for quantile regression."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Pinball loss with optional penalty for quantile crossing.

    Shapes:
    - pred:   (B, H, N, Q)
    - target: (B, H, N)
    """

    def __init__(self, quantiles: Sequence[float], crossing_penalty_weight: float = 0.0):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(list(quantiles), dtype=torch.float32))
        self.penalty_weight = float(crossing_penalty_weight)

        sorted_indices = torch.argsort(self.quantiles)
        self.register_buffer("sorted_indices", sorted_indices)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        quantiles = self.quantiles.to(pred.device)

        error = target.unsqueeze(-1) - pred
        q_view = quantiles.view(1, 1, 1, -1)
        pinball = torch.max((q_view - 1) * error, q_view * error)
        loss = pinball.mean()

        if self.penalty_weight > 0 and quantiles.numel() > 1:
            pred_sorted = pred[..., self.sorted_indices]
            diffs = pred_sorted[..., 1:] - pred_sorted[..., :-1]
            crossing = torch.relu(-diffs)
            loss = loss + self.penalty_weight * crossing.mean()

        return loss
