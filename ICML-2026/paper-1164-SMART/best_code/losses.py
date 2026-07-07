"""Charbonnier-SoftECE."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierSoftECE(nn.Module):
    def __init__(
        self,
        n_bins: int = 15,
        sigma: float = 0.05,
        delta: float = 1e-3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.n_bins = n_bins
        self.sigma = sigma
        self.delta = delta
        self.eps = eps

    def _soft_bin_stats(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels).float()

        centers = (torch.arange(self.n_bins, device=logits.device, dtype=logits.dtype) + 0.5) / self.n_bins
        weights = torch.exp(-0.5 * ((confidences[:, None] - centers[None, :]) / self.sigma) ** 2)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

        masses = weights.sum(dim=0).clamp_min(self.eps)
        bin_confidences = (weights * confidences[:, None]).sum(dim=0) / masses
        bin_accuracies = (weights * accuracies[:, None]).sum(dim=0) / masses
        bin_weights = masses / logits.size(0)
        return bin_confidences, bin_accuracies, bin_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bin_confidences, bin_accuracies, bin_weights = self._soft_bin_stats(logits, labels)
        gaps = bin_confidences - bin_accuracies
        penalty = torch.sqrt(gaps.square() + self.delta**2)
        return (bin_weights * penalty).sum()


SmoothSoftECE = CharbonnierSoftECE


__all__ = ["CharbonnierSoftECE", "SmoothSoftECE"]
