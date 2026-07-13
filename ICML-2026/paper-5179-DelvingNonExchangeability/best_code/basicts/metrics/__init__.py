"""Metrics helpers used by conformal runners."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def winkler_score(lower: torch.Tensor, upper: torch.Tensor, obs: torch.Tensor, alpha: float) -> torch.Tensor:
    width = (upper - lower).abs()
    penalty = (2.0 / max(alpha, 1e-6)) * (torch.relu(lower - obs) + torch.relu(obs - upper))
    return (width + penalty).mean()


def interval_metrics(lower: torch.Tensor, upper: torch.Tensor, targets: torch.Tensor, alpha: float) -> dict:
    width_tensor = (upper - lower).abs()
    covered_mask = ((targets >= lower) & (targets <= upper)).float()
    covered = covered_mask.mean().item()
    width = width_tensor.mean().item()
    try:
        pi_width_median = float(torch.median(width_tensor).item())
    except Exception:
        pi_width_median = float("nan")
    winkler = winkler_score(lower, upper, targets, alpha).item()
    return {
        "target_coverage": 1.0 - alpha,
        "observed_coverage": covered,
        "delta_cov": covered - (1.0 - alpha),
        "pi_width": width,
        "pi_width_median": pi_width_median,
        "winkler": winkler,
    }


__all__ = [
    "interval_metrics",
    "winkler_score",
]
