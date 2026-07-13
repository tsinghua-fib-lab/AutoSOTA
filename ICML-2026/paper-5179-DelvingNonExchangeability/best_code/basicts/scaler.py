"""Lightweight tensor standardization helpers."""

from __future__ import annotations

import torch


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardize tensor with broadcastable mean/std."""
    return (x - mean) / (std + 1e-8)


def destandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of standardize."""
    return x * (std + 1e-8) + mean


__all__ = ["standardize", "destandardize"]
