"""Metrics."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_tensor(x: ArrayLike, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def expected_calibration_error(
    logits: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    probs: Optional[ArrayLike] = None,
    n_bins: int = 15,
) -> float:
    if labels is None:
        raise ValueError("labels must be provided")
    if (logits is None) == (probs is None):
        raise ValueError("provide exactly one of logits or probs")

    labels_tensor = _as_tensor(labels, torch.long)
    if probs is None:
        probs_tensor = F.softmax(_as_tensor(logits, torch.float32), dim=1)
    else:
        probs_tensor = _as_tensor(probs, torch.float32)
    labels_tensor = labels_tensor.to(probs_tensor.device)

    confidences, predictions = probs_tensor.max(dim=1)
    accuracies = predictions.eq(labels_tensor).float()
    ece = torch.zeros((), dtype=torch.float32)

    boundaries = torch.linspace(0, 1, n_bins + 1, device=probs_tensor.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            ece = ece + prop * (confidences[in_bin].mean() - accuracies[in_bin].mean()).abs()
    return float(ece.item())


def negative_log_likelihood(logits: ArrayLike, labels: ArrayLike) -> float:
    logits_tensor = _as_tensor(logits, torch.float32)
    labels_tensor = _as_tensor(labels, torch.long).to(logits_tensor.device)
    return float(F.cross_entropy(logits_tensor, labels_tensor).item())


def accuracy(logits: ArrayLike, labels: ArrayLike) -> float:
    logits_tensor = _as_tensor(logits, torch.float32)
    labels_tensor = _as_tensor(labels, torch.long).to(logits_tensor.device)
    return float(logits_tensor.argmax(dim=1).eq(labels_tensor).float().mean().item())


__all__ = ["expected_calibration_error", "negative_log_likelihood", "accuracy"]
