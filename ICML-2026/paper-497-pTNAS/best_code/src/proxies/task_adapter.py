from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


_REGRESSION_TOKENS = ("regression", "mae", "l1")
_BINARY_TOKENS = ("binary", "binclass", "bce", "auc")


def infer_proxy_task(space_name: str) -> str:
    name = (space_name or "").lower()
    if any(token in name for token in _REGRESSION_TOKENS):
        return "regression"
    if any(token in name for token in _BINARY_TOKENS):
        return "binary"
    return "binary"


def proxy_forward(arch: nn.Module, batch_data):
    if isinstance(batch_data, torch.Tensor) and hasattr(arch, "forward_wo_embedding"):
        return arch.forward_wo_embedding(batch_data)
    return arch(batch_data)


def proxy_batch_size(batch_data) -> int:
    if isinstance(batch_data, torch.Tensor):
        return int(batch_data.shape[0])
    if hasattr(batch_data, "y"):
        return int(batch_data.y.shape[0])
    if isinstance(batch_data, dict):
        for value in batch_data.values():
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return int(value.shape[0])
    if hasattr(batch_data, "shape") and len(batch_data.shape) >= 1:
        return int(batch_data.shape[0])
    raise TypeError(f"Unsupported batch_data type for batch size inference: {type(batch_data)!r}")


def proxy_targets(batch_labels: torch.Tensor, task: str, device: torch.device | str):
    if task == "regression":
        return batch_labels.float().to(device).view(-1)
    if task == "binary":
        return batch_labels.float().to(device).view(-1)
    return batch_labels.long().to(device).view(-1)


def proxy_loss(outputs: torch.Tensor,
               batch_labels: torch.Tensor,
               space_name: str,
               reduction: str = "mean") -> torch.Tensor:
    task = infer_proxy_task(space_name)
    if task == "regression":
        preds = outputs.view(-1)
        target = batch_labels.float().to(outputs.device).view(-1)
        return F.l1_loss(preds, target, reduction=reduction)

    if outputs.ndim == 1 or outputs.shape[-1] == 1:
        preds = outputs.view(-1)
        target = batch_labels.float().to(outputs.device).view(-1)
        return F.binary_cross_entropy_with_logits(preds, target, reduction=reduction)

    target = batch_labels.long().to(outputs.device).view(-1)
    return F.cross_entropy(outputs, target, reduction=reduction)


def synflow_input(batch_data):
    if isinstance(batch_data, torch.Tensor):
        return torch.ones_like(batch_data[:1])
    raise TypeError("SynFlow currently expects a tensor batch or a model exposing forward_wo_embedding.")


def activation_module_types() -> tuple[type[nn.Module], ...]:
    return (
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.LeakyReLU,
        nn.ELU,
    )
