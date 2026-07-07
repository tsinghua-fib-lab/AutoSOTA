"""Checkpoint snapshot helpers.

PyTorch ``state_dict()`` values share storage with the live module.  Training
validation temporarily swaps EMA weights into the model, then restores live
weights afterwards.  Any checkpoint payload that is kept in memory across that
restore must therefore own detached tensor clones, not references into the live
model.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, MutableMapping, Optional

import torch
from torch import nn


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def clone_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Return a detached CPU clone of a state dict.

    This preserves keys/order while ensuring later mutations of the source model
    or EMA dictionary cannot change the checkpoint payload in memory.
    """

    return OrderedDict((name, _clone_value(value)) for name, value in state_dict.items())


def _trainable_state_dict(model: nn.Module) -> OrderedDict[str, Any]:
    trainable_names = {
        name
        for name, param in model.named_parameters()
        if getattr(param, "requires_grad", False)
    }
    full_state = model.state_dict()
    return OrderedDict((name, value) for name, value in full_state.items() if name in trainable_names)


def build_checkpoint_payload(
    *,
    model: nn.Module,
    epoch: int,
    batch: int,
    score: float,
    ema_state: Optional[Mapping[str, Any]] = None,
    checkpoint_save_mode: str = "full",
) -> MutableMapping[str, Any]:
    """Build an immutable-in-memory checkpoint payload for a validation point."""
    mode = str(checkpoint_save_mode).lower()
    if mode not in {"full", "trainable_only"}:
        raise ValueError(f"Unsupported checkpoint_save_mode={checkpoint_save_mode!r}")
    model_state = model.state_dict() if mode == "full" else _trainable_state_dict(model)

    payload: MutableMapping[str, Any] = {
        "model": clone_state_dict(model_state),
        "epoch": epoch,
        "batch": batch,
        "score": float(score),
        "checkpoint_save_mode": mode,
    }
    if ema_state is not None and mode == "full":
        payload["ema"] = clone_state_dict(ema_state)
    return payload


def load_matching_state_dict(model: nn.Module, state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Load only checkpoint tensors whose keys and shapes match ``model``.

    ``strict=False`` still raises on same-key shape mismatches.  Architecture
    diagnostics such as LAD→MLDC intentionally change the first operator
    projection shape, so they need a controlled partial load that keeps all
    compatible pretrained weights and reports skipped tensors.
    """

    current = model.state_dict()
    filtered = OrderedDict()
    skipped_shape_mismatch: list[str] = []
    adapted_shape_mismatch: list[str] = []
    unexpected_keys: list[str] = []
    for name, value in state_dict.items():
        target = current.get(name)
        if target is None:
            unexpected_keys.append(name)
            continue
        if not torch.is_tensor(value):
            skipped_shape_mismatch.append(name)
            continue
        if tuple(target.shape) != tuple(value.shape):
            if (
                name == "ferret_backbone.cbr1.0.weight"
                and value.ndim == 4
                and target.ndim == 4
                and int(value.shape[1]) == 1
                and int(target.shape[1]) > 1
                and tuple(value.shape[0:1] + value.shape[2:]) == tuple(target.shape[0:1] + target.shape[2:])
            ):
                expanded = value.repeat(1, int(target.shape[1]), 1, 1) / float(target.shape[1])
                filtered[name] = expanded.to(dtype=target.dtype)
                adapted_shape_mismatch.append(name)
                continue
            skipped_shape_mismatch.append(name)
            continue
        filtered[name] = value
    missing_keys, unexpected_after_load = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "missing_keys": list(missing_keys),
        "unexpected_keys": unexpected_keys + list(unexpected_after_load),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "adapted_shape_mismatch": adapted_shape_mismatch,
    }
