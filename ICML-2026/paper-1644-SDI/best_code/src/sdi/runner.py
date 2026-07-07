"""
Checkpoint runners for TracIn/SDI.

This module is intentionally lightweight: it defines common dataclasses and
default checkpoint loading / weighting helpers used by both estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CheckpointSpec:
    """
    A checkpoint path and its TracIn weight (eta_k).

    If `weight` is None, callers may derive it from the checkpoint (e.g., optimizer lr).
    """

    path: Union[str, Path]
    weight: Optional[float] = None


@dataclass(frozen=True)
class InfluenceOutputs:
    """
    Aggregated influence outputs across checkpoints.

    - sdi: (N_train, N_query, steps_query)  [query-step decomposition] or None
    - tracin: (N_train, N_query)
    - sdi_matrix: optional (N_train, N_query, steps_train, steps_query) [step×step]
    """

    sdi: Optional[torch.Tensor]
    tracin: torch.Tensor
    sdi_matrix: Optional[torch.Tensor] = None


def default_load_checkpoint(
    checkpoint_path: Union[str, Path], *, map_location: str = "cpu"
) -> Dict[str, Any]:
    return torch.load(
        str(checkpoint_path), map_location=map_location, weights_only=False
    )


def default_load_model_state_dict(model: nn.Module, checkpoint: Dict[str, Any]) -> None:
    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    else:
        # Best-effort fallback: assume the checkpoint itself is a state dict
        sd = checkpoint
    model.load_state_dict(sd)


def default_eta_from_checkpoint(
    checkpoint: Dict[str, Any], *, fallback: float = 1.0
) -> float:
    """
    Best-effort extraction of a scalar eta_k from common checkpoint formats.

    If an `optimizer_state_dict` is present and contains `param_groups[0]['lr']`,
    that value is used. Otherwise, returns `fallback`.
    """
    opt = checkpoint.get("optimizer_state_dict", None)
    if isinstance(opt, dict):
        pgs = opt.get("param_groups", None)
        if isinstance(pgs, (list, tuple)) and len(pgs) > 0 and isinstance(pgs[0], dict):
            lr = pgs[0].get("lr", None)
            if lr is not None:
                try:
                    return float(lr)
                except Exception:
                    pass
    return float(fallback)


def resolve_checkpoint_weights(
    checkpoints: Sequence[CheckpointSpec],
    *,
    load_checkpoint_fn: Callable[
        [Union[str, Path]], Dict[str, Any]
    ] = default_load_checkpoint,
    eta_from_checkpoint_fn: Callable[
        [Dict[str, Any]], float
    ] = default_eta_from_checkpoint,
) -> Sequence[CheckpointSpec]:
    """
    Fill in missing CheckpointSpec.weight values by reading the checkpoint.
    """
    out: list[CheckpointSpec] = []
    for spec in checkpoints:
        if spec.weight is not None:
            out.append(spec)
            continue
        ckpt = load_checkpoint_fn(spec.path)
        out.append(CheckpointSpec(path=spec.path, weight=eta_from_checkpoint_fn(ckpt)))
    return out
