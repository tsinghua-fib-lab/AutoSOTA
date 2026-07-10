from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..merge.task_vectors import TaskVector, apply_task_vector, default_key_filter
from .registry import get_method


def merge_task_vectors(
    *,
    base: Mapping[str, torch.Tensor],
    tuned: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    strict: bool = False,
    key_filter=default_key_filter,
) -> TaskVector:
    if weights is None:
        weights = [1.0] * len(tuned)
    if len(weights) != len(tuned):
        raise ValueError('weights and tuned must have the same length.')

    merged: dict[str, torch.Tensor] = {}
    for weight, tuned_sd in zip(weights, tuned, strict=True):
        tv = TaskVector.from_checkpoints(base, tuned_sd, strict=strict, key_filter=key_filter)
        for key, tensor in tv.delta.items():
            if key not in merged:
                merged[key] = torch.zeros_like(tensor)
            merged[key] = merged[key] + float(weight) * tensor
    return TaskVector(delta=merged)



def transport_task_vector(
    *,
    source_base: Mapping[str, torch.Tensor],
    target_base: Mapping[str, torch.Tensor],
    task_vector: TaskVector,
    method: str | Any = 'identity',
    strict: bool = False,
    **kwargs,
) -> TaskVector:
    transport = get_method(method) if isinstance(method, str) else method
    transported = transport.transport(
        source_base=source_base,
        target_base=target_base,
        delta=task_vector.delta,
        strict=strict,
        **kwargs,
    )
    return TaskVector(delta=transported)



def rebase_merged_task_vectors(
    *,
    source_base: Mapping[str, torch.Tensor],
    target_base: Mapping[str, torch.Tensor],
    tuned: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    transport_method: str | Any = 'identity',
    strict: bool = False,
    key_filter=default_key_filter,
    **kwargs,
) -> dict[str, torch.Tensor]:
    merged = merge_task_vectors(base=source_base, tuned=tuned, weights=weights, strict=strict, key_filter=key_filter)
    transported = transport_task_vector(
        source_base=source_base,
        target_base=target_base,
        task_vector=merged,
        method=transport_method,
        strict=strict,
        **kwargs,
    )
    return apply_task_vector(target_base, transported, alpha=alpha, strict=strict, key_filter=key_filter)
