from __future__ import annotations

from collections.abc import Sequence

import torch

from ..base import TensorDict
from ..task_vectors import TaskVector


def default_weights(n: int, weights: Sequence[float] | None) -> torch.Tensor:
    if weights is None:
        return torch.ones(n, dtype=torch.float32)
    if len(weights) != n:
        raise ValueError("weights length must match tuned checkpoints")
    return torch.tensor([float(w) for w in weights], dtype=torch.float32)


def resolve_merge_weights(n: int, weights: Sequence[float] | None) -> list[float]:
    return [float(weight) for weight in default_weights(int(n), weights).tolist()]


def axpy_state_dict(base: TensorDict, delta: TensorDict, alpha: float) -> TensorDict:
    out: TensorDict = dict(base)
    for k in TaskVector.common_keys(base, [delta]):
        b = base[k]
        d = delta[k].to(dtype=b.dtype, device=b.device)
        out[k] = b + float(alpha) * d
    return out


def get_method_params(kwargs: dict) -> dict:
    method_params = kwargs.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("method_params must be a dict.")
    return method_params
