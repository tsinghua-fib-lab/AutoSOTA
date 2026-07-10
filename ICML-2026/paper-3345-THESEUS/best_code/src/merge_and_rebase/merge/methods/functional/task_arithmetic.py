from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._registry import register_impl


def task_arithmetic_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    _ = params
    out = torch.zeros_like(matrices[0])
    for weight, matrix in zip(weights, matrices, strict=True):
        out = out + float(weight) * matrix.to(dtype=out.dtype, device=out.device)
    return out


register_impl("task_arithmetic", task_arithmetic_impl)
