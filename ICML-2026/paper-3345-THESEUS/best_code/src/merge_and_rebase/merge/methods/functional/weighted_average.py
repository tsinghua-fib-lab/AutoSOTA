from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._registry import register_impl


def weighted_average_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    normalize = str(params.get("normalize", "sumw"))
    if normalize == "sumw":
        denom = float(weights.sum().clamp_min(1e-12).item())
    elif normalize == "n":
        denom = float(len(matrices))
    else:
        raise ValueError("normalize must be 'sumw' or 'n'")

    out = torch.zeros_like(matrices[0])
    for weight, matrix in zip(weights, matrices, strict=True):
        out = out + float(weight) * matrix.to(dtype=out.dtype, device=out.device)
    return out / max(1e-12, denom)


register_impl("weighted_average", weighted_average_impl)
