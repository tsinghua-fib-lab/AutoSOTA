from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, require_2d
from ._registry import register_impl
from .weighted_average import weighted_average_impl


def isoc_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("isoc_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return weighted_average_impl(matrices, weights, {"normalize": "sumw"})

    require_2d(matrices, "isoc_merge")
    svd_dtype = parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("isoc_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")

    combined = torch.zeros_like(matrices[0])
    for weight, matrix in zip(weights, matrices, strict=True):
        combined = combined + float(weight) * matrix.to(dtype=combined.dtype, device=combined.device)

    u, s, vh = torch.linalg.svd(combined.to(dtype=svd_dtype), full_matrices=False)
    if s.numel() == 0:
        return torch.zeros_like(combined)
    s_iso = torch.ones_like(s) * s.mean()
    out = u @ torch.diag(s_iso) @ vh
    return out.to(dtype=combined.dtype, device=combined.device)


register_impl("isoc_merge", isoc_merge_impl)
