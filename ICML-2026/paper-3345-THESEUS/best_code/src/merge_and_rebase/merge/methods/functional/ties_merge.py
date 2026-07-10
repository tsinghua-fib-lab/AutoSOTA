from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, stack_flatten
from ._registry import register_impl


def _topk_mask(matrix: torch.Tensor, topk: float) -> tuple[torch.Tensor, torch.Tensor]:
    if topk > 1.0:
        topk = topk / 100.0
    topk = float(topk)

    if topk >= 1.0:
        mask = torch.ones_like(matrix, dtype=torch.bool)
        return matrix, mask

    _, dim = matrix.shape
    k = max(1, int(dim * topk))
    vals, _ = torch.topk(matrix.abs(), k=k, dim=1, largest=True, sorted=False)
    threshold = vals.min(dim=1, keepdim=True).values
    mask = matrix.abs() >= threshold
    return matrix * mask, mask


def _resolve_sign(matrix: torch.Tensor) -> torch.Tensor:
    if torch.all(matrix == 0):
        return torch.ones(matrix.shape[1], device=matrix.device, dtype=torch.float32)
    sign = torch.sign(matrix.sum(dim=0))
    global_majority = torch.sign(sign.sum())
    global_majority = global_majority if global_majority != 0 else torch.tensor(1.0, device=sign.device)
    sign[sign == 0] = global_majority
    return sign


def _disjoint_merge(matrix: torch.Tensor, ref_sign: torch.Tensor, *, weights: torch.Tensor, merge: str) -> torch.Tensor:
    keep = torch.where(ref_sign.unsqueeze(0) > 0, matrix > 0, matrix < 0)
    selected = matrix * keep

    weight_row = weights.to(selected.device, selected.dtype).view(-1, 1)
    selected = selected * weight_row

    if merge == "mean":
        denom = (keep.to(selected.dtype) * weight_row).sum(dim=0).clamp_min(1e-12)
        return selected.sum(dim=0) / denom
    if merge == "sum":
        return selected.sum(dim=0)
    if merge == "max":
        vals, _ = selected.abs().max(dim=0)
        return vals * ref_sign.to(vals.dtype)
    raise ValueError(f"Unknown TIES merge type '{merge}'")


def ties_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    merging_type = str(params.get("merging_type", "mean"))
    topk = float(params.get("topk", 1.0))
    work_dtype = parse_dtype(str(params.get("work_dtype", "float32")))

    ref = matrices[0]
    flat = stack_flatten(matrices, dtype=work_dtype)
    pruned, _mask = _topk_mask(flat, topk=topk)
    sign = _resolve_sign(pruned)
    merged_flat = _disjoint_merge(pruned, sign, weights=weights, merge=merging_type)
    return merged_flat.view_as(ref).to(dtype=ref.dtype, device=ref.device)


register_impl("ties_merge", ties_merge_impl)
