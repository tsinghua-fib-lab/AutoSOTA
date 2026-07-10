from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, stack_flatten
from ._registry import register_alias, register_impl


def _validate_ratios(*, clamp_min_ratio: float, clamp_max_ratio: float, att_ratio: float) -> None:
    if not (0.0 <= clamp_min_ratio < 1.0):
        raise ValueError("clamp_min_ratio must be in [0, 1).")
    if not (0.0 <= clamp_max_ratio < 1.0):
        raise ValueError("clamp_max_ratio must be in [0, 1).")
    if clamp_min_ratio + clamp_max_ratio >= 1.0:
        raise ValueError("clamp_min_ratio + clamp_max_ratio must be < 1.")
    if not (0.0 < att_ratio <= 1.0):
        raise ValueError("att_ratio must be in (0, 1].")


def _normalize_minmax(x: torch.Tensor, *, dim: int, eps: float = 1e-12) -> torch.Tensor:
    min_values = x.amin(dim=dim, keepdim=True)
    max_values = x.amax(dim=dim, keepdim=True)
    denom = (max_values - min_values).clamp_min(eps)
    return (x - min_values) / denom


def _clamp_by_ratio(x: torch.Tensor, *, min_ratio: float, max_ratio: float) -> torch.Tensor:
    if x.ndim == 1:
        dim = x.shape[0]
        sorted_x, _ = torch.sort(x)
        lo_idx = int(dim * min_ratio)
        hi_idx = int(dim * (1.0 - max_ratio) - 1)
        hi_idx = max(lo_idx, hi_idx)
        min_v = sorted_x[lo_idx]
        max_v = sorted_x[hi_idx]
        return torch.clamp(x, min=min_v, max=max_v)

    if x.ndim == 2:
        dim = x.shape[1]
        sorted_x, _ = torch.sort(x, dim=1)
        lo_idx = int(dim * min_ratio)
        hi_idx = int(dim * (1.0 - max_ratio) - 1)
        hi_idx = max(lo_idx, hi_idx)
        min_v = sorted_x[:, lo_idx].unsqueeze(1)
        max_v = sorted_x[:, hi_idx].unsqueeze(1)
        return torch.clamp(x, min=min_v, max=max_v)

    raise ValueError(f"Expected x to be 1D or 2D, got shape {tuple(x.shape)}")


def pcb_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    clamp_min_ratio = float(params.get("clamp_min_ratio", 0.01))
    clamp_max_ratio = float(params.get("clamp_max_ratio", 0.01))
    att_ratio = float(params.get("att_ratio", 0.05))
    lam = float(params.get("lam", 1.2))

    _validate_ratios(
        clamp_min_ratio=clamp_min_ratio,
        clamp_max_ratio=clamp_max_ratio,
        att_ratio=att_ratio,
    )

    work_dtype = parse_dtype(str(params.get("work_dtype", "float32")))

    ref = matrices[0]
    flat = stack_flatten(matrices, dtype=work_dtype)

    abs_flat = flat.abs()
    abs_clamped = _clamp_by_ratio(abs_flat, min_ratio=clamp_min_ratio, max_ratio=clamp_max_ratio)
    clamped = flat.sign() * abs_clamped

    norm_abs = _normalize_minmax(abs_clamped, dim=1)
    intra = torch.exp(float(flat.shape[0]) * norm_abs.square())
    signed_norm = flat.sign() * norm_abs
    inter = torch.tanh(flat * signed_norm.sum(dim=0))
    balancing = intra * inter

    scale_seed = _clamp_by_ratio(balancing, min_ratio=1.0 - att_ratio, max_ratio=0.0)
    scale = _normalize_minmax(scale_seed, dim=1)

    lam_weights = (float(lam) * weights.to(device=flat.device, dtype=flat.dtype)).view(-1, 1)
    num = (clamped * lam_weights * scale).sum(dim=0)
    den = scale.sum(dim=0).clamp_min(1e-12)
    merged_flat = num / den

    return merged_flat.view_as(ref).to(dtype=ref.dtype, device=ref.device)


register_impl("pcb", pcb_impl)
register_alias("pcb_merge", "pcb")
