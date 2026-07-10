from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, stack_flatten
from ._registry import register_impl


def dare_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if "drop_rate" in params:
        drop_rate = float(params["drop_rate"])
    elif "p" in params:
        drop_rate = float(params["p"])
    elif "keep_ratio" in params:
        drop_rate = 1.0 - float(params["keep_ratio"])
    else:
        drop_rate = 0.9

    if not (0.0 <= drop_rate < 1.0):
        raise ValueError("drop_rate must satisfy 0 <= drop_rate < 1.")

    seed_val = params.get("seed", None)
    seed = None if seed_val is None else int(seed_val)
    rescale = bool(params.get("rescale", True))
    work_dtype = parse_dtype(str(params.get("work_dtype", "float32")))

    ref = matrices[0]
    flat = stack_flatten(matrices, dtype=work_dtype)
    keep_prob = 1.0 - float(drop_rate)

    if keep_prob == 1.0:
        sparse = flat
    else:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=flat.device)
            generator.manual_seed(seed)
        mask = (torch.rand(flat.shape, device=flat.device, generator=generator) < keep_prob).to(flat.dtype)
        sparse = flat * mask
        if rescale:
            sparse = sparse / keep_prob

    merged_flat = (sparse * weights.to(device=flat.device, dtype=flat.dtype).view(-1, 1)).sum(dim=0)
    return merged_flat.view_as(ref).to(dtype=ref.dtype, device=ref.device)


register_impl("dare_merge", dare_merge_impl)
