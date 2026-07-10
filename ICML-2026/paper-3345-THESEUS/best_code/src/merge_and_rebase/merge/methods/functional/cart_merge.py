from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from ._common import require_2d
from ._registry import register_impl


def cart_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    require_2d(matrices, "cart_merge")

    pruning_rank = float(params.get("pruning_rank", 4))
    scaling_coeffs = float(params.get("scaling_coeffs", 0.5))

    theta_avg = torch.stack(matrices).mean(dim=0)
    sum_term = torch.zeros_like(theta_avg)

    for i, matrix in enumerate(matrices):
        tau = matrix - theta_avg
        u, s, vh = torch.linalg.svd(tau.to(torch.float64), full_matrices=False)
        rank_k = int(math.ceil(float(pruning_rank) * float(s.shape[0])))
        rank_k = max(1, min(int(s.shape[0]), rank_k))
        recon = u[:, :rank_k] @ torch.diag(s[:rank_k]) @ vh[:rank_k, :]
        sum_term = sum_term + recon.to(dtype=theta_avg.dtype, device=theta_avg.device) * float(weights[i])

    return theta_avg + float(scaling_coeffs) * sum_term


register_impl("cart_merge", cart_merge_impl)
