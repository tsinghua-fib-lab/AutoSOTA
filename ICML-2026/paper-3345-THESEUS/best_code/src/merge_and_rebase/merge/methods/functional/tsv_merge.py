from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, rank_from_singular_values, require_2d
from ._registry import register_impl
from .weighted_average import weighted_average_impl


def tsv_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("tsv_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return weighted_average_impl(matrices, weights, {"normalize": "sumw"})

    require_2d(matrices, "tsv_merge")
    sv_reduction = float(params.get("sv_reduction", 1.0 / max(1, len(matrices))))
    if not (0.0 < sv_reduction <= 1.0):
        raise ValueError("tsv_merge method_params['sv_reduction'] must be in (0, 1].")

    max_rank_raw = params.get("max_rank", None)
    max_rank = None if max_rank_raw is None else int(max_rank_raw)
    if max_rank is not None and max_rank <= 0:
        raise ValueError("tsv_merge method_params['max_rank'] must be > 0.")

    svd_dtype = parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("tsv_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")
    accum_dtype = parse_dtype(str(params.get("accum_dtype", "float32")))

    ref = matrices[0]
    min_dim = min(int(ref.shape[0]), int(ref.shape[1]))
    rank = rank_from_singular_values(min_dim, sv_reduction=sv_reduction, max_rank=max_rank)
    n_tasks = len(matrices)
    total_rank = rank * n_tasks

    sum_u = torch.zeros((int(ref.shape[0]), total_rank), dtype=accum_dtype, device="cpu")
    sum_s = torch.zeros((total_rank,), dtype=accum_dtype, device="cpu")
    sum_v = torch.zeros((total_rank, int(ref.shape[1])), dtype=accum_dtype, device="cpu")

    for i, matrix in enumerate(matrices):
        mat = matrix.detach().to(device="cpu", dtype=svd_dtype)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        lo = i * rank
        hi = lo + rank
        sum_u[:, lo:hi] = u[:, :rank].to(dtype=accum_dtype, device="cpu")
        sum_s[lo:hi] = (s[:rank] * float(weights[i])).to(dtype=accum_dtype, device="cpu")
        sum_v[lo:hi, :] = vh[:rank, :].to(dtype=accum_dtype, device="cpu")

    u_u, _, vh_u = torch.linalg.svd(sum_u.to(dtype=svd_dtype), full_matrices=False)
    u_v, _, vh_v = torch.linalg.svd(sum_v.to(dtype=svd_dtype), full_matrices=False)
    merged = torch.linalg.multi_dot((u_u, vh_u, torch.diag(sum_s.to(dtype=svd_dtype)), u_v, vh_v))
    return merged.to(dtype=ref.dtype, device=ref.device)


register_impl("tsv_merge", tsv_merge_impl)
