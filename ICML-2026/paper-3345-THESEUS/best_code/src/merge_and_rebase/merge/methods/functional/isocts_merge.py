from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, require_2d
from ._registry import register_impl
from .weighted_average import weighted_average_impl


def isocts_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("isocts_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return weighted_average_impl(matrices, weights, {"normalize": "sumw"})

    require_2d(matrices, "isocts_merge")
    common_space_fraction = float(params.get("common_space_fraction", 0.8))
    svd_dtype = parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("isocts_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")

    ref = matrices[0]
    mats = [matrix.to(dtype=svd_dtype) for matrix in matrices]
    combined_w = sum(float(weight) * matrix for weight, matrix in zip(weights, mats, strict=True))

    n_tasks = len(mats)
    min_dim = min(combined_w.shape)
    if min_dim == 0:
        return torch.zeros_like(ref)

    common_space_dim = int(min_dim * common_space_fraction)
    common_space_dim = max(0, min(common_space_dim, min_dim))
    task_specific_total_dim = max(0, min_dim - common_space_dim)
    task_dims_per_task = int(task_specific_total_dim // max(1, n_tasks))
    task_specific_total_dim = task_dims_per_task * n_tasks
    common_space_dim = min_dim - task_specific_total_dim

    u, s, vh = torch.linalg.svd(combined_w, full_matrices=False)
    common_u = u[:, :common_space_dim]
    common_s = s[:common_space_dim]
    common_v = vh[:common_space_dim, :]

    combined_space_u = torch.zeros_like(u)
    combined_space_s = torch.zeros_like(s)
    combined_space_v = torch.zeros_like(vh)

    if common_space_dim > 0:
        common_proj = common_u @ common_u.T
    else:
        common_proj = torch.zeros((combined_w.shape[0], combined_w.shape[0]), dtype=svd_dtype, device=combined_w.device)

    for task_idx, matrix in enumerate(mats):
        mat_task_space = matrix - (common_proj @ matrix)
        u_ts, s_ts, vh_ts = torch.linalg.svd(mat_task_space, full_matrices=False)

        start = task_idx * task_dims_per_task
        end = (task_idx + 1) * task_dims_per_task
        if task_dims_per_task > 0:
            combined_space_u[:, start:end] = u_ts[:, :task_dims_per_task]
            combined_space_s[start:end] = s_ts[:task_dims_per_task]
            combined_space_v[start:end, :] = vh_ts[:task_dims_per_task, :]

    common_start = n_tasks * task_dims_per_task
    common_end = common_start + common_space_dim
    if common_space_dim > 0:
        combined_space_u[:, common_start:common_end] = common_u
        combined_space_s[common_start:common_end] = common_s
        combined_space_v[common_start:common_end, :] = common_v

    u_u, _, vh_u = torch.linalg.svd(combined_space_u, full_matrices=False)
    u_v, _, vh_v = torch.linalg.svd(combined_space_v, full_matrices=False)
    ortho_u = u_u @ vh_u
    ortho_v = u_v @ vh_v

    if combined_space_s.numel() > 0:
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

    out = ortho_u @ torch.diag(combined_space_s) @ ortho_v
    return out.to(dtype=ref.dtype, device=ref.device)


register_impl("isocts_merge", isocts_merge_impl)
