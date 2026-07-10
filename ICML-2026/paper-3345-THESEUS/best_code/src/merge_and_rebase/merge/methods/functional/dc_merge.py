from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype, rank_from_singular_values
from ._registry import merge_functional, register_impl
from .weighted_average import weighted_average_impl


def _matrix_view(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim <= 1:
        return matrix.reshape(1, -1)
    return matrix.reshape(int(matrix.shape[0]), -1)


def _resolve_energy_smoothing(params: Mapping[str, Any]) -> str:
    mode = str(params.get("energy_smoothing", "average")).strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "no": "none",
        "average": "average",
        "avg": "average",
        "mean": "average",
        "linear": "linear",
    }
    if mode not in aliases:
        raise ValueError("dc_merge method_params['energy_smoothing'] must be one of: none, average, linear.")
    return aliases[mode]


def _resolve_mask_mode(params: Mapping[str, Any]) -> str:
    mode = str(params.get("mask_mode", "block")).strip().lower()
    aliases = {
        "block": "block",
        "block_diag": "block",
        "block-diag": "block",
        "none": "none",
        "off": "none",
    }
    if mode not in aliases:
        raise ValueError("dc_merge method_params['mask_mode'] must be one of: block, none.")
    return aliases[mode]


def _resolve_cover_merge_method(params: Mapping[str, Any]) -> str:
    method = str(params.get("cover_merge_method", "task_arithmetic")).strip().lower()
    aliases = {
        "task_arithmetic": "task_arithmetic",
        "ta": "task_arithmetic",
        "weighted_average": "weighted_average",
        "average": "weighted_average",
        "ties_merge": "ties_merge",
        "ties": "ties_merge",
        "wudi": "wudi",
        "dare_merge": "dare_merge",
        "dare": "dare_merge",
        "pcb": "pcb",
        "pcb_merge": "pcb",
        "cart_merge": "cart_merge",
        "cart": "cart_merge",
    }
    if method not in aliases:
        raise ValueError("dc_merge method_params['cover_merge_method'] must resolve to a supported functional method.")
    resolved = aliases[method]
    if resolved == "dc_merge":
        raise ValueError("dc_merge cannot recursively use itself as cover_merge_method.")
    return resolved


def _smooth_singular_values(
    singular_values: torch.Tensor,
    *,
    mode: str,
    strength: float,
) -> torch.Tensor:
    if singular_values.numel() == 0 or mode == "none" or strength == 0.0:
        return singular_values

    if mode == "average":
        target = torch.ones_like(singular_values) * singular_values.mean()
    elif mode == "linear":
        weights = torch.linspace(
            float(singular_values.numel()),
            1.0,
            singular_values.numel(),
            dtype=singular_values.dtype,
            device=singular_values.device,
        )
        target = weights / weights.sum().clamp_min(1e-12)
        target = target * singular_values.sum()
    else:
        raise AssertionError(f"Unhandled dc_merge smoothing mode: {mode}")

    return singular_values.lerp(target, float(strength))


def _inverse_sqrt_psd(matrix: torch.Tensor, *, eps: float) -> torch.Tensor:
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = torch.linalg.eigh(sym)
    inv_sqrt = evals.clamp_min(float(eps)).rsqrt()
    return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T


def _whiten_column_basis(matrix: torch.Tensor, *, eps: float) -> torch.Tensor:
    gram = matrix.T @ matrix
    whitening = _inverse_sqrt_psd(gram, eps=eps)
    return matrix @ whitening


def _block_mask(*, n_tasks: int, rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    size = int(n_tasks) * int(rank)
    mask = torch.zeros((size, size), dtype=dtype, device=device)
    for task_idx in range(int(n_tasks)):
        lo = task_idx * int(rank)
        hi = lo + int(rank)
        mask[lo:hi, lo:hi] = 1
    return mask


def dc_merge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
    if vector_1d_merge not in {"zero", "average"}:
        raise ValueError("dc_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
    if matrices[0].ndim == 1:
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return weighted_average_impl(matrices, weights, {"normalize": "sumw"})

    sv_reduction = float(params.get("sv_reduction", 1.0 / max(1, len(matrices))))
    if not (0.0 < sv_reduction <= 1.0):
        raise ValueError("dc_merge method_params['sv_reduction'] must be in (0, 1].")

    max_rank_raw = params.get("max_rank", None)
    max_rank = None if max_rank_raw is None else int(max_rank_raw)
    if max_rank is not None and max_rank <= 0:
        raise ValueError("dc_merge method_params['max_rank'] must be > 0.")

    svd_dtype = parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("dc_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")

    smooth_mode = _resolve_energy_smoothing(params)
    smooth_strength = float(params.get("energy_smoothing_strength", 1.0))
    if not (0.0 <= smooth_strength <= 1.0):
        raise ValueError("dc_merge method_params['energy_smoothing_strength'] must be in [0, 1].")

    mask_mode = _resolve_mask_mode(params)
    whiten_eps = float(params.get("whiten_eps", 1e-6))
    if whiten_eps <= 0.0:
        raise ValueError("dc_merge method_params['whiten_eps'] must be > 0.")

    cover_merge_method = _resolve_cover_merge_method(params)
    cover_merge_params = params.get("cover_merge_params", {})
    if cover_merge_params is None:
        cover_merge_params = {}
    if not isinstance(cover_merge_params, Mapping):
        raise ValueError("dc_merge method_params['cover_merge_params'] must be a mapping when provided.")

    ref = matrices[0]
    views = [_matrix_view(matrix).to(dtype=svd_dtype) for matrix in matrices]
    min_dim = min(int(views[0].shape[0]), int(views[0].shape[1]))
    rank = rank_from_singular_values(min_dim, sv_reduction=sv_reduction, max_rank=max_rank)
    n_tasks = len(views)

    u_cols: list[torch.Tensor] = []
    v_cols: list[torch.Tensor] = []
    smoothed_deltas: list[torch.Tensor] = []
    for view in views:
        u, s, vh = torch.linalg.svd(view, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        s_smooth = _smooth_singular_values(s_r, mode=smooth_mode, strength=smooth_strength)
        delta = u_r @ torch.diag(s_smooth) @ vh_r
        u_cols.append(u_r)
        v_cols.append(vh_r.T)
        smoothed_deltas.append(delta)

    u_cover = _whiten_column_basis(torch.cat(u_cols, dim=1), eps=whiten_eps)
    v_cover = _whiten_column_basis(torch.cat(v_cols, dim=1), eps=whiten_eps)
    core_mats = [u_cover.T @ delta @ v_cover for delta in smoothed_deltas]

    dc_param_keys = {
        "vector_1d_merge",
        "sv_reduction",
        "max_rank",
        "svd_dtype",
        "energy_smoothing",
        "energy_smoothing_strength",
        "whiten_eps",
        "cover_merge_method",
        "cover_merge_params",
        "mask_mode",
    }
    inner_params = {k: v for k, v in params.items() if k not in dc_param_keys}
    inner_params.update(dict(cover_merge_params))

    merged_core = merge_functional(
        cover_merge_method,
        matrices=core_mats,
        weights=weights.tolist(),
        alpha=1.0,
        method_params=inner_params,
    ).to(dtype=svd_dtype)

    if mask_mode == "block":
        merged_core = merged_core * _block_mask(
            n_tasks=n_tasks,
            rank=rank,
            dtype=merged_core.dtype,
            device=merged_core.device,
        )

    merged = u_cover @ merged_core @ v_cover.T
    return merged.reshape_as(ref).to(dtype=ref.dtype, device=ref.device)


register_impl("dc_merge", dc_merge_impl)
