from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype
from ._registry import register_alias, register_impl
from .weighted_average import weighted_average_impl

_VECTOR_1D_MODES = {"zero", "average"}
_FORM_MODES = {"delta", "absolute"}


def _resolve_form(params: Mapping[str, Any]) -> str:
    form = str(params.get("form", "delta")).strip().lower()
    if form not in _FORM_MODES:
        raise ValueError("actmerge method_params['form'] must be 'delta' or 'absolute'.")
    return form


def actmerge_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "average")).strip().lower()
        if vector_1d_merge not in _VECTOR_1D_MODES:
            raise ValueError("actmerge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return weighted_average_impl(matrices, weights, {"normalize": "sumw"})

    if matrices[0].ndim != 2:
        fallback = str(params.get("non_matrix_merge", "average")).strip().lower()
        if fallback not in _VECTOR_1D_MODES:
            raise ValueError("actmerge method_params['non_matrix_merge'] must be 'zero' or 'average'.")
        if fallback == "zero":
            return torch.zeros_like(matrices[0])
        return weighted_average_impl(matrices, weights, {"normalize": "sumw"})

    if bool((weights < 0).any().item()):
        raise ValueError("actmerge requires non-negative weights.")

    work_dtype = parse_dtype(str(params.get("work_dtype", "float64")))
    if work_dtype not in {torch.float32, torch.float64}:
        raise ValueError("actmerge method_params['work_dtype'] must be float32/fp32 or float64/fp64.")

    ridge = float(params.get("ridge", 0.0))
    if ridge < 0.0:
        raise ValueError("actmerge method_params['ridge'] must be >= 0.")

    pinv_rtol_raw = params.get("pinv_rtol", None)
    pinv_atol = float(params.get("pinv_atol", 0.0))
    if pinv_atol < 0.0:
        raise ValueError("actmerge method_params['pinv_atol'] must be >= 0.")
    pinv_rtol = None if pinv_rtol_raw is None else float(pinv_rtol_raw)
    if pinv_rtol is not None and pinv_rtol < 0.0:
        raise ValueError("actmerge method_params['pinv_rtol'] must be >= 0 when provided.")

    form = _resolve_form(params)
    base_matrix = params.get("base_matrix", None)

    ref = matrices[0]
    work_weights = weights.to(device=ref.device, dtype=work_dtype)
    work_mats = [matrix.to(device=ref.device, dtype=work_dtype) for matrix in matrices]

    denom = torch.zeros((int(ref.shape[1]), int(ref.shape[1])), dtype=work_dtype, device=ref.device)
    numer = torch.zeros_like(ref, dtype=work_dtype, device=ref.device)

    for weight, matrix in zip(work_weights, work_mats, strict=True):
        cov = matrix.T @ matrix
        denom = denom + weight * cov
        numer = numer + weight * (matrix @ cov)

    if ridge > 0.0:
        denom = denom + ridge * torch.eye(int(ref.shape[1]), dtype=work_dtype, device=ref.device)

    if pinv_rtol is None:
        denom_pinv = torch.linalg.pinv(denom, atol=pinv_atol)
    else:
        denom_pinv = torch.linalg.pinv(denom, atol=pinv_atol, rtol=pinv_rtol)

    merged = numer @ denom_pinv
    if form == "absolute":
        return merged.to(dtype=ref.dtype, device=ref.device)

    if base_matrix is None:
        return merged.to(dtype=ref.dtype, device=ref.device)

    base = base_matrix.to(device=ref.device, dtype=work_dtype)
    return (base + merged).to(dtype=ref.dtype, device=ref.device) - base_matrix.to(dtype=ref.dtype, device=ref.device)


register_impl("actmerge", actmerge_impl)
register_alias("actmat", "actmerge")
