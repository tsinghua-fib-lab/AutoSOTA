from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._common import parse_dtype
from ._registry import register_alias, register_impl


def _matrix_view(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim == 0:
        return matrix.reshape(1, 1)
    if matrix.ndim == 1:
        return matrix.reshape(1, -1)
    return matrix.reshape(int(matrix.shape[0]), -1)


def _resolve_solver(params: Mapping[str, Any]) -> str:
    solver = str(params.get("solver", params.get("variant", params.get("mode", "closed_form")))).strip().lower()
    aliases = {
        "closed_form": "closed_form",
        "closed-form": "closed_form",
        "closed": "closed_form",
        "analytic": "closed_form",
        "solve": "closed_form",
        "gd": "gd",
        "gradient_descent": "gd",
        "gradient-descent": "gd",
        "optimizer": "gd",
        "optim": "gd",
    }
    if solver not in aliases:
        raise ValueError("wudi method_params['solver'] must be 'closed_form' or 'gd'.")
    return aliases[solver]


def _prepare_terms(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    *,
    work_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    ref = matrices[0]
    weight_device = weights.to(device=ref.device, dtype=work_dtype)
    terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for weight, matrix in zip(weight_device, matrices, strict=True):
        view = _matrix_view(matrix).to(dtype=work_dtype, device=ref.device)
        scaled = weight * view
        norm_sq = (scaled * scaled).sum()
        if bool(norm_sq <= 1e-12):
            continue
        gram = scaled.T @ scaled
        coeff = norm_sq.reciprocal()
        terms.append((scaled, gram, coeff))
    return ref, terms


def _closed_form(
    ref: torch.Tensor,
    terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    work_dtype: torch.dtype,
    ridge: float,
) -> torch.Tensor:
    ref_view = _matrix_view(ref)
    out_dim, in_dim = int(ref_view.shape[0]), int(ref_view.shape[1])
    a = torch.zeros((in_dim, in_dim), dtype=work_dtype, device=ref.device)
    c = torch.zeros((out_dim, in_dim), dtype=work_dtype, device=ref.device)

    for scaled, gram, coeff in terms:
        a = a + coeff * gram
        c = c + coeff * (scaled @ gram)

    if ridge > 0.0:
        a = a + torch.eye(in_dim, dtype=work_dtype, device=a.device) * ridge

    try:
        merged = torch.linalg.solve(a, c.T).T
    except RuntimeError:
        merged = c @ torch.linalg.pinv(a)

    return merged.reshape_as(ref_view).reshape_as(ref)


def _gradient_descent(
    ref: torch.Tensor,
    terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    work_dtype: torch.dtype,
    params: Mapping[str, Any],
) -> torch.Tensor:
    steps = int(params.get("steps", params.get("num_steps", params.get("iters", 300))))
    lr = float(params.get("lr", params.get("learning_rate", 1e-5)))
    weight_decay = float(params.get("weight_decay", 0.0))
    if steps <= 0:
        raise ValueError("wudi method_params['steps'] must be > 0.")
    if lr <= 0.0:
        raise ValueError("wudi method_params['lr'] must be > 0.")
    if weight_decay < 0.0:
        raise ValueError("wudi method_params['weight_decay'] must be >= 0.")

    ref_view = _matrix_view(ref)
    merged = torch.zeros_like(ref_view, dtype=work_dtype, device=ref.device)
    for scaled, _gram, _coeff in terms:
        merged = merged + scaled

    merging_vector = merged.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([merging_vector], lr=lr, weight_decay=weight_decay)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=work_dtype, device=ref.device)
        for scaled, _gram, coeff in terms:
            residual = (merging_vector - scaled) @ scaled.T
            loss = loss + coeff * residual.square().sum()
        loss.backward()
        optimizer.step()

    return merging_vector.detach().reshape_as(ref_view).reshape_as(ref)


def wudi_impl(
    matrices: list[torch.Tensor],
    weights: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    solver = _resolve_solver(params)
    ridge = float(params.get("ridge", 1e-8))
    if ridge < 0.0:
        raise ValueError("wudi method_params['ridge'] must be >= 0.")

    work_dtype = parse_dtype(str(params.get("work_dtype", "float32")))
    if work_dtype not in {torch.float32, torch.float64}:
        raise ValueError("wudi method_params['work_dtype'] must be float32/fp32 or float64/fp64.")

    ref, terms = _prepare_terms(matrices, weights, work_dtype=work_dtype)
    if not terms:
        return torch.zeros_like(ref)

    if solver == "closed_form":
        merged = _closed_form(ref, terms, work_dtype=work_dtype, ridge=ridge)
    else:
        merged = _gradient_descent(ref, terms, work_dtype=work_dtype, params=params)

    return merged.to(dtype=ref.dtype, device=ref.device)


register_impl("wudi", wudi_impl)
register_alias("wudi_merge", "wudi")
