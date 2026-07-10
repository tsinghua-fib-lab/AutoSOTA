from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    key = str(name).strip().lower()
    if key.startswith("torch."):
        key = key[len("torch.") :]
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if key not in mapping:
        raise ValueError("Unknown dtype. Use one of: fp16, bf16, fp32, fp64 (or float16/32/64).")
    return mapping[key]


def default_weights(n: int, weights: Sequence[float] | None) -> torch.Tensor:
    if weights is None:
        return torch.ones(n, dtype=torch.float32)
    if len(weights) != n:
        raise ValueError("weights length must match number of matrices")
    return torch.tensor([float(w) for w in weights], dtype=torch.float32)


def merge_method_params(
    method_params: Mapping[str, Any] | None,
    technical_params: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(method_params or {})
    out.update(dict(technical_params))
    return out


def validate_matrices(matrices: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    mats = list(matrices)
    if not mats:
        raise ValueError("At least one matrix is required.")
    ref_shape = tuple(mats[0].shape)
    for i, matrix in enumerate(mats):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"Matrix #{i} is not a torch.Tensor.")
        if tuple(matrix.shape) != ref_shape:
            raise ValueError(f"All matrices must have the same shape. got {tuple(matrix.shape)} vs {ref_shape}")
    return mats


def require_2d(matrices: Sequence[torch.Tensor], method_name: str) -> None:
    if matrices[0].ndim != 2:
        raise ValueError(f"{method_name} requires 2D matrices. got shape {tuple(matrices[0].shape)}")


def rank_from_singular_values(
    num_singular_values: int,
    *,
    sv_reduction: float,
    max_rank: int | None,
) -> int:
    rank = max(1, int(num_singular_values * float(sv_reduction)))
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return max(1, int(rank))


def stack_flatten(matrices: Sequence[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
    rows = [matrix.reshape(-1).to(dtype=dtype) for matrix in matrices]
    return torch.stack(rows, dim=0)
