from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from ._common import default_weights, merge_method_params, validate_matrices

FunctionalMergeImpl = Callable[[list[torch.Tensor], torch.Tensor, Mapping[str, Any]], torch.Tensor]

_IMPLS: dict[str, FunctionalMergeImpl] = {}
_ALIASES: dict[str, str] = {}


def register_impl(name: str, impl: FunctionalMergeImpl) -> None:
    if name in _IMPLS:
        raise KeyError(f"Functional merge method '{name}' already registered.")
    _IMPLS[name] = impl


def register_alias(alias: str, canonical: str) -> None:
    if alias in _ALIASES:
        raise KeyError(f"Functional merge alias '{alias}' already registered.")
    _ALIASES[alias] = canonical


def list_functional_methods() -> list[str]:
    return sorted(set(_IMPLS.keys()) | set(_ALIASES.keys()))


def merge_functional(
    method_name: str,
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    canonical = _ALIASES.get(method_name, method_name)
    if canonical not in _IMPLS:
        raise KeyError(f"Unknown functional merge method '{method_name}'. Available: {list_functional_methods()}")

    mats = validate_matrices(matrices)
    merge_weights = default_weights(len(mats), weights)
    params = merge_method_params(method_params, technical_params)

    merged = _IMPLS[canonical](mats, merge_weights, params)
    return (float(alpha) * merged).to(dtype=mats[0].dtype, device=mats[0].device)


def merge_raw_matrices(
    method_name: str,
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        method_name,
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )
