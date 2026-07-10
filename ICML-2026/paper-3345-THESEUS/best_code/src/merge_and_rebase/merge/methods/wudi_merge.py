from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, get_method_params
from .functional import merge_functional


@dataclass(frozen=True)
class WUDIMerge:
    """
    Minimal WUDI implementation.

    We apply the closed-form zero-regularization variant per tensor:
      argmin_M Σ_i ||(M - Δ_i) Δ_i^T||_F^2 / ||Δ_i||_F^2

    Non-matrix 1D tensors use a simple fallback because the paper focuses on
    linear weights. By default we average 1D deltas for practical stability.
    """

    name: str = "wudi"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        if len(tuned) == 0:
            raise ValueError("tuned must be non-empty")

        method_params = get_method_params(kwargs)
        vector_1d_merge = str(method_params.get("vector_1d_merge", "average")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("wudi method_params['vector_1d_merge'] must be 'zero' or 'average'.")

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for key in tqdm(keys, desc="Processing keys"):
            ref = base[key]
            matrices = [delta[key] for delta in deltas]

            if ref.ndim == 1:
                if vector_1d_merge == "zero":
                    direction[key] = torch.zeros_like(ref)
                else:
                    direction[key] = merge_functional(
                        "weighted_average",
                        matrices=matrices,
                        weights=weights,
                    ).to(dtype=ref.dtype, device=ref.device)
                continue

            direction[key] = merge_functional(
                "wudi",
                matrices=matrices,
                weights=weights,
                method_params=method_params,
            ).to(dtype=ref.dtype, device=ref.device)

        return base, direction

    def apply(self, prepared: tuple[TensorDict, TensorDict], *, alpha: float, **kwargs) -> TensorDict:
        base, direction = prepared
        return axpy_state_dict(base, direction, alpha=float(alpha))

    def merge(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        prepared = self.prepare(
            base=base,
            tuned=tuned,
            weights=weights,
            strict=strict,
            **kwargs,
        )
        return self.apply(prepared, alpha=float(alpha))


register(WUDIMerge())
register(WUDIMerge(name="wudi_merge"))
