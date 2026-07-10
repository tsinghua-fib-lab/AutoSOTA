from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params
from .functional import merge_functional


def _should_actmerge_key(key: str, tensor: torch.Tensor, *, merge_all_2d: bool) -> bool:
    if tensor.ndim != 2 or not tensor.is_floating_point():
        return False

    lower = key.lower()
    if "text_projection" in lower:
        return False
    if merge_all_2d:
        return True
    if "embedding" in lower or "embed_tokens" in lower or lower.endswith("lm_head.weight"):
        return False
    return lower.endswith(".weight") or lower == "weight" or lower.endswith("_weight")


@dataclass(frozen=True)
class ActMerge:
    """
    ACTMat / ACTMerge on dense task vectors.

    For linear 2D weights we merge deltas with
      Δ* = Σ_i w_i Δ_i (Δ_i^T Δ_i) (Σ_j w_j Δ_j^T Δ_j)^†
    and then apply the resulting direction on top of the shared base model.

    Non-matrix tensors default to weighted averaging, which matches the
    practical recipe described in the paper for parameters outside linear
    layer weights.
    """

    name: str = "actmerge"

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

        w = default_weights(len(tuned), weights)
        if bool((w < 0).any().item()):
            raise ValueError("actmerge requires non-negative weights.")

        method_params = get_method_params(kwargs)
        merge_all_2d = bool(method_params.get("merge_all_2d", False))
        fallback_merge = str(method_params.get("non_matrix_merge", "average")).strip().lower()
        if fallback_merge not in {"zero", "average"}:
            raise ValueError("actmerge method_params['non_matrix_merge'] must be 'zero' or 'average'.")

        vector_1d_merge = str(method_params.get("vector_1d_merge", fallback_merge)).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("actmerge method_params['vector_1d_merge'] must be 'zero' or 'average'.")

        tvs = [TaskVector.from_checkpoints(base, checkpoint, strict=strict) for checkpoint in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for key in tqdm(keys, desc="Processing keys"):
            ref = base[key]
            matrices = [delta[key] for delta in deltas]

            if _should_actmerge_key(key, ref, merge_all_2d=merge_all_2d):
                direction[key] = merge_functional(
                    "actmerge",
                    matrices=matrices,
                    weights=w.tolist(),
                    method_params={
                        **method_params,
                        "form": "delta",
                        "base_matrix": ref,
                    },
                ).to(dtype=ref.dtype, device=ref.device)
                continue

            if ref.ndim == 1 and vector_1d_merge == "zero":
                direction[key] = torch.zeros_like(ref)
                continue
            if ref.ndim != 2 and fallback_merge == "zero":
                direction[key] = torch.zeros_like(ref)
                continue

            direction[key] = merge_functional(
                "weighted_average",
                matrices=matrices,
                weights=w.tolist(),
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


register(ActMerge())
register(ActMerge(name="actmat"))
