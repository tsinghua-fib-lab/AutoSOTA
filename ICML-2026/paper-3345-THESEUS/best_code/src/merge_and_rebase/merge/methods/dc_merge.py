from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tqdm import tqdm

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, get_method_params
from .functional import merge_functional


@dataclass(frozen=True)
class DCMerge:
    """
    DC-Merge over dense task deltas.

    LoRA checkpoints are expected to be materialized into dense model deltas by
    the existing evaluation pipeline before reaching this method.
    """

    name: str = "dc_merge"

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
        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for key in tqdm(keys, desc="Processing keys"):
            ref = base[key]
            matrices = [delta[key] for delta in deltas]
            direction[key] = merge_functional(
                "dc_merge",
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


register(DCMerge())
