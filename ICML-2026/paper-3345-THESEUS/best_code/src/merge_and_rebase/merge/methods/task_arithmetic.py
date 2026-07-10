from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights


@dataclass(frozen=True)
class TaskArithmeticMerge:
    """
    merged = base + alpha * Σ_i w_i * (tuned_i - base)
    Optionally masks each task vector by magnitude first (unstructured).
    """

    name: str = "task_arithmetic"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        w = default_weights(len(tuned), weights)

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]

        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for k in keys:
            acc = torch.zeros_like(base[k])
            for wi, d in zip(w, deltas, strict=True):
                acc = acc + float(wi) * d[k].to(dtype=acc.dtype, device=acc.device)
            direction[k] = acc

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
        prepared = self.prepare(base=base, tuned=tuned, weights=weights, strict=strict)
        return self.apply(prepared, alpha=float(alpha))


register(TaskArithmeticMerge())
