from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class WeightedAverageMerge:
    """
    merged = base + alpha * (avg(tuned) - base)

    If weights are all 1: avg is simple mean of checkpoints.
    If weights provided: avg is weighted mean.
    """

    name: str = "weighted_average"

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
        normalize = str(method_params.get("normalize", "sumw"))

        w = default_weights(len(tuned), weights)
        keys = TaskVector.common_keys(base, tuned)

        if normalize == "sumw":
            denom = float(w.sum().clamp_min(1e-12).item())
        elif normalize == "n":
            denom = float(len(tuned))
        else:
            raise ValueError("normalize must be 'sumw' or 'n'")

        direction: TensorDict = {}
        for k in keys:
            b = base[k]
            acc = torch.zeros_like(b)
            for wi, t in zip(w, tuned, strict=True):
                acc = acc + float(wi) * t[k].to(dtype=acc.dtype, device=acc.device)
            avg = acc / denom
            direction[k] = avg - b

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


register(WeightedAverageMerge())
