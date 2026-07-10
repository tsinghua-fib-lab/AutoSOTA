from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class CARTMerge:
    name: str = "cart_merge"

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

        method_params = get_method_params(kwargs)

        pruning_rank = int(method_params.get("pruning_rank", 4))
        scaling_coeffs = float(method_params.get("scaling_coeffs", 0.5))

        direction: TensorDict = {}
        for k in keys:
            b = base[k]
            if b.ndim == 2 and "text_projection" not in k:
                direction[k] = self._cart_delta([d[k] for d in deltas], w, pruning_rank, scaling_coeffs).to(
                    dtype=b.dtype, device=b.device
                )
            else:
                direction[k] = torch.zeros_like(b)
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

    @staticmethod
    def _cart_delta(
        mats: list[torch.Tensor], w: torch.Tensor, pruning_rank: int, scaling_coeffs: float
    ) -> torch.Tensor:
        theta_avg = torch.stack(mats).mean(dim=0)
        sum = torch.zeros_like(theta_avg)
        for i in range(len(mats)):
            tau = mats[i] - theta_avg
            U, S, Vh = torch.linalg.svd(tau.to(torch.float64), full_matrices=False)
            pruning_rank_k = math.ceil(pruning_rank * S.shape[0])
            sum += U[:, :pruning_rank_k] @ torch.diag(S[:pruning_rank_k]) @ Vh[:pruning_rank_k, :] * w[i]
        return theta_avg + scaling_coeffs * sum


register(CARTMerge())
