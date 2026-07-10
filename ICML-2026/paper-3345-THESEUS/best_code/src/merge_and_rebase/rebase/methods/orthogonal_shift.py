from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ...merge.task_vectors import TaskVector
from ..base import TensorDict
from ..registry import register


@dataclass(frozen=True)
class OrthogonalShiftTransport:
    """
    Transport that removes the component of Δ along the base-shift direction:
      s = target_base - source_base
      Δ' = Δ - beta * proj_s(Δ)

    When beta=1.0 this yields a delta orthogonal to s (per-parameter tensor block).
    """

    name: str = "orthogonal_shift"

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        beta: float = 1.0,
        eps: float = 1e-12,
        **kwargs,
    ) -> TensorDict:
        keys = TaskVector.common_keys(source_base, [target_base, delta])

        if strict and set(keys) != set(delta.keys()):
            missing = sorted(set(delta.keys()) - set(keys))
            raise KeyError(f"Target/source base missing keys from delta. Example: {missing[:10]}")

        b = float(beta)
        tol = float(eps)

        out: TensorDict = {}
        for k in keys:
            d = delta[k]
            shift = target_base[k] - source_base[k]

            shift_f = shift.float()
            d_f = d.float()

            denom = float((shift_f * shift_f).sum().item())
            if denom <= tol:
                transported = d
            else:
                coeff = float((d_f * shift_f).sum().item()) / denom
                proj = coeff * shift
                transported = d - b * proj.to(dtype=d.dtype, device=d.device)

            out[k] = transported.to(dtype=target_base[k].dtype, device=target_base[k].device)

        return out


register(OrthogonalShiftTransport())
