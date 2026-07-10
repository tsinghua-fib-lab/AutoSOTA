from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ...merge.task_vectors import TaskVector
from ..base import TensorDict
from ..registry import register


@dataclass(frozen=True)
class IdentityTransport:
    """
    No-op transport: Δ' = Δ (restricted to compatible shared keys).
    """

    name: str = "identity"

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        keys = TaskVector.common_keys(source_base, [target_base, delta])

        if strict and set(keys) != set(delta.keys()):
            missing = sorted(set(delta.keys()) - set(keys))
            raise KeyError(f"Target/source base missing keys from delta. Example: {missing[:10]}")

        out: TensorDict = {}
        for k in keys:
            out[k] = delta[k].to(dtype=target_base[k].dtype, device=target_base[k].device)
        return out


register(IdentityTransport())
