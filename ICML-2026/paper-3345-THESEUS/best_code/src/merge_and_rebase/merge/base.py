from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

TensorDict = dict[str, torch.Tensor]


@runtime_checkable
class MergeMethod(Protocol):
    """
    Minimal API: merge base + tuned -> merged
    """

    name: str

    def merge(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> TensorDict: ...


@runtime_checkable
class PreparedMergeMethod(Protocol):
    """
    Optional API for alpha search:
      - prepare(): do heavy work once
      - apply(alpha): cheap scaling / composition
    """

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        """
        Returns:
          (base_sd, direction_delta_sd) where merged(alpha) = base_sd + alpha * direction_delta_sd
        """
        ...

    def apply(
        self,
        prepared: tuple[TensorDict, TensorDict],
        *,
        alpha: float,
        **kwargs,
    ) -> TensorDict: ...
