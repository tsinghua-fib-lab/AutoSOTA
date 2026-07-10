from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from .task_delta_bank import TaskDeltaBank

TensorDict = dict[str, torch.Tensor]
EntropyLossFn = Callable[[TaskDeltaBank, torch.Tensor, str], torch.Tensor]
BackwardEntropyLossFn = Callable[[TaskDeltaBank, torch.Tensor, str], torch.Tensor]


@dataclass(frozen=True)
class PostMergeContext:
    kind: str
    model: torch.nn.Module
    base: Mapping[str, torch.Tensor]
    tuned: Sequence[Mapping[str, torch.Tensor]]
    tasks: Sequence[str]
    weights: Sequence[float] | None = None
    peft_subspace: str = "full"
    config: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    entropy_loss_fn: EntropyLossFn | None = None
    backward_entropy_loss_fn: BackwardEntropyLossFn | None = None


@dataclass(frozen=True)
class PostMergeResult:
    merged_state: TensorDict
    metadata: dict[str, Any]


@runtime_checkable
class PostMergeMethod(Protocol):
    name: str

    def run(self, context: PostMergeContext) -> PostMergeResult: ...
