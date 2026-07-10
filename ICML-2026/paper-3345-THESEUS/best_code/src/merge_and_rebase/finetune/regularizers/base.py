from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class OptimizerBundle:
    name: str
    optimizer: optim.Optimizer
    scheduler: Callable[[int], None] | None = None
    grad_clip_norm: float = -1.0


@dataclass(frozen=True)
class BatchOverride:
    outputs: torch.Tensor
    primary_loss: torch.Tensor
    context: Any = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class CheckpointArtifact:
    output_dir: str | Path
    filename: str
    payload: dict[str, Any]
    summary_filename: str | None = None
    summary: dict[str, Any] | None = None


@runtime_checkable
class PreparedRegularizer(Protocol):
    optimizer_bundles: tuple[OptimizerBundle, ...]

    def prepare_batch(self, **kwargs) -> BatchOverride | None: ...

    def checkpoint_payload(self, *, kind: str) -> dict[str, Any]: ...

    def checkpoint_artifacts(self, *, kind: str, **kwargs) -> tuple[CheckpointArtifact, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class Regularizer(Protocol):
    name: str

    def finalize_model(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]: ...

    def prepare(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[Any, dict[str, int]]: ...

    def apply(
        self,
        prepared: Any,
        *,
        model: nn.Module,
        step: int,
        batch_index: int,
        **kwargs,
    ) -> torch.Tensor: ...


def finalize_model_for_regularizer(
    regularizer: Any,
    *,
    model: nn.Module,
    device: torch.device,
    regularization_cfg: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    finalizer = getattr(regularizer, "finalize_model", None)
    if not callable(finalizer):
        return {}
    info = finalizer(
        model=model,
        device=device,
        regularization_cfg=regularization_cfg,
        **kwargs,
    )
    if info is None:
        return {}
    if not isinstance(info, dict):
        raise TypeError("regularizer.finalize_model must return a dict.")
    return dict(info)


def iter_optimizer_bundles(
    *,
    student_optimizer: optim.Optimizer,
    student_scheduler: Callable[[int], None] | None,
    student_grad_clip_norm: float,
    prepared: Any,
) -> tuple[OptimizerBundle, ...]:
    bundles = [
        OptimizerBundle(
            name="student",
            optimizer=student_optimizer,
            scheduler=student_scheduler,
            grad_clip_norm=float(student_grad_clip_norm),
        )
    ]
    auxiliary = getattr(prepared, "optimizer_bundles", ()) or ()
    bundles.extend(bundle for bundle in auxiliary if isinstance(bundle, OptimizerBundle))
    return tuple(bundles)


def prepare_batch_override(prepared: Any, **kwargs) -> BatchOverride | None:
    prepare_batch = getattr(prepared, "prepare_batch", None)
    if not callable(prepare_batch):
        return None
    override = prepare_batch(**kwargs)
    if override is not None and not isinstance(override, BatchOverride):
        raise TypeError("prepared_regularizer.prepare_batch must return BatchOverride or None.")
    return override


def checkpoint_payload_from_prepared(prepared: Any, *, kind: str) -> dict[str, Any]:
    builder = getattr(prepared, "checkpoint_payload", None)
    if not callable(builder):
        return {}
    payload = builder(kind=kind) or {}
    if not isinstance(payload, dict):
        raise TypeError("prepared_regularizer.checkpoint_payload must return a dict.")
    return dict(payload)


def collect_checkpoint_artifacts(prepared: Any, *, kind: str, **kwargs) -> tuple[CheckpointArtifact, ...]:
    builder = getattr(prepared, "checkpoint_artifacts", None)
    if not callable(builder):
        return ()
    artifacts = builder(kind=kind, **kwargs)
    if artifacts is None:
        return ()
    out: list[CheckpointArtifact] = []
    for artifact in tuple(artifacts):
        if not isinstance(artifact, CheckpointArtifact):
            raise TypeError("prepared_regularizer.checkpoint_artifacts must return CheckpointArtifact items.")
        out.append(artifact)
    return tuple(out)


def close_prepared_regularizer(prepared: Any) -> None:
    closer = getattr(prepared, "close", None)
    if callable(closer):
        closer()
