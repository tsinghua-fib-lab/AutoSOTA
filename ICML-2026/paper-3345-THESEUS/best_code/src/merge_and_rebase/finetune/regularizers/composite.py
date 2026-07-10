from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from merge_and_rebase.finetune.regularizers.base import (
    BatchOverride,
    CheckpointArtifact,
    OptimizerBundle,
    checkpoint_payload_from_prepared,
    close_prepared_regularizer,
    collect_checkpoint_artifacts,
    finalize_model_for_regularizer,
    prepare_batch_override,
)
from merge_and_rebase.finetune.regularizers.registry import get_regularizer, register


@dataclass
class _CompositeChild:
    name: str
    impl: Any
    prepared: Any
    cfg: dict[str, Any]


@dataclass
class PreparedComposite:
    children: tuple[_CompositeChild, ...]
    optimizer_bundles: tuple[OptimizerBundle, ...]

    def close(self) -> None:
        for child in self.children:
            close_prepared_regularizer(child.prepared)

    def prepare_batch(self, **kwargs) -> BatchOverride | None:
        selected: BatchOverride | None = None
        selected_name: str | None = None
        for child in self.children:
            override = prepare_batch_override(child.prepared, **kwargs)
            if override is None:
                continue
            if selected is not None:
                raise ValueError(
                    "At most one child regularizer may override a batch. "
                    f"Got overrides from '{selected_name}' and '{child.name}'."
                )
            selected = override
            selected_name = child.name
        return selected

    def checkpoint_payload(self, *, kind: str) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for child in self.children:
            payload = checkpoint_payload_from_prepared(child.prepared, kind=kind)
            if not payload:
                continue
            collision = set(merged).intersection(payload)
            if collision:
                raise ValueError(
                    f"Checkpoint payload key collision from child regularizer '{child.name}': {sorted(collision)}"
                )
            merged.update(payload)
        return merged

    def checkpoint_artifacts(self, *, kind: str, **kwargs) -> tuple[CheckpointArtifact, ...]:
        out: list[CheckpointArtifact] = []
        for child in self.children:
            for artifact in collect_checkpoint_artifacts(child.prepared, kind=kind, **kwargs):
                out.append(artifact)
        return tuple(out)


@dataclass(frozen=True)
class CompositeRegularizer:
    name: str = "composite"

    def finalize_model(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        cfg = dict(regularization_cfg or {})
        children_raw = cfg.get("regularizers", None)
        if not isinstance(children_raw, list) or not children_raw:
            raise ValueError("regularization.regularizers must be a non-empty list for composite regularizer.")

        info: dict[str, Any] = {"composite_children": int(len(children_raw))}
        for idx, child_raw in enumerate(children_raw):
            if not isinstance(child_raw, dict):
                raise ValueError(f"regularization.regularizers[{idx}] must be a mapping.")
            child_cfg = dict(child_raw)
            child_name = str(child_cfg.get("name", "")).strip()
            if not child_name:
                raise ValueError(f"regularization.regularizers[{idx}] is missing 'name'.")
            child_impl = get_regularizer(child_name)
            child_info = finalize_model_for_regularizer(
                child_impl,
                model=model,
                device=device,
                regularization_cfg=child_cfg,
                **kwargs,
            )
            for key, value in child_info.items():
                info[f"{child_name}.{key}"] = value
        return info

    def prepare(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[PreparedComposite, dict[str, Any]]:
        cfg = dict(regularization_cfg or {})
        children_raw = cfg.get("regularizers", None)
        if not isinstance(children_raw, list) or not children_raw:
            raise ValueError("regularization.regularizers must be a non-empty list for composite regularizer.")

        children: list[_CompositeChild] = []
        bundles: list[OptimizerBundle] = []
        info: dict[str, Any] = {"composite_children": int(len(children_raw))}
        for idx, child_raw in enumerate(children_raw):
            if not isinstance(child_raw, dict):
                raise ValueError(f"regularization.regularizers[{idx}] must be a mapping.")
            child_cfg = dict(child_raw)
            child_name = str(child_cfg.get("name", "")).strip()
            if not child_name:
                raise ValueError(f"regularization.regularizers[{idx}] is missing 'name'.")
            child_impl = get_regularizer(child_name)
            prepared, child_info = child_impl.prepare(
                model=model,
                device=device,
                regularization_cfg=child_cfg,
                **kwargs,
            )
            children.append(_CompositeChild(name=child_name, impl=child_impl, prepared=prepared, cfg=child_cfg))
            bundles.extend(tuple(getattr(prepared, "optimizer_bundles", ()) or ()))
            for key, value in dict(child_info or {}).items():
                info[f"{child_name}.{key}"] = value
        return PreparedComposite(children=tuple(children), optimizer_bundles=tuple(bundles)), info

    def apply(
        self,
        prepared: PreparedComposite,
        *,
        model: nn.Module,
        step: int,
        batch_index: int,
        **kwargs,
    ) -> torch.Tensor:
        loss: torch.Tensor | None = None
        for child in prepared.children:
            value = child.impl.apply(
                child.prepared,
                model=model,
                step=step,
                batch_index=batch_index,
                **kwargs,
            )
            loss = value if loss is None else loss + value
        if loss is None:
            ref = kwargs.get("outputs", None)
            if isinstance(ref, torch.Tensor):
                return ref.new_zeros(())
            return torch.zeros(())
        return loss


register(CompositeRegularizer())
