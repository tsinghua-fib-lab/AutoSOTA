from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

_VISION_BLOCK_RE = re.compile(r"(visual\.transformer\.resblocks\.\d+)(?:\.|$)")
_LLM_LAYER_PATTERNS = (
    re.compile(r"(model\.layers\.\d+)(?:\.|$)"),
    re.compile(r"(encoder\.block\.\d+)(?:\.|$)"),
    re.compile(r"(decoder\.block\.\d+)(?:\.|$)"),
    re.compile(r"(transformer\.h\.\d+)(?:\.|$)"),
)


def layer_group_for_key(key: str, *, kind: str) -> str:
    key = str(key)
    kind = str(kind).strip().lower()
    if kind == "vision":
        match = _VISION_BLOCK_RE.search(key)
        return match.group(1) if match is not None else "__other__"
    if kind == "llm":
        for pattern in _LLM_LAYER_PATTERNS:
            match = pattern.search(key)
            if match is not None:
                return match.group(1)
        return "__other__"
    return "__other__"


@dataclass(frozen=True)
class TaskDeltaBank:
    base: dict[str, torch.Tensor]
    deltas_by_task: list[dict[str, torch.Tensor]]
    weights: list[float]
    tasks: list[str]
    kind: str
    tensor_keys: list[str]
    tensor_index_by_key: dict[str, int]
    layer_for_key: dict[str, str]
    layer_groups: list[str]

    @classmethod
    def build(
        cls,
        *,
        base: Mapping[str, torch.Tensor],
        tuned: Sequence[Mapping[str, torch.Tensor]],
        tasks: Sequence[str],
        weights: Sequence[float] | None,
        kind: str,
    ) -> TaskDeltaBank:
        if not tuned:
            raise ValueError("AdaMerging requires at least one tuned checkpoint.")
        if len(tasks) != len(tuned):
            raise ValueError("tasks length must match tuned checkpoint count for AdaMerging.")
        if weights is None:
            resolved_weights = [1.0] * len(tuned)
        else:
            if len(weights) != len(tuned):
                raise ValueError("weights length must match tuned checkpoint count for AdaMerging.")
            resolved_weights = [float(w) for w in weights]

        shared_keys = {k for k, v in base.items() if torch.is_tensor(v) and torch.is_floating_point(v)}
        for task_sd in tuned:
            current = {
                k
                for k, v in task_sd.items()
                if k in base
                and torch.is_tensor(v)
                and torch.is_floating_point(v)
                and tuple(v.shape) == tuple(base[k].shape)
            }
            shared_keys &= current
        if not shared_keys:
            raise ValueError("AdaMerging found no common floating-point tensors across base and tuned checkpoints.")

        ordered_keys = sorted(shared_keys)
        base_out = {k: v.detach().cpu() for k, v in base.items() if torch.is_tensor(v)}
        deltas_by_task: list[dict[str, torch.Tensor]] = []
        for task_sd in tuned:
            deltas: dict[str, torch.Tensor] = {}
            for key in ordered_keys:
                b = base_out[key]
                t = task_sd[key].detach().cpu().to(dtype=b.dtype)
                deltas[key] = t - b
            deltas_by_task.append(deltas)

        layer_for_key = {k: layer_group_for_key(k, kind=kind) for k in ordered_keys}
        layer_groups = sorted(set(layer_for_key.values()))
        return cls(
            base=base_out,
            deltas_by_task=deltas_by_task,
            weights=resolved_weights,
            tasks=[str(t) for t in tasks],
            kind=str(kind),
            tensor_keys=ordered_keys,
            tensor_index_by_key={k: i for i, k in enumerate(ordered_keys)},
            layer_for_key=layer_for_key,
            layer_groups=layer_groups,
        )

    def alpha_shape(self, mode: str) -> tuple[int, ...]:
        mode = str(mode).strip().lower()
        if mode == "task":
            return (len(self.deltas_by_task),)
        if mode == "layer":
            return (len(self.tensor_keys), len(self.deltas_by_task))
        raise ValueError("postmerge.alpha_mode must be one of: task, layer")

    def alpha_for(self, alpha_values: torch.Tensor, *, task_index: int, key: str, mode: str) -> torch.Tensor:
        if mode == "task":
            return alpha_values[int(task_index)]
        if mode == "layer":
            key_idx = self.tensor_index_by_key[str(key)]
            return alpha_values[key_idx, int(task_index)]
        raise ValueError("alpha mode must be one of: task, layer")

    def materialize(self, alpha_values: torch.Tensor, *, mode: str) -> dict[str, torch.Tensor]:
        mode = str(mode).strip().lower()
        alpha_cpu = alpha_values.detach().cpu()
        out: dict[str, torch.Tensor] = {}
        for key, base_tensor in self.base.items():
            if key not in self.layer_for_key:
                out[key] = base_tensor
                continue
            acc = torch.zeros_like(base_tensor)
            for task_idx, deltas in enumerate(self.deltas_by_task):
                alpha = self.alpha_for(alpha_cpu, task_index=task_idx, key=key, mode=mode)
                acc = acc + float(self.weights[task_idx]) * alpha.to(dtype=acc.dtype) * deltas[key]
            out[key] = base_tensor + acc
        return out

    def merged_parameter_dict(
        self,
        model: torch.nn.Module,
        alpha_values: torch.Tensor,
        *,
        mode: str,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        mode = str(mode).strip().lower()
        dev = torch.device(device)
        out: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name not in self.base or name not in self.layer_for_key:
                continue
            base_tensor = self.base[name].to(device=dev, dtype=param.dtype)
            acc = torch.zeros_like(base_tensor)
            for task_idx, deltas in enumerate(self.deltas_by_task):
                alpha = self.alpha_for(alpha_values, task_index=task_idx, key=name, mode=mode)
                delta = deltas[name].to(device=dev, dtype=param.dtype)
                acc = acc + float(self.weights[task_idx]) * alpha.to(device=dev, dtype=param.dtype) * delta
            out[name] = base_tensor + acc
        return out

    def trainable_delta_parameters(
        self,
        *,
        device: str | torch.device,
    ) -> list[dict[str, torch.nn.Parameter]]:
        dev = torch.device(device)
        out: list[dict[str, torch.nn.Parameter]] = []
        for deltas in self.deltas_by_task:
            task_params: dict[str, torch.nn.Parameter] = {}
            for key in self.tensor_keys:
                task_params[key] = torch.nn.Parameter(deltas[key].detach().clone().to(device=dev))
            out.append(task_params)
        return out

    def materialize_trainable_deltas(
        self,
        trainable_deltas: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if len(trainable_deltas) != len(self.deltas_by_task):
            raise ValueError("trainable_deltas length must match task count.")
        out: dict[str, torch.Tensor] = {}
        for key, base_tensor in self.base.items():
            if key not in self.layer_for_key:
                out[key] = base_tensor
                continue
            acc = torch.zeros_like(base_tensor)
            for task_idx, deltas in enumerate(trainable_deltas):
                delta = deltas[key].detach().cpu().to(dtype=acc.dtype)
                acc = acc + float(self.weights[task_idx]) * delta
            out[key] = base_tensor + acc
        return out

    def trainable_merged_delta_parameters(
        self,
        *,
        device: str | torch.device,
        trainable_keys: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        dev = torch.device(device)
        selected = None if trainable_keys is None else set(trainable_keys)
        out: dict[str, torch.Tensor] = {}
        for key in self.tensor_keys:
            acc = torch.zeros_like(self.base[key])
            for task_idx, deltas in enumerate(self.deltas_by_task):
                delta = deltas[key].detach().to(dtype=acc.dtype)
                acc = acc + float(self.weights[task_idx]) * delta
            if selected is None or key in selected:
                out[key] = torch.nn.Parameter(acc.to(device=dev))
            else:
                out[key] = acc
        return out

    def materialize_trainable_merged_delta(
        self,
        trainable_merged_delta: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, base_tensor in self.base.items():
            if key not in self.layer_for_key:
                out[key] = base_tensor
                continue
            delta = trainable_merged_delta[key].detach().cpu().to(dtype=base_tensor.dtype)
            out[key] = base_tensor + delta
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "num_tasks": len(self.tasks),
            "tasks": list(self.tasks),
            "num_tensors": len(self.layer_for_key),
            "tensor_keys": list(self.tensor_keys),
            "num_layer_groups": len(self.layer_groups),
            "layer_groups": list(self.layer_groups),
        }
