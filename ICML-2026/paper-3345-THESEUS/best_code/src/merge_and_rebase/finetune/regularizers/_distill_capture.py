from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from merge_and_rebase.finetune._vision_runtime import ImageEncoder
from merge_and_rebase.finetune.regularizers._distill_config import EndpointConfig


@dataclass
class EndpointRuntime:
    config: EndpointConfig
    capture_id: str | None = None
    projection: nn.Module | None = None


@dataclass
class LocationRuntime:
    name: str
    student: EndpointRuntime
    teacher: EndpointRuntime
    loss: dict[str, Any]
    weight: float


class CaptureStore:
    def __init__(self) -> None:
        self.handles: list[Any] = []
        self.values: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()
        self.values.clear()


def coerce_capture_tensor(value: Any, *, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise ValueError(f"Unable to capture tensor for {label}.")


def resolve_hook_module(model: nn.Module, path: str) -> nn.Module:
    linearized_ref = getattr(model, "_linearized_visual_ref", None)
    if isinstance(linearized_ref, nn.Module):
        for prefix in ("clip_model.model.visual", "model.visual", "visual"):
            if path == prefix:
                return linearized_ref
            if path.startswith(prefix + "."):
                return linearized_ref.get_submodule(path[len(prefix) + 1 :])
    return model.get_submodule(path)


def register_endpoint_hook(store: CaptureStore, model: nn.Module, capture_id: str, config: EndpointConfig) -> None:
    if config.source != "module":
        return
    module = resolve_hook_module(model, str(config.path))

    if config.capture == "input":

        def _pre_hook(_module, inputs):
            store.values[capture_id] = coerce_capture_tensor(inputs, label=capture_id)

        handle = module.register_forward_pre_hook(_pre_hook)
    else:

        def _hook(_module, _inputs, output):
            store.values[capture_id] = coerce_capture_tensor(output, label=capture_id)

        handle = module.register_forward_hook(_hook)
    store.handles.append(handle)


def resolve_sequence_layout(tensor: torch.Tensor, *, batch_size: int, layout: str | None) -> str:
    if layout in {"batch_first", "sequence_first"}:
        return str(layout)
    if tensor.ndim != 3:
        raise ValueError("Token transforms require a 3D tensor.")
    if tensor.shape[0] == batch_size and tensor.shape[1] != batch_size:
        return "batch_first"
    if tensor.shape[1] == batch_size and tensor.shape[0] != batch_size:
        return "sequence_first"
    return "batch_first"


def apply_transform(tensor: torch.Tensor, spec: Mapping[str, Any], *, batch_size: int) -> torch.Tensor:
    name = str(spec.get("name", "identity")).strip().lower()
    if name == "identity":
        return tensor
    if name == "softmax":
        return torch.softmax(tensor, dim=int(spec.get("dim", -1)))
    if name == "log_softmax":
        return torch.log_softmax(tensor, dim=int(spec.get("dim", -1)))
    if name == "normalize":
        return F.normalize(tensor, dim=int(spec.get("dim", -1)), eps=float(spec.get("eps", 1e-12)))
    if name == "flatten":
        return tensor.flatten(start_dim=int(spec.get("start_dim", 1)))
    if name in {"mean_pool_tokens", "cls_token"}:
        layout = resolve_sequence_layout(tensor, batch_size=batch_size, layout=spec.get("layout"))
        seq_dim = 1 if layout == "batch_first" else 0
        if name == "mean_pool_tokens":
            return tensor.mean(dim=seq_dim)
        index = 0
        return tensor[:, index, :] if layout == "batch_first" else tensor[index, :, :]
    raise ValueError(f"Unknown transform '{name}'.")


def apply_transforms(tensor: torch.Tensor, transforms: Sequence[Mapping[str, Any]], *, batch_size: int) -> torch.Tensor:
    out = tensor
    for spec in transforms:
        out = apply_transform(out, spec, batch_size=batch_size)
    return out


def make_projection(raw: Mapping[str, Any] | None, *, in_features: int) -> nn.Module | None:
    if raw is None:
        return None
    kind = str(raw.get("kind", "identity")).strip().lower()
    if kind == "identity":
        return None
    out_features = int(raw.get("out_features", 0))
    if out_features <= 0:
        raise ValueError("Projection out_features must be > 0.")
    if kind == "linear":
        return nn.Linear(in_features, out_features, bias=bool(raw.get("bias", True)))
    if kind == "mlp":
        hidden = int(raw.get("hidden_features", max(in_features, out_features)))
        activation = str(raw.get("activation", "gelu")).strip().lower()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        return nn.Sequential(
            nn.Linear(in_features, hidden, bias=bool(raw.get("bias", True))),
            act,
            nn.Linear(hidden, out_features, bias=bool(raw.get("bias", True))),
        )
    raise ValueError(f"Unknown projection kind '{kind}'.")


def endpoint_tensor_from_symbolic(model: ImageEncoder, name: str, *, outputs: torch.Tensor | None) -> torch.Tensor:
    key = str(name).strip().lower()
    if key == "logits":
        tensor = outputs if outputs is not None else getattr(model, "_last_logits", None)
    elif key == "image_features":
        tensor = getattr(model, "_last_image_features", None)
    elif key == "visual_features":
        tensor = getattr(model, "_last_visual_features", None)
    else:
        raise ValueError(f"Unknown symbolic endpoint '{name}'.")
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"Endpoint '{name}' did not produce a tensor.")
    return tensor


def resolve_endpoint_tensor(
    *,
    endpoint: EndpointRuntime,
    model: ImageEncoder,
    store: CaptureStore,
    outputs: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    if endpoint.config.source == "symbolic":
        tensor = endpoint_tensor_from_symbolic(model, str(endpoint.config.name), outputs=outputs)
    else:
        if endpoint.capture_id is None or endpoint.capture_id not in store.values:
            raise RuntimeError(f"Missing captured activation for endpoint '{endpoint.config.path}'.")
        tensor = store.values[endpoint.capture_id]
    tensor = apply_transforms(tensor, endpoint.config.transforms, batch_size=batch_size)
    if endpoint.projection is not None:
        tensor = endpoint.projection.to(device=tensor.device, dtype=tensor.dtype)(tensor)
    return tensor


def infer_projection_modules(
    *,
    student_model: ImageEncoder,
    teacher_model: ImageEncoder,
    student_store: CaptureStore,
    teacher_store: CaptureStore,
    locations: Sequence[LocationRuntime],
    sample_inputs: torch.Tensor,
    device: torch.device,
) -> tuple[nn.ModuleList, dict[str, int]]:
    adapter_modules = nn.ModuleList()
    student_was_training = student_model.training
    teacher_was_training = teacher_model.training
    student_store.clear()
    teacher_store.clear()
    with torch.no_grad():
        student_model.eval()
        teacher_model.eval()
        sample_inputs = sample_inputs.to(device=device)
        student_outputs = student_model(sample_inputs)
        teacher_outputs = teacher_model(sample_inputs)
        batch_size = int(sample_inputs.shape[0])
        for location in locations:
            student_tensor = resolve_endpoint_tensor(
                endpoint=location.student,
                model=student_model,
                store=student_store,
                outputs=student_outputs,
                batch_size=batch_size,
            )
            teacher_tensor = resolve_endpoint_tensor(
                endpoint=location.teacher,
                model=teacher_model,
                store=teacher_store,
                outputs=teacher_outputs,
                batch_size=batch_size,
            )
            student_proj = make_projection(location.student.config.projection, in_features=int(student_tensor.shape[-1]))
            if student_proj is not None:
                location.student.projection = student_proj.to(device=device, dtype=student_tensor.dtype)
                adapter_modules.append(location.student.projection)
                student_tensor = location.student.projection(student_tensor)
            teacher_proj = make_projection(location.teacher.config.projection, in_features=int(teacher_tensor.shape[-1]))
            if teacher_proj is not None:
                location.teacher.projection = teacher_proj.to(device=device, dtype=teacher_tensor.dtype)
                adapter_modules.append(location.teacher.projection)
                teacher_tensor = location.teacher.projection(teacher_tensor)
            if tuple(student_tensor.shape) != tuple(teacher_tensor.shape):
                raise ValueError(
                    f"Location '{location.name}' produced mismatched shapes after transforms/projections: "
                    f"student={tuple(student_tensor.shape)} teacher={tuple(teacher_tensor.shape)}"
                )
    student_store.clear()
    teacher_store.clear()
    student_model.train(student_was_training)
    teacher_model.train(teacher_was_training)
    info = {
        "adapter_params": int(sum(p.numel() for p in adapter_modules.parameters())),
        "adapter_modules": int(len(adapter_modules)),
    }
    return adapter_modules, info
