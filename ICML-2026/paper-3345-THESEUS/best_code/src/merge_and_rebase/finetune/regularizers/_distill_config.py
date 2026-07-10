from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig


def as_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided.")
    return dict(raw)


def normalize_transform_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"name": str(raw)}
    if isinstance(raw, Mapping):
        data = dict(raw)
        if "name" not in data:
            data["name"] = str(data.get("type", "identity"))
        return data
    raise ValueError("Transform entries must be strings or mappings.")


def normalize_loss_cfg(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"name": str(raw)}
    if isinstance(raw, Mapping):
        data = dict(raw)
        if "name" not in data:
            raise ValueError("loss mappings must include 'name'.")
        return data
    raise ValueError("loss must be a string or a mapping.")


@dataclass(frozen=True)
class AlongPathConfig:
    enabled: bool = False
    alpha_start: float = 1.0
    alpha_end: float = 1.0
    sampling: str = "curriculum"
    temperature: float = 1.0


def normalize_along_path_cfg(raw: Any) -> AlongPathConfig:
    data = as_mapping(raw, field_name="regularization.along_path")
    if not data:
        return AlongPathConfig()
    enabled = bool(data.get("enabled", False))
    alpha_range = data.get("alpha_range", [1.0, 1.0])
    if (
        not isinstance(alpha_range, Sequence)
        or isinstance(alpha_range, (str, bytes))
        or len(alpha_range) != 2
    ):
        raise ValueError("regularization.along_path.alpha_range must be a 2-item sequence.")
    alpha_start = float(alpha_range[0])
    alpha_end = float(alpha_range[1])
    sampling = str(data.get("sampling", "curriculum")).strip().lower()
    if sampling not in {"curriculum", "uniform"}:
        raise ValueError("regularization.along_path.sampling must be 'curriculum' or 'uniform'.")
    return AlongPathConfig(
        enabled=enabled,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        sampling=sampling,
        temperature=float(data.get("temperature", 1.0)),
    )


@dataclass(frozen=True)
class EndpointConfig:
    source: str
    name: str | None = None
    path: str | None = None
    capture: str = "output"
    transforms: tuple[dict[str, Any], ...] = ()
    projection: dict[str, Any] | None = None


@dataclass(frozen=True)
class LocationConfig:
    name: str
    student: EndpointConfig
    teacher: EndpointConfig
    loss: dict[str, Any]
    weight: float


def normalize_endpoint(raw: Any, *, field_name: str) -> EndpointConfig:
    if isinstance(raw, str):
        return EndpointConfig(source="symbolic", name=str(raw))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a string or mapping.")
    data = dict(raw)
    if "path" in data:
        source = str(data.get("source", "module")).strip().lower()
    else:
        source = str(data.get("source", "symbolic")).strip().lower()
    if source == "symbolic":
        name = data.get("name", data.get("endpoint", None))
        if name is None:
            raise ValueError(f"{field_name} symbolic endpoint requires 'name'.")
        return EndpointConfig(
            source="symbolic",
            name=str(name),
            transforms=tuple(normalize_transform_spec(item) for item in list(data.get("transforms", []) or [])),
            projection=dict(data["projection"]) if isinstance(data.get("projection"), Mapping) else None,
        )
    if source != "module":
        raise ValueError(f"{field_name}.source must be 'symbolic' or 'module'.")
    path = data.get("path", None)
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{field_name} module endpoint requires a non-empty 'path'.")
    capture = str(data.get("capture", "output")).strip().lower()
    if capture not in {"input", "output"}:
        raise ValueError(f"{field_name}.capture must be 'input' or 'output'.")
    return EndpointConfig(
        source="module",
        path=path.strip(),
        capture=capture,
        transforms=tuple(normalize_transform_spec(item) for item in list(data.get("transforms", []) or [])),
        projection=dict(data["projection"]) if isinstance(data.get("projection"), Mapping) else None,
    )


def normalize_locations(raw: Any, *, shared_weight: float) -> tuple[LocationConfig, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("regularization.locations must be a list.")
    out: list[LocationConfig] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("Each distillation location must be a mapping.")
        data = dict(item)
        out.append(
            LocationConfig(
                name=str(data.get("name", f"location_{idx}")),
                student=normalize_endpoint(data.get("student", "logits"), field_name=f"locations[{idx}].student"),
                teacher=normalize_endpoint(data.get("teacher", "logits"), field_name=f"locations[{idx}].teacher"),
                loss=normalize_loss_cfg(data.get("loss", "kl_div")),
                weight=float(data.get("weight", shared_weight)),
            )
        )
    if not out:
        raise ValueError("regularization.locations must contain at least one location.")
    return tuple(out)


def merge_build_cfg(base: OpenClipBuildConfig, raw: Mapping[str, Any] | None) -> OpenClipBuildConfig:
    build = dict(raw or {})
    prompt_templates = base.prompt_templates
    if "prompt_templates" in build:
        raw_templates = build.get("prompt_templates")
        prompt_templates = list(raw_templates) if raw_templates is not None else None
    return OpenClipBuildConfig(
        loader=str(build.get("loader", base.loader)),
        model_name=str(build.get("model_name", base.model_name)),
        pretrained=str(build.get("pretrained", base.pretrained)),
        device=str(build.get("device", base.device)),
        dtype=build.get("dtype", base.dtype),
        normalize=bool(build.get("normalize", base.normalize)),
        logit_scale=float(build.get("logit_scale", base.logit_scale)),
        prompt_template=str(build.get("prompt_template", base.prompt_template)),
        prompt_templates=prompt_templates,
    )


def adapter_train_cfg(raw: Mapping[str, Any] | None, *, defaults: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw or {})
    optimizer_cfg = as_mapping(data.get("optimizer"), field_name="regularization.adapter_train.optimizer")
    scheduler_cfg = as_mapping(data.get("lr_scheduler"), field_name="regularization.adapter_train.lr_scheduler")
    return {
        "lr": float(data.get("lr", defaults["lr"])),
        "weight_decay": float(data.get("weight_decay", defaults["weight_decay"])),
        "optimizer_name": str(optimizer_cfg.get("name", defaults["optimizer_name"])),
        "scheduler_name": str(scheduler_cfg.get("name", defaults.get("scheduler_name", "cosine"))),
        "warmup_length": int(scheduler_cfg.get("warmup_steps", defaults["warmup_length"])),
        "grad_clip_norm": float(data.get("grad_clip_norm", defaults["grad_clip_norm"])),
    }


def teacher_train_cfg(raw: Mapping[str, Any] | None, *, defaults: Mapping[str, Any]) -> dict[str, Any]:
    data = as_mapping(raw, field_name="regularization.teacher.train")
    optimizer_cfg = as_mapping(data.get("optimizer"), field_name="regularization.teacher.train.optimizer")
    scheduler_cfg = as_mapping(data.get("lr_scheduler"), field_name="regularization.teacher.train.lr_scheduler")
    optimizer_dense_lr = optimizer_cfg.get("dense_lr", None)
    dense_lr = optimizer_dense_lr if optimizer_dense_lr is not None else data.get("dense_lr", defaults["dense_lr"])
    return {
        "lr": float(data.get("lr", defaults["lr"])),
        "dense_lr": float(dense_lr),
        "weight_decay": float(data.get("weight_decay", defaults["weight_decay"])),
        "optimizer_name": str(optimizer_cfg.get("name", defaults["optimizer_name"])),
        "scheduler_name": str(scheduler_cfg.get("name", defaults.get("scheduler_name", "cosine"))),
        "warmup_length": int(scheduler_cfg.get("warmup_steps", defaults["warmup_length"])),
        "grad_clip_norm": float(data.get("grad_clip_norm", defaults["grad_clip_norm"])),
    }
