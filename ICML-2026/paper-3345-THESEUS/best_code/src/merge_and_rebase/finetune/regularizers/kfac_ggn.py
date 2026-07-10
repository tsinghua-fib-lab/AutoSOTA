from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from merge_and_rebase.finetune.reference_tasks import resolve_reference_tasks_from_kwargs
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig
from merge_and_rebase.models.patch_openclip_attention import split_openclip_vit_attn
from merge_and_rebase.utils.peft_materialization import (
    is_lora_parameter_name as _shared_is_lora_parameter_name,
    materialized_peft_param_map as _shared_materialized_peft_param_map,
    is_peft_linear_module as _shared_is_peft_linear_module,
    materialize_peft_lora_weight as _shared_materialize_peft_lora_weight,
)

from ._vision_collection import build_vision_regularizer_task_context
from .registry import register

_VERSION = 3
_RUNTIME_LAYOUT_POLICY = "batch_size_inferred_v1"
_RUNTIME_SEQUENCE_LAYOUT = "runtime_batch_size_inferred"
_LN_POST_FULL_BLOCK_POLICY = "cls_only_v1"
_CACHE_METADATA_IGNORED_KEYS = frozenset(
    {
        "train_percent",
        "fisher_seed",
        "fisher_num_samples_expectation",
        "precision",
    }
)


@dataclass
class TaskCurvatureStats:
    aaT: dict[str, torch.Tensor]
    ggT: dict[str, torch.Tensor]
    ffT: dict[str, torch.Tensor]
    num_examples_aaT: int
    num_examples_ggT: int
    metadata: dict[str, Any] | None = None


@dataclass
class AggregatedCurvature:
    aaT: dict[str, list[torch.Tensor]]
    ggT: dict[str, list[torch.Tensor]]
    ffT: dict[str, list[torch.Tensor]]
    coeffs: list[float]
    reference_tasks: list[str]


@dataclass
class PenaltyBreakdown:
    loss_reg_matrix: torch.Tensor
    loss_reg_ffT: torch.Tensor
    loss_reg_proj: torch.Tensor
    loss_reg_cls: torch.Tensor

    @property
    def total_unscaled(self) -> torch.Tensor:
        return self.loss_reg_matrix + self.loss_reg_ffT + self.loss_reg_proj + self.loss_reg_cls


@dataclass(frozen=True)
class PreparedKfacGgn:
    config: KfacGgnConfig
    plan: TrackedCurvaturePlan
    base: dict[str, torch.Tensor]
    aggregated: AggregatedCurvature
    ignored_trainable: int


@dataclass(frozen=True)
class KfacGgnConfig:
    cache_dir: Path = Path("src/checkpoints/kfac_ggn")
    precision: str = "fp32"
    reg_lambda: float = 0.0
    full_block_scaler: float = 1.0e4
    projection_scaler: float = 1.0e-3
    cadence: int = 1
    force_recompute: bool = False
    train_percent: float | int = 1.0
    fisher_seed: int | None = None
    fisher_num_samples_expectation: int = 1


@dataclass(frozen=True)
class MatrixBlock:
    key: str
    module_name: str | None
    bias_key: str | None
    layout: str
    is_projection: bool = False


@dataclass(frozen=True)
class FullBlock:
    key: str
    module_name: str | None
    kind: str
    layout: str = "non_sequence"


@dataclass
class TrackedCurvaturePlan:
    matrix_blocks: dict[str, MatrixBlock]
    full_blocks: dict[str, FullBlock]
    param_shapes: dict[str, tuple[int, ...]]
    ignored_trainable: list[str]
    actual_param_names: dict[str, str] = field(default_factory=dict)


@dataclass
class _Acc:
    value: torch.Tensor
    compensation: torch.Tensor


def _as_config(raw: Mapping[str, Any] | None) -> KfacGgnConfig:
    cfg = dict(raw or {})
    cfg.pop("name", None)
    cfg.pop("reference_suite", None)
    cfg.pop("reference_datasets", None)
    allowed = set(KfacGgnConfig.__dataclass_fields__)
    unknown = sorted(k for k in cfg if k not in allowed)
    if unknown:
        raise ValueError(f"Unknown kfac_ggn config keys: {unknown}")
    if "cache_dir" in cfg:
        cfg["cache_dir"] = Path(str(cfg["cache_dir"]))
    if "train_percent" in cfg and isinstance(cfg["train_percent"], str):
        raw_value = cfg["train_percent"].strip()
        cfg["train_percent"] = float(raw_value) if any(c in raw_value for c in ".eE") else int(raw_value)
    out = KfacGgnConfig(**cfg)
    if out.precision not in {"fp32", "fp64"}:
        raise ValueError("kfac_ggn precision must be one of: fp32, fp64")
    if out.cadence < 1:
        raise ValueError("kfac_ggn cadence must be >= 1")
    if isinstance(out.train_percent, float) and not (0 < out.train_percent <= 1.0):
        raise ValueError("kfac_ggn train_percent float must be in (0, 1].")
    if isinstance(out.train_percent, int) and out.train_percent < 1:
        raise ValueError("kfac_ggn train_percent int must be >= 1.")
    if out.fisher_num_samples_expectation < 0:
        raise ValueError("kfac_ggn fisher_num_samples_expectation must be >= 0.")
    return out


def _safe_tag(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "__", text)
    return text.strip("_") or "unknown"


def _format_cache_status(
    *,
    regularizer: str,
    task: str,
    stage: str,
    path: Path,
    cached: bool,
) -> str:
    verb = "found in cache" if cached else "computing"
    return f"[{regularizer}] {task} {stage}: {verb} ({path})"


def _format_cache_completed(
    *,
    regularizer: str,
    task: str,
    stage: str,
    path: Path,
) -> str:
    return f"[{regularizer}] {task} {stage}: completed ({path})"


def _progress_total(data_loader: Iterable[Any], max_batches: int | None) -> int | None:
    try:
        total = len(data_loader)  # type: ignore[arg-type]
    except TypeError:
        return max_batches
    return total if max_batches is None else min(total, max_batches)


def task_cache_path(
    *,
    cache_dir: str | Path,
    build_cfg: OpenClipBuildConfig,
    task: str,
) -> Path:
    loader = str(getattr(build_cfg, "loader", "openclip")).strip().lower() or "openclip"
    return (
        Path(cache_dir)
        / _safe_tag(loader)
        / _safe_tag(build_cfg.model_name)
        / _safe_tag(build_cfg.pretrained)
        / _safe_tag(task)
        / "curvature.pt"
    )


def _dtype_from_precision(precision: str) -> torch.dtype:
    return torch.float64 if precision == "fp64" else torch.float32


def _visual_module(model: nn.Module) -> nn.Module:
    clip = getattr(model, "clip_model", None)
    if clip is not None and hasattr(clip, "model") and hasattr(clip.model, "visual"):
        return clip.model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    if hasattr(model, "visual"):
        return model.visual
    return model


def _tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected tensor module output, got {type(output)}")


def _images_from_batch(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("pixel_values", "images", "image", "inputs", "x"):
            value = batch.get(key, None)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    raise TypeError("Unsupported batch format for kfac_ggn collection.")


def _module_by_name(root: nn.Module, name: str | None) -> nn.Module | None:
    if name is None:
        return None
    if name == "":
        return root
    return dict(root.named_modules()).get(name)


def _kahan_add(accs: dict[str, _Acc], key: str, value: torch.Tensor) -> None:
    value = value.detach()
    if key not in accs:
        accs[key] = _Acc(value=value.clone(), compensation=torch.zeros_like(value))
        return
    acc = accs[key]
    y = value - acc.compensation
    t = acc.value + y
    acc.compensation = (t - acc.value) - y
    acc.value = t


def _acc_values(accs: dict[str, _Acc]) -> dict[str, torch.Tensor]:
    return {k: v.value for k, v in accs.items()}


def _infer_sequence_layout(
    x: torch.Tensor,
    *,
    layout: str,
    current_batch_size: int | None,
    target: str,
) -> str:
    if x.ndim != 3:
        raise ValueError(f"KFAC layout inference expects 3D tensors, got shape={tuple(x.shape)} for {target}")
    if layout == "batch_first_sequence":
        return layout
    if layout == "sequence_first":
        return layout
    if layout != _RUNTIME_SEQUENCE_LAYOUT:
        raise ValueError(f"Unknown sequence layout policy: {layout!r} for {target}")
    if current_batch_size is None:
        raise RuntimeError(f"Cannot infer KFAC layout for {target}: current batch size is unknown.")
    matches: list[str] = []
    if x.shape[0] == current_batch_size:
        matches.append("batch_first_sequence")
    if x.shape[1] == current_batch_size:
        matches.append("sequence_first")
    if len(matches) != 1:
        raise ValueError(
            f"Ambiguous KFAC layout for {target}: shape={tuple(x.shape)}, current_batch_size={current_batch_size}, "
            f"matches={matches or 'none'}"
        )
    return matches[0]


def _flatten_sequence(
    x: torch.Tensor,
    layout: str,
    *,
    current_batch_size: int | None = None,
    target: str,
) -> tuple[torch.Tensor, int]:
    if x.ndim == 2:
        return x, 1
    if x.ndim != 3:
        raise ValueError(f"KFAC hooks expect 2D or 3D tensors, got shape={tuple(x.shape)}")
    inferred_layout = _infer_sequence_layout(
        x,
        layout=layout,
        current_batch_size=current_batch_size,
        target=target,
    )
    if inferred_layout == "batch_first_sequence":
        _, seq_len, channels = x.shape
    elif inferred_layout == "sequence_first":
        seq_len, _, channels = x.shape
    else:
        raise ValueError(f"Unknown sequence layout: {inferred_layout}")
    return x.reshape(-1, channels), int(seq_len)


def _sum_over_sequence_axis(
    x: torch.Tensor,
    *,
    layout: str,
    current_batch_size: int | None,
    target: str,
) -> torch.Tensor:
    if x.ndim <= 2:
        return x
    inferred_layout = _infer_sequence_layout(
        x,
        layout=layout,
        current_batch_size=current_batch_size,
        target=target,
    )
    return x.sum(dim=1) if inferred_layout == "batch_first_sequence" else x.sum(dim=0)


def _cls_only_from_sequence(
    x: torch.Tensor,
    *,
    layout: str,
    current_batch_size: int | None,
    target: str,
) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(f"CLS-only reduction expects 2D or 3D tensors, got shape={tuple(x.shape)} for {target}")
    inferred_layout = _infer_sequence_layout(
        x,
        layout=layout,
        current_batch_size=current_batch_size,
        target=target,
    )
    return x[:, 0, :] if inferred_layout == "batch_first_sequence" else x[0, :, :]


def _projection_rows_from_ln_post(x: torch.Tensor, visual: nn.Module) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(f"Projection KFAC hook expects 2D or 3D ln_post output, got shape={tuple(x.shape)}")
    pool_type = str(getattr(visual, "pool_type", "tok"))
    if pool_type == "avg":
        return x[:, 1:].mean(dim=1)
    if pool_type == "tok":
        return x[:, 0]
    raise ValueError(f"Unsupported visual pool_type for projection KFAC hook: {pool_type!r}")


def _matrix_gram_from_rows(rows: torch.Tensor, *, normalize_by: int) -> torch.Tensor:
    denom = max(1, int(normalize_by))
    return rows.T @ (rows / float(denom))


def normalize_attn_patch_cfg(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    attn_impl = str(cfg.get("attn_impl", "softmax")).strip().lower()
    if attn_impl not in {"softmax", "linear"}:
        raise ValueError("attention attn_impl must be one of: softmax, linear")
    linear_rule = str(cfg.get("linear_rule", "kernel")).strip().lower()
    if linear_rule not in {"kernel", "delta"}:
        raise ValueError("attention linear_rule must be one of: kernel, delta")
    return {
        "attn_impl": attn_impl,
        "kernel": str(cfg.get("kernel", "elu_plus_one")),
        "eps": float(cfg.get("eps", 1e-6)),
        "linear_rule": linear_rule,
        "delta_eta": float(cfg.get("delta_eta", 1.0)),
        "delta_exclude_cls_from_store": bool(cfg.get("delta_exclude_cls_from_store", True)),
        "delta_cls_only_readout": bool(cfg.get("delta_cls_only_readout", False)),
        "delta_learn_w0": bool(cfg.get("delta_learn_w0", False)),
        "delta_w0_rank": int(cfg.get("delta_w0_rank", 0)),
    }


def ensure_openclip_kfac_surface(
    model_or_visual: nn.Module,
    *,
    attn_patch_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    visual = _visual_module(model_or_visual)
    cfg = normalize_attn_patch_cfg(attn_patch_cfg)
    patched = split_openclip_vit_attn(
        visual,
        proj_dropout=0.0,
        attn_impl=str(cfg["attn_impl"]),
        kernel=str(cfg["kernel"]),
        eps=float(cfg["eps"]),
        linear_rule=str(cfg["linear_rule"]),
        delta_eta=float(cfg["delta_eta"]),
        delta_exclude_cls_from_store=bool(cfg["delta_exclude_cls_from_store"]),
        delta_cls_only_readout=bool(cfg["delta_cls_only_readout"]),
        delta_learn_w0=bool(cfg["delta_learn_w0"]),
        delta_w0_rank=int(cfg["delta_w0_rank"]),
    )
    setattr(model_or_visual, "peft_patched_attn", True)
    setattr(model_or_visual, "peft_attn_patch_cfg", cfg)
    return {"attn_patch_cfg": cfg, "patched_blocks": int(patched)}


def _is_matrix_module_name(name: str) -> bool:
    normalized = f".{name}."
    return ".attn." in normalized or ".mlp." in normalized


def _normalize_visual_name(name: str) -> str:
    out = str(name)
    while out.startswith("base_model.model."):
        out = out[len("base_model.model.") :]
    out = out.replace(".base_layer.", ".")
    return out


def _is_lora_parameter_name(name: str) -> bool:
    return _shared_is_lora_parameter_name(name)


def _is_peft_linear_module(module: nn.Module) -> bool:
    return _shared_is_peft_linear_module(module)


def _normalized_visual_param_key(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    local_name = _normalize_visual_name(name)
    if local_name == "lin_proj.weight":
        return "visual.proj", tensor.T
    return f"visual.{local_name}", tensor


def _materialize_peft_lora_weight(module: nn.Module) -> torch.Tensor:
    return _shared_materialize_peft_lora_weight(module)


def select_tracked_parameters(model_or_visual: nn.Module) -> TrackedCurvaturePlan:
    visual = _visual_module(model_or_visual)
    matrix_blocks: dict[str, MatrixBlock] = {}
    full_blocks: dict[str, FullBlock] = {}
    param_shapes: dict[str, tuple[int, ...]] = {}
    actual_param_names: dict[str, str] = {}

    named_params = dict(visual.named_parameters())
    modules = dict(visual.named_modules())

    for module_name, module in modules.items():
        normalized_module_name = _normalize_visual_name(module_name)

        weight: torch.Tensor | None = None
        bias: torch.Tensor | None = None
        if _is_peft_linear_module(module):
            if not _is_matrix_module_name(normalized_module_name):
                continue
            base_layer = getattr(module, "base_layer")
            weight = getattr(base_layer, "weight", None)
            bias = getattr(base_layer, "bias", None)
        elif isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            if module_name.endswith(".base_layer") or not _is_matrix_module_name(normalized_module_name):
                continue
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                continue
            key = f"visual.{normalized_module_name}.weight"
            bias_key = None
            if isinstance(bias, torch.Tensor):
                bias_key = f"visual.{normalized_module_name}.bias"
                param_shapes[bias_key] = tuple(bias.shape)
                if _is_peft_linear_module(module):
                    actual_param_names[bias_key] = f"{module_name}.base_layer.bias"
                else:
                    actual_param_names[bias_key] = f"{module_name}.bias" if module_name else "bias"
            matrix_blocks[key] = MatrixBlock(
                key=key,
                module_name=module_name,
                bias_key=bias_key,
                layout=_RUNTIME_SEQUENCE_LAYOUT,
            )
            param_shapes[key] = tuple(weight.shape)
            if _is_peft_linear_module(module):
                actual_param_names[key] = f"{module_name}.base_layer.weight"
            else:
                actual_param_names[key] = f"{module_name}.weight" if module_name else "weight"

        if isinstance(module, nn.LayerNorm):
            layout = _RUNTIME_SEQUENCE_LAYOUT
            if normalized_module_name == "ln_post":
                weight_kind = "layer_norm_post_weight"
                bias_kind = "layer_norm_post_bias"
            else:
                weight_kind = "layer_norm_weight"
                bias_kind = "layer_norm_bias"
            if getattr(module, "weight", None) is not None:
                key = f"visual.{normalized_module_name}.weight" if normalized_module_name else "visual.weight"
                full_blocks[key] = FullBlock(key=key, module_name=module_name, kind=weight_kind, layout=layout)
                param_shapes[key] = tuple(module.weight.shape)  # type: ignore[union-attr]
                actual_param_names[key] = f"{module_name}.weight" if module_name else "weight"
            if getattr(module, "bias", None) is not None:
                key = f"visual.{normalized_module_name}.bias" if normalized_module_name else "visual.bias"
                full_blocks[key] = FullBlock(key=key, module_name=module_name, kind=bias_kind, layout=layout)
                param_shapes[key] = tuple(module.bias.shape)  # type: ignore[union-attr]
                actual_param_names[key] = f"{module_name}.bias" if module_name else "bias"

    proj_param_name = None
    proj = None
    for name, param in named_params.items():
        normalized_local_name = _normalize_visual_name(name)
        if normalized_local_name not in {"proj", "lin_proj.weight"} or _is_lora_parameter_name(name):
            continue
        proj_param_name = name
        proj = param
        break
    if proj is not None and proj.ndim == 2:
        matrix_blocks["visual.proj"] = MatrixBlock(
            key="visual.proj",
            module_name=None,
            bias_key=None,
            layout="non_sequence",
            is_projection=True,
        )
        param_shapes["visual.proj"] = tuple(proj.T.shape) if _normalize_visual_name(str(proj_param_name)) == "lin_proj.weight" else tuple(proj.shape)
        assert proj_param_name is not None
        actual_param_names["visual.proj"] = str(proj_param_name)

    class_embedding_name = None
    class_embedding = None
    for name, param in named_params.items():
        if _normalize_visual_name(name) != "class_embedding" or _is_lora_parameter_name(name):
            continue
        class_embedding_name = name
        class_embedding = param
        break
    if class_embedding is not None and class_embedding.ndim == 1:
        cls_module_name = next(
            (name for name in modules if _normalize_visual_name(name) == "ln_pre"),
            None,
        )
        full_blocks["visual.class_embedding"] = FullBlock(
            key="visual.class_embedding",
            module_name=cls_module_name,
            kind="class_embedding",
            layout="batch_first_sequence",
        )
        param_shapes["visual.class_embedding"] = tuple(class_embedding.shape)
        assert class_embedding_name is not None
        actual_param_names["visual.class_embedding"] = str(class_embedding_name)

    ignored_trainable: list[str] = []
    tracked_param_names = set(param_shapes)
    for name, param in visual.named_parameters():
        if not param.requires_grad or _is_lora_parameter_name(name):
            continue
        full_name, _ = _normalized_visual_param_key(name, param)
        if full_name not in tracked_param_names:
            ignored_trainable.append(full_name)

    return TrackedCurvaturePlan(
        matrix_blocks=matrix_blocks,
        full_blocks=full_blocks,
        param_shapes=param_shapes,
        ignored_trainable=sorted(ignored_trainable),
        actual_param_names=actual_param_names,
    )


def _metadata(
    *,
    task: str,
    build_cfg: OpenClipBuildConfig,
    config: KfacGgnConfig,
    plan: TrackedCurvaturePlan,
    attn_patch_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "task": str(task),
        "backbone_kind": "openclip",
        "model_name": str(build_cfg.model_name),
        "pretrained": str(build_cfg.pretrained),
        "precision": config.precision,
        "train_percent": config.train_percent,
        "fisher_seed": config.fisher_seed,
        "fisher_num_samples_expectation": int(config.fisher_num_samples_expectation),
        "layout_policy": _RUNTIME_LAYOUT_POLICY,
        "ln_post_full_block_policy": _LN_POST_FULL_BLOCK_POLICY,
        "attn_patch_cfg": normalize_attn_patch_cfg(attn_patch_cfg),
        "ignored_trainable": list(plan.ignored_trainable),
        "matrix_blocks": {
            key: {
                "shape": list(plan.param_shapes[key]),
                "bias_key": block.bias_key,
                "bias_shape": list(plan.param_shapes[block.bias_key]) if block.bias_key in plan.param_shapes else None,
                "layout": block.layout,
                "is_projection": bool(block.is_projection),
            }
            for key, block in sorted(plan.matrix_blocks.items())
        },
        "full_blocks": {
            key: {
                "shape": list(plan.param_shapes[key]),
                "kind": block.kind,
                "layout": block.layout,
            }
            for key, block in sorted(plan.full_blocks.items())
        },
    }


def metadata_compatible(existing: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    existing_meta = {k: v for k, v in dict(existing or {}).items() if k not in _CACHE_METADATA_IGNORED_KEYS}
    expected_meta = {k: v for k, v in dict(expected).items() if k not in _CACHE_METADATA_IGNORED_KEYS}
    return existing_meta == expected_meta


def save_task_curvature(path: str | Path, stats: TaskCurvatureStats) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aaT": {k: v.detach().cpu() for k, v in stats.aaT.items()},
        "ggT": {k: v.detach().cpu() for k, v in stats.ggT.items()},
        "ffT": {k: v.detach().cpu() for k, v in stats.ffT.items()},
        "num_examples_aaT": int(stats.num_examples_aaT),
        "num_examples_ggT": int(stats.num_examples_ggT),
        "metadata": dict(stats.metadata or {}),
    }
    torch.save(payload, p)


def load_task_curvature(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    precision: str = "fp32",
) -> TaskCurvatureStats:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    dtype = _dtype_from_precision(precision)
    metadata = dict(payload.get("metadata", {}))

    def _load_dict(name: str) -> dict[str, torch.Tensor]:
        raw = payload.get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid curvature cache: '{name}' is not a dict")
        return {str(k): v.to(dtype=dtype) for k, v in raw.items()}

    return TaskCurvatureStats(
        aaT=_load_dict("aaT"),
        ggT=_load_dict("ggT"),
        ffT=_load_dict("ffT"),
        num_examples_aaT=int(payload["num_examples_aaT"]),
        num_examples_ggT=int(payload["num_examples_ggT"]),
        metadata=metadata,
    )


def _run_visual(model_or_visual: nn.Module, images: torch.Tensor) -> torch.Tensor:
    visual = _visual_module(model_or_visual)
    return visual(images)


def _limited_loader(data_loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    for idx, batch in enumerate(data_loader):
        if max_batches is not None and idx >= max_batches:
            break
        yield batch


def _resolve_num_batches(data_loader: Iterable[Any], cfg: KfacGgnConfig) -> int | None:
    train_percent = cfg.train_percent
    if isinstance(train_percent, float):
        if train_percent >= 1.0:
            return None
        try:
            return int(train_percent * len(data_loader))  # type: ignore[arg-type]
        except TypeError:
            return None
    return int(train_percent)


def collect_curvature(
    model: nn.Module,
    data_loader: Iterable[Any],
    tracked_params: TrackedCurvaturePlan | None = None,
    config: KfacGgnConfig | Mapping[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
    progress_label: str | None = None,
) -> TaskCurvatureStats:
    cfg = config if isinstance(config, KfacGgnConfig) else _as_config(config)
    visual = _visual_module(model)
    plan = tracked_params or select_tracked_parameters(visual)
    dtype = _dtype_from_precision(cfg.precision)
    dev = torch.device(device) if device is not None else next(visual.parameters()).device

    orig_training = visual.training
    orig_requires_grad = {name: param.requires_grad for name, param in visual.named_parameters()}
    handles: list[Any] = []
    aaT_accs: dict[str, _Acc] = {}
    ggT_accs: dict[str, _Acc] = {}
    ffT_accs: dict[str, _Acc] = {}
    modules = dict(visual.named_modules())
    max_batches = _resolve_num_batches(data_loader, cfg)
    current_batch_size: int | None = None
    if cfg.fisher_seed is not None:
        torch.manual_seed(int(cfg.fisher_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.fisher_seed))

    def _register_activation_hooks() -> None:
        for block in plan.matrix_blocks.values():
            if block.is_projection and block.module_name is None:
                ln_post = modules.get("ln_post", None)
                if ln_post is None:
                    raise RuntimeError("Cannot collect visual.proj KFAC block: visual.ln_post was not found.")

                def proj_hook(_module, _inputs, output, *, key=block.key):
                    with torch.no_grad():
                        hook_input = _tensor_from_output(output).detach().to(dtype=dtype)
                        rows = _projection_rows_from_ln_post(hook_input, visual)
                        _kahan_add(aaT_accs, key, _matrix_gram_from_rows(rows, normalize_by=1))

                handles.append(ln_post.register_forward_hook(proj_hook))
                continue

            module = _module_by_name(visual, block.module_name)
            if module is None:
                raise RuntimeError(f"Tracked KFAC module not found: {block.module_name}")

            def hook(_module, inputs, _output, *, b=block):
                with torch.no_grad():
                    x = inputs[0].detach().to(dtype=dtype)
                    if b.bias_key is not None:
                        ones = torch.ones_like(x[..., :1])
                        x = torch.cat([x, ones], dim=-1)
                    rows, denom = _flatten_sequence(
                        x,
                        b.layout,
                        current_batch_size=current_batch_size,
                        target=b.key,
                    )
                    _kahan_add(aaT_accs, b.key, _matrix_gram_from_rows(rows, normalize_by=denom))

            handles.append(module.register_forward_hook(hook))

    def _register_gradient_hooks() -> None:
        ln_inputs: dict[str, torch.Tensor] = {}

        for block in plan.matrix_blocks.values():
            if block.is_projection and block.module_name is None:
                continue
            module = _module_by_name(visual, block.module_name)
            if module is None:
                raise RuntimeError(f"Tracked KFAC module not found: {block.module_name}")

            def hook(_module, _grad_input, grad_output, *, b=block):
                with torch.no_grad():
                    grad = grad_output[0].detach().to(dtype=dtype)
                    rows, denom = _flatten_sequence(
                        grad,
                        b.layout,
                        current_batch_size=current_batch_size,
                        target=b.key,
                    )
                    _kahan_add(ggT_accs, b.key, _matrix_gram_from_rows(rows, normalize_by=denom))

            handles.append(module.register_full_backward_hook(hook))

        norm_blocks = [b for b in plan.full_blocks.values() if b.kind.startswith("layer_norm")]
        by_module: dict[str, list[FullBlock]] = {}
        for block in norm_blocks:
            if block.module_name is not None:
                by_module.setdefault(block.module_name, []).append(block)

        for module_name, blocks in by_module.items():
            module = _module_by_name(visual, module_name)
            if not isinstance(module, nn.LayerNorm):
                raise RuntimeError(f"Tracked LayerNorm module not found: {module_name}")

            def fwd_hook(_module, inputs, _output, *, name=module_name):
                ln_inputs[name] = inputs[0].detach()

            def bwd_hook(_module, _grad_input, grad_output, *, name=module_name, cur_blocks=tuple(blocks)):
                with torch.no_grad():
                    grad = grad_output[0].detach().to(dtype=dtype)
                    inputs = ln_inputs[name].to(dtype=dtype)
                    layout = cur_blocks[0].layout
                    ln_post_only = any(block.kind.startswith("layer_norm_post") for block in cur_blocks)
                    if ln_post_only:
                        pooled_inputs = _cls_only_from_sequence(
                            inputs,
                            layout=layout,
                            current_batch_size=current_batch_size,
                            target=f"{name}.inputs",
                        )
                        pooled_grad = _cls_only_from_sequence(
                            grad,
                            layout=layout,
                            current_batch_size=current_batch_size,
                            target=f"{name}.grad",
                        )
                        normalized = F.layer_norm(pooled_inputs, _module.normalized_shape, None, None, _module.eps)
                        grad_weight = pooled_grad * normalized
                        grad_bias = pooled_grad
                    else:
                        normalized = F.layer_norm(inputs, _module.normalized_shape, None, None, _module.eps)
                        grad_weight = grad * normalized
                        grad_bias = grad
                        grad_weight = _sum_over_sequence_axis(
                            grad_weight,
                            layout=layout,
                            current_batch_size=current_batch_size,
                            target=name,
                        )
                        grad_bias = _sum_over_sequence_axis(
                            grad_bias,
                            layout=layout,
                            current_batch_size=current_batch_size,
                            target=name,
                        )
                    for block in cur_blocks:
                        if block.kind in {"layer_norm_weight", "layer_norm_post_weight"}:
                            _kahan_add(ffT_accs, block.key, grad_weight.T @ grad_weight)
                        elif block.kind in {"layer_norm_bias", "layer_norm_post_bias"}:
                            _kahan_add(ffT_accs, block.key, grad_bias.T @ grad_bias)

            handles.append(module.register_forward_hook(fwd_hook))
            handles.append(module.register_full_backward_hook(bwd_hook))

        cls_block = next((b for b in plan.full_blocks.values() if b.kind == "class_embedding"), None)
        if cls_block is not None and cls_block.module_name is not None:
            module = _module_by_name(visual, cls_block.module_name)
            if module is not None:

                def cls_hook(_module, grad_input, _grad_output, *, key=cls_block.key):
                    with torch.no_grad():
                        grad = grad_input[0]
                        if grad is None:
                            return
                        grad = grad.detach().to(dtype=dtype)
                        if grad.ndim != 3:
                            return
                        # ln_pre sees the class token in batch-first form before the transformer transpose,
                        # so grad[:, 0, :] matches Mammoth's cls_token_layer backward statistic.
                        if current_batch_size is None or grad.shape[0] != current_batch_size:
                            raise RuntimeError(
                                "visual.class_embedding KFAC hook expected batch-first ln_pre gradients "
                                f"with batch_size={current_batch_size}, got shape={tuple(grad.shape)}"
                            )
                        token_grad = grad[:, 0, :]
                        _kahan_add(ffT_accs, key, token_grad.T @ token_grad)

                handles.append(module.register_full_backward_hook(cls_hook))

    try:
        visual.eval()
        _register_activation_hooks()
        num_aaT = 0
        aaT_desc = f"[{progress_label}] KFAC activations" if progress_label else "KFAC activations"
        with torch.no_grad(), tqdm(
            _limited_loader(data_loader, max_batches),
            total=_progress_total(data_loader, max_batches),
            desc=aaT_desc,
            unit="batch",
        ) as pbar:
            for batch in pbar:
                images = _images_from_batch(batch).to(dev)
                current_batch_size = int(images.shape[0])
                num_aaT += int(images.shape[0])
                _ = _run_visual(visual, images)

        for handle in handles:
            handle.remove()
        handles.clear()

        for param in visual.parameters():
            param.requires_grad_(False)
        named_visual_params = dict(visual.named_parameters())
        for key in plan.param_shapes:
            actual_name = plan.actual_param_names.get(key, None)
            param = named_visual_params.get(actual_name, None) if actual_name is not None else None
            if param is not None:
                param.requires_grad_(True)

        _register_gradient_hooks()
        num_ggT = 0
        ggT_desc = f"[{progress_label}] KFAC gradients" if progress_label else "KFAC gradients"
        with tqdm(
            _limited_loader(data_loader, max_batches),
            total=_progress_total(data_loader, max_batches),
            desc=ggT_desc,
            unit="batch",
        ) as pbar:
            for batch in pbar:
                visual.zero_grad(set_to_none=True)
                images = _images_from_batch(batch).to(dev)
                current_batch_size = int(images.shape[0])
                num_ggT += int(images.shape[0])
                fake_param = torch.tensor([1.0], device=dev, requires_grad=True)
                raw_features = _run_visual(visual, images * fake_param)
                if raw_features.ndim != 2:
                    raise RuntimeError(
                        "KFAC curvature expects visual(image) to produce pooled features with shape (batch, channels); "
                        f"got shape={tuple(raw_features.shape)}"
                    )
                proj_block = next((b for b in plan.matrix_blocks.values() if b.is_projection and b.module_name is None), None)
                if proj_block is not None:

                    def proj_grad_hook(grad, *, key=proj_block.key):
                        with torch.no_grad():
                            hook_input = grad.detach().to(dtype=dtype)
                            rows, denom = _flatten_sequence(
                                hook_input,
                                "non_sequence",
                                current_batch_size=current_batch_size,
                                target=key,
                            )
                            _kahan_add(ggT_accs, key, _matrix_gram_from_rows(rows, normalize_by=denom))

                    raw_features.register_hook(proj_grad_hook)
                features = F.normalize(raw_features, dim=-1)
                if cfg.fisher_num_samples_expectation > 0:
                    for sample_idx in range(int(cfg.fisher_num_samples_expectation)):
                        probe = torch.randn_like(features)
                        backward_source = features * probe
                        backward_target = backward_source.sum()
                        backward_target.backward(retain_graph=sample_idx < int(cfg.fisher_num_samples_expectation) - 1)
                else:
                    summed = features.sum(0)
                    for feat_idx, feat in enumerate(summed):
                        visual.zero_grad(set_to_none=True)
                        feat.backward(retain_graph=feat_idx < summed.shape[0] - 1)

        visual.zero_grad(set_to_none=True)
        stats = TaskCurvatureStats(
            aaT=_acc_values(aaT_accs),
            ggT=_acc_values(ggT_accs),
            ffT=_acc_values(ffT_accs),
            num_examples_aaT=num_aaT,
            num_examples_ggT=num_ggT,
        )
        missing_aaT = sorted(set(plan.matrix_blocks) - set(stats.aaT))
        missing_ggT = sorted(set(plan.matrix_blocks) - set(stats.ggT))
        missing_ffT = sorted(set(plan.full_blocks) - set(stats.ffT))
        if missing_aaT or missing_ggT or missing_ffT:
            raise RuntimeError(
                "KFAC/GGN collection missed tracked blocks: "
                f"aaT={missing_aaT}, ggT={missing_ggT}, ffT={missing_ffT}"
            )
        return stats
    finally:
        for handle in handles:
            handle.remove()
        for name, param in visual.named_parameters():
            if name in orig_requires_grad:
                param.requires_grad_(orig_requires_grad[name])
        visual.train(orig_training)


def aggregate_curvature(
    all_task_stats: Mapping[str, TaskCurvatureStats],
    current_task_id: str,
    *,
    task_order: list[str] | None = None,
) -> AggregatedCurvature:
    ordered = list(task_order or all_task_stats.keys())
    refs = [t for t in ordered if t != current_task_id and t in all_task_stats]
    if not refs:
        return AggregatedCurvature(aaT={}, ggT={}, ffT={}, coeffs=[], reference_tasks=[])

    total = sum(int(all_task_stats[t].num_examples_ggT) for t in refs)
    total = max(1, total)
    first = all_task_stats[refs[0]]
    aaT = {k: [torch.zeros_like(v)] for k, v in first.aaT.items()}
    ggT = {k: [torch.zeros_like(v)] for k, v in first.ggT.items()}
    ffT = {k: [torch.zeros_like(v)] for k, v in first.ffT.items()}
    for task in refs:
        stats = all_task_stats[task]
        n = max(1, int(stats.num_examples_ggT))
        for key in aaT:
            aaT[key][0] = aaT[key][0] + (float(n) / float(total)) * (stats.aaT[key] / float(n))
            ggT[key][0] = ggT[key][0] + (stats.ggT[key] / float(n))
        for key in ffT:
            ffT[key][0] = ffT[key][0] + (stats.ffT[key] / float(total))
    return AggregatedCurvature(aaT=aaT, ggT=ggT, ffT=ffT, coeffs=[1.0], reference_tasks=refs)


def _zero_like_context(delta_params: Mapping[str, torch.Tensor], curvature: AggregatedCurvature) -> torch.Tensor:
    for tensor in delta_params.values():
        return tensor.sum() * 0.0
    for groups in (curvature.aaT, curvature.ffT):
        for tensors in groups.values():
            if tensors:
                return tensors[0].sum() * 0.0
    return torch.tensor(0.0)


def compute_curvature_penalty(
    delta_params: Mapping[str, torch.Tensor],
    aggregated_curvature: AggregatedCurvature,
) -> PenaltyBreakdown:
    zero = _zero_like_context(delta_params, aggregated_curvature)
    loss_reg_matrix = zero
    loss_reg_ffT = zero
    loss_reg_proj = zero
    loss_reg_cls = zero

    for key, aaT_list in aggregated_curvature.aaT.items():
        if key not in delta_params:
            continue
        delta = delta_params[key]
        is_projection = key == "visual.proj"
        if key == "visual.proj":
            delta_w = delta.T
        else:
            delta_w = delta
            if key.endswith(".weight"):
                bias_key = key[: -len(".weight")] + ".bias"
                if bias_key in delta_params and aaT_list and aaT_list[0].shape[0] == delta_w.shape[1] + 1:
                    delta_w = torch.cat([delta_w, delta_params[bias_key].unsqueeze(1)], dim=1)
        terms = []
        for idx, aaT in enumerate(aaT_list):
            ggT = aggregated_curvature.ggT[key][idx]
            coeff = aggregated_curvature.coeffs[idx] if idx < len(aggregated_curvature.coeffs) else 1.0
            aaT_d = aaT.to(device=delta_w.device, dtype=delta_w.dtype)
            ggT_d = ggT.to(device=delta_w.device, dtype=delta_w.dtype)
            terms.append(float(coeff) * torch.trace(ggT_d @ delta_w @ aaT_d @ delta_w.T))
        if not terms:
            continue
        value = torch.stack(terms).sum()
        if is_projection:
            loss_reg_proj = loss_reg_proj + value
        else:
            loss_reg_matrix = loss_reg_matrix + value

    for key, ffT_list in aggregated_curvature.ffT.items():
        if key not in delta_params:
            continue
        delta = delta_params[key].reshape(1, -1)
        terms = []
        for idx, ffT in enumerate(ffT_list):
            coeff = aggregated_curvature.coeffs[idx] if idx < len(aggregated_curvature.coeffs) else 1.0
            ffT_d = ffT.to(device=delta.device, dtype=delta.dtype)
            terms.append(float(coeff) * torch.trace(delta @ ffT_d @ delta.T))
        if not terms:
            continue
        value = torch.stack(terms).sum()
        if key == "visual.class_embedding":
            loss_reg_cls = loss_reg_cls + value
        else:
            loss_reg_ffT = loss_reg_ffT + value

    return PenaltyBreakdown(
        loss_reg_matrix=loss_reg_matrix,
        loss_reg_ffT=loss_reg_ffT,
        loss_reg_proj=loss_reg_proj,
        loss_reg_cls=loss_reg_cls,
    )


def _visual_param_map(model_or_visual: nn.Module) -> dict[str, torch.Tensor]:
    visual = _visual_module(model_or_visual)
    getter = getattr(model_or_visual, "_current_param_map", None)
    if callable(getter):
        raw = getter()
        if isinstance(raw, Mapping):
            raw_local: dict[str, torch.Tensor] = {}
            prefixed_local: dict[str, torch.Tensor] = {}
            for key, value in raw.items():
                if not isinstance(key, str) or not torch.is_tensor(value):
                    continue
                if key.startswith("clip_model.model.visual."):
                    prefixed_local[key[len("clip_model.model.visual.") :]] = value
                elif key.startswith("model.visual."):
                    prefixed_local[key[len("model.visual.") :]] = value
                elif key.startswith("visual."):
                    prefixed_local[key[len("visual.") :]] = value
                else:
                    raw_local[key] = value
            local_map = prefixed_local or raw_local
            if local_map:
                materialized = _shared_materialized_peft_param_map(visual, raw_current_params=local_map)
                normalized: dict[str, torch.Tensor] = {}
                for name, param in materialized.items():
                    key, value = _normalized_visual_param_key(name, param)
                    normalized[key] = value
                return normalized
    materialized = _shared_materialized_peft_param_map(visual)
    out: dict[str, torch.Tensor] = {}
    for name, param in materialized.items():
        key, value = _normalized_visual_param_key(name, param)
        out[key] = value
    return out


def _base_snapshot(model_or_visual: nn.Module, plan: TrackedCurvaturePlan) -> dict[str, torch.Tensor]:
    params = _visual_param_map(model_or_visual)
    return {key: params[key].detach().clone() for key in plan.param_shapes if key in params}


def _delta_params(model_or_visual: nn.Module, base: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = _visual_param_map(model_or_visual)
    out: dict[str, torch.Tensor] = {}
    for key, base_value in base.items():
        if key in current:
            out[key] = current[key] - base_value.to(device=current[key].device, dtype=current[key].dtype)
    return out


def _load_cache_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    meta = payload.get("metadata", None)
    return dict(meta) if isinstance(meta, dict) else None


class KfacGgnRegularizer:
    name = "kfac_ggn"

    def finalize_model(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del regularization_cfg
        build_cfg = kwargs.get("build_cfg", None)
        if not isinstance(build_cfg, OpenClipBuildConfig):
            return {}
        strategy_cfg = kwargs.get("strategy_cfg", None)
        attn_patch_cfg = self._resolve_attn_patch_cfg(model=model, strategy_cfg=strategy_cfg)
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        model.to(device)
        return {
            "patched_blocks": int(surface["patched_blocks"]),
            "patched_attn_impl": str(surface["attn_patch_cfg"]["attn_impl"]),
        }

    def _resolve_attn_patch_cfg(
        self,
        *,
        model: nn.Module,
        strategy_cfg: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        existing_attn_cfg = getattr(model, "peft_attn_patch_cfg", None)
        attn_patch_cfg = existing_attn_cfg if isinstance(existing_attn_cfg, dict) else None
        if isinstance(strategy_cfg, Mapping) and isinstance(strategy_cfg.get("attention", None), dict):
            attn_patch_cfg = attn_patch_cfg or dict(strategy_cfg["attention"])
        return attn_patch_cfg

    def _expected_cache_metadata(
        self,
        *,
        model: nn.Module,
        task: str,
        build_cfg: OpenClipBuildConfig,
        config: KfacGgnConfig,
        attn_patch_cfg: Mapping[str, Any] | None,
    ) -> tuple[TrackedCurvaturePlan, dict[str, Any]]:
        visual = _visual_module(model)
        plan = select_tracked_parameters(visual)
        meta = _metadata(
            task=task,
            build_cfg=build_cfg,
            config=config,
            plan=plan,
            attn_patch_cfg=attn_patch_cfg,
        )
        return plan, meta

    def _collect_and_store(
        self,
        *,
        model: nn.Module,
        loader: Iterable[Any],
        task: str,
        build_cfg: OpenClipBuildConfig,
        config: KfacGgnConfig,
        attn_patch_cfg: Mapping[str, Any] | None,
        device: torch.device,
    ) -> tuple[Path, TaskCurvatureStats]:
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        plan, meta = self._expected_cache_metadata(
            model=model,
            task=task,
            build_cfg=build_cfg,
            config=config,
            attn_patch_cfg=surface["attn_patch_cfg"],
        )
        stats = collect_curvature(
            model,
            loader,
            tracked_params=plan,
            config=config,
            device=device,
            progress_label=task,
        )
        stats.metadata = meta
        path = task_cache_path(cache_dir=config.cache_dir, build_cfg=build_cfg, task=task)
        save_task_curvature(path, stats)
        print(
            _format_cache_completed(
                regularizer="kfac_ggn",
                task=task,
                stage="curvature",
                path=path,
            )
        )
        return path, stats

    def _ensure_cache(
        self,
        *,
        model: nn.Module,
        loader: Iterable[Any],
        task: str,
        build_cfg: OpenClipBuildConfig,
        config: KfacGgnConfig,
        attn_patch_cfg: Mapping[str, Any] | None,
        device: torch.device,
    ) -> tuple[Path, bool]:
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        _, expected = self._expected_cache_metadata(
            model=model,
            task=task,
            build_cfg=build_cfg,
            config=config,
            attn_patch_cfg=surface["attn_patch_cfg"],
        )
        path = task_cache_path(cache_dir=config.cache_dir, build_cfg=build_cfg, task=task)
        if path.exists() and not config.force_recompute:
            existing = _load_cache_metadata(path)
            if metadata_compatible(existing, expected):
                print(
                    _format_cache_status(
                        regularizer="kfac_ggn",
                        task=task,
                        stage="curvature",
                        path=path,
                        cached=True,
                    )
                )
                return path, False
        print(
            _format_cache_status(
                regularizer="kfac_ggn",
                task=task,
                stage="curvature",
                path=path,
                cached=False,
            )
        )
        self._collect_and_store(
            model=model,
            loader=loader,
            task=task,
            build_cfg=build_cfg,
            config=config,
            attn_patch_cfg=surface["attn_patch_cfg"],
            device=device,
        )
        return path, True

    def prepare(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[PreparedKfacGgn, dict[str, int]]:
        config = _as_config(regularization_cfg)
        task = str(kwargs.get("task", "")).strip()
        build_cfg = kwargs.get("build_cfg", None)
        if not isinstance(build_cfg, OpenClipBuildConfig):
            raise ValueError("kfac_ggn.prepare requires build_cfg from train_vision.")
        loaders = kwargs.get("loaders", None)
        train_loader = getattr(loaders, "train", None)
        if train_loader is None:
            raise ValueError("kfac_ggn.prepare requires loaders.train from train_vision.")
        run_logger = kwargs.get("run_logger", None)
        strategy_cfg = kwargs.get("strategy_cfg", None)
        attn_patch_cfg = self._resolve_attn_patch_cfg(model=model, strategy_cfg=strategy_cfg)
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        model.to(device)
        plan = select_tracked_parameters(model)
        base = _base_snapshot(model, plan)

        selected_tasks = list(
            resolve_reference_tasks_from_kwargs(
                regularization_cfg=regularization_cfg,
                kwargs=kwargs,
                task=task,
                require_reference=True,
            )
        )
        required_tasks = list(dict.fromkeys(selected_tasks))
        batch_size = int(kwargs.get("batch_size", getattr(train_loader, "batch_size", 128) or 128))
        num_workers = int(kwargs.get("num_workers", getattr(train_loader, "num_workers", 0)))
        val_fraction = float(kwargs.get("val_fraction", 0.1))
        seed = int(kwargs.get("seed", 42))

        stats_by_task: dict[str, TaskCurvatureStats] = {}
        for cache_task in required_tasks:
            if cache_task == task:
                cache_model = model
                cache_loader = train_loader
            else:
                ctx = build_vision_regularizer_task_context(
                    task=cache_task,
                    build_cfg=build_cfg,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    val_fraction=val_fraction,
                    seed=seed,
                )
                cache_model = ctx.model
                cache_loader = ctx.loader
            path, recomputed = self._ensure_cache(
                model=cache_model,
                loader=cache_loader,
                task=cache_task,
                build_cfg=build_cfg,
                config=config,
                attn_patch_cfg=surface["attn_patch_cfg"],
                device=device,
            )
            if run_logger is not None:
                run_logger.log_event(
                    "kfac_ggn_cache",
                    metrics={},
                    context={"task": cache_task, "path": str(path), "recomputed": bool(recomputed)},
                )

        for ref_task in selected_tasks:
            if ref_task in stats_by_task:
                continue
            path = task_cache_path(cache_dir=config.cache_dir, build_cfg=build_cfg, task=ref_task)
            if not path.exists():
                continue
            stats_by_task[ref_task] = load_task_curvature(
                path,
                device=device,
                precision=config.precision,
            )
        aggregated = aggregate_curvature(
            stats_by_task,
            current_task_id=task,
            task_order=selected_tasks,
        )
        ignored_trainable = len(plan.ignored_trainable)
        prepared = PreparedKfacGgn(
            config=config,
            plan=plan,
            base=base,
            aggregated=aggregated,
            ignored_trainable=int(ignored_trainable),
        )
        info = {
            "kfac_reference_tasks": len(aggregated.reference_tasks),
            "kfac_matrix_blocks": len(aggregated.aaT),
            "kfac_full_blocks": len(aggregated.ffT),
            "kfac_ignored_trainable": int(ignored_trainable),
        }
        return prepared, info

    def apply(
        self,
        prepared: PreparedKfacGgn,
        *,
        model: nn.Module,
        step: int,
        batch_index: int,
        **kwargs,
    ) -> torch.Tensor:
        del batch_index, kwargs
        if (
            not prepared.aggregated.reference_tasks
            or prepared.config.reg_lambda == 0.0
            or (int(step) % prepared.config.cadence) != 0
        ):
            return next(model.parameters()).sum() * 0.0
        deltas = _delta_params(model, prepared.base)
        breakdown = compute_curvature_penalty(deltas, prepared.aggregated)
        loss = (
            prepared.config.reg_lambda * breakdown.loss_reg_matrix
            + prepared.config.reg_lambda * prepared.config.full_block_scaler * breakdown.loss_reg_ffT
            + prepared.config.reg_lambda * prepared.config.projection_scaler * breakdown.loss_reg_proj
            + prepared.config.reg_lambda * prepared.config.full_block_scaler * breakdown.loss_reg_cls
        )
        model._kfac_ggn_last_breakdown = {  # type: ignore[attr-defined]
            "matrix": float(breakdown.loss_reg_matrix.detach().cpu()),
            "ffT": float(breakdown.loss_reg_ffT.detach().cpu()),
            "projection": float(breakdown.loss_reg_proj.detach().cpu()),
            "class_embedding": float(breakdown.loss_reg_cls.detach().cpu()),
        }
        return loss


register(KfacGgnRegularizer())
