from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from merge_and_rebase.finetune._vision_runtime import ImageEncoder
from merge_and_rebase.finetune._vision_scaled_forward import run_scaled_image_encoder
from merge_and_rebase.finetune.regularizers._distill_capture import (
    CaptureStore,
    LocationRuntime,
    resolve_endpoint_tensor,
)
from merge_and_rebase.finetune.regularizers._distill_config import AlongPathConfig, normalize_loss_cfg
from merge_and_rebase.finetune.regularizers.base import BatchOverride, CheckpointArtifact, OptimizerBundle


def _clear_breakdown_attrs(module: torch.nn.Module) -> None:
    for attr_name in ("_distillation_last_breakdown", "_ekfac_ggn_last_breakdown", "_kfac_ggn_last_breakdown"):
        if hasattr(module, attr_name):
            delattr(module, attr_name)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            return None
    return None


def _record_curvature_breakdown(
    breakdown: dict[str, float],
    raw_breakdown: Mapping[str, Any] | None,
    *,
    suffix: str,
) -> None:
    if not isinstance(raw_breakdown, Mapping):
        return
    mapping = {
        "matrix": f"loss_penalty_{suffix}",
        "ffT": f"loss_reg_ffT_{suffix}",
        "projection": f"loss_ft_proj_{suffix}",
        "class_embedding": f"loss_reg_cls_emb_{suffix}",
    }
    for source_key, metric_name in mapping.items():
        value = _to_float(raw_breakdown.get(source_key))
        if value is not None:
            breakdown[metric_name] = value


def compute_distillation_loss(student: torch.Tensor, teacher: torch.Tensor, loss_cfg: Mapping[str, Any]) -> torch.Tensor:
    name = str(loss_cfg.get("name", "kl_div")).strip().lower()
    if name in {"kl_div", "cross_entropy"}:
        temperature = float(loss_cfg.get("temperature", 1.0))
        student_logits = student / temperature
        teacher_logits = teacher / temperature
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        if name == "kl_div":
            return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
        return (-(teacher_probs * student_log_probs).sum(dim=-1)).mean() * (temperature**2)
    if name in {"mse", "mse_sum_features_mean_batch"}:
        if student.ndim < 2 or teacher.ndim < 2:
            raise ValueError("mse requires tensors with an explicit batch dimension.")
        return (student - teacher).pow(2).reshape(student.shape[0], -1).sum(dim=1).mean()
    if name == "l1":
        return F.l1_loss(student, teacher)
    if name == "cosine":
        student_flat = student.reshape(student.shape[0], -1)
        teacher_flat = teacher.reshape(teacher.shape[0], -1)
        return 1.0 - F.cosine_similarity(student_flat, teacher_flat, dim=-1).mean()
    raise ValueError(f"Unknown distillation loss '{name}'.")


def compute_supervised_loss(logits: torch.Tensor, targets: torch.Tensor, loss_cfg: Mapping[str, Any]) -> torch.Tensor:
    name = str(loss_cfg.get("name", "cross_entropy")).strip().lower()
    if name == "cross_entropy":
        return F.cross_entropy(logits, targets)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(dtype=logits.dtype, device=logits.device)
    if name == "mse":
        return F.mse_loss(torch.softmax(logits, dim=-1), one_hot)
    if name == "l1":
        return F.l1_loss(torch.softmax(logits, dim=-1), one_hot)
    if name == "cosine":
        return 1.0 - F.cosine_similarity(torch.softmax(logits, dim=-1), one_hot, dim=-1).mean()
    raise ValueError("teacher.supervised.loss.name must be one of: cross_entropy, mse, l1, cosine")


@dataclass
class PreparedDistillation:
    teacher: Any
    locations: tuple[LocationRuntime, ...]
    student_store: CaptureStore
    teacher_store: CaptureStore
    optimizer_bundles: tuple[OptimizerBundle, ...]
    adapter_modules: torch.nn.ModuleList = field(default_factory=torch.nn.ModuleList)
    along_path: AlongPathConfig = field(default_factory=AlongPathConfig)
    student_base: dict[str, torch.Tensor] = field(default_factory=dict)
    total_steps: int = 0
    current_alpha: float | None = None
    current_virtual_batch_id: int | None = None
    info: dict[str, int | float | str] = field(default_factory=dict)

    def close(self) -> None:
        self.student_store.close()
        self.teacher_store.close()
        closer = getattr(self.teacher.regularizer_prepared, "close", None)
        if callable(closer):
            closer()

    def checkpoint_payload(self, *, kind: str) -> dict[str, Any]:
        del kind
        return {}

    def checkpoint_artifacts(
        self,
        *,
        kind: str,
        epoch_i: int,
        val_acc_i: float,
        test_acc_i: float,
        zero_shot_metrics: Mapping[str, float] | None = None,
    ) -> tuple[CheckpointArtifact, ...]:
        builder = getattr(self.teacher, "checkpoint_artifacts", None)
        if not callable(builder):
            return ()
        return tuple(
            builder(
                kind=kind,
                epoch_i=epoch_i,
                val_acc_i=val_acc_i,
                test_acc_i=test_acc_i,
                zero_shot_metrics=zero_shot_metrics,
            )
        )


def sample_alpha(prepared: PreparedDistillation, *, step: int) -> float:
    cfg = prepared.along_path
    if not cfg.enabled:
        return 1.0
    if cfg.sampling == "uniform":
        return random.uniform(cfg.alpha_start, cfg.alpha_end)
    total_steps = max(1, int(prepared.total_steps))
    progress = min(float(step) / float(total_steps), 1.0)
    adjusted_progress = math.pow(progress, float(cfg.temperature))
    current_low = 1.0 + (cfg.alpha_start - 1.0) * adjusted_progress
    current_high = 1.0 + (cfg.alpha_end - 1.0) * adjusted_progress
    return random.uniform(current_low, current_high)


def prepare_batch_override(
    prepared: PreparedDistillation,
    *,
    model: ImageEncoder,
    step: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    virtual_batch_start: bool,
) -> BatchOverride | None:
    if not prepared.along_path.enabled:
        return None
    virtual_batch_id = int(step)
    if virtual_batch_start or prepared.current_virtual_batch_id != virtual_batch_id or prepared.current_alpha is None:
        prepared.current_alpha = float(sample_alpha(prepared, step=int(step)))
        prepared.current_virtual_batch_id = virtual_batch_id
        prepared.info["distillation_along_path_last_alpha"] = float(prepared.current_alpha)
    alpha = float(prepared.current_alpha)

    prepared.student_store.clear()
    prepared.teacher_store.clear()
    student_outputs = run_scaled_image_encoder(
        model=model,
        images=inputs,
        alpha=alpha,
        base_params=prepared.student_base,
    )
    primary_loss = F.cross_entropy(student_outputs, targets)

    teacher_outputs: torch.Tensor | None = None
    teacher_model = prepared.teacher.model
    teacher_base = prepared.teacher.along_path_base
    if prepared.teacher.along_path_enabled and teacher_base is not None:
        if prepared.teacher.mode == "online":
            teacher_model.train()
            teacher_outputs = run_scaled_image_encoder(
                model=teacher_model,
                images=inputs,
                alpha=alpha,
                base_params=teacher_base,
            )
        else:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = run_scaled_image_encoder(
                    model=teacher_model,
                    images=inputs,
                    alpha=alpha,
                    base_params=teacher_base,
                )
    else:
        if prepared.teacher.mode == "online":
            teacher_model.train()
            teacher_outputs = teacher_model(inputs)
        else:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

    context = {
        "along_path_enabled": True,
        "alpha": alpha,
        "virtual_batch_id": virtual_batch_id,
        "teacher_outputs": teacher_outputs,
        "teacher_along_path_enabled": bool(prepared.teacher.along_path_enabled),
    }
    return BatchOverride(outputs=student_outputs, primary_loss=primary_loss, context=context)


def apply_prepared_distillation(
    prepared: PreparedDistillation,
    *,
    model: ImageEncoder,
    step: int,
    batch_index: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    batch_context: Any = None,
    forced_alpha: float | None = None,
) -> torch.Tensor:
    _clear_breakdown_attrs(model)
    alpha: float | None = None
    teacher_outputs_from_context: torch.Tensor | None = None
    if isinstance(batch_context, Mapping):
        alpha_raw = batch_context.get("alpha", None)
        if alpha_raw is not None:
            alpha = float(alpha_raw)
        teacher_outputs_raw = batch_context.get("teacher_outputs", None)
        if isinstance(teacher_outputs_raw, torch.Tensor):
            teacher_outputs_from_context = teacher_outputs_raw
    if forced_alpha is not None:
        alpha = float(forced_alpha)
    along_path_active = bool(prepared.along_path.enabled and alpha is not None)

    teacher_model = prepared.teacher.model
    if teacher_outputs_from_context is None:
        prepared.teacher_store.clear()
    batch_size = int(inputs.shape[0])
    if teacher_outputs_from_context is not None:
        teacher_outputs = teacher_outputs_from_context
    elif along_path_active and prepared.teacher.along_path_enabled and prepared.teacher.along_path_base is not None:
        if prepared.teacher.mode == "online":
            teacher_model.train()
            teacher_outputs = run_scaled_image_encoder(
                model=teacher_model,
                images=inputs,
                alpha=float(alpha),
                base_params=prepared.teacher.along_path_base,
            )
        else:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = run_scaled_image_encoder(
                    model=teacher_model,
                    images=inputs,
                    alpha=float(alpha),
                    base_params=prepared.teacher.along_path_base,
                )
    else:
        if prepared.teacher.mode == "online":
            teacher_model.train()
            teacher_outputs = teacher_model(inputs)
        else:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

    total_loss = outputs.new_zeros(())
    breakdown: dict[str, float] = {}
    if alpha is not None:
        breakdown["sampled_alpha"] = float(alpha)
    teacher_total_value = 0.0
    teacher_total_active = False
    for index, location in enumerate(prepared.locations):
        student_tensor = resolve_endpoint_tensor(
            endpoint=location.student,
            model=model,
            store=prepared.student_store,
            outputs=outputs,
            batch_size=batch_size,
        )
        teacher_tensor = resolve_endpoint_tensor(
            endpoint=location.teacher,
            model=teacher_model,
            store=prepared.teacher_store,
            outputs=teacher_outputs,
            batch_size=batch_size,
        )
        if prepared.teacher.mode == "online" and prepared.teacher.stop_gradient:
            teacher_tensor = teacher_tensor.detach()
        component_loss = float(location.weight) * compute_distillation_loss(student_tensor, teacher_tensor, location.loss)
        total_loss = total_loss + component_loss
        component_value = _to_float(component_loss)
        if component_value is not None:
            location_name = str(location.name).strip() or f"location_{index}"
            breakdown.setdefault("loss_distill", 0.0)
            breakdown["loss_distill"] += component_value
            breakdown[f"loss_distill_{index}_{location_name}"] = component_value

    if prepared.teacher.mode == "online":
        supervised_cfg = dict(prepared.teacher.supervised_cfg or {})
        if bool(supervised_cfg.get("enabled", True)):
            supervised_loss_cfg = normalize_loss_cfg(supervised_cfg.get("loss", {"name": "cross_entropy"}))
            teacher_supervised_loss = float(supervised_cfg.get("weight", 1.0)) * compute_supervised_loss(
                teacher_outputs,
                targets,
                supervised_loss_cfg,
            )
            total_loss = total_loss + teacher_supervised_loss
            teacher_supervised_value = _to_float(teacher_supervised_loss)
            if teacher_supervised_value is not None:
                teacher_total_active = True
                teacher_total_value += teacher_supervised_value
                breakdown["loss_teacher_task"] = teacher_supervised_value
                breakdown["loss_teacher_supervised"] = teacher_supervised_value
        if prepared.teacher.regularizer_impl is not None and prepared.teacher.regularizer_prepared is not None:
            nested_batch_context: dict[str, Any] | None = None
            if along_path_active:
                nested_batch_context = {
                    "along_path_enabled": True,
                    "alpha": float(alpha),
                    "virtual_batch_id": (
                        batch_context.get("virtual_batch_id", None) if isinstance(batch_context, Mapping) else None
                    ),
                }
            _clear_breakdown_attrs(teacher_model)
            teacher_reg_loss = prepared.teacher.regularizer_impl.apply(
                prepared.teacher.regularizer_prepared,
                model=teacher_model,
                step=int(step),
                batch_index=int(batch_index),
                inputs=inputs,
                targets=targets,
                outputs=teacher_outputs,
                forced_alpha=float(alpha) if along_path_active else None,
                batch_context=nested_batch_context,
            )
            total_loss = total_loss + teacher_reg_loss
            teacher_reg_value = _to_float(teacher_reg_loss)
            if teacher_reg_value is not None:
                teacher_total_active = True
                teacher_total_value += teacher_reg_value
                breakdown["loss_reg_teacher"] = teacher_reg_value
            _record_curvature_breakdown(
                breakdown,
                getattr(teacher_model, "_ekfac_ggn_last_breakdown", None),
                suffix="teacher",
            )
            _record_curvature_breakdown(
                breakdown,
                getattr(teacher_model, "_kfac_ggn_last_breakdown", None),
                suffix="teacher",
            )
        if teacher_total_active:
            breakdown["loss_teacher_total"] = float(teacher_total_value)

    prepared.student_store.clear()
    prepared.teacher_store.clear()
    model._distillation_last_breakdown = breakdown  # type: ignore[attr-defined]
    return total_loss
