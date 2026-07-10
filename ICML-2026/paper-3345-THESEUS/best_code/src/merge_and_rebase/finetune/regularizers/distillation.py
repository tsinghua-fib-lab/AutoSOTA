from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from merge_and_rebase.finetune._vision_runtime import (
    ImageEncoder,
    build_image_encoder,
    initialize_task_text_features,
    load_model_init_checkpoint,
    snapshot_parameter_map,
)
from merge_and_rebase.finetune.regularizers._distill_capture import (
    CaptureStore as _CaptureStore,
)
from merge_and_rebase.finetune.regularizers._distill_capture import (
    EndpointRuntime,
    LocationRuntime,
)
from merge_and_rebase.finetune.regularizers._distill_capture import (
    infer_projection_modules as _infer_projection_modules_shared,
)
from merge_and_rebase.finetune.regularizers._distill_capture import (
    register_endpoint_hook as _register_endpoint_hook_shared,
)
from merge_and_rebase.finetune.regularizers._distill_config import (
    adapter_train_cfg as _adapter_train_cfg_shared,
)
from merge_and_rebase.finetune.regularizers._distill_config import (
    as_mapping as _as_mapping,
)
from merge_and_rebase.finetune.regularizers._distill_config import (
    normalize_along_path_cfg as _normalize_along_path_cfg,
)
from merge_and_rebase.finetune.regularizers._distill_config import (
    normalize_locations as _normalize_locations_shared,
)
from merge_and_rebase.finetune.regularizers._distill_runtime import (
    PreparedDistillation,
)
from merge_and_rebase.finetune.regularizers._distill_runtime import (
    apply_prepared_distillation as _apply_prepared_distillation,
)
from merge_and_rebase.finetune.regularizers._distill_runtime import (
    prepare_batch_override as _prepare_batch_override,
)
from merge_and_rebase.finetune.regularizers._distill_teacher import (
    build_optimizer as _build_optimizer,
)
from merge_and_rebase.finetune.regularizers._distill_teacher import (
    prepare_teacher_runtime as _prepare_teacher_runtime,
)
from merge_and_rebase.finetune.schedulers import build_lr_scheduler
from merge_and_rebase.finetune.regularizers.base import BatchOverride, OptimizerBundle
from merge_and_rebase.finetune.regularizers.registry import register
from merge_and_rebase.finetune.strategies.registry import get_strategy
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig


@dataclass(frozen=True)
class DistillationRegularizer:
    name: str = "distillation"

    def prepare(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[PreparedDistillation, dict[str, int | float | str]]:
        if not isinstance(model, ImageEncoder):
            raise TypeError("distillation regularizer expects an ImageEncoder model.")
        config = _as_mapping(regularization_cfg, field_name="regularization")
        along_path = _normalize_along_path_cfg(config.get("along_path"))
        shared_weight = float(config.get("shared_weight", 1.0))
        run_logger = kwargs.get("run_logger", None)
        locations = [
            LocationRuntime(
                name=cfg.name,
                student=EndpointRuntime(config=cfg.student, capture_id=f"student:{idx}" if cfg.student.source == "module" else None),
                teacher=EndpointRuntime(config=cfg.teacher, capture_id=f"teacher:{idx}" if cfg.teacher.source == "module" else None),
                loss=cfg.loss,
                weight=cfg.weight,
            )
            for idx, cfg in enumerate(_normalize_locations_shared(config.get("locations", []), shared_weight=shared_weight))
        ]

        teacher_cfg = _as_mapping(config.get("teacher"), field_name="regularization.teacher")
        student_base = snapshot_parameter_map(model)
        task_name = str(kwargs.get("task", ""))

        build_cfg = kwargs.get("build_cfg", None)
        if not isinstance(build_cfg, OpenClipBuildConfig):
            raise ValueError("distillation.prepare requires build_cfg from train_vision.")
        loaders = kwargs.get("loaders", None)
        classnames = list(getattr(loaders, "classnames", []) or [])
        if not classnames:
            raise ValueError("distillation.prepare requires loaders.classnames from train_vision.")

        student_defaults = {
            "lr": float(kwargs.get("train_lr", 1e-4)),
            "dense_lr": float(kwargs.get("train_dense_lr", kwargs.get("train_lr", 1e-4))),
            "weight_decay": float(kwargs.get("train_weight_decay", 0.0)),
            "optimizer_name": str(kwargs.get("train_optimizer_name", "adamw")),
            "scheduler_name": str(kwargs.get("train_scheduler_name", "cosine")),
            "warmup_length": int(kwargs.get("warmup_length", 0)),
            "grad_clip_norm": float(kwargs.get("train_grad_clip_norm", -1.0)),
        }

        teacher_runtime, teacher_bundles = _prepare_teacher_runtime(
            student_model=model,
            student_base=student_base,
            teacher_cfg=teacher_cfg,
            build_cfg=build_cfg,
            classnames=classnames,
            device=device,
            task_name=task_name,
            run_logger=run_logger,
            student_defaults=student_defaults,
            along_path_enabled=bool(along_path.enabled),
            kwargs=kwargs,
            build_image_encoder_fn=build_image_encoder,
            initialize_task_text_features_fn=initialize_task_text_features,
            load_model_init_checkpoint_fn=load_model_init_checkpoint,
            get_strategy_fn=get_strategy,
        )
        student_store = _CaptureStore()
        teacher_store = _CaptureStore()
        for location in locations:
            _register_endpoint_hook_shared(student_store, model, str(location.student.capture_id), location.student.config)
            _register_endpoint_hook_shared(
                teacher_store,
                teacher_runtime.model,
                str(location.teacher.capture_id),
                location.teacher.config,
            )

        train_loader = getattr(loaders, "train", None)
        if train_loader is None:
            raise ValueError("distillation.prepare requires loaders.train from train_vision.")
        try:
            sample_inputs, _sample_targets = next(iter(train_loader))
        except StopIteration as exc:
            raise ValueError("distillation.prepare requires a non-empty train loader.") from exc

        adapter_modules, adapter_info = _infer_projection_modules_shared(
            student_model=model,
            teacher_model=teacher_runtime.model,
            student_store=student_store,
            teacher_store=teacher_store,
            locations=locations,
            sample_inputs=sample_inputs,
            device=device,
        )
        optimizer_bundles: list[OptimizerBundle] = list(teacher_bundles)
        if adapter_info["adapter_params"] > 0:
            adapter_cfg = _adapter_train_cfg_shared(config.get("adapter_train"), defaults=student_defaults)
            adapter_opt = _build_optimizer(
                adapter_modules.parameters(),
                optimizer_name=str(adapter_cfg["optimizer_name"]),
                lr=float(adapter_cfg["lr"]),
                weight_decay=float(adapter_cfg["weight_decay"]),
            )
            adapter_sched = build_lr_scheduler(
                adapter_opt,
                name=str(adapter_cfg["scheduler_name"]),
                base_lrs=float(adapter_cfg["lr"]),
                warmup_length=int(adapter_cfg["warmup_length"]),
                steps=int(kwargs.get("total_steps", 0)),
            )
            optimizer_bundles.append(
                OptimizerBundle(
                    name="distillation_adapters",
                    optimizer=adapter_opt,
                    scheduler=adapter_sched,
                    grad_clip_norm=float(adapter_cfg["grad_clip_norm"]),
                )
            )

        info: dict[str, int | float | str] = {
            "distillation_locations": int(len(locations)),
            "distillation_teacher_online": int(teacher_runtime.mode == "online"),
            "distillation_aux_optimizers": int(len(optimizer_bundles)),
            "distillation_adapter_params": int(adapter_info["adapter_params"]),
            "distillation_along_path_enabled": int(along_path.enabled),
            "distillation_along_path_sampling": along_path.sampling,
            "distillation_along_path_alpha_start": float(along_path.alpha_start),
            "distillation_along_path_alpha_end": float(along_path.alpha_end),
            "distillation_along_path_teacher_enabled": int(teacher_runtime.along_path_enabled),
            "distillation_along_path_teacher_fallback": int(along_path.enabled and not teacher_runtime.along_path_enabled),
            "distillation_along_path_last_alpha": float("nan"),
        }
        prepared = PreparedDistillation(
            teacher=teacher_runtime,
            locations=tuple(locations),
            student_store=student_store,
            teacher_store=teacher_store,
            optimizer_bundles=tuple(optimizer_bundles),
            adapter_modules=adapter_modules,
            along_path=along_path,
            student_base=student_base,
            total_steps=int(kwargs.get("total_steps", 0)),
            info=info,
        )
        return prepared, info

    def prepare_batch(
        self,
        prepared: PreparedDistillation,
        *,
        model: nn.Module,
        step: int,
        batch_index: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        virtual_batch_start: bool,
        **kwargs,
    ) -> BatchOverride | None:
        del batch_index, kwargs
        if not isinstance(model, ImageEncoder):
            raise TypeError("distillation regularizer expects an ImageEncoder model.")
        return _prepare_batch_override(
            prepared,
            model=model,
            step=step,
            inputs=inputs,
            targets=targets,
            virtual_batch_start=virtual_batch_start,
        )

    def apply(
        self,
        prepared: PreparedDistillation,
        *,
        model: nn.Module,
        step: int,
        batch_index: int,
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(model, ImageEncoder):
            raise TypeError("distillation regularizer expects an ImageEncoder model.")
        inputs = kwargs.get("inputs", None)
        targets = kwargs.get("targets", None)
        outputs = kwargs.get("outputs", None)
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError("distillation.apply requires inputs and targets tensors.")
        if not isinstance(outputs, torch.Tensor):
            raise ValueError("distillation.apply requires outputs tensor.")
        return _apply_prepared_distillation(
            prepared,
            model=model,
            step=step,
            batch_index=batch_index,
            inputs=inputs,
            targets=targets,
            outputs=outputs,
            batch_context=kwargs.get("batch_context", None),
            forced_alpha=kwargs.get("forced_alpha", None),
        )


register(DistillationRegularizer())
