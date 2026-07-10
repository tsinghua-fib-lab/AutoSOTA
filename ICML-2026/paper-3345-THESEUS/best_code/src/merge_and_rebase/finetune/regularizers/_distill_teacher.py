from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from merge_and_rebase.finetune._vision_runtime import (
    ImageEncoder,
    build_image_encoder,
    build_vision_model_payload,
    initialize_task_text_features,
    load_model_init_checkpoint,
)
from merge_and_rebase.finetune._vision_scaled_forward import parameter_maps_compatible, snapshot_parameter_map
from merge_and_rebase.finetune.forward_mode import resolve_training_forward_mode
from merge_and_rebase.finetune.reference_tasks import (
    ReferenceTaskResolutionContext,
    apply_reference_tags_to_out_dir,
)
from merge_and_rebase.finetune.regularizers._distill_config import as_mapping, merge_build_cfg, teacher_train_cfg
from merge_and_rebase.finetune.regularizers.base import CheckpointArtifact, OptimizerBundle, finalize_model_for_regularizer
from merge_and_rebase.finetune.regularizers.registry import get_regularizer
from merge_and_rebase.finetune.strategies.registry import get_strategy
from merge_and_rebase.models.forward_modes import bind_training_forward_mode, normalize_forward_mode_params
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig

def build_optimizer(params, *, optimizer_name: str, lr: float, weight_decay: float) -> optim.Optimizer:
    name = str(optimizer_name).strip().lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def emit_along_path_warning(*, run_logger: Any, task: str, message: str) -> None:
    print(message)
    if run_logger is not None:
        run_logger.log_event(
            "distillation_along_path_warning",
            metrics={},
            context={"message": message, "task": task},
        )


@dataclass
class TeacherRuntime:
    model: ImageEncoder
    build_cfg: OpenClipBuildConfig
    mode: str
    stop_gradient: bool
    save_checkpoint: bool
    strategy_name: str
    strategy_cfg: dict[str, Any]
    forward_mode: str
    forward_mode_params: dict[str, Any]
    checkpoint_init_summary: dict[str, Any] | None
    text_features_init_source: str
    supervised_cfg: dict[str, Any]
    regularizer_name: str
    regularizer_impl: Any | None
    regularizer_prepared: Any | None
    task: str
    classnames: tuple[str, ...]
    num_classes: int
    output_dir: str | None = None
    checkpoint_stem: str | None = None
    along_path_enabled: bool = False
    along_path_base: dict[str, torch.Tensor] | None = None

    def checkpoint_artifacts(
        self,
        *,
        kind: str,
        epoch_i: int,
        val_acc_i: float,
        test_acc_i: float,
        zero_shot_metrics: Mapping[str, float] | None = None,
    ) -> tuple[CheckpointArtifact, ...]:
        if not self.save_checkpoint or not self.output_dir or not self.checkpoint_stem:
            return ()
        payload = build_vision_model_payload(
            model=self.model,
            build_cfg=self.build_cfg,
            forward_mode=self.forward_mode,
            forward_mode_params=self.forward_mode_params,
            strategy=self.strategy_name,
            task=self.task,
            classnames=list(self.classnames),
            num_classes=int(self.num_classes),
            checkpoint_init_summary=self.checkpoint_init_summary,
            text_features_init_source=self.text_features_init_source,
            include_weights=True,
        )
        payload["metrics"] = {"val_top1": float(val_acc_i), "test_top1": float(test_acc_i)}
        payload["zero_shot_metrics"] = dict(zero_shot_metrics or {})
        if kind == "best_ep":
            payload["best_epoch"] = int(epoch_i)
        elif kind == "last_ep":
            payload["last_epoch"] = int(epoch_i)
        else:
            raise ValueError("kind must be 'best_ep' or 'last_ep'")

        summary = {
            "task": self.task,
            "strategy": self.strategy_name,
            "forward_mode": self.forward_mode,
            "forward_mode_params": dict(self.forward_mode_params),
            "metrics": payload["metrics"],
            "zero_shot_metrics": payload["zero_shot_metrics"],
            "checkpoint_init_summary": self.checkpoint_init_summary,
            "text_features_init_source": self.text_features_init_source,
        }
        return (
            CheckpointArtifact(
                output_dir=self.output_dir,
                filename=f"{self.checkpoint_stem}_{kind}.pt",
                payload=payload,
                summary_filename=f"{self.checkpoint_stem}_{kind}.json",
                summary=summary,
            ),
        )


def prepare_teacher_runtime(
    *,
    student_model: ImageEncoder,
    student_base: dict[str, torch.Tensor],
    teacher_cfg: Mapping[str, Any],
    build_cfg: OpenClipBuildConfig,
    classnames: list[str],
    device: torch.device,
    task_name: str,
    run_logger: Any,
    student_defaults: Mapping[str, Any],
    along_path_enabled: bool,
    kwargs: Mapping[str, Any],
    build_image_encoder_fn=build_image_encoder,
    initialize_task_text_features_fn=initialize_task_text_features,
    load_model_init_checkpoint_fn=load_model_init_checkpoint,
    get_strategy_fn=get_strategy,
) -> tuple[TeacherRuntime, tuple[OptimizerBundle, ...]]:
    teacher_mode = str(teacher_cfg.get("mode", "frozen")).strip().lower()
    if teacher_mode not in {"frozen", "online"}:
        raise ValueError("regularization.teacher.mode must be 'frozen' or 'online'.")

    teacher_build_cfg = merge_build_cfg(build_cfg, teacher_cfg.get("build"))
    teacher_model = build_image_encoder_fn(build_cfg=teacher_build_cfg, device=device)

    teacher_checkpoint_init_summary = None
    teacher_checkpoint_init_obj = None
    init_cfg = as_mapping(teacher_cfg.get("initialization"), field_name="regularization.teacher.initialization")
    init_checkpoint = init_cfg.get("checkpoint", None)
    if init_checkpoint is not None:
        teacher_checkpoint_init_obj, teacher_checkpoint_init_summary = load_model_init_checkpoint_fn(
            model=teacher_model,
            ckpt_path=str(init_checkpoint),
        )
    teacher_text_features_source = initialize_task_text_features_fn(
        model=teacher_model,
        classnames=classnames,
        build_cfg=teacher_build_cfg,
        device=device,
        ckpt_obj=teacher_checkpoint_init_obj,
        ckpt_path=str(init_checkpoint) if init_checkpoint is not None else None,
        text_features_source=str(init_cfg.get("text_features_source", "zero_shot")),
    )

    teacher_strategy_cfg = as_mapping(teacher_cfg.get("strategy"), field_name="regularization.teacher.strategy")
    teacher_strategy_name = str(teacher_strategy_cfg.get("name", "full")).strip()
    teacher_forward_mode = resolve_training_forward_mode(teacher_strategy_cfg) if teacher_strategy_cfg else "standard"
    teacher_forward_mode_params = normalize_forward_mode_params(
        teacher_forward_mode,
        teacher_strategy_cfg.get("forward_mode_params", None),
    )
    nested_regularizer_cfg = as_mapping(
        teacher_cfg.get("regularization"),
        field_name="regularization.teacher.regularization",
    )
    nested_regularizer_name = str(nested_regularizer_cfg.get("name", "")).strip()
    nested_regularizer_impl = None
    if nested_regularizer_name:
        nested_regularizer_impl = get_regularizer(nested_regularizer_name)
        finalize_model_for_regularizer(
            nested_regularizer_impl,
            model=teacher_model,
            device=device,
            regularization_cfg=nested_regularizer_cfg,
            task=task_name,
            strategy_cfg=teacher_strategy_cfg,
            build_cfg=teacher_build_cfg,
            loaders=kwargs.get("loaders"),
            all_tasks=list(kwargs.get("all_tasks", []) or []),
            reference_tasks=list(kwargs.get("reference_tasks", []) or []),
            reference_resolution_context=kwargs.get("reference_resolution_context", None),
            batch_size=int(kwargs.get("batch_size", getattr(kwargs.get("loaders"), "batch_size", 128) or 128)),
            num_workers=int(kwargs.get("num_workers", 0)),
            val_fraction=float(kwargs.get("val_fraction", 0.1)),
            seed=int(kwargs.get("seed", 42)),
            run_logger=run_logger,
            total_steps=int(kwargs.get("total_steps", 0)),
            warmup_length=int(kwargs.get("warmup_length", 0)),
            train_lr=float(kwargs.get("train_lr", student_defaults.get("lr", 1e-4))),
            train_dense_lr=float(kwargs.get("train_dense_lr", student_defaults.get("dense_lr", student_defaults.get("lr", 1e-4)))),
            train_weight_decay=float(kwargs.get("train_weight_decay", student_defaults.get("weight_decay", 0.0))),
            train_optimizer_name=str(kwargs.get("train_optimizer_name", student_defaults.get("optimizer_name", "adamw"))),
            train_grad_clip_norm=float(kwargs.get("train_grad_clip_norm", student_defaults.get("grad_clip_norm", -1.0))),
            accumulate_grad_batches=int(kwargs.get("accumulate_grad_batches", 1)),
            student_forward_mode=teacher_forward_mode,
            student_forward_mode_params=dict(teacher_forward_mode_params),
        )

    teacher_base = snapshot_parameter_map(teacher_model)
    teacher_along_path_enabled = bool(along_path_enabled)
    optimizer_bundles: list[OptimizerBundle] = []

    if teacher_mode == "online":
        if not teacher_strategy_cfg:
            raise ValueError("regularization.teacher.strategy is required for online teachers.")
        teacher_train = teacher_train_cfg(teacher_cfg.get("train"), defaults=student_defaults)
        total_steps = int(kwargs.get("total_steps", 0))
        teacher_opt, teacher_sched, _teacher_info = get_strategy_fn(teacher_strategy_name).configure(
            model=teacher_model,
            lr=float(teacher_train["lr"]),
            dense_lr=float(teacher_train["dense_lr"]),
            weight_decay=float(teacher_train["weight_decay"]),
            warmup_length=int(teacher_train["warmup_length"]),
            scheduler_name=str(teacher_train["scheduler_name"]),
            optimizer=str(teacher_train["optimizer_name"]),
            steps=total_steps,
            device=device,
            peft_cfg=teacher_strategy_cfg.get("peft", None),
            strategy_cfg=teacher_strategy_cfg,
            task=task_name,
        )
        teacher_base_sd = {k: v.detach().clone() for k, v in teacher_model.clip_model.model.state_dict().items()}
        bind_training_forward_mode(
            model=teacher_model,
            forward_mode=teacher_forward_mode,
            base_sd=teacher_base_sd,
            strict_load=True,
            params=teacher_forward_mode_params,
        )
        optimizer_bundles.append(
            OptimizerBundle(
                name="distillation_teacher",
                optimizer=teacher_opt,
                scheduler=teacher_sched,
                grad_clip_norm=float(teacher_train["grad_clip_norm"]),
            )
        )
        teacher_base = snapshot_parameter_map(teacher_model)
    else:
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        if along_path_enabled:
            if parameter_maps_compatible(student_base, teacher_base):
                teacher_base = {key: value.detach().clone() for key, value in student_base.items()}
            else:
                teacher_along_path_enabled = False
                emit_along_path_warning(
                    run_logger=run_logger,
                    task=task_name,
                    message=(
                        "Frozen teacher along-path distillation is disabled because student and teacher "
                        "do not share compatible parameter names/shapes."
                    ),
                )

    nested_prepared = None
    if nested_regularizer_name:
        if teacher_mode != "online":
            raise ValueError("regularization.teacher.regularization requires teacher.mode='online'.")
        teacher_train = teacher_train_cfg(teacher_cfg.get("train"), defaults=student_defaults)
        nested_prepared, _nested_info = nested_regularizer_impl.prepare(
            model=teacher_model,
            device=device,
            regularization_cfg=nested_regularizer_cfg,
            task=task_name,
            strategy_cfg=teacher_strategy_cfg,
            build_cfg=teacher_build_cfg,
            loaders=kwargs.get("loaders"),
            all_tasks=list(kwargs.get("all_tasks", []) or []),
            reference_tasks=list(kwargs.get("reference_tasks", []) or []),
            reference_resolution_context=kwargs.get("reference_resolution_context", None),
            batch_size=int(kwargs.get("batch_size", getattr(kwargs.get("loaders"), "batch_size", 128) or 128)),
            num_workers=int(kwargs.get("num_workers", 0)),
            val_fraction=float(kwargs.get("val_fraction", 0.1)),
            seed=int(kwargs.get("seed", 42)),
            run_logger=run_logger,
            total_steps=int(kwargs.get("total_steps", 0)),
            warmup_length=int(teacher_train["warmup_length"]),
            train_lr=float(teacher_train["lr"]),
            train_dense_lr=float(teacher_train["dense_lr"]),
            train_weight_decay=float(teacher_train["weight_decay"]),
            train_optimizer_name=str(teacher_train["optimizer_name"]),
            train_grad_clip_norm=float(teacher_train["grad_clip_norm"]),
            accumulate_grad_batches=int(kwargs.get("accumulate_grad_batches", 1)),
            student_forward_mode=teacher_forward_mode,
            student_forward_mode_params=dict(teacher_forward_mode_params),
        )
        optimizer_bundles.extend(tuple(getattr(nested_prepared, "optimizer_bundles", ()) or ()))

    teacher_runtime = TeacherRuntime(
        model=teacher_model,
        build_cfg=teacher_build_cfg,
        mode=teacher_mode,
        stop_gradient=bool(teacher_cfg.get("stop_gradient", True)),
        save_checkpoint=bool(teacher_cfg.get("save_checkpoint", False)),
        strategy_name=teacher_strategy_name,
        strategy_cfg=teacher_strategy_cfg,
        forward_mode=teacher_forward_mode,
        forward_mode_params=teacher_forward_mode_params,
        checkpoint_init_summary=teacher_checkpoint_init_summary,
        text_features_init_source=teacher_text_features_source,
        supervised_cfg=as_mapping(teacher_cfg.get("supervised"), field_name="regularization.teacher.supervised"),
        regularizer_name=nested_regularizer_name,
        regularizer_impl=nested_regularizer_impl,
        regularizer_prepared=nested_prepared,
        task=task_name,
        classnames=tuple(classnames),
        num_classes=int(len(classnames)),
        output_dir=(
            str(
                Path(
                    apply_reference_tags_to_out_dir(
                        out_dir=str(teacher_cfg.get("output_dir")).strip(),
                        regularization_cfg=nested_regularizer_cfg,
                        context=kwargs["reference_resolution_context"],
                    )
                    if isinstance(kwargs.get("reference_resolution_context", None), ReferenceTaskResolutionContext)
                    else str(teacher_cfg.get("output_dir")).strip()
                )
                / teacher_build_cfg.model_name
                / teacher_build_cfg.pretrained
                / task_name
            )
            if teacher_cfg.get("output_dir", None) is not None
            else None
        ),
        checkpoint_stem=(
            teacher_strategy_name if teacher_forward_mode == "standard" else f"{teacher_strategy_name}__{teacher_forward_mode}"
        ) + (f"__{nested_regularizer_name}" if nested_regularizer_name else ""),
        along_path_enabled=teacher_along_path_enabled,
        along_path_base=teacher_base,
    )
    return teacher_runtime, tuple(optimizer_bundles)
