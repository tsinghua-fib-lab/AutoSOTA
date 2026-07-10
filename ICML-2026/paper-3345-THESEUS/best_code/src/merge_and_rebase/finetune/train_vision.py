# src/merge_and_rebase/finetune/train_vision.py
from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import yaml  # type: ignore
from tqdm import tqdm

from merge_and_rebase.cli_args import add_logging_args, build_logging_overrides
from merge_and_rebase.data.templates import get_templates
from merge_and_rebase.io.peft_helpers import state_dict_looks_patched_attn
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from merge_and_rebase.utils.helpers import parse_csv

from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.datasets.vision8_14_20 import SUITES, VISION_SUPPORTED_TASKS, _vision_spec
from ..models.forward_modes import bind_training_forward_mode, normalize_forward_mode_params
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..models.patch_openclip_attention import set_linear_attention_ramp_step, split_openclip_vit_attn
from ._vision_runtime import (
    ImageEncoder,
    initialize_task_text_features,
    load_model_init_checkpoint,
    materialized_model_state_dict,
)
from .forward_mode import resolve_training_forward_mode
from .reference_tasks import (
    ReferenceTaskResolutionContext,
)
from .reference_tasks import (
    apply_reference_tags_to_out_dir as _apply_reference_tags_to_out_dir,
)
from .reference_tasks import (
    build_reference_task_resolution_context as _build_reference_task_resolution_context,
)
from .reference_tasks import (
    parse_reference_datasets as _parse_reference_datasets_shared,
)
from .reference_tasks import (
    resolve_reference_tasks as _resolve_reference_tasks_shared,
)
from .reference_tasks import (
    validate_vision_tasks as _validate_vision_tasks_shared,
)
from .regularizers.base import (
    BatchOverride,
    checkpoint_payload_from_prepared,
    close_prepared_regularizer,
    collect_checkpoint_artifacts,
    finalize_model_for_regularizer,
    iter_optimizer_bundles,
    prepare_batch_override,
)
from .regularizers.registry import get_regularizer, list_regularizers
from .strategies.registry import get_strategy, list_strategies
from .text_prestages import (
    _resolve_text_embeddings_finetune_cfg,
    _resolve_text_prompt_tuning_cfg,
    _run_text_embeddings_finetune_stage,
    _run_text_prompt_tuning_stage,
)

# Backward-compatible aliases for tests and internal callers that still import these helpers
# from train_vision directly.
_load_model_init_checkpoint = load_model_init_checkpoint
_initialize_task_text_features = initialize_task_text_features


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _to_metric_float(value: Any) -> float | None:
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


def _sanitize_metric_name(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()]
    sanitized = "".join(chars)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def _clear_regularizer_breakdowns(model: nn.Module) -> None:
    for attr_name in ("_distillation_last_breakdown", "_ekfac_ggn_last_breakdown", "_kfac_ggn_last_breakdown"):
        if hasattr(model, attr_name):
            delattr(model, attr_name)


def _extend_curvature_breakdown_metrics(
    metrics: dict[str, float],
    breakdown: Mapping[str, Any] | None,
    *,
    suffix: str = "",
) -> None:
    if not isinstance(breakdown, Mapping):
        return
    mapping = {
        "matrix": "loss_penalty",
        "ffT": "loss_reg_ffT",
        "projection": "loss_ft_proj",
        "class_embedding": "loss_reg_cls_emb",
    }
    suffix_text = f"_{suffix}" if suffix else ""
    for source_key, metric_name in mapping.items():
        value = _to_metric_float(breakdown.get(source_key))
        if value is not None:
            metrics[f"{metric_name}{suffix_text}"] = value


def _collect_regularizer_loss_metrics(model: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}
    distillation_breakdown = getattr(model, "_distillation_last_breakdown", None)
    if isinstance(distillation_breakdown, Mapping):
        for key, value in distillation_breakdown.items():
            metric_value = _to_metric_float(value)
            if metric_value is not None:
                metrics[_sanitize_metric_name(str(key))] = metric_value
    _extend_curvature_breakdown_metrics(metrics, getattr(model, "_ekfac_ggn_last_breakdown", None))
    if "loss_penalty" not in metrics:
        _extend_curvature_breakdown_metrics(metrics, getattr(model, "_kfac_ggn_last_breakdown", None))
    return metrics


def _optimizer_lr_metrics(*, task: str, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    metrics = {f"train/{task}/lr": float(optimizer.param_groups[0]["lr"])}
    for group in optimizer.param_groups:
        group_name = str(group.get("name", "")).strip().lower()
        if not group_name:
            continue
        metrics[f"train/{task}/lr_{group_name}"] = float(group["lr"])
    return metrics


def _is_teacher_loss_metric_name(metric_name: str) -> bool:
    name = str(metric_name).strip().lower()
    return name.startswith("loss_teacher_") or name.endswith("_teacher")


def _get_teacher_model(prepared_regularizer: Any) -> nn.Module | None:
    if prepared_regularizer is None:
        return None

    teacher_runtime = getattr(prepared_regularizer, "teacher", None)
    teacher_model = getattr(teacher_runtime, "model", None)
    if isinstance(teacher_model, nn.Module):
        return teacher_model

    children = getattr(prepared_regularizer, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            nested_prepared = getattr(child, "prepared", None)
            nested_model = _get_teacher_model(nested_prepared)
            if isinstance(nested_model, nn.Module):
                return nested_model
    return None


def _build_backward_loss_metrics(
    *,
    task: str,
    raw_loss: torch.Tensor,
    reg_loss: torch.Tensor,
    total_loss: torch.Tensor,
    regularizer_metrics: Mapping[str, float],
) -> dict[str, float]:
    metrics = {
        f"train_backward/{task}/loss_task": float(raw_loss.item()),
        f"train_backward/{task}/loss_reg": float(reg_loss.item()),
        f"train_backward/{task}/loss_total": float(total_loss.item()),
    }
    for metric_name, metric_value in regularizer_metrics.items():
        metrics[f"train_backward/{task}/{metric_name}"] = float(metric_value)
    return metrics


def _device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """
    Recursive dict merge: src overwrites dst. Returns dst (mutated).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_config(path: str) -> dict[str, Any]:
    """
    Load a single config file (YAML preferred, JSON supported).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    if p.suffix.lower() in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError("PyYAML not available. Install pyyaml or use a .json config.")
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML config must be a mapping at the top-level.")
        return cfg

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("JSON config must be an object at the top-level.")
        return cfg

    raise ValueError(f"Unsupported config extension: {p.suffix} (use .yaml/.yml or .json)")


def _get_common_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    common = cfg.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("config['common'] must be a dict.")
    return common


def _get_dataset_override(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    ds = cfg.get("datasets", {})
    if ds is None:
        return {}
    if not isinstance(ds, dict):
        raise ValueError("config['datasets'] must be a dict mapping dataset_name -> overrides.")
    ov = ds.get(task, {})
    if ov is None:
        return {}
    if not isinstance(ov, dict):
        raise ValueError(f"config['datasets']['{task}'] must be a dict.")
    return ov


def _resolve_tasks_from_cfg(cfg: dict[str, Any]) -> list[str] | None:
    order = cfg.get("datasets_order", None)
    if order is None:
        return None
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise ValueError("config['datasets_order'] must be a list of strings.")
    return list(order)


def _parse_reference_datasets(raw: Any, *, field_name: str) -> list[str] | None:
    return _parse_reference_datasets_shared(raw, field_name=field_name)


def _validate_vision_tasks(tasks: list[str], *, field_name: str) -> list[str]:
    return _validate_vision_tasks_shared(tasks, field_name=field_name)


def resolve_reference_tasks(
    args,
    *,
    training_tasks: list[str],
    regularization_cfg: dict[str, Any] | None = None,
    require_reference: bool = True,
) -> tuple[list[str], bool]:
    context = _build_reference_task_resolution_context(
        training_tasks=training_tasks,
        suite=getattr(args, "suite", None),
        cli_reference_suite=getattr(args, "reference_suite", None),
        cli_reference_datasets=getattr(args, "reference_datasets", None),
    )
    return _resolve_reference_tasks_shared(
        context=context,
        regularization_cfg=regularization_cfg,
        require_reference=require_reference,
    )


def _resolve_attention_patch_cfg(strategy_cfg: dict[str, Any] | None, *, total_steps: int) -> dict[str, Any] | None:
    if not isinstance(strategy_cfg, dict):
        return None
    attention_cfg = strategy_cfg.get("attention", None)
    if attention_cfg is None:
        return None
    if not isinstance(attention_cfg, dict):
        raise ValueError("strategy.attention must be a dict when provided.")

    attn_impl = str(attention_cfg.get("attn_impl", "softmax")).strip().lower()
    if attn_impl not in {"softmax", "linear"}:
        raise ValueError("attention.attn_impl must be one of: softmax, linear")
    ramp_fraction_default = 0.2 if attn_impl == "linear" else 0.0
    ramp_fraction = float(attention_cfg.get("ramp_fraction", ramp_fraction_default))
    if ramp_fraction < 0.0 or ramp_fraction > 1.0:
        raise ValueError("attention.ramp_fraction must be in [0, 1].")

    linear_rule = str(attention_cfg.get("linear_rule", "kernel")).strip().lower()
    if linear_rule not in {"kernel", "delta"}:
        raise ValueError("attention.linear_rule must be one of: kernel, delta")

    ramp_steps = int(round(ramp_fraction * max(1, int(total_steps))))
    return {
        "attn_impl": attn_impl,
        "kernel": str(attention_cfg.get("kernel", "elu_plus_one")),
        "eps": float(attention_cfg.get("eps", 1e-6)),
        "ramp_fraction": ramp_fraction,
        "ramp_steps": ramp_steps,
        "linear_rule": linear_rule,
        "delta_eta": float(attention_cfg.get("delta_eta", 1.0)),
        "delta_exclude_cls_from_store": bool(attention_cfg.get("delta_exclude_cls_from_store", True)),
        "delta_cls_only_readout": bool(attention_cfg.get("delta_cls_only_readout", False)),
        "delta_learn_w0": bool(attention_cfg.get("delta_learn_w0", False)),
        "delta_w0_rank": int(attention_cfg.get("delta_w0_rank", 0)),
    }


def _save_peft_visual_adapter(
    *,
    model: nn.Module,
    task_dir: Path,
    strategy: str,
    suffix: str | None,
    peft_cfg: dict[str, Any] | None,
    patched_attn: bool,
    attn_patch_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Save PEFT adapter using PEFT's native API on model.clip_model.model.visual.

    Returns a dict to be inserted into the checkpoint payload.
    """
    # We expect PEFT to wrap ONLY the visual module:
    visual = model.clip_model.model.visual  # type: ignore[attr-defined]
    if not hasattr(visual, "save_pretrained"):
        raise ValueError(
            "save_format='peft' expects model.clip_model.model.visual to be a PEFT-wrapped module "
            "(must have .save_pretrained())."
        )

    adapter_name = f"{strategy}_adapter" if suffix is None else f"{strategy}_{suffix}_adapter"
    adapter_dir = task_dir / adapter_name
    _ensure_dir(adapter_dir)
    visual.save_pretrained(adapter_dir)

    resolved_peft_cfg = getattr(model, "peft_cfg_resolved", None)
    if not isinstance(resolved_peft_cfg, dict):
        resolved_peft_cfg = peft_cfg if peft_cfg is not None else {}
    dense_trainable_keys = getattr(model, "peft_dense_trainable_visual_keys", ())
    dense_key_set = set(dense_trainable_keys)
    dense_state: dict[str, torch.Tensor] = {}
    current_param_getter = getattr(model, "_current_param_map", None)
    if callable(current_param_getter):
        current_param_map = current_param_getter()
        if isinstance(current_param_map, Mapping):
            dense_state = {
                key: value.detach().cpu()
                for key, value in (
                    (
                        key,
                        current_param_map[f"clip_model.model.visual.{key}"],
                    )
                    for key in dense_trainable_keys
                    if f"clip_model.model.visual.{key}" in current_param_map
                )
            }
    if not dense_state:
        dense_state = {
            key: value.detach().cpu()
            for key, value in visual.state_dict().items()
            if key in dense_key_set
        }
    trainable_plan = getattr(model, "peft_trainable_plan", None)

    meta = {
        "format": "peft",
        "peft_target": "visual",
        "peft_adapter_dir": str(adapter_dir),
        "peft_cfg": resolved_peft_cfg,
        "patched_attn": bool(patched_attn),
        "attn_patch_cfg": dict(attn_patch_cfg or {}),
        "patched_proj": bool(getattr(model, "peft_patched_proj", False)),
        "dense_trainable_keys": list(dense_state.keys()),
        "peft_trainable_plan": dict(trainable_plan) if isinstance(trainable_plan, dict) else {},
    }
    _save_json(adapter_dir / "merge_and_rebase_meta.json", meta)
    payload = dict(meta)
    payload["peft_dense_state"] = dense_state
    return payload


# ---------------------------
# Training loop
# ---------------------------


def train_task(
    *,
    task: str,
    hf_path: str,
    hf_config: str | None,
    split_map: dict[str, str],
    build_cfg: OpenClipBuildConfig,
    strategy: str,
    epochs: int,
    lr: float,
    optimizer_name: str = "adamw",
    weight_decay: float,
    dense_lr: float | None = None,
    warmup_length: int,
    scheduler_name: str = "cosine",
    clip_grad_norm: float,
    accumulate_grad_batches: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    loader_profile: str = "hf",
    data_root: str | None = None,
    early_stopping: bool,
    early_stopping_patience: int,
    text_only: bool,
    seed: int,
    device: str,
    out_dir: Path,
    save_format: str,  # "full"|"head"|"peft"
    save_checkpoints: bool = True,
    save_last_epoch: bool = False,
    train_preprocess: str = "eval",
    init_checkpoint: str | None = None,
    init_text_features_source: str = "zero_shot",
    peft_cfg: dict[str, Any] | None = None,
    strategy_cfg: dict[str, Any] | None = None,
    regularization_cfg: dict[str, Any] | None = None,
    all_tasks: list[str] | None = None,
    reference_tasks: list[str] | None = None,
    reference_resolution_context: ReferenceTaskResolutionContext | None = None,
    log_every_n_steps: int = 50,
    run_logger: Any | None = None,
) -> dict[str, Any]:
    dev = _device(device)
    _set_seed(seed)
    if accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be >= 1.")
    strategy_cfg = dict(strategy_cfg or {})
    forward_mode = resolve_training_forward_mode(strategy_cfg)
    forward_mode_params = normalize_forward_mode_params(forward_mode, strategy_cfg.get("forward_mode_params", None))

    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    clf = OpenClipClassifier.build(build_cfg)
    train_preprocess = _resolve_train_preprocess(train_preprocess, task=task)
    loader_profile = _resolve_loader_profile(loader_profile, task=task)
    train_transform = getattr(clf, "train_preprocess", clf.preprocess) if train_preprocess == "train" else None

    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=clf.preprocess,
        train_preprocess=train_transform,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        val_fraction=val_fraction,
        seed=seed,
    )

    num_classes = len(loaders.classnames)

    task_dir = out_dir / build_cfg.model_name / build_cfg.pretrained / task
    _ensure_dir(task_dir)

    if run_logger is not None:
        run_logger.log_event(
            "task_start",
            metrics={},
            context={
                "task": task,
                "strategy": strategy,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "effective_batch_size": int(batch_size * accumulate_grad_batches),
                "train_preprocess": train_preprocess,
                "task_dir": str(task_dir),
                "text_only": bool(text_only),
                "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
                "lr": float(lr),
                "dense_lr": float(lr if dense_lr is None else dense_lr),
            },
        )

    # model = encoder + head
    model = ImageEncoder(clf).to(dev)
    checkpoint_init_summary: dict[str, Any] | None = None
    checkpoint_init_obj: Any = None
    if init_checkpoint is not None:
        checkpoint_init_obj, checkpoint_init_summary = load_model_init_checkpoint(
            model=model, ckpt_path=init_checkpoint
        )
        print(
            f"[{task}] initialized model from checkpoint {init_checkpoint} "
            f"(loaded_tensors={checkpoint_init_summary['loaded_tensors']}, "
            f"target={checkpoint_init_summary['load_target']}, mode={checkpoint_init_summary['load_mode']})."
        )
        if run_logger is not None:
            run_logger.log_event(
                "checkpoint_init",
                metrics={},
                context={
                    "task": task,
                    "initialization": checkpoint_init_summary,
                },
            )
    text_features_init_source = initialize_task_text_features(
        model=model,
        classnames=list(loaders.classnames),
        build_cfg=build_cfg,
        device=dev,
        ckpt_obj=checkpoint_init_obj,
        ckpt_path=init_checkpoint,
        text_features_source=init_text_features_source,
    )
    zero_shot_val = (
        model.top1(loaders.val, str(dev)) if hasattr(loaders, "val") and loaders.val is not None else float("nan")
    )
    zero_shot_test = model.top1(loaders.test, str(dev))
    print(f"[{task}] zero-shot before finetuning  val={zero_shot_val:.4f}  test={zero_shot_test:.4f}")
    if run_logger is not None:
        run_logger.log_event(
            "zero_shot_eval",
            metrics={
                f"zero_shot/{task}/val_top1": float(zero_shot_val),
                f"zero_shot/{task}/test_top1": float(zero_shot_test),
            },
            context={"task": task},
        )

    text_emb_ft_cfg = _resolve_text_embeddings_finetune_cfg(
        strategy_cfg,
        default_epochs=epochs,
        default_lr=lr,
        default_weight_decay=weight_decay,
        default_warmup_length=warmup_length,
        default_clip_grad_norm=clip_grad_norm,
        default_accumulate_grad_batches=accumulate_grad_batches,
    )
    text_prompt_ft_cfg = _resolve_text_prompt_tuning_cfg(
        strategy_cfg,
        default_epochs=epochs,
        default_lr=lr,
        default_weight_decay=weight_decay,
        default_warmup_length=warmup_length,
        default_clip_grad_norm=clip_grad_norm,
        default_accumulate_grad_batches=accumulate_grad_batches,
    )
    if text_emb_ft_cfg is not None and text_prompt_ft_cfg is not None:
        raise ValueError(
            "strategy.text_embeddings_finetune and strategy.text_prompt_tuning are mutually exclusive. "
            "Enable only one text pre-stage."
        )

    text_emb_ft_summary: dict[str, Any] | None = None
    text_prompt_ft_summary: dict[str, Any] | None = None
    if text_prompt_ft_cfg is not None:
        print(
            f"[{task}] Running text prompt-tuning pre-stage "
            f"(epochs={text_prompt_ft_cfg['epochs']}, ctx_len={text_prompt_ft_cfg['context_length']}, lr={text_prompt_ft_cfg['lr']:.2e})."
        )
        text_prompt_ft_summary = _run_text_prompt_tuning_stage(
            task=task,
            model=model,
            loaders=loaders,
            device=dev,
            cfg=text_prompt_ft_cfg,
        )
        if run_logger is not None:
            run_logger.log_event(
                "text_prestage_end",
                metrics={},
                context={
                    "task": task,
                    "stage": "text_prompt_tuning",
                    "summary": text_prompt_ft_summary,
                },
            )
    elif text_emb_ft_cfg is not None:
        print(
            f"[{task}] Running text-embedding pre-stage "
            f"(epochs={text_emb_ft_cfg['epochs']}, lr={text_emb_ft_cfg['lr']:.2e})."
        )
        text_emb_ft_summary = _run_text_embeddings_finetune_stage(
            task=task,
            model=model,
            loaders=loaders,
            device=dev,
            cfg=text_emb_ft_cfg,
        )
        if run_logger is not None:
            run_logger.log_event(
                "text_prestage_end",
                metrics={},
                context={
                    "task": task,
                    "stage": "text_embeddings_finetune",
                    "summary": text_emb_ft_summary,
                },
            )

    if text_only:
        if forward_mode != "standard":
            raise ValueError(f"[{task}] train.text_only=True requires strategy.forward_mode='standard'.")
        if text_prompt_ft_summary is None and text_emb_ft_summary is None:
            raise ValueError(
                f"[{task}] train.text_only=True requires enabling either "
                "strategy.text_prompt_tuning or strategy.text_embeddings_finetune."
            )

        text_stage_summary = text_prompt_ft_summary if text_prompt_ft_summary is not None else text_emb_ft_summary
        assert text_stage_summary is not None

        model_sd = model.state_dict()
        patched_attn = state_dict_looks_patched_attn(model_sd)
        best_epoch = int(text_stage_summary.get("best_epoch", 0))
        last_epoch = int(text_stage_summary.get("last_epoch", best_epoch))
        metrics = {
            "val_top1": float(text_stage_summary.get("best_val_top1", float("nan"))),
            "test_top1": float(text_stage_summary.get("best_test_top1", float("nan"))),
        }
        best_state: dict[str, Any] = {
            "task": task,
            "strategy": strategy,
            "forward_mode": forward_mode,
            "forward_mode_params": dict(forward_mode_params),
            "backbone": {
                "kind": "openclip",
                "model_name": build_cfg.model_name,
                "pretrained": build_cfg.pretrained,
                "dtype": build_cfg.dtype,
            },
            "num_classes": num_classes,
            "classnames": list(loaders.classnames),
            "metrics": metrics,
            "zero_shot_metrics": {
                "val_top1": float(zero_shot_val),
                "test_top1": float(zero_shot_test),
            },
            "best_epoch": best_epoch,
            "patched_attn": patched_attn,
        }
        if checkpoint_init_summary is not None:
            best_state["initialization"] = dict(checkpoint_init_summary)
        best_state["text_features_init_source"] = text_features_init_source
        if text_emb_ft_summary is not None:
            best_state["text_embeddings_finetune"] = dict(text_emb_ft_summary)
        if text_prompt_ft_summary is not None:
            best_state["text_prompt_tuning"] = dict(text_prompt_ft_summary)
        best_state["tuned_text_features"] = model.clip_model._zs_text_features.detach().cpu()
        tuned_prompt_context = getattr(model.clip_model, "_tuned_prompt_context", None)
        if text_prompt_ft_summary is not None and isinstance(tuned_prompt_context, torch.Tensor):
            best_state["tuned_prompt_context"] = tuned_prompt_context.detach().cpu()

        ckpt_stem = strategy if forward_mode == "standard" else f"{strategy}__{forward_mode}"
        best_ckpt_path = task_dir / f"{ckpt_stem}_best_ep.pt"
        if save_format != "full":
            raise ValueError(
                f"[{task}] train.text_only=True currently supports output.save_format='full' only; got '{save_format}'."
            )
        if save_checkpoints:
            best_state["state_dict"] = {k: v.detach().cpu() for k, v in model_sd.items()}
            best_state["format"] = "full"
            torch.save(best_state, best_ckpt_path)

        last_ckpt_path: Path | None = None
        if save_checkpoints and save_last_epoch:
            last_state = dict(best_state)
            last_state["last_epoch"] = last_epoch
            last_state["best_epoch"] = best_epoch
            last_ckpt_path = task_dir / f"{ckpt_stem}_last_ep.pt"
            torch.save(last_state, last_ckpt_path)

        text_stage_best_seconds = float(
            text_stage_summary.get("best_elapsed_seconds", text_stage_summary.get("seconds", 0.0))
        )
        text_stage_last_seconds = float(
            text_stage_summary.get("last_elapsed_seconds", text_stage_summary.get("seconds", 0.0))
        )

        summary = {
            "task": task,
            "strategy": strategy,
            "forward_mode": forward_mode,
            "save_format": save_format,
            "save_checkpoints": bool(save_checkpoints),
            "save_last_epoch": bool(save_last_epoch),
            "ckpt_path": str(best_ckpt_path) if save_checkpoints else None,
            "best_ckpt_path": str(best_ckpt_path) if save_checkpoints else None,
            "last_ckpt_path": str(last_ckpt_path) if last_ckpt_path is not None else None,
            "metrics": metrics,
            "zero_shot_metrics": best_state["zero_shot_metrics"],
            "seconds": float(text_stage_summary.get("seconds", 0.0)),
            "best_elapsed_seconds": text_stage_best_seconds,
            "last_elapsed_seconds": text_stage_last_seconds,
            "selected_timing": {
                "text_prestage_seconds": text_stage_best_seconds,
                "vision_seconds": 0.0,
                "total_seconds": text_stage_best_seconds,
            },
            "trainable": {
                "trainable_params": int(text_stage_summary.get("trainable_params", 0)),
                "mode": "text_only",
            },
            "text_features_init_source": text_features_init_source,
            "initialization": checkpoint_init_summary,
            "text_embeddings_finetune": text_emb_ft_summary,
            "text_prompt_tuning": text_prompt_ft_summary,
            "regularization": {"name": "", "info": {}},
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            "vision_training_skipped": True,
            "hparams": {
                "epochs": int(epochs),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "warmup_length": int(warmup_length),
                "clip_grad_norm": float(clip_grad_norm),
                "accumulate_grad_batches": int(accumulate_grad_batches),
                "batch_size": int(batch_size),
                "effective_batch_size": int(batch_size * accumulate_grad_batches),
                "num_workers": int(num_workers),
                "val_fraction": float(val_fraction),
                "train_preprocess": train_preprocess,
                "seed": int(seed),
            },
        }
        _save_json(task_dir / f"{ckpt_stem}.json", summary)
        if save_checkpoints:
            print(f"[{task}] saved best: {best_ckpt_path}")
        if save_checkpoints and last_ckpt_path is not None:
            print(f"[{task}] saved last: {last_ckpt_path}")
        if run_logger is not None:
            run_logger.log_event(
                "task_end",
                metrics={
                    f"val/{task}/top1": float(summary["metrics"].get("val_top1", float("nan"))),
                    f"test/{task}/top1": float(summary["metrics"].get("test_top1", float("nan"))),
                    f"train/{task}/seconds": float(summary["seconds"]),
                },
                context={
                    "task": task,
                    "summary": summary,
                },
            )
        return summary

    loss_fn = nn.CrossEntropyLoss()
    steps_per_epoch = math.ceil(len(loaders.train) / accumulate_grad_batches)
    total_steps = epochs * steps_per_epoch

    # For non-PEFT strategies, optional strategy.attention patching is applied here.
    # PEFT handles its own attention patching inside PeftLoraVision.configure().
    if strategy != "peft_lora":
        attn_patch_cfg = _resolve_attention_patch_cfg(strategy_cfg, total_steps=total_steps)
        if attn_patch_cfg is not None:
            patched = split_openclip_vit_attn(
                model.clip_model.model.visual,
                proj_dropout=0.0,
                attn_impl=str(attn_patch_cfg.get("attn_impl", "softmax")),
                kernel=str(attn_patch_cfg.get("kernel", "elu_plus_one")),
                eps=float(attn_patch_cfg.get("eps", 1e-6)),
                ramp_steps=int(attn_patch_cfg.get("ramp_steps", 0)),
                linear_rule=str(attn_patch_cfg.get("linear_rule", "kernel")),
                delta_eta=float(attn_patch_cfg.get("delta_eta", 1.0)),
                delta_exclude_cls_from_store=bool(attn_patch_cfg.get("delta_exclude_cls_from_store", True)),
                delta_cls_only_readout=bool(attn_patch_cfg.get("delta_cls_only_readout", False)),
                delta_learn_w0=bool(attn_patch_cfg.get("delta_learn_w0", False)),
                delta_w0_rank=int(attn_patch_cfg.get("delta_w0_rank", 0)),
            )
            if patched == 0:
                raise RuntimeError("Requested strategy.attention patching but patched 0 blocks.")
            model.peft_patched_attn = True  # type: ignore[attr-defined]
            model.peft_attn_patch_cfg = dict(attn_patch_cfg)  # type: ignore[attr-defined]
            print(f"[{task}] Patched {patched} attention blocks (attn_impl={attn_patch_cfg['attn_impl']}).")

    regularizer_cfg = dict(regularization_cfg or {})
    regularizer_name = str(regularizer_cfg.get("name", "")).strip()
    regularizer_impl = None
    regularizer_info: dict[str, int] = {}
    prepared_regularizer: Any | None = None
    regularizer_finalize_info: dict[str, Any] = {}

    if regularizer_name:
        regularizer_impl = get_regularizer(regularizer_name)
        regularizer_finalize_info = finalize_model_for_regularizer(
            regularizer_impl,
            model=model,
            device=dev,
            regularization_cfg=regularizer_cfg,
            task=task,
            strategy_cfg=strategy_cfg,
            build_cfg=build_cfg,
            loaders=loaders,
            all_tasks=list(all_tasks or [task]),
            reference_tasks=list(reference_tasks or []),
            reference_resolution_context=reference_resolution_context,
            batch_size=batch_size,
            num_workers=num_workers,
            val_fraction=val_fraction,
            loader_profile=loader_profile,
            data_root=data_root,
            seed=seed,
            run_logger=run_logger,
            total_steps=total_steps,
            warmup_length=warmup_length,
            train_scheduler_name=scheduler_name,
            train_lr=lr,
            train_dense_lr=float(lr if dense_lr is None else dense_lr),
            train_weight_decay=weight_decay,
            train_optimizer_name=optimizer_name,
            train_grad_clip_norm=clip_grad_norm,
            accumulate_grad_batches=accumulate_grad_batches,
            student_forward_mode=forward_mode,
            student_forward_mode_params=dict(forward_mode_params),
        )

    strategy_impl = get_strategy(strategy)
    configured = strategy_impl.configure(
        model=model,
        lr=lr,
        dense_lr=float(lr if dense_lr is None else dense_lr),
        weight_decay=weight_decay,
        warmup_length=warmup_length,
        scheduler_name=scheduler_name,
        optimizer=optimizer_name,
        steps=total_steps,
        device=dev,
        peft_cfg=peft_cfg,
        strategy_cfg=strategy_cfg,
        task=task,
    )
    if len(configured) != 3:
        raise ValueError("Strategy.configure() must return (opt, scheduler, info).")
    opt, scheduler, trainable_info = configured
    trainable_info = dict(trainable_info)
    trainable_info["forward_mode"] = forward_mode
    base_model_sd = {k: v.detach().clone() for k, v in model.clip_model.model.state_dict().items()}
    trainable_info.update(
        bind_training_forward_mode(
            model=model,
            forward_mode=forward_mode,
            base_sd=base_model_sd,
            strict_load=True,
            params=forward_mode_params,
        )
    )
    if regularizer_name:
        assert regularizer_impl is not None
        prepared_regularizer, regularizer_info = regularizer_impl.prepare(
            model=model,
            device=dev,
            regularization_cfg=regularizer_cfg,
            task=task,
            strategy_cfg=strategy_cfg,
            build_cfg=build_cfg,
            loaders=loaders,
            all_tasks=list(all_tasks or [task]),
            reference_tasks=list(reference_tasks or []),
            reference_resolution_context=reference_resolution_context,
            batch_size=batch_size,
            num_workers=num_workers,
            val_fraction=val_fraction,
            seed=seed,
            run_logger=run_logger,
            total_steps=total_steps,
            warmup_length=warmup_length,
            train_scheduler_name=scheduler_name,
            train_lr=lr,
            train_dense_lr=float(lr if dense_lr is None else dense_lr),
            train_weight_decay=weight_decay,
            train_optimizer_name=optimizer_name,
            train_grad_clip_norm=clip_grad_norm,
            accumulate_grad_batches=accumulate_grad_batches,
            student_forward_mode=forward_mode,
            student_forward_mode_params=dict(forward_mode_params),
        )
        for key, value in regularizer_finalize_info.items():
            regularizer_info[f"finalize_model.{key}"] = value

    def _close_prepared_regularizer() -> None:
        close_prepared_regularizer(prepared_regularizer)

    best_val = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_elapsed_seconds = 0.0
    last_epoch = 0
    last_val = float("nan")
    last_test = float("nan")
    last_elapsed_seconds = 0.0
    early_stopping_patience_current = early_stopping_patience
    model.to(dev)

    t_start = time.time()
    global_update_step = 0
    backward_log_step = 0
    optimizer_bundles = iter_optimizer_bundles(
        student_optimizer=opt,
        student_scheduler=scheduler,
        student_grad_clip_norm=float(clip_grad_norm),
        prepared=prepared_regularizer,
    )

    ckpt_paths = _task_checkpoint_paths(
        out_dir=out_dir,
        build_cfg=build_cfg,
        task=task,
        strategy=strategy,
        strategy_cfg=strategy_cfg,
        regularization_cfg=regularization_cfg,
    )
    ckpt_stem = str(ckpt_paths["ckpt_stem"])

    def _build_checkpoint_payload(
        *,
        epoch_i: int,
        val_acc_i: float,
        test_acc_i: float,
        kind: str,  # "best_ep" | "last_ep"
        include_weights: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": task,
            "strategy": strategy,
            "forward_mode": forward_mode,
            "forward_mode_params": dict(forward_mode_params),
            "backbone": {
                "kind": "openclip",
                "model_name": build_cfg.model_name,
                "pretrained": build_cfg.pretrained,
                "dtype": build_cfg.dtype,
            },
            "num_classes": num_classes,
            "classnames": list(loaders.classnames),
            "metrics": {"val_top1": float(val_acc_i), "test_top1": float(test_acc_i)},
            "zero_shot_metrics": {
                "val_top1": float(zero_shot_val),
                "test_top1": float(zero_shot_test),
            },
            "text_features_init_source": text_features_init_source,
        }
        if checkpoint_init_summary is not None:
            payload["initialization"] = dict(checkpoint_init_summary)
        if text_emb_ft_summary is not None:
            payload["text_embeddings_finetune"] = dict(text_emb_ft_summary)
        if text_prompt_ft_summary is not None:
            payload["text_prompt_tuning"] = dict(text_prompt_ft_summary)
        if text_emb_ft_summary is not None or text_prompt_ft_summary is not None:
            payload["tuned_text_features"] = model.clip_model._zs_text_features.detach().cpu()
        tuned_prompt_context = getattr(model.clip_model, "_tuned_prompt_context", None)
        if text_prompt_ft_summary is not None and isinstance(tuned_prompt_context, torch.Tensor):
            payload["tuned_prompt_context"] = tuned_prompt_context.detach().cpu()
        if kind == "best_ep":
            payload["best_epoch"] = int(epoch_i)
        elif kind == "last_ep":
            payload["last_epoch"] = int(epoch_i)
            payload["best_epoch"] = int(best_epoch)
        else:
            raise ValueError("kind must be 'best_ep' or 'last_ep'")

        model_sd = materialized_model_state_dict(model)
        patched_attn = bool(getattr(model, "peft_patched_attn", False)) or state_dict_looks_patched_attn(model_sd)
        attn_patch_cfg_raw = getattr(model, "peft_attn_patch_cfg", None)
        attn_patch_cfg = dict(attn_patch_cfg_raw) if isinstance(attn_patch_cfg_raw, dict) else None
        if patched_attn and attn_patch_cfg is None:
            # Fallback for non-PEFT paths that patched q/k/v attention without explicit cfg metadata.
            attn_patch_cfg = {
                "attn_impl": "softmax",
                "kernel": "elu_plus_one",
                "eps": 1e-6,
                "linear_rule": "kernel",
                "delta_eta": 1.0,
                "delta_exclude_cls_from_store": True,
                "delta_cls_only_readout": False,
                "delta_learn_w0": False,
                "delta_w0_rank": 0,
            }
        payload["patched_attn"] = patched_attn
        if attn_patch_cfg is not None:
            payload["attn_patch_cfg"] = attn_patch_cfg

        if include_weights:
            if save_format == "full":
                payload["state_dict"] = {k: v.detach().cpu() for k, v in model_sd.items()}
                payload["format"] = "full"
            elif save_format == "head":
                payload["head"] = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
                payload["format"] = "head"
            elif save_format == "peft":
                payload.update(
                    _save_peft_visual_adapter(
                        model=model,
                        task_dir=task_dir,
                        strategy=ckpt_stem,
                        suffix=kind,
                        peft_cfg=peft_cfg,
                        patched_attn=patched_attn,
                        attn_patch_cfg=attn_patch_cfg,
                    )
                )
            else:
                raise ValueError("save_format must be 'full', 'head', or 'peft'")
        payload.update(checkpoint_payload_from_prepared(prepared_regularizer, kind=kind))
        return payload

    def _save_regularizer_checkpoint_artifacts(
        *,
        kind: str,
        epoch_i: int,
        val_acc_i: float,
        test_acc_i: float,
    ) -> None:
        artifacts = collect_checkpoint_artifacts(
            prepared_regularizer,
            kind=kind,
            epoch_i=epoch_i,
            val_acc_i=float(val_acc_i),
            test_acc_i=float(test_acc_i),
            zero_shot_metrics={"val_top1": float(zero_shot_val), "test_top1": float(zero_shot_test)},
        )
        for artifact in artifacts:
            artifact_dir = Path(artifact.output_dir)
            _ensure_dir(artifact_dir)
            torch.save(artifact.payload, artifact_dir / artifact.filename)
            if artifact.summary is not None and artifact.summary_filename is not None:
                _save_json(artifact_dir / artifact.summary_filename, dict(artifact.summary))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for bundle in optimizer_bundles:
            bundle.optimizer.zero_grad(set_to_none=True)
        window_batch_count = 0
        window_size = 1

        with tqdm(total=len(loaders.train), desc=f"[{task}] Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for i, (x, y) in enumerate(loaders.train):
                if window_batch_count == 0:
                    remaining = len(loaders.train) - i
                    window_size = min(accumulate_grad_batches, remaining)
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)

                # Blend softmax -> linear attention during warmup ramp (if enabled).
                set_linear_attention_ramp_step(model, step=global_update_step)
                batch_override: BatchOverride | None = None
                try:
                    if regularizer_impl is not None and prepared_regularizer is not None:
                        batch_override = prepare_batch_override(
                            prepared_regularizer,
                            model=model,
                            step=global_update_step,
                            batch_index=i,
                            inputs=x,
                            targets=y,
                            virtual_batch_start=(window_batch_count == 0),
                            window_size=window_size,
                            accumulate_grad_batches=accumulate_grad_batches,
                        )
                    if batch_override is not None:
                        logits = batch_override.outputs
                        raw_loss = batch_override.primary_loss
                    else:
                        logits = model(x)
                        raw_loss = loss_fn(logits, y)
                    reg_loss = raw_loss.new_zeros(())
                    if regularizer_impl is not None and prepared_regularizer is not None:
                        _clear_regularizer_breakdowns(model)
                        reg_loss = reg_loss + regularizer_impl.apply(
                            prepared_regularizer,
                            model=model,
                            step=global_update_step,
                            batch_index=i,
                            inputs=x,
                            targets=y,
                            outputs=logits,
                            batch_context=(batch_override.context if batch_override is not None else None),
                        )
                    total_loss = raw_loss + reg_loss
                    loss = total_loss / window_size

                    loss.backward()
                finally:
                    if batch_override is not None and callable(batch_override.close):
                        batch_override.close()

                backward_log_step += 1
                regularizer_metrics = _collect_regularizer_loss_metrics(model)
                if run_logger is not None:
                    run_logger.log_event(
                        "train_backward_loss",
                        metrics=_build_backward_loss_metrics(
                            task=task,
                            raw_loss=raw_loss,
                            reg_loss=reg_loss,
                            total_loss=total_loss,
                            regularizer_metrics=regularizer_metrics,
                        ),
                        step=int(backward_log_step),
                        context={
                            "task": task,
                            "epoch": int(epoch),
                            "update_step": int(global_update_step),
                            "_wandb_only": True,
                        },
                    )

                window_batch_count += 1
                should_step = window_batch_count == window_size
                if should_step:
                    for bundle in optimizer_bundles:
                        if bundle.grad_clip_norm > 0:
                            params = [p for group in bundle.optimizer.param_groups for p in group["params"] if p.grad is not None]
                            if params:
                                torch.nn.utils.clip_grad_norm_(params, max_norm=float(bundle.grad_clip_norm))
                        if callable(bundle.scheduler):
                            bundle.scheduler(global_update_step)
                        bundle.optimizer.step()
                        bundle.optimizer.zero_grad(set_to_none=True)
                    global_update_step += 1
                    window_batch_count = 0

                bs = int(y.numel())
                running_loss += float(total_loss.item()) * bs
                n_seen += bs

                train_loss = running_loss / max(1, n_seen)
                if (
                    run_logger is not None
                    and should_step
                    and log_every_n_steps > 0
                    and global_update_step > 0
                    and global_update_step % log_every_n_steps == 0
                ):
                    teacher_loss_metrics = {
                        metric_name: float(metric_value)
                        for metric_name, metric_value in regularizer_metrics.items()
                        if _is_teacher_loss_metric_name(metric_name)
                    }
                    metrics = {
                        f"train/{task}/loss": float(train_loss),
                        f"train/{task}/reg_loss": float(reg_loss.item()) if regularizer_impl is not None else 0.0,
                        f"train/{task}/loss_task": float(raw_loss.item()),
                        f"train/{task}/loss_reg": float(reg_loss.item()) if regularizer_impl is not None else 0.0,
                        f"train/{task}/loss_total_step": float(total_loss.item()),
                    }
                    metrics.update(_optimizer_lr_metrics(task=task, optimizer=opt))
                    for metric_name, metric_value in regularizer_metrics.items():
                        if _is_teacher_loss_metric_name(metric_name):
                            continue
                        metrics[f"train/{task}/{metric_name}"] = float(metric_value)
                    run_logger.log_event(
                        "train_step",
                        metrics=metrics,
                        step=int(global_update_step),
                        context={
                            "task": task,
                            "epoch": int(epoch),
                        },
                    )
                    if teacher_loss_metrics:
                        run_logger.log_event(
                            "train_step_teacher",
                            metrics={
                                f"train/{task}/{metric_name}": float(metric_value)
                                for metric_name, metric_value in teacher_loss_metrics.items()
                            },
                            step=int(global_update_step),
                            context={
                                "task": task,
                                "epoch": int(epoch),
                                "_wandb_only": True,
                            },
                        )
                pbar.update(1)
                postfix = {"loss": f"{train_loss:.4f}", "lr": f"{opt.param_groups[0]['lr']:.6f}"}
                if regularizer_impl is not None:
                    postfix["reg"] = f"{float(reg_loss.item()):.2e}"
                pbar.set_postfix(postfix)

        # val/test
        set_linear_attention_ramp_step(model, step=global_update_step)
        val_acc = (
            model.top1(loaders.val, str(dev)) if hasattr(loaders, "val") and loaders.val is not None else float("nan")
        )
        test_acc = model.top1(loaders.test, str(dev))

        last_epoch = epoch
        last_val = float(val_acc)
        last_test = float(test_acc)
        epoch_elapsed_seconds = float(time.time() - t_start)
        last_elapsed_seconds = epoch_elapsed_seconds

        if not math.isnan(val_acc) and val_acc > best_val:
            early_stopping_patience_current = early_stopping_patience
            best_epoch = epoch
            best_val = val_acc
            best_elapsed_seconds = epoch_elapsed_seconds
            best_state = _build_checkpoint_payload(
                epoch_i=best_epoch,
                val_acc_i=float(val_acc),
                test_acc_i=float(test_acc),
                kind="best_ep",
                include_weights=save_checkpoints,
            )
            if save_checkpoints:
                torch.save(best_state, task_dir / f"{ckpt_stem}_best_ep.pt")
        else:
            early_stopping_patience_current -= 1
            if early_stopping_patience_current <= 0 and early_stopping:
                print(f"[{task}] Early stopping triggered. No improvement in validation for several epochs.")
                break

        print(
            f"[{task}] epoch {epoch:03d}/{epochs}  loss={train_loss:.4f}  val={val_acc:.4f}  test={test_acc:.4f}  patience={early_stopping_patience_current}/{early_stopping_patience}"
        )
        if run_logger is not None:
            run_logger.log_event(
                "epoch_end",
                metrics={
                    f"train/{task}/loss": float(train_loss),
                    f"val/{task}/top1": float(val_acc),
                    f"test/{task}/top1": float(test_acc),
                    f"train/{task}/seconds": float(time.time() - t_start),
                    **_optimizer_lr_metrics(task=task, optimizer=opt),
                },
                step=int(epoch),
                context={
                    "task": task,
                    "epoch": int(epoch),
                    "patience_left": int(early_stopping_patience_current),
                },
            )

    seconds = time.time() - t_start

    if best_state is None:
        fallback_best_epoch = best_epoch if best_epoch > 0 else last_epoch
        fallback_test = last_test if not math.isnan(last_test) else float(model.top1(loaders.test, str(dev)))
        best_state = _build_checkpoint_payload(
            epoch_i=fallback_best_epoch,
            val_acc_i=last_val,
            test_acc_i=fallback_test,
            kind="best_ep",
            include_weights=save_checkpoints,
        )
        best_state["regularization"] = {"name": regularizer_name, "info": regularizer_info}
        if best_elapsed_seconds <= 0.0:
            best_elapsed_seconds = float(last_elapsed_seconds)

    best_ckpt_path = task_dir / f"{ckpt_stem}_best_ep.pt"
    if save_checkpoints:
        torch.save(best_state, best_ckpt_path)
        _save_regularizer_checkpoint_artifacts(
            kind="best_ep",
            epoch_i=int(best_state.get("best_epoch", best_epoch if best_epoch > 0 else last_epoch)),
            val_acc_i=float(best_state.get("metrics", {}).get("val_top1", last_val)),
            test_acc_i=float(best_state.get("metrics", {}).get("test_top1", last_test)),
        )

    last_ckpt_path: Path | None = None
    if save_checkpoints and save_last_epoch:
        if last_epoch <= 0:
            last_epoch = epochs
        last_state = _build_checkpoint_payload(
            epoch_i=last_epoch,
            val_acc_i=last_val,
            test_acc_i=last_test,
            kind="last_ep",
            include_weights=True,
        )
        last_state["regularization"] = {"name": regularizer_name, "info": regularizer_info}
        last_ckpt_path = task_dir / f"{ckpt_stem}_last_ep.pt"
        torch.save(last_state, last_ckpt_path)
        _save_regularizer_checkpoint_artifacts(
            kind="last_ep",
            epoch_i=last_epoch,
            val_acc_i=last_val,
            test_acc_i=last_test,
        )

    text_stage_best_seconds = 0.0
    if text_prompt_ft_summary is not None:
        text_stage_best_seconds = float(
            text_prompt_ft_summary.get("best_elapsed_seconds", text_prompt_ft_summary.get("seconds", 0.0))
        )
    elif text_emb_ft_summary is not None:
        text_stage_best_seconds = float(
            text_emb_ft_summary.get("best_elapsed_seconds", text_emb_ft_summary.get("seconds", 0.0))
        )

    teacher_metrics: dict[str, float] = {}
    teacher_model = _get_teacher_model(prepared_regularizer)
    if teacher_model is not None and hasattr(teacher_model, "top1"):
        teacher_val = (
            float(teacher_model.top1(loaders.val, str(dev)))
            if hasattr(loaders, "val") and loaders.val is not None
            else float("nan")
        )
        teacher_test = float(teacher_model.top1(loaders.test, str(dev)))
        teacher_metrics = {
            "val_top1": teacher_val,
            "test_top1": teacher_test,
        }

    summary = {
        "task": task,
        "strategy": strategy,
        "forward_mode": forward_mode,
        "forward_mode_params": dict(forward_mode_params),
        "save_format": save_format,
        "save_checkpoints": bool(save_checkpoints),
        "save_last_epoch": bool(save_last_epoch),
        "ckpt_path": str(best_ckpt_path) if save_checkpoints else None,
        "best_ckpt_path": str(best_ckpt_path) if save_checkpoints else None,
        "last_ckpt_path": str(last_ckpt_path) if last_ckpt_path is not None else None,
        "metrics": best_state.get("metrics", {}),
        "teacher_metrics": teacher_metrics,
        "zero_shot_metrics": best_state.get("zero_shot_metrics", {}),
        "seconds": float(seconds),
        "best_elapsed_seconds": float(best_elapsed_seconds),
        "last_elapsed_seconds": float(last_elapsed_seconds),
        "selected_timing": {
            "text_prestage_seconds": float(text_stage_best_seconds),
            "vision_seconds": float(best_elapsed_seconds),
            "total_seconds": float(text_stage_best_seconds + best_elapsed_seconds),
        },
        "trainable": trainable_info,
        "text_features_init_source": text_features_init_source,
        "initialization": checkpoint_init_summary,
        "text_embeddings_finetune": text_emb_ft_summary,
        "text_prompt_tuning": text_prompt_ft_summary,
        "regularization": {"name": regularizer_name, "info": regularizer_info},
        "best_epoch": best_state.get("best_epoch", -1),
        "last_epoch": int(last_epoch),
        "vision_training_skipped": False,
        "hparams": {
            "epochs": int(epochs),
            "lr": float(lr),
            "optimizer": str(optimizer_name),
            "weight_decay": float(weight_decay),
            # "scheduler": strategy_impl.__name__,
            "warmup_length": int(warmup_length),
            "scheduler_name": str(scheduler_name),
            "clip_grad_norm": float(clip_grad_norm),
            "accumulate_grad_batches": int(accumulate_grad_batches),
            "batch_size": int(batch_size),
            "effective_batch_size": int(batch_size * accumulate_grad_batches),
            "num_workers": int(num_workers),
            "val_fraction": float(val_fraction),
            "train_preprocess": train_preprocess,
            "seed": int(seed),
        },
    }
    _save_json(task_dir / f"{ckpt_stem}.json", summary)

    if save_checkpoints:
        print(f"[{task}] saved best: {best_ckpt_path}")
    if save_checkpoints and last_ckpt_path is not None:
        print(f"[{task}] saved last: {last_ckpt_path}")
    if run_logger is not None:
        run_logger.log_event(
            "task_end",
            metrics={
                f"val/{task}/top1": float(summary["metrics"].get("val_top1", float("nan"))),
                f"test/{task}/top1": float(summary["metrics"].get("test_top1", float("nan"))),
                f"val/{task}/top1_teacher": float(summary["teacher_metrics"].get("val_top1", float("nan"))),
                f"test/{task}/top1_teacher": float(summary["teacher_metrics"].get("test_top1", float("nan"))),
                f"train/{task}/seconds": float(summary["seconds"]),
            },
            context={
                "task": task,
                "summary": summary,
            },
        )
    _close_prepared_regularizer()
    return summary


# ---------------------------
# Main
# ---------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fine-tune from a vision config file (YAML/JSON).")

    g = p.add_argument_group("Config")
    g.add_argument("--vision-config", type=str, required=True, help="Path to vision config (.yaml/.yml/.json).")

    g = p.add_argument_group("Task selection overrides (optional)")
    g.add_argument("--suite", type=str, default=None, choices=sorted(SUITES.keys()))
    g.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names (overrides suite/order).")
    g.add_argument("--reference-suite", type=str, default=None, choices=sorted(SUITES.keys()))
    g.add_argument("--reference-datasets", type=str, default=None, help="Comma-separated regularization/reference dataset names.")

    g = p.add_argument_group("Runtime overrides (optional)")
    g.add_argument("--device", type=str, default=None, help="Override config device, e.g. cuda, cuda:0, cpu, mps.")
    g.add_argument(
        "--train-preprocess",
        type=str,
        default=None,
        choices=("eval", "train"),
        help="Override common.data.train_preprocess for task train loaders.",
    )
    g.add_argument(
        "--vision-loader-profile",
        type=str,
        default=None,
        choices=("hf",),
        help="Override common.data.loader_profile for vision datasets.",
    )
    g.add_argument(
        "--vision-data-root",
        type=str,
        default=None,
        help="Override common.data.data_root for vision datasets.",
    )
    g.add_argument(
        "--skip-existing-task-vectors",
        action="store_true",
        help="Skip datasets whose final task-vector checkpoint already exists.",
    )
    g.add_argument(
        "--force-recompute",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override regularization.force_recompute for cache-backed regularizers.",
    )
    add_logging_args(p)

    return p


def resolve_tasks(args, cfg_file: dict[str, Any]) -> list[str]:
    if args.datasets and args.datasets.strip():
        return parse_csv(args.datasets)
    if args.suite is not None:
        return list(SUITES[args.suite].tasks)

    tasks = _resolve_tasks_from_cfg(cfg_file)
    return tasks if tasks is not None else list(SUITES["vision8"].tasks)


def _resolve_task_init_checkpoint(raw: Any, task: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        path = raw.strip()
        return path or None
    if isinstance(raw, dict):
        value = raw.get(task, None)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                f"initialization.checkpoint['{task}'] must be a string when provided; got {type(value).__name__}."
            )
        path = value.strip()
        return path or None
    raise ValueError("initialization.checkpoint must be a string path or a dict mapping task -> path.")


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Tiny helper to read nested dicts with dot paths."""
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _resolve_dense_lr(cfg: dict[str, Any], *, default_lr: float) -> float:
    optimizer_dense_lr = _get(cfg, "train.optimizer.dense_lr", None)
    if optimizer_dense_lr is not None:
        return float(optimizer_dense_lr)
    train_dense_lr = _get(cfg, "train.dense_lr", None)
    if train_dense_lr is not None:
        return float(train_dense_lr)
    return float(default_lr)


def _resolve_train_preprocess(raw: Any = None, *, cli_value: str | None = None, task: str | None = None) -> str:
    value = cli_value if cli_value is not None else raw
    if value is None:
        return "eval"
    resolved = str(value).strip().lower()
    if resolved not in {"eval", "train"}:
        prefix = f"[{task}] " if task else ""
        raise ValueError(f"{prefix}data.train_preprocess must be one of: eval, train.")
    return resolved


def _resolve_loader_profile(raw: Any = None, *, cli_value: str | None = None, task: str | None = None) -> str:
    value = cli_value if cli_value is not None else raw
    if value is None:
        return "hf"
    resolved = str(value).strip().lower()
    if resolved != "hf":
        prefix = f"[{task}] " if task else ""
        raise ValueError(f"{prefix}data.loader_profile must be 'hf'.")
    return resolved


def _resolve_data_root(
    raw: Any = None,
    *,
    cli_value: str | None = None,
    loader_profile: str,
) -> str | None:
    value = cli_value if cli_value is not None else raw
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved or None


def _task_checkpoint_stem(
    *,
    strategy: str,
    strategy_cfg: dict[str, Any] | None = None,
    regularization_cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    forward_mode = resolve_training_forward_mode(dict(strategy_cfg or {}))
    ckpt_stem = strategy if forward_mode == "standard" else f"{strategy}__{forward_mode}"
    regularization_name = str((regularization_cfg or {}).get("name", "")).strip()
    if regularization_name:
        ckpt_stem = f"{ckpt_stem}__{regularization_name}"
    return ckpt_stem, regularization_name


def _task_checkpoint_paths(
    *,
    out_dir: Path,
    build_cfg: OpenClipBuildConfig,
    task: str,
    strategy: str,
    strategy_cfg: dict[str, Any] | None = None,
    regularization_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ckpt_stem, regularization_name = _task_checkpoint_stem(
        strategy=strategy,
        strategy_cfg=strategy_cfg,
        regularization_cfg=regularization_cfg,
    )
    task_dir = out_dir / build_cfg.model_name / build_cfg.pretrained / task
    return {
        "task_dir": task_dir,
        "ckpt_stem": ckpt_stem,
        "best_ckpt_path": task_dir / f"{ckpt_stem}_best_ep.pt",
        "last_ckpt_path": task_dir / f"{ckpt_stem}_last_ep.pt",
        "summary_path": task_dir / f"{ckpt_stem}.json",
        "regularization_name": regularization_name,
    }


def _build_skipped_existing_task_summary(
    *,
    task: str,
    strategy: str,
    strategy_cfg: dict[str, Any] | None,
    regularization_cfg: dict[str, Any] | None,
    save_format: str,
    save_checkpoints: bool,
    save_last_epoch: bool,
    checkpoint_paths: dict[str, Any],
    existing_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    forward_mode = resolve_training_forward_mode(dict(strategy_cfg or {}))
    forward_mode_params = normalize_forward_mode_params(
        forward_mode,
        (strategy_cfg or {}).get("forward_mode_params", None),
    )
    regularization_name = str((regularization_cfg or {}).get("name", "")).strip()
    summary = existing_summary or {}
    regularization_summary = summary.get("regularization", {})
    regularization_info = {}
    if isinstance(regularization_summary, dict):
        maybe_info = regularization_summary.get("info", {})
        if isinstance(maybe_info, dict):
            regularization_info = dict(maybe_info)
    return {
        "task": task,
        "strategy": strategy,
        "forward_mode": forward_mode,
        "forward_mode_params": dict(forward_mode_params),
        "save_format": save_format,
        "save_checkpoints": bool(save_checkpoints),
        "save_last_epoch": bool(save_last_epoch),
        "ckpt_path": str(checkpoint_paths["best_ckpt_path"]) if save_checkpoints else None,
        "best_ckpt_path": str(checkpoint_paths["best_ckpt_path"]) if save_checkpoints else None,
        "last_ckpt_path": (
            str(checkpoint_paths["last_ckpt_path"]) if checkpoint_paths["last_ckpt_path"].exists() else None
        ),
        "metrics": dict(summary.get("metrics", {})) if isinstance(summary.get("metrics", {}), dict) else {},
        "zero_shot_metrics": (
            dict(summary.get("zero_shot_metrics", {}))
            if isinstance(summary.get("zero_shot_metrics", {}), dict)
            else {}
        ),
        "seconds": 0.0,
        "best_elapsed_seconds": float(summary.get("best_elapsed_seconds", 0.0) or 0.0),
        "last_elapsed_seconds": float(summary.get("last_elapsed_seconds", 0.0) or 0.0),
        "selected_timing": (
            dict(summary.get("selected_timing", {}))
            if isinstance(summary.get("selected_timing", {}), dict)
            else {}
        ),
        "trainable": summary.get("trainable", None),
        "text_features_init_source": summary.get("text_features_init_source", None),
        "initialization": summary.get("initialization", None),
        "text_embeddings_finetune": summary.get("text_embeddings_finetune", None),
        "text_prompt_tuning": summary.get("text_prompt_tuning", None),
        "regularization": {
            "name": regularization_name,
            "info": regularization_info,
        },
        "best_epoch": int(summary.get("best_epoch", -1) or -1),
        "last_epoch": int(summary.get("last_epoch", -1) or -1),
        "vision_training_skipped": True,
        "skip_reason": "existing_task_vector_checkpoint",
        "existing_summary_path": (
            str(checkpoint_paths["summary_path"]) if checkpoint_paths["summary_path"].exists() else None
        ),
        "task_dir": str(checkpoint_paths["task_dir"]),
    }


def main() -> None:
    run_logger = None
    try:
        parser = build_parser()
        args = parser.parse_args()

        cfg_file = _load_config(args.vision_config)
        common = _get_common_cfg(cfg_file)

        # Resolve tasks (datasets/suite override config order)
        training_tasks = resolve_tasks(args, cfg_file)

        # Global cfg is common with selected CLI overrides.
        global_cfg = deepcopy(common)
        reference_resolution_context = _build_reference_task_resolution_context(
            training_tasks=list(training_tasks),
            suite=getattr(args, "suite", None),
            cli_reference_suite=getattr(args, "reference_suite", None),
            cli_reference_datasets=getattr(args, "reference_datasets", None),
        )

        backbone_name = str(_get(global_cfg, "backbone.name", "openclip")).strip().lower()
        if backbone_name not in {"openclip", "openai_clip"}:
            raise ValueError(f"Unsupported backbone '{backbone_name}' (supported: openclip, openai_clip).")

        clip_model = _get(global_cfg, "backbone.clip_model", "ViT-B-32")
        clip_pretrained = _get(global_cfg, "backbone.clip_pretrained", "openai")
        device = str(args.device) if args.device is not None else _get(global_cfg, "device", "cuda")
        dtype = _get(global_cfg, "dtype", None)

        out_dir = Path(
            _apply_reference_tags_to_out_dir(
                out_dir=str(_get(global_cfg, "output.out_dir", "src/checkpoints/finetune")),
                regularization_cfg=_get(global_cfg, "regularization", {}),
                context=reference_resolution_context,
            )
        )
        save_format_default = str(_get(global_cfg, "output.save_format", "full"))
        save_last_epoch_default = bool(_get(global_cfg, "output.save_last_epoch", False))
        logging_cfg = merge_logging_config(_get(global_cfg, "logging", {}), build_logging_overrides(args))
        run_ts = int(time.time())
        run_path = default_summary_path(
            entrypoint="finetune.train_vision",
            logging_cfg=logging_cfg,
            default_parent=out_dir / str(clip_model) / str(clip_pretrained),
            timestamp=run_ts,
        )
        startup_cfg = deepcopy(common)
        startup_cfg["config"] = args.vision_config
        startup_cfg["tasks"] = list(training_tasks)
        startup_cfg["device"] = device
        startup_cfg["dtype"] = dtype
        startup_cfg["logging"] = logging_cfg
        startup_cfg["summary"] = str(run_path)
        startup_cfg.setdefault("backbone", {})
        startup_cfg["backbone"]["name"] = backbone_name
        startup_cfg["backbone"]["clip_model"] = clip_model
        startup_cfg["backbone"]["clip_pretrained"] = clip_pretrained
        if args.train_preprocess is not None:
            startup_cfg.setdefault("data", {})
            startup_cfg["data"]["train_preprocess"] = args.train_preprocess
        if args.vision_loader_profile is not None:
            startup_cfg.setdefault("data", {})
            startup_cfg["data"]["loader_profile"] = args.vision_loader_profile
        if args.vision_data_root is not None:
            startup_cfg.setdefault("data", {})
            startup_cfg["data"]["data_root"] = args.vision_data_root
        if args.force_recompute is not None:
            startup_cfg.setdefault("regularization", {})
            startup_cfg["regularization"]["force_recompute"] = bool(args.force_recompute)
        startup_cfg.setdefault("output", {})
        startup_cfg["output"]["out_dir"] = str(out_dir)
        startup_cfg["output"]["save_format"] = save_format_default
        startup_cfg["output"]["save_last_epoch"] = save_last_epoch_default
        startup_cfg["output"]["skip_existing_task_vectors"] = bool(args.skip_existing_task_vectors)

        all_summaries: dict[str, Any] = {
            "config_path": args.vision_config,
            "common": common,
            "cli": {
                "suite": args.suite,
                "datasets": args.datasets,
                "reference_suite": args.reference_suite,
                "reference_datasets": args.reference_datasets,
                "device": args.device,
                "train_preprocess": args.train_preprocess,
                "vision_loader_profile": args.vision_loader_profile,
                "vision_data_root": args.vision_data_root,
                "force_recompute": args.force_recompute,
                "skip_existing_task_vectors": bool(args.skip_existing_task_vectors),
                "logging": build_logging_overrides(args),
            },
            "resolved": {
                "tasks": training_tasks,
                "training_tasks": training_tasks,
                "build_cfg": {
                    "backbone": backbone_name,
                    "clip_model": clip_model,
                    "clip_pretrained": clip_pretrained,
                    "dtype": dtype,
                    "device": device,
                    "skip_existing_task_vectors": bool(args.skip_existing_task_vectors),
                },
                "run_path": str(run_path),
            },
            "results": {},
        }
        run_logger = start_run(
            entrypoint="finetune.train_vision",
            logging_cfg=logging_cfg,
            summary_path=run_path,
            metadata={
                "config_path": args.vision_config,
                "summary_path": str(run_path),
                "resolved_config": startup_cfg,
            },
        )

        task_runs: list[dict[str, Any]] = []
        for task in training_tasks:
            if task not in VISION_SUPPORTED_TASKS:
                raise ValueError(f"Unknown task '{task}'. Supported: {VISION_SUPPORTED_TASKS}")

            # task_cfg = common -> per-dataset override
            task_cfg = deepcopy(common)
            _deep_update(task_cfg, _get_dataset_override(cfg_file, task))
            task_logging_cfg = merge_logging_config(_get(task_cfg, "logging", {}), build_logging_overrides(args))

            epochs = _get(task_cfg, "train.epochs", None)
            if epochs is None:
                raise ValueError(
                    f"[{task}] train.epochs missing. Set common.train.epochs or datasets.{task}.train.epochs."
                )
            epochs = int(epochs)

            strategy = str(_get(task_cfg, "strategy.name", "full"))
            if strategy not in list_strategies():
                raise ValueError(f"[{task}] Unknown strategy '{strategy}'. Available: {list_strategies()}")
            strategy_cfg = _get(task_cfg, "strategy", {})
            if not isinstance(strategy_cfg, dict):
                raise ValueError(f"[{task}] strategy must be a dict.")
            resolve_training_forward_mode(strategy_cfg)
            regularization_cfg = _get(task_cfg, "regularization", {})
            if regularization_cfg is None:
                regularization_cfg = {}
            if not isinstance(regularization_cfg, dict):
                raise ValueError(f"[{task}] regularization must be a dict when provided.")
            regularization_cfg = dict(regularization_cfg)
            if args.force_recompute is not None:
                regularization_cfg["force_recompute"] = bool(args.force_recompute)
            regularization_name = str(regularization_cfg.get("name", "")).strip()
            if regularization_name and regularization_name not in list_regularizers():
                raise ValueError(
                    f"[{task}] Unknown regularizer '{regularization_name}'. Available: {list_regularizers()}"
                )
            reference_tasks = []
            if regularization_name:
                reference_tasks, _ = resolve_reference_tasks(
                    args,
                    training_tasks=list(training_tasks),
                    regularization_cfg=regularization_cfg,
                    require_reference=True,
                )

            lr = float(_get(task_cfg, "train.lr", 1e-4))
            dense_lr = _resolve_dense_lr(task_cfg, default_lr=lr)
            optimizer_name = str(_get(task_cfg, "train.optimizer.name", "adamw"))
            weight_decay = float(_get(task_cfg, "train.weight_decay", 0.0))
            clip_grad_norm = float(_get(task_cfg, "train.grad_clip_norm", 1.0))
            accumulate_grad_batches = int(_get(task_cfg, "train.accumulate_grad_batches", 1))
            if accumulate_grad_batches <= 0:
                raise ValueError(f"[{task}] train.accumulate_grad_batches must be >= 1.")
            text_only = bool(_get(task_cfg, "train.text_only", False))

            batch_size = int(_get(task_cfg, "data.batch_size", 64))
            num_workers = int(_get(task_cfg, "data.num_workers", 6))
            val_fraction = float(_get(task_cfg, "data.val_fraction", 0.1))
            train_preprocess = _resolve_train_preprocess(
                _get(task_cfg, "data.train_preprocess", "eval"),
                cli_value=args.train_preprocess,
                task=task,
            )
            loader_profile = _resolve_loader_profile(
                _get(task_cfg, "data.loader_profile", None),
                cli_value=args.vision_loader_profile,
                task=task,
            )
            data_root = _resolve_data_root(
                _get(task_cfg, "data.data_root", None),
                cli_value=args.vision_data_root,
                loader_profile=loader_profile,
            )
            seed = int(_get(task_cfg, "seed", 42))
            early_stopping = bool(_get(task_cfg, "train.early_stopping", False))
            early_stopping_patience = int(_get(task_cfg, "train.early_stopping_patience", 5))

            task_out_dir = Path(
                _apply_reference_tags_to_out_dir(
                    out_dir=str(_get(task_cfg, "output.out_dir", str(out_dir))),
                    regularization_cfg=regularization_cfg,
                    context=reference_resolution_context,
                )
            )
            save_format = str(_get(task_cfg, "output.save_format", save_format_default))
            save_checkpoints = bool(_get(task_cfg, "output.save_checkpoints", True))
            save_last_epoch = bool(_get(task_cfg, "output.save_last_epoch", save_last_epoch_default))
            init_checkpoint = _resolve_task_init_checkpoint(_get(task_cfg, "initialization.checkpoint", None), task)
            init_text_features_source = str(_get(task_cfg, "initialization.text_features_source", "zero_shot"))

            hf_path, hf_config, split_map = _vision_spec(task)

            build_cfg = OpenClipBuildConfig(
                loader=backbone_name,
                model_name=str(clip_model),
                pretrained=str(clip_pretrained),
                device=str(device),
                dtype=dtype,
                prompt_templates=get_templates(task),
            )

            task_runs.append(
                {
                    "task": task,
                    "hf_path": hf_path,
                    "hf_config": hf_config,
                    "split_map": split_map,
                    "build_cfg": build_cfg,
                    "strategy": strategy,
                    "epochs": epochs,
                    "lr": lr,
                    "dense_lr": dense_lr,
                    "optimizer_name": optimizer_name,
                    "weight_decay": weight_decay,
                    "warmup_length": int(_get(task_cfg, "train.lr_scheduler.warmup_steps", 500)),
                    "scheduler_name": str(_get(task_cfg, "train.lr_scheduler.name", "cosine")),
                    "clip_grad_norm": clip_grad_norm,
                    "accumulate_grad_batches": accumulate_grad_batches,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "val_fraction": val_fraction,
                    "loader_profile": loader_profile,
                    "data_root": data_root,
                    "train_preprocess": train_preprocess,
                    "early_stopping": early_stopping,
                    "early_stopping_patience": early_stopping_patience,
                    "text_only": text_only,
                    "seed": seed,
                    "device": str(device),
                    "out_dir": task_out_dir,
                    "save_format": save_format,
                    "save_checkpoints": save_checkpoints,
                    "save_last_epoch": save_last_epoch,
                    "init_checkpoint": init_checkpoint,
                    "init_text_features_source": init_text_features_source,
                    "peft_cfg": strategy_cfg.get("peft") if strategy_cfg else None,
                    "strategy_cfg": strategy_cfg,
                    "regularization_cfg": regularization_cfg,
                    "reference_tasks": list(reference_tasks),
                    "reference_resolution_context": reference_resolution_context,
                    "log_every_n_steps": int(task_logging_cfg.get("log_every_n_steps", 50)),
                    "run_logger": run_logger,
                }
            )

        train_preprocess_by_task = {str(run["task"]): str(run["train_preprocess"]) for run in task_runs}
        unique_train_preprocess = set(train_preprocess_by_task.values())
        all_summaries["resolved"]["train_preprocess"] = (
            next(iter(unique_train_preprocess)) if len(unique_train_preprocess) == 1 else train_preprocess_by_task
        )
        loader_profile_by_task = {str(run["task"]): str(run["loader_profile"]) for run in task_runs}
        unique_loader_profiles = set(loader_profile_by_task.values())
        all_summaries["resolved"]["loader_profile"] = (
            next(iter(unique_loader_profiles)) if len(unique_loader_profiles) == 1 else loader_profile_by_task
        )
        data_root_by_task = {str(run["task"]): run.get("data_root", None) for run in task_runs}
        unique_data_roots = {str(value) for value in data_root_by_task.values()}
        all_summaries["resolved"]["data_root"] = (
            next(iter(unique_data_roots)) if len(unique_data_roots) == 1 else data_root_by_task
        )

        reference_tasks_by_task = {
            str(run["task"]): list(run.get("reference_tasks") or [])
            for run in task_runs
            if str(run.get("regularization_cfg", {}).get("name", "")).strip()
        }
        if reference_tasks_by_task:
            unique_reference_sets = {tuple(value) for value in reference_tasks_by_task.values()}
            all_summaries["resolved"]["reference_tasks"] = (
                list(next(iter(unique_reference_sets)))
                if len(unique_reference_sets) == 1
                else reference_tasks_by_task
            )
        else:
            all_summaries["resolved"]["reference_tasks"] = []

        for task_run in task_runs:
            checkpoint_paths = _task_checkpoint_paths(
                out_dir=Path(task_run["out_dir"]),
                build_cfg=task_run["build_cfg"],
                task=str(task_run["task"]),
                strategy=str(task_run["strategy"]),
                strategy_cfg=task_run.get("strategy_cfg"),
                regularization_cfg=task_run.get("regularization_cfg"),
            )
            if bool(args.skip_existing_task_vectors) and bool(task_run["save_checkpoints"]):
                best_ckpt_path = checkpoint_paths["best_ckpt_path"]
                if best_ckpt_path.exists():
                    existing_summary = _load_json_if_exists(checkpoint_paths["summary_path"])
                    summary = _build_skipped_existing_task_summary(
                        task=str(task_run["task"]),
                        strategy=str(task_run["strategy"]),
                        strategy_cfg=task_run.get("strategy_cfg"),
                        regularization_cfg=task_run.get("regularization_cfg"),
                        save_format=str(task_run["save_format"]),
                        save_checkpoints=bool(task_run["save_checkpoints"]),
                        save_last_epoch=bool(task_run["save_last_epoch"]),
                        checkpoint_paths=checkpoint_paths,
                        existing_summary=existing_summary,
                    )
                    print(
                        f"[{task_run['task']}] skipping training because existing task vector checkpoint was found at "
                        f"{best_ckpt_path}"
                    )
                    if run_logger is not None:
                        run_logger.log_event(
                            "task_skipped",
                            metrics={},
                            context={
                                "task": str(task_run["task"]),
                                "reason": "existing_task_vector_checkpoint",
                                "best_ckpt_path": str(best_ckpt_path),
                                "summary": summary,
                            },
                        )
                    all_summaries["results"][str(task_run["task"])] = summary
                    continue
            summary = train_task(
                **task_run,
                all_tasks=list(training_tasks),
            )
            all_summaries["results"][str(task_run["task"])] = summary

        _save_json(run_path, all_summaries)
        run_logger.log_summary(all_summaries)
        run_logger.finish("success")
        print(f"\nSaved run summary: {run_path}")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
