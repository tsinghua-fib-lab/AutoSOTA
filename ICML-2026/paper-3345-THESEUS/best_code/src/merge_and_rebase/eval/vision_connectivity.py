from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from merge_and_rebase.io.peft_helpers import (
    is_peft_adapter_dir_ckpt,
    load_peft_adapter_dir_components,
    normalize_peft_adapter_dir_checkpoint,
)
from merge_and_rebase.io.utils import atomic_write_json
from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import (
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_suite_arg,
    add_tasks_arg,
    build_common_eval_overrides,
    build_logging_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import (
    TaskAttentionMeta,
    assert_qkv_patched_before_linearizing,
    extract_checkpoint_attn_patch_info,
    extract_peft_components,
    get_peft_cfg,
    humanize,
    is_peft_checkpoint,
    load_vision_checkpoint_reference,
    materialize_peft_sd_from_adapter,
    maybe_patch_base_for_task_attn,
    to_cpu_fp32,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..models.forward_modes import (
    get_forward_mode,
    list_forward_modes,
    normalize_forward_mode_params,
    resolve_auto_forward_mode,
    resolve_shared_forward_mode_params,
)
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..run_logging import default_summary_path, merge_logging_config, start_run
from .datasets.vision8_14_20 import SUITES


@dataclass(frozen=True)
class _CheckpointPayload:
    path: str
    obj: Any
    strategy: str | None
    forward_mode: str | None
    forward_mode_params: dict[str, Any] | None
    attn_meta: TaskAttentionMeta
    tuned_text_features: torch.Tensor | None


@dataclass(frozen=True)
class _TaskEvalContext:
    task: str
    loaders: Any
    classnames: list[str]
    build_cfg_task: OpenClipBuildConfig
    text_features: torch.Tensor | None
    text_features_mode: str


@dataclass(frozen=True)
class _TaskStaticContext:
    task: str
    loaders: Any
    classnames: list[str]
    build_cfg_task: OpenClipBuildConfig


def _resolve_two_checkpoints(cfg: dict[str, Any]) -> tuple[str, str]:
    ckpt_a = cfg.get("checkpoint_a", cfg.get("ckpt_a", None))
    ckpt_b = cfg.get("checkpoint_b", cfg.get("ckpt_b", None))
    if ckpt_a is not None and ckpt_b is not None:
        return str(ckpt_a), str(ckpt_b)

    tuned = cfg.get("tuned_ckpts", None)
    if isinstance(tuned, (list, tuple)):
        if len(tuned) != 2:
            raise ValueError("For connectivity, tuned_ckpts must contain exactly two checkpoints.")
        return str(tuned[0]), str(tuned[1])

    if isinstance(tuned, dict):
        for ka, kb in (("a", "b"), ("ckpt_a", "ckpt_b"), ("checkpoint_a", "checkpoint_b")):
            if ka in tuned and kb in tuned:
                return str(tuned[ka]), str(tuned[kb])
        if len(tuned) == 2:
            vals = list(tuned.values())
            return str(vals[0]), str(vals[1])

    raise ValueError(
        "Provide exactly two checkpoints via --checkpoint-a/--checkpoint-b, "
        "or config['checkpoint_a'/'checkpoint_b'], or config['tuned_ckpts'] with 2 entries."
    )


def _build_factor_grid(*, minimum: float, maximum: float, step: float, label: str) -> list[float]:
    if step <= 0:
        raise ValueError(f"{label} step must be > 0. Got {step}.")
    if maximum < minimum:
        raise ValueError(f"{label} maximum must be >= minimum. Got min={minimum}, max={maximum}.")
    vals = torch.arange(minimum, maximum + 1e-12, step).tolist()
    out = sorted({round(float(v), 10) for v in vals})
    if not out:
        raise ValueError(f"{label} grid is empty.")
    return out


def _ensure_line_endpoints(alphas: list[float]) -> list[float]:
    out = sorted({*alphas, 0.0, 1.0})
    return [round(float(v), 10) for v in out]


def _find_alpha_index(alphas: list[float], target: float, *, tol: float = 1e-8) -> int:
    for i, a in enumerate(alphas):
        if abs(float(a) - float(target)) <= tol:
            return i
    raise ValueError(f"Could not find alpha={target} in {alphas}.")


def _load_checkpoint_payload(path: str) -> _CheckpointPayload:
    resolved_path, obj = load_vision_checkpoint_reference(ckpt_ref=path)
    obj = normalize_peft_adapter_dir_checkpoint(obj, checkpoint_path=resolved_path)
    strategy = obj.get("strategy", None) if isinstance(obj, dict) else None
    forward_mode = obj.get("forward_mode", None) if isinstance(obj, dict) else None
    forward_mode_params = (
        normalize_forward_mode_params(str(forward_mode), obj.get("forward_mode_params", None))
        if isinstance(obj, dict) and forward_mode is not None
        else None
    )
    attn_meta = extract_checkpoint_attn_patch_info(obj=obj, ckpt_path=resolved_path)
    tuned_text_features = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(
        obj=obj,
        ckpt_path=resolved_path,
    )
    return _CheckpointPayload(
        path=resolved_path,
        obj=obj,
        strategy=strategy,
        forward_mode=forward_mode,
        forward_mode_params=forward_mode_params,
        attn_meta=attn_meta,
        tuned_text_features=tuned_text_features,
    )


def _check_attn_consistency(ckpt_a: _CheckpointPayload, ckpt_b: _CheckpointPayload) -> None:
    if ckpt_a.attn_meta.patched_attn != ckpt_b.attn_meta.patched_attn:
        raise ValueError("Inconsistent patched_attn flags across the two checkpoints.")
    if not ckpt_a.attn_meta.patched_attn:
        return
    cfg_a = ckpt_a.attn_meta.attn_patch_cfg or {}
    cfg_b = ckpt_b.attn_meta.attn_patch_cfg or {}
    if cfg_a != cfg_b:
        raise ValueError("Inconsistent attn_patch_cfg across the two checkpoints.")


def _load_full_checkpoint_state(
    *,
    ckpt: _CheckpointPayload,
    base_sd: dict[str, torch.Tensor],
    build_cfg: OpenClipBuildConfig,
    strict_load: bool,
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor]
    is_peft = False
    peft_state: dict[str, torch.Tensor] | None = None
    peft_cfg_map: dict[str, Any] | None = None

    obj = ckpt.obj
    if is_peft_adapter_dir_ckpt(obj):
        peft_state, peft_cfg_map = load_peft_adapter_dir_components(obj["peft_adapter_dir"], checkpoint_path=ckpt.path)
        is_peft = True
    elif is_peft_checkpoint(obj) and isinstance(obj, dict):
        peft_state, peft_cfg_map = extract_peft_components(obj)
        is_peft = True

    if is_peft:
        assert peft_state is not None and peft_cfg_map is not None
        sd = materialize_peft_sd_from_adapter(
            peft_state=peft_state,
            base_sd=base_sd,
            build_cfg=build_cfg,
            peft_cfg=get_peft_cfg(peft_cfg_map),
            peft_dense_state=dict(obj.get("peft_dense_state", {})) if isinstance(obj, dict) and isinstance(obj.get("peft_dense_state", {}), dict) else None,
            strict_load=strict_load,
            patched_attn=ckpt.attn_meta.patched_attn,
            attn_patch_cfg=ckpt.attn_meta.attn_patch_cfg,
        )
    else:
        sd = load_ckpt(ckpt.path)

    aligned = align_to_base_keys(sd, base_sd)
    if not aligned:
        raise ValueError(
            f"No tensors from checkpoint aligned to base keys: {ckpt.path}. Check model compatibility or key prefixes."
        )
    full = to_cpu_fp32(base_sd)
    full.update(to_cpu_fp32(aligned))
    return full


def _lerp_states(
    *,
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, va in sd_a.items():
        vb = sd_b[k]
        out[k] = torch.lerp(va, vb, float(alpha))
    return out


def _state_delta(
    *,
    tuned_sd: dict[str, torch.Tensor],
    base_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: tuned_sd[k] - base_sd[k] for k in base_sd}


def _plane_state(
    *,
    base_sd: dict[str, torch.Tensor],
    delta_a: dict[str, torch.Tensor],
    delta_b: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, base_v in base_sd.items():
        out[k] = base_v + float(alpha) * delta_a[k] + float(beta) * delta_b[k]
    return out


@torch.no_grad()
def _eval_loader_top1_and_loss(
    *,
    clf: OpenClipClassifier,
    loader: Any,
    device: str,
    text_features: torch.Tensor | None,
) -> tuple[float, float]:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    clf.to(dev)
    clf.eval()

    prev_text = clf._zs_text_features
    prev_fingerprint = clf._zs_text_fingerprint
    if text_features is not None:
        feats = text_features.to(device=dev)
        if clf.normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        clf._zs_text_features = feats
        clf._zs_text_fingerprint = None

    total = 0
    correct = 0
    loss_sum = 0.0
    try:
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = clf(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    finally:
        if text_features is not None:
            clf._zs_text_features = prev_text
            clf._zs_text_fingerprint = prev_fingerprint

    acc = float(correct / max(1, total))
    avg_loss = float(loss_sum / max(1, total))
    return acc, avg_loss


@torch.no_grad()
def _eval_task_metrics(
    *,
    clf: OpenClipClassifier,
    item: _TaskEvalContext,
    device: str,
    split: str,
) -> tuple[float, float]:
    if split == "val":
        loader = item.loaders.val
    elif split == "test":
        loader = item.loaders.test
    else:
        raise ValueError(f"Unknown split '{split}'. Expected one of: val, test.")

    if item.text_features is None:
        clf.build_zeroshot_text_features(
            item.classnames,
            item.build_cfg_task,
            cache_dir="src/.cache/zs_cache",
            force_rebuild=False,
        )
    return _eval_loader_top1_and_loss(
        clf=clf,
        loader=loader,
        device=device,
        text_features=item.text_features,
    )


@torch.no_grad()
def _eval_suite_metrics(
    *,
    clf: OpenClipClassifier,
    per_task: list[_TaskEvalContext],
    device: str,
    split: str,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    per_task_metrics: dict[str, dict[str, float]] = {}
    avg_acc = 0.0
    avg_loss = 0.0
    for item in per_task:
        acc, loss = _eval_task_metrics(
            clf=clf,
            item=item,
            device=device,
            split=split,
        )
        per_task_metrics[item.task] = {"acc": float(acc), "loss": float(loss)}
        avg_acc += float(acc)
        avg_loss += float(loss)
    n = max(1, len(per_task))
    return float(avg_acc / n), float(avg_loss / n), per_task_metrics


def _write_line_csv(
    *,
    path: Path,
    alphas: list[float],
    avg_acc: list[float],
    avg_loss: list[float],
    per_task_acc: dict[str, list[float]],
    per_task_loss: dict[str, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task_names = sorted(per_task_acc)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["alpha", "avg_acc", "avg_loss"]
        header += [f"acc::{t}" for t in task_names]
        header += [f"loss::{t}" for t in task_names]
        writer.writerow(header)
        for i, alpha in enumerate(alphas):
            row: list[Any] = [alpha, avg_acc[i], avg_loss[i]]
            row.extend(per_task_acc[t][i] for t in task_names)
            row.extend(per_task_loss[t][i] for t in task_names)
            writer.writerow(row)


def _write_heatmap_csv(
    *,
    path: Path,
    factors: list[float],
    matrix: list[list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta\\alpha", *factors])
        for beta, row in zip(factors, matrix, strict=True):
            writer.writerow([beta, *row])


def _maybe_save_line_plots(
    *,
    loss_curve_png: Path,
    accuracy_curve_png: Path,
    line_alphas: list[float],
    line_avg_acc: list[float],
    line_avg_loss: list[float],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Could not import plotting dependencies. Skipping plots. ({exc})")
        return False

    loss_curve_png.parent.mkdir(parents=True, exist_ok=True)
    accuracy_curve_png.parent.mkdir(parents=True, exist_ok=True)

    idx0 = _find_alpha_index(line_alphas, 0.0)
    idx1 = _find_alpha_index(line_alphas, 1.0)
    l0 = float(line_avg_loss[idx0])
    l1 = float(line_avg_loss[idx1])
    baseline = [(1.0 - a) * l0 + a * l1 for a in line_alphas]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(line_alphas, line_avg_loss, marker="o", linewidth=2.0, label="avg CE loss")
    ax.plot(line_alphas, baseline, linestyle="--", linewidth=1.2, label="endpoint interpolation")
    ax.set_xlabel("Interpolation factor (alpha)")
    ax.set_ylabel("Average CE loss")
    ax.set_title("Loss Barrier on Linear Interpolation Path")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_curve_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(line_alphas, line_avg_acc, marker="o", linewidth=2.0, color="tab:green")
    ax.set_xlabel("Interpolation factor (alpha)")
    ax.set_ylabel("Average top-1 accuracy")
    ax.set_title("Accuracy on Linear Interpolation Path")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(accuracy_curve_png, dpi=180)
    plt.close(fig)
    return True


def _maybe_save_heatmap_plots(
    *,
    loss_heatmap_png: Path,
    accuracy_heatmap_png: Path,
    heatmap_factors: list[float],
    heatmap_acc: list[list[float]],
    heatmap_loss: list[list[float]],
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[warn] Could not import plotting dependencies. Skipping plots. ({exc})")
        return False

    loss_heatmap_png.parent.mkdir(parents=True, exist_ok=True)
    accuracy_heatmap_png.parent.mkdir(parents=True, exist_ok=True)

    factors = np.asarray(heatmap_factors, dtype=float)
    loss_mat = np.asarray(heatmap_loss, dtype=float)
    acc_mat = np.asarray(heatmap_acc, dtype=float)
    extent = [float(factors[0]), float(factors[-1]), float(factors[0]), float(factors[-1])]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(loss_mat, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel("alpha (delta from checkpoint A)")
    ax.set_ylabel("beta (delta from checkpoint B)")
    ax.set_title("Linear Mode Connectivity (Average CE loss)")
    fig.colorbar(im, ax=ax, shrink=0.95, label="CE loss")
    fig.tight_layout()
    fig.savefig(loss_heatmap_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(acc_mat, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel("alpha (delta from checkpoint A)")
    ax.set_ylabel("beta (delta from checkpoint B)")
    ax.set_title("Linear Mode Connectivity (Average top-1)")
    fig.colorbar(im, ax=ax, shrink=0.95, label="top-1 accuracy")
    fig.tight_layout()
    fig.savefig(accuracy_heatmap_png, dpi=180)
    plt.close(fig)
    return True


def _resolve_suite_tasks(*, cfg: dict[str, Any], suite_name: str) -> list[str]:
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    allowed = set(SUITES[suite_name].tasks)

    tasks_raw = cfg.get("tasks", "all")
    if isinstance(tasks_raw, str):
        tasks = list(SUITES[suite_name].tasks) if tasks_raw == "all" else parse_csv(tasks_raw)
    elif isinstance(tasks_raw, (list, tuple)):
        tasks = [str(x) for x in tasks_raw]
    else:
        raise ValueError("tasks must be 'all', a CSV string, or a list.")

    bad = [t for t in tasks if t not in allowed]
    if bad:
        raise ValueError(f"Unknown tasks for suite '{suite_name}': {bad}. Allowed: {sorted(allowed)}")

    dedup: list[str] = []
    seen: set[str] = set()
    for t in tasks:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
    return dedup


def _normalize_task_checkpoint_map(cfg: dict[str, Any]) -> dict[str, str]:
    raw = cfg.get("tuned_ckpts_map", None)
    if raw is None and isinstance(cfg.get("tuned_ckpts", None), dict):
        raw = cfg.get("tuned_ckpts")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("tuned_ckpts_map must be a task->checkpoint JSON object.")
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[str(k)] = str(v)
    return out


def _resolve_single_pair_tasks(
    *,
    cfg: dict[str, Any],
    selected_tasks: list[str],
    checkpoint_map: dict[str, str],
) -> tuple[str, str]:
    raw = cfg.get("pair_tasks", None)
    pair: list[str] | None = None
    if raw is not None:
        if isinstance(raw, str):
            pair = parse_csv(raw)
        elif isinstance(raw, (list, tuple)):
            pair = [str(x) for x in raw]
        else:
            raise ValueError("pair_tasks must be a CSV string or a list of exactly two task names.")
    elif str(cfg.get("tasks", "all")) != "all" and len(selected_tasks) == 2:
        pair = list(selected_tasks)
    elif len(checkpoint_map) == 2:
        pair = list(checkpoint_map.keys())

    if pair is None:
        raise ValueError(
            "Single-pair mode requires exactly two tasks. Provide --pair-tasks, "
            "or --tasks with two entries, or tuned_ckpts_map with two task keys."
        )
    if len(pair) != 2:
        raise ValueError(f"pair_tasks must contain exactly two task names. Got {pair}.")
    if pair[0] == pair[1]:
        raise ValueError("pair_tasks must contain two distinct task names.")

    allowed = set(selected_tasks)
    bad = [t for t in pair if t not in allowed]
    if bad:
        raise ValueError(f"pair_tasks must be in selected tasks {selected_tasks}. Invalid: {bad}")
    return pair[0], pair[1]


def _resolve_pair_checkpoints(
    *,
    cfg: dict[str, Any],
    task_a: str,
    task_b: str,
    checkpoint_map: dict[str, str],
    all_pairs: bool,
) -> tuple[str, str]:
    if all_pairs:
        if task_a not in checkpoint_map or task_b not in checkpoint_map:
            raise ValueError(
                f"all_pairs mode requires checkpoint map entries for both tasks in each pair. "
                f"Missing for pair ({task_a}, {task_b})."
            )
        return checkpoint_map[task_a], checkpoint_map[task_b]

    ckpt_a = cfg.get("checkpoint_a", cfg.get("ckpt_a", None))
    ckpt_b = cfg.get("checkpoint_b", cfg.get("ckpt_b", None))
    if ckpt_a is not None and ckpt_b is not None:
        return str(ckpt_a), str(ckpt_b)
    if task_a in checkpoint_map and task_b in checkpoint_map:
        return checkpoint_map[task_a], checkpoint_map[task_b]
    return _resolve_two_checkpoints(cfg)


def _build_task_static_contexts(
    *,
    tasks: list[str],
    suite_name: str,
    preprocess: Any,
    build_cfg: OpenClipBuildConfig,
    cfg: dict[str, Any],
    use_humanized_classnames: bool,
) -> dict[str, _TaskStaticContext]:
    suite = SUITES[suite_name]
    out: dict[str, _TaskStaticContext] = {}
    for task in tasks:
        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=preprocess,
            ft_epochs=1,
            split_map=split_map,
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 6)),
            pin_memory=True,
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            seed=int(cfg.get("seed", 42)),
        )

        classnames = list(loaders.classnames)
        if use_humanized_classnames:
            classnames = [humanize(c) for c in classnames]

        templates = get_templates(task)
        if not templates:
            raise ValueError(f"get_templates('{task}') returned empty list")
        build_cfg_task = OpenClipBuildConfig(
            model_name=build_cfg.model_name,
            pretrained=build_cfg.pretrained,
            device=build_cfg.device,
            dtype=build_cfg.dtype,
            prompt_templates=templates,
        )
        out[task] = _TaskStaticContext(
            task=str(task),
            loaders=loaders,
            classnames=classnames,
            build_cfg_task=build_cfg_task,
        )
    return out


def _build_eval_contexts_for_pair(
    *,
    clf: OpenClipClassifier,
    pair_tasks: tuple[str, str],
    task_static_by_name: dict[str, _TaskStaticContext],
    text_features_source: str,
    tuned_text_features_by_task: dict[str, torch.Tensor | None],
    tuned_ckpt_by_task: dict[str, str],
) -> list[_TaskEvalContext]:
    per_task: list[_TaskEvalContext] = []
    for task in pair_tasks:
        static = task_static_by_name[task]
        task_text_features, mode = clf.resolve_eval_text_features(
            text_features_source=text_features_source,
            classnames=static.classnames,
            build_cfg=static.build_cfg_task,
            tuned_text_features=tuned_text_features_by_task.get(task, None),
            cache_dir="src/.cache/zs_cache",
            force_rebuild_zeroshot=False,
            task_name=task,
            ckpt_path=tuned_ckpt_by_task.get(task, "<unknown>"),
            verbose=True,
        )
        if task_text_features is not None and int(task_text_features.shape[0]) != int(len(static.classnames)):
            if text_features_source == "auto":
                print(
                    f"{task}: tuned_text_features rows ({int(task_text_features.shape[0])}) do not match "
                    f"class count ({len(static.classnames)}), falling back to zero-shot text features."
                )
                task_text_features, mode = clf.resolve_eval_text_features(
                    text_features_source="zero_shot",
                    classnames=static.classnames,
                    build_cfg=static.build_cfg_task,
                    tuned_text_features=None,
                    cache_dir="src/.cache/zs_cache",
                    force_rebuild_zeroshot=False,
                    task_name=task,
                    ckpt_path=None,
                    verbose=False,
                )
            else:
                raise ValueError(
                    f"Task '{task}' class count ({len(static.classnames)}) does not match tuned_text_features rows "
                    f"({int(task_text_features.shape[0])}). "
                    "Use text_features_source='zero_shot' or text_features_checkpoint='pair_task'."
                )

        per_task.append(
            _TaskEvalContext(
                task=task,
                loaders=static.loaders,
                classnames=static.classnames,
                build_cfg_task=static.build_cfg_task,
                text_features=task_text_features,
                text_features_mode=mode,
            )
        )
    return per_task


def _safe_name(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(s))
    out = out.strip("_")
    return out if out else "task"


def _paper_barrier(
    *,
    values: list[float],
    alphas: list[float],
    idx_alpha0: int,
    idx_alpha1: int,
) -> tuple[float, float, list[float]]:
    """Barrier from the interpolation chord, as in linear mode connectivity papers."""
    if len(values) != len(alphas):
        raise ValueError("values and alphas must have the same length.")
    v0 = float(values[idx_alpha0])
    v1 = float(values[idx_alpha1])

    curve = [float(v - ((1.0 - a) * v0 + a * v1)) for v, a in zip(values, alphas, strict=True)]
    in_01 = [i for i, a in enumerate(alphas) if 0.0 <= float(a) <= 1.0]
    if not in_01:
        raise ValueError("Need at least one alpha in [0, 1] to compute paper barrier.")
    best_idx = max(in_01, key=lambda i: curve[i])
    return float(curve[best_idx]), float(alphas[best_idx]), [float(x) for x in curve]


def _run_pair_analysis(
    *,
    suite_name: str,
    pair_tasks: tuple[str, str],
    pair_ckpt_by_task: dict[str, str],
    ckpt_a_path: str,
    ckpt_b_path: str,
    task_static_by_name: dict[str, _TaskStaticContext],
    build_cfg: OpenClipBuildConfig,
    base_ckpt: str | None,
    strict_load: bool,
    requested_forward_mode: str,
    text_features_source: str,
    text_features_checkpoint: str,
    classnames_mode: str,
    eval_split: str,
    device: str,
    line_alphas: list[float],
    heatmap_factors: list[float],
    run_heatmap: bool,
    output_dir: Path,
    save_plots: bool,
) -> dict[str, Any]:
    task_a, task_b = pair_tasks
    pair_tag = f"{_safe_name(task_a)}__{_safe_name(task_b)}"
    metrics_json = output_dir / f"metrics__{pair_tag}.json"
    line_csv = output_dir / f"line_metrics__{pair_tag}.csv"
    heat_loss_csv = output_dir / f"heatmap_avg_loss__{pair_tag}.csv"
    heat_acc_csv = output_dir / f"heatmap_avg_acc__{pair_tag}.csv"
    loss_curve_png = output_dir / f"loss_barrier_curve__{pair_tag}.png"
    line_acc_curve_png = output_dir / f"line_accuracy_curve__{pair_tag}.png"
    lmc_loss_png = output_dir / f"lmc_loss_heatmap__{pair_tag}.png"
    lmc_acc_png = output_dir / f"lmc_accuracy_heatmap__{pair_tag}.png"
    for ckpt in (ckpt_a_path, ckpt_b_path):
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")

    print(f"\nPair tasks: {task_a}, {task_b}")
    print(f"Checkpoints: A={ckpt_a_path}  B={ckpt_b_path}")

    clf = OpenClipClassifier.build(build_cfg)

    if base_ckpt is None:
        print(f"Using open_clip {build_cfg.model_name} (pretrain={build_cfg.pretrained}) weights as base checkpoint")
        base_sd = to_cpu_fp32({k: v.detach().cpu() for k, v in clf.model.state_dict().items()})
    else:
        print(f"Loading base checkpoint from {base_ckpt}")
        base_sd = to_cpu_fp32(load_ckpt(str(base_ckpt)))

    payload_a = _load_checkpoint_payload(ckpt_a_path)
    payload_b = _load_checkpoint_payload(ckpt_b_path)
    _check_attn_consistency(payload_a, payload_b)

    base_patched_for_attn = False
    base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
        task_meta=payload_a.attn_meta,
        base_patched_for_attn=base_patched_for_attn,
        clf=clf,
        base_ckpt=base_ckpt,
        strict_load=strict_load,
        base_sd=base_sd,
    )
    base_sd = to_cpu_fp32(base_sd)

    tuned_sd_a = _load_full_checkpoint_state(
        ckpt=payload_a,
        base_sd=base_sd,
        build_cfg=build_cfg,
        strict_load=strict_load,
    )
    tuned_sd_b = _load_full_checkpoint_state(
        ckpt=payload_b,
        base_sd=base_sd,
        build_cfg=build_cfg,
        strict_load=strict_load,
    )

    if set(tuned_sd_a) != set(tuned_sd_b):
        raise ValueError("Checkpoint A/B keyspaces differ after alignment; cannot interpolate safely.")
    if set(tuned_sd_a) != set(base_sd):
        raise ValueError("Aligned checkpoint keyspace differs from base keyspace; cannot build interpolation path.")

    needs_linear_attention = bool(payload_a.attn_meta.linearized_attn or payload_b.attn_meta.linearized_attn)
    assert_qkv_patched_before_linearizing(
        needs_linear_attention=needs_linear_attention,
        base_patched_for_attn=base_patched_for_attn,
        model_state_dict=clf.model.state_dict(),
    )
    if needs_linear_attention:
        print("Verified q/k/v attention patch is active before linearized attention evaluation.")

    if requested_forward_mode == "auto":
        resolved_forward_mode = resolve_auto_forward_mode([payload_a.forward_mode, payload_b.forward_mode])
    else:
        resolved_forward_mode = requested_forward_mode
    resolved_forward_mode_params = resolve_shared_forward_mode_params(
        resolved_forward_mode,
        [
            payload.forward_mode_params
            for payload in (payload_a, payload_b)
            if payload.forward_mode == "linearized_ntk"
        ],
    )
    forward_mode = get_forward_mode(resolved_forward_mode)
    forward_mode.bind(clf=clf, base_sd=base_sd, strict_load=strict_load, params=resolved_forward_mode_params)
    print(f"Using forward mode: {resolved_forward_mode} params={resolved_forward_mode_params}")

    if text_features_checkpoint == "pair_task":
        if task_a not in pair_ckpt_by_task or task_b not in pair_ckpt_by_task:
            raise ValueError(
                "pair_task text-feature mode requires an explicit task->checkpoint mapping for the current pair."
            )

        payload_by_ckpt: dict[str, _CheckpointPayload] = {
            ckpt_a_path: payload_a,
            ckpt_b_path: payload_b,
        }
        tuned_text_features_by_task: dict[str, torch.Tensor | None] = {}
        tuned_ckpt_by_task: dict[str, str] = {}
        for t in (task_a, task_b):
            t_ckpt = str(pair_ckpt_by_task[t])
            payload = payload_by_ckpt.get(t_ckpt, None)
            if payload is None:
                payload = _load_checkpoint_payload(t_ckpt)
                payload_by_ckpt[t_ckpt] = payload
            tuned_text_features_by_task[t] = payload.tuned_text_features
            tuned_ckpt_by_task[t] = t_ckpt
    else:
        shared = payload_a.tuned_text_features if text_features_checkpoint == "a" else payload_b.tuned_text_features
        shared_path = ckpt_a_path if text_features_checkpoint == "a" else ckpt_b_path
        tuned_text_features_by_task = {
            task_a: shared,
            task_b: shared,
        }
        tuned_ckpt_by_task = {
            task_a: shared_path,
            task_b: shared_path,
        }

    per_task = _build_eval_contexts_for_pair(
        clf=clf,
        pair_tasks=pair_tasks,
        task_static_by_name=task_static_by_name,
        text_features_source=text_features_source,
        tuned_text_features_by_task=tuned_text_features_by_task,
        tuned_ckpt_by_task=tuned_ckpt_by_task,
    )

    line_avg_acc: list[float] = []
    line_avg_loss: list[float] = []
    line_per_task_acc: dict[str, list[float]] = {item.task: [] for item in per_task}
    line_per_task_loss: dict[str, list[float]] = {item.task: [] for item in per_task}

    print(f"\nEvaluating interpolation line on {len(line_alphas)} points (split={eval_split})")
    t_line_0 = time.time()
    for i, alpha in enumerate(line_alphas, start=1):
        merged_sd = _lerp_states(sd_a=tuned_sd_a, sd_b=tuned_sd_b, alpha=alpha)
        load_into_model(clf.model, merged_sd, strict=strict_load)
        del merged_sd
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()

        avg_acc, avg_loss, per_task_metrics = _eval_suite_metrics(
            clf=clf,
            per_task=per_task,
            device=device,
            split=eval_split,
        )
        line_avg_acc.append(float(avg_acc))
        line_avg_loss.append(float(avg_loss))
        for t in line_per_task_acc:
            line_per_task_acc[t].append(float(per_task_metrics[t]["acc"]))
            line_per_task_loss[t].append(float(per_task_metrics[t]["loss"]))
        print(f"[line {i:>3}/{len(line_alphas)}] alpha={alpha:.4f}  avg_acc={avg_acc:.6f}  avg_loss={avg_loss:.6f}")
    line_seconds = time.time() - t_line_0

    idx_alpha0 = _find_alpha_index(line_alphas, 0.0)
    idx_alpha1 = _find_alpha_index(line_alphas, 1.0)

    avg_loss_barrier, avg_loss_barrier_alpha, avg_loss_barrier_curve = _paper_barrier(
        values=line_avg_loss,
        alphas=line_alphas,
        idx_alpha0=idx_alpha0,
        idx_alpha1=idx_alpha1,
    )
    avg_error = [1.0 - a for a in line_avg_acc]
    avg_error_barrier, avg_error_barrier_alpha, avg_error_barrier_curve = _paper_barrier(
        values=avg_error,
        alphas=line_alphas,
        idx_alpha0=idx_alpha0,
        idx_alpha1=idx_alpha1,
    )

    line_barrier_by_task: dict[str, dict[str, Any]] = {}
    for t in line_per_task_loss:
        losses = line_per_task_loss[t]
        errs = [1.0 - a for a in line_per_task_acc[t]]
        l_barrier, l_alpha, l_curve = _paper_barrier(
            values=losses,
            alphas=line_alphas,
            idx_alpha0=idx_alpha0,
            idx_alpha1=idx_alpha1,
        )
        e_barrier, e_alpha, e_curve = _paper_barrier(
            values=errs,
            alphas=line_alphas,
            idx_alpha0=idx_alpha0,
            idx_alpha1=idx_alpha1,
        )
        line_barrier_by_task[t] = {
            "loss_barrier": float(l_barrier),
            "loss_barrier_alpha": float(l_alpha),
            "error_barrier": float(e_barrier),
            "error_barrier_alpha": float(e_alpha),
            "loss_barrier_curve": [float(x) for x in l_curve],
            "error_barrier_curve": [float(x) for x in e_curve],
            "endpoint_loss_alpha0": float(losses[idx_alpha0]),
            "endpoint_loss_alpha1": float(losses[idx_alpha1]),
            "endpoint_error_alpha0": float(errs[idx_alpha0]),
            "endpoint_error_alpha1": float(errs[idx_alpha1]),
            "max_loss": float(max(losses)),
            "max_error": float(max(errs)),
        }

    print(
        f"\nLine summary: avg_loss_barrier={avg_loss_barrier:.6f} "
        f"avg_error_barrier={avg_error_barrier:.6f}  seconds={line_seconds:.2f}"
    )

    line_plots_written = False
    if save_plots:
        line_plots_written = _maybe_save_line_plots(
            loss_curve_png=loss_curve_png,
            accuracy_curve_png=line_acc_curve_png,
            line_alphas=line_alphas,
            line_avg_acc=line_avg_acc,
            line_avg_loss=line_avg_loss,
        )
        if line_plots_written:
            print(f"Saved line plots to: {loss_curve_png.parent}")

    heatmap_avg_acc: list[list[float]] = []
    heatmap_avg_loss: list[list[float]] = []
    heatmap_task_acc: dict[str, list[list[float]]] = {}
    heatmap_task_loss: dict[str, list[list[float]]] = {}
    heatmap_seconds = 0.0
    consistency: dict[str, float] = {}

    if run_heatmap:
        print(
            f"\nEvaluating connectivity heatmap on {len(heatmap_factors)}x{len(heatmap_factors)} points "
            f"(split={eval_split})"
        )
        delta_a = _state_delta(tuned_sd=tuned_sd_a, base_sd=base_sd)
        delta_b = _state_delta(tuned_sd=tuned_sd_b, base_sd=base_sd)

        heatmap_task_acc = {item.task: [[0.0 for _ in heatmap_factors] for _ in heatmap_factors] for item in per_task}
        heatmap_task_loss = {item.task: [[0.0 for _ in heatmap_factors] for _ in heatmap_factors] for item in per_task}

        total_points = len(heatmap_factors) * len(heatmap_factors)
        point_idx = 0
        t_heat_0 = time.time()
        for iy, beta in enumerate(heatmap_factors):
            row_acc: list[float] = []
            row_loss: list[float] = []
            for ix, alpha in enumerate(heatmap_factors):
                point_idx += 1
                merged_sd = _plane_state(
                    base_sd=base_sd,
                    delta_a=delta_a,
                    delta_b=delta_b,
                    alpha=alpha,
                    beta=beta,
                )
                load_into_model(clf.model, merged_sd, strict=strict_load)
                del merged_sd
                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.empty_cache()

                avg_acc, avg_loss, per_task_metrics = _eval_suite_metrics(
                    clf=clf,
                    per_task=per_task,
                    device=device,
                    split=eval_split,
                )
                row_acc.append(float(avg_acc))
                row_loss.append(float(avg_loss))
                for t in heatmap_task_acc:
                    heatmap_task_acc[t][iy][ix] = float(per_task_metrics[t]["acc"])
                    heatmap_task_loss[t][iy][ix] = float(per_task_metrics[t]["loss"])
                print(
                    f"[heat {point_idx:>4}/{total_points}] alpha={alpha:.4f} beta={beta:.4f} "
                    f"avg_acc={avg_acc:.6f} avg_loss={avg_loss:.6f}"
                )
            heatmap_avg_acc.append(row_acc)
            heatmap_avg_loss.append(row_loss)
        heatmap_seconds = time.time() - t_heat_0
        print(f"Heatmap evaluation completed in {heatmap_seconds:.2f}s")

        # Same-state consistency checks between line and heatmap paths:
        # (alpha=1,beta=0) should match line(alpha=0) == checkpoint A
        # (alpha=0,beta=1) should match line(alpha=1) == checkpoint B
        # for deterministic evaluation.
        try:
            line_i0 = _find_alpha_index(line_alphas, 0.0)
            line_i1 = _find_alpha_index(line_alphas, 1.0)
            heat_ix0 = _find_alpha_index(heatmap_factors, 0.0)
            heat_ix1 = _find_alpha_index(heatmap_factors, 1.0)
            heat_iy0 = heat_ix0
            heat_iy1 = heat_ix1

            d_a = abs(float(heatmap_avg_loss[heat_iy0][heat_ix1]) - float(line_avg_loss[line_i0]))
            d_b = abs(float(heatmap_avg_loss[heat_iy1][heat_ix0]) - float(line_avg_loss[line_i1]))
            consistency["loss_absdiff_heat10_vs_line0"] = float(d_a)
            consistency["loss_absdiff_heat01_vs_line1"] = float(d_b)
            consistency["max_loss_absdiff"] = float(max(d_a, d_b))
            if max(d_a, d_b) > 1e-5:
                print(
                    "[warn] Heatmap/line consistency check failed: "
                    f"|H(1,0)-L(0)|={d_a:.6e}, |H(0,1)-L(1)|={d_b:.6e}. "
                    "This suggests evaluation-path non-determinism or hidden state."
                )
        except ValueError:
            pass
    else:
        print("\nSkipping connectivity heatmap (run_heatmap=false).")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_line_csv(
        path=line_csv,
        alphas=line_alphas,
        avg_acc=line_avg_acc,
        avg_loss=line_avg_loss,
        per_task_acc=line_per_task_acc,
        per_task_loss=line_per_task_loss,
    )
    if run_heatmap:
        _write_heatmap_csv(path=heat_loss_csv, factors=heatmap_factors, matrix=heatmap_avg_loss)
        _write_heatmap_csv(path=heat_acc_csv, factors=heatmap_factors, matrix=heatmap_avg_acc)

    line_per_task_serialized: dict[str, dict[str, Any]] = {}
    for t in line_per_task_acc:
        line_per_task_serialized[t] = {
            "acc": [float(x) for x in line_per_task_acc[t]],
            "loss": [float(x) for x in line_per_task_loss[t]],
            **line_barrier_by_task[t],
        }

    results: dict[str, Any] = {
        "suite": suite_name,
        "tasks": [task_a, task_b],
        "checkpoints": {
            "a": ckpt_a_path,
            "b": ckpt_b_path,
            "base": str(base_ckpt) if base_ckpt is not None else "open_clip_pretrained",
        },
        "settings": {
            "device": device,
            "dtype": build_cfg.dtype,
            "eval_split": eval_split,
            "text_features_source": text_features_source,
            "text_features_checkpoint": text_features_checkpoint,
            "forward_mode": resolved_forward_mode,
            "forward_mode_params": dict(resolved_forward_mode_params),
            "classnames_mode": classnames_mode,
        },
        "line": {
            "alphas": [float(a) for a in line_alphas],
            "avg_acc": [float(v) for v in line_avg_acc],
            "avg_loss": [float(v) for v in line_avg_loss],
            "avg_error": [float(v) for v in avg_error],
            "barrier": {
                "avg_loss_barrier": float(avg_loss_barrier),
                "avg_loss_barrier_alpha": float(avg_loss_barrier_alpha),
                "avg_error_barrier": float(avg_error_barrier),
                "avg_error_barrier_alpha": float(avg_error_barrier_alpha),
                "avg_loss_barrier_curve": [float(x) for x in avg_loss_barrier_curve],
                "avg_error_barrier_curve": [float(x) for x in avg_error_barrier_curve],
                "avg_endpoint_loss_alpha0": float(line_avg_loss[idx_alpha0]),
                "avg_endpoint_loss_alpha1": float(line_avg_loss[idx_alpha1]),
                "avg_endpoint_error_alpha0": float(avg_error[idx_alpha0]),
                "avg_endpoint_error_alpha1": float(avg_error[idx_alpha1]),
                "avg_max_loss": float(max(line_avg_loss)),
                "avg_max_error": float(max(avg_error)),
            },
            "per_task": line_per_task_serialized,
            "seconds": float(line_seconds),
        },
        "heatmap": {
            "enabled": bool(run_heatmap),
            "factors": [float(x) for x in heatmap_factors] if run_heatmap else [],
            "avg_acc": [[float(v) for v in row] for row in heatmap_avg_acc] if run_heatmap else [],
            "avg_loss": [[float(v) for v in row] for row in heatmap_avg_loss] if run_heatmap else [],
            "per_task_acc": heatmap_task_acc,
            "per_task_loss": heatmap_task_loss,
            "seconds": float(heatmap_seconds),
        },
        "consistency_checks": consistency,
        "files": {
            "metrics_json": str(metrics_json),
            "line_csv": str(line_csv),
        },
        "created_unix": int(time.time()),
    }
    if run_heatmap:
        results["files"]["heatmap_avg_loss_csv"] = str(heat_loss_csv)
        results["files"]["heatmap_avg_acc_csv"] = str(heat_acc_csv)
    atomic_write_json(str(metrics_json), results)

    heatmap_plots_written = False
    if save_plots and run_heatmap:
        heatmap_plots_written = _maybe_save_heatmap_plots(
            loss_heatmap_png=lmc_loss_png,
            accuracy_heatmap_png=lmc_acc_png,
            heatmap_factors=heatmap_factors,
            heatmap_acc=heatmap_avg_acc,
            heatmap_loss=heatmap_avg_loss,
        )
        if line_plots_written:
            results["files"]["loss_barrier_curve_png"] = str(loss_curve_png)
            results["files"]["line_accuracy_curve_png"] = str(line_acc_curve_png)
        if heatmap_plots_written:
            results["files"]["lmc_loss_heatmap_png"] = str(lmc_loss_png)
            results["files"]["lmc_accuracy_heatmap_png"] = str(lmc_acc_png)
        if line_plots_written or heatmap_plots_written:
            atomic_write_json(str(metrics_json), results)

    print(f"\nSaved raw metrics JSON to: {metrics_json}")
    if run_heatmap:
        print(f"Saved CSVs to: {line_csv}, {heat_loss_csv}, {heat_acc_csv}")
    else:
        print(f"Saved CSV to: {line_csv}")
    if save_plots:
        print(f"Saved plots (if available) to: {output_dir}")

    del clf
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.empty_cache()
    return results


def _write_pairs_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_a",
                "task_b",
                "checkpoint_a",
                "checkpoint_b",
                "avg_loss_barrier",
                "avg_error_barrier",
                "metrics_json",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["task_a"],
                    r["task_b"],
                    r["checkpoint_a"],
                    r["checkpoint_b"],
                    r["avg_loss_barrier"],
                    r["avg_error_barrier"],
                    r["metrics_json"],
                ]
            )


def main() -> None:
    run_logger = None
    p = argparse.ArgumentParser(
        "Analyze interpolation barriers and linear mode connectivity between two vision checkpoints."
    )
    add_config_arg(p)
    add_suite_arg(p, choices=sorted(SUITES.keys()))
    add_tasks_arg(
        p,
        help_text="CSV task list or 'all'. In single mode these are candidate tasks; in all_pairs mode this is the pair pool.",
    )

    p.add_argument("--clip-model", type=str, default=None)
    p.add_argument("--clip-pretrained", type=str, default=None)
    add_device_dtype_args(p, device_default="cuda", dtype_default=None)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-split", type=str, choices=["val", "test"], default=None)
    p.add_argument("--all-pairs", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--pair-tasks",
        type=str,
        default=None,
        help="Single-pair mode only. CSV with exactly two tasks defining the joint evaluation dataset.",
    )
    p.add_argument(
        "--no-humanize",
        action="store_true",
        default=None,
        help="Use raw classnames. Default is raw classnames for training/eval consistency.",
    )

    p.add_argument("--base-ckpt", type=str, default=None)
    p.add_argument("--checkpoint-a", type=str, default=None)
    p.add_argument("--checkpoint-b", type=str, default=None)
    p.add_argument(
        "--tuned-ckpts",
        type=str,
        nargs="+",
        default=None,
        help="Single-pair fallback. Two checkpoint paths. For all_pairs use tuned_ckpts map in config or --tuned-ckpts-map.",
    )
    p.add_argument(
        "--tuned-ckpts-map",
        type=str,
        default=None,
        help="JSON object task->checkpoint, used by all_pairs mode (and optional in single-pair mode).",
    )
    p.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument(
        "--text-features-source",
        type=str,
        choices=["auto", "zero_shot", "tuned_ckpt"],
        default=None,
        help="Text features source for evaluation.",
    )
    p.add_argument(
        "--text-features-checkpoint",
        type=str,
        choices=["pair_task", "a", "b"],
        default=None,
        help=(
            "Which checkpoint provides tuned_text_features in tuned_ckpt/auto mode: "
            "'pair_task' (task A uses checkpoint A, task B uses checkpoint B), or force 'a'/'b'."
        ),
    )

    p.add_argument("--alpha-min", type=float, default=None)
    p.add_argument("--alpha-max", type=float, default=None)
    p.add_argument("--alpha-step", type=float, default=None)
    p.add_argument("--heatmap-alpha-min", type=float, default=None)
    p.add_argument("--heatmap-alpha-max", type=float, default=None)
    p.add_argument("--heatmap-alpha-step", type=float, default=None)
    p.add_argument(
        "--run-heatmap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run 2D alpha/beta connectivity heatmap. Disable for line/barrier-only runs.",
    )

    p.add_argument("--forward-mode", type=str, choices=["auto", *list_forward_modes()], default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=None)
    add_logging_args(p)

    args = p.parse_args()
    tuned_ckpts_map_cli = parse_json_object_arg(args.tuned_ckpts_map, arg_name="--tuned-ckpts-map")

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)

    cli_overrides: dict[str, Any] = {
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "all_pairs": args.all_pairs,
        "pair_tasks": args.pair_tasks,
        "no_humanize": args.no_humanize,
        "base_ckpt": args.base_ckpt,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "tuned_ckpts": args.tuned_ckpts,
        "tuned_ckpts_map": tuned_ckpts_map_cli,
        "strict_load": args.strict_load,
        "text_features_source": args.text_features_source,
        "text_features_checkpoint": args.text_features_checkpoint,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "heatmap_alpha_min": args.heatmap_alpha_min,
        "heatmap_alpha_max": args.heatmap_alpha_max,
        "heatmap_alpha_step": args.heatmap_alpha_step,
        "run_heatmap": args.run_heatmap,
        "forward_mode": args.forward_mode,
        "output_dir": args.output_dir,
        "save_plots": args.save_plots,
    }
    cli_overrides = merge_non_none(cli_overrides, build_common_eval_overrides(args))
    cfg = merge_non_none(cfg, cli_overrides)
    logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
    cfg["logging"] = logging_cfg

    suite_name = str(cfg.get("suite", "vision8"))
    selected_tasks = _resolve_suite_tasks(cfg=cfg, suite_name=suite_name)
    checkpoint_map = _normalize_task_checkpoint_map(cfg)
    all_pairs = bool(cfg.get("all_pairs", False))

    if all_pairs:
        if len(selected_tasks) < 2:
            raise ValueError("all_pairs mode requires at least 2 tasks.")
        missing = [t for t in selected_tasks if t not in checkpoint_map]
        if missing:
            raise ValueError(
                "all_pairs mode requires checkpoint paths for every selected task. "
                f"Missing: {missing}. Present keys: {sorted(checkpoint_map)}"
            )
        pair_list = list(combinations(selected_tasks, 2))
    else:
        pair_list = [_resolve_single_pair_tasks(cfg=cfg, selected_tasks=selected_tasks, checkpoint_map=checkpoint_map)]

    alpha_min = float(cfg.get("alpha_min", 0.0))
    alpha_max = float(cfg.get("alpha_max", 1.0))
    alpha_step = float(cfg.get("alpha_step", 0.1))
    line_alphas = _ensure_line_endpoints(
        _build_factor_grid(minimum=alpha_min, maximum=alpha_max, step=alpha_step, label="alpha")
    )
    run_heatmap = bool(cfg.get("run_heatmap", True))
    if run_heatmap:
        hm_min = float(cfg.get("heatmap_alpha_min", 0.0))
        hm_max = float(cfg.get("heatmap_alpha_max", 1.0))
        hm_step = float(cfg.get("heatmap_alpha_step", 0.1))
        heatmap_factors = _build_factor_grid(minimum=hm_min, maximum=hm_max, step=hm_step, label="heatmap alpha")
    else:
        heatmap_factors = []

    eval_split = str(cfg.get("eval_split", "val")).strip().lower()
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be one of: val, test")
    text_features_source = str(cfg.get("text_features_source", "auto")).strip().lower()
    if text_features_source not in {"auto", "zero_shot", "tuned_ckpt"}:
        raise ValueError("text_features_source must be one of: auto, zero_shot, tuned_ckpt")
    text_features_checkpoint = str(cfg.get("text_features_checkpoint", "pair_task")).strip().lower()
    if text_features_checkpoint not in {"pair_task", "a", "b"}:
        raise ValueError("text_features_checkpoint must be one of: pair_task, a, b")

    use_humanized_classnames = not bool(cfg.get("no_humanize", True))
    classnames_mode = "humanized" if use_humanized_classnames else "raw"
    print(f"Classname mode: {classnames_mode}")
    print(f"Text features source: {text_features_source} (checkpoint mode={text_features_checkpoint})")
    print(f"Heatmap enabled: {run_heatmap}")

    build_cfg = OpenClipBuildConfig(
        model_name=cfg.get("clip_model", "ViT-B-32"),
        pretrained=cfg.get("clip_pretrained", "openai"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", None),
    )
    base_ckpt = cfg.get("base_ckpt", None)
    strict_load = bool(cfg.get("strict_load", False))
    requested_forward_mode = str(cfg.get("forward_mode", "auto"))
    device = str(cfg.get("device", "cuda"))
    output_root = Path(str(cfg.get("output_dir", "src/.cache/vision_connectivity")))
    save_plots = bool(cfg.get("save_plots", True))
    run_summary_path = default_summary_path(
        entrypoint="eval.vision_connectivity",
        logging_cfg=logging_cfg,
        default_parent=output_root,
    )
    run_logger = start_run(
        entrypoint="eval.vision_connectivity",
        logging_cfg=logging_cfg,
        summary_path=run_summary_path,
        metadata={
            "config_path": args.config,
            "resolved_config": cfg,
            "summary_path": str(run_summary_path),
        },
    )

    tasks_to_prepare = sorted({t for pair in pair_list for t in pair})
    print(f"Preparing static task contexts for: {tasks_to_prepare}")
    data_cfg = OpenClipBuildConfig(
        model_name=build_cfg.model_name,
        pretrained=build_cfg.pretrained,
        device="cpu",
        dtype="fp32",
    )
    clf_data = OpenClipClassifier.build(data_cfg)
    task_static_by_name = _build_task_static_contexts(
        tasks=tasks_to_prepare,
        suite_name=suite_name,
        preprocess=clf_data.preprocess,
        build_cfg=build_cfg,
        cfg=cfg,
        use_humanized_classnames=use_humanized_classnames,
    )
    del clf_data

    pair_summaries: list[dict[str, Any]] = []
    for idx, pair in enumerate(pair_list, start=1):
        task_a, task_b = pair
        print(f"\n=== Pair {idx}/{len(pair_list)}: {task_a} vs {task_b} ===")
        ckpt_a_path, ckpt_b_path = _resolve_pair_checkpoints(
            cfg=cfg,
            task_a=task_a,
            task_b=task_b,
            checkpoint_map=checkpoint_map,
            all_pairs=all_pairs,
        )
        if task_a in checkpoint_map and task_b in checkpoint_map:
            pair_ckpt_by_task = {
                task_a: checkpoint_map[task_a],
                task_b: checkpoint_map[task_b],
            }
        else:
            pair_ckpt_by_task = {
                task_a: ckpt_a_path,
                task_b: ckpt_b_path,
            }
        pair_tag = f"{_safe_name(task_a)}__{_safe_name(task_b)}"
        # Backward compatible: if the user passes a pair-specific output_dir, avoid nesting twice.
        pair_output_dir = output_root if output_root.name == pair_tag else output_root / pair_tag
        pair_results = _run_pair_analysis(
            suite_name=suite_name,
            pair_tasks=pair,
            pair_ckpt_by_task=pair_ckpt_by_task,
            ckpt_a_path=ckpt_a_path,
            ckpt_b_path=ckpt_b_path,
            task_static_by_name=task_static_by_name,
            build_cfg=build_cfg,
            base_ckpt=base_ckpt,
            strict_load=strict_load,
            requested_forward_mode=requested_forward_mode,
            text_features_source=text_features_source,
            text_features_checkpoint=text_features_checkpoint,
            classnames_mode=classnames_mode,
            eval_split=eval_split,
            device=device,
            line_alphas=line_alphas,
            heatmap_factors=heatmap_factors,
            run_heatmap=run_heatmap,
            output_dir=pair_output_dir,
            save_plots=save_plots,
        )
        pair_summaries.append(
            {
                "task_a": task_a,
                "task_b": task_b,
                "checkpoint_a": ckpt_a_path,
                "checkpoint_b": ckpt_b_path,
                "avg_loss_barrier": float(pair_results["line"]["barrier"]["avg_loss_barrier"]),
                "avg_error_barrier": float(pair_results["line"]["barrier"]["avg_error_barrier"]),
                "metrics_json": str(pair_results["files"]["metrics_json"]),
            }
        )
        if run_logger is not None:
            run_logger.log_event(
                "pair_end",
                metrics={
                    "connectivity/avg_loss_barrier": float(pair_results["line"]["barrier"]["avg_loss_barrier"]),
                    "connectivity/avg_error_barrier": float(pair_results["line"]["barrier"]["avg_error_barrier"]),
                },
                context={
                    "pair": [task_a, task_b],
                    "metrics_json": str(pair_results["files"]["metrics_json"]),
                    "line_csv": str(pair_results["files"]["line_csv"]),
                    "heatmap_enabled": bool(pair_results["heatmap"]["enabled"]),
                },
            )

    if all_pairs:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_json = output_root / "all_pairs_summary.json"
        summary_csv = output_root / "all_pairs_summary.csv"
        summary = {
            "suite": suite_name,
            "tasks": selected_tasks,
            "num_pairs": len(pair_summaries),
            "pairs": pair_summaries,
            "created_unix": int(time.time()),
        }
        atomic_write_json(str(summary_json), summary)
        _write_pairs_summary_csv(summary_csv, pair_summaries)
        print(f"\nSaved all-pairs summary JSON to: {summary_json}")
        print(f"Saved all-pairs summary CSV to: {summary_csv}")
        if run_logger is not None:
            run_logger.log_summary(summary)
            run_logger.finish("success")
    elif run_logger is not None:
        run_logger.log_summary(
            {
                "suite": suite_name,
                "tasks": selected_tasks,
                "num_pairs": len(pair_summaries),
                "pairs": pair_summaries,
                "created_unix": int(time.time()),
            }
        )
        run_logger.finish("success")


if __name__ == "__main__":
    main()
