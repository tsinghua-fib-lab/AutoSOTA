from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import islice
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
    add_merge_io_args,
    add_suite_arg,
    add_tasks_arg,
    build_common_eval_overrides,
    build_common_merge_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import (
    TaskAttentionMeta,
    assert_qkv_patched_before_linearizing,
    build_dense_delta_branch,
    ensure_peft_cfg_map,
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
from ..merge.base import PreparedMergeMethod
from ..merge.methods._common import resolve_merge_weights
from ..merge.registry import get_method, list_methods
from ..merge.runtime import build_merged_state_for_alpha
from ..merge.subspaces.registry import get_subspace, list_subspaces
from ..models.forward_modes import (
    get_forward_mode,
    list_forward_modes,
    normalize_forward_mode_params,
    resolve_auto_forward_mode,
    resolve_shared_forward_mode_params,
)
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..run_logging import print_config_args
from .datasets.vision8_14_20 import SUITES


@dataclass(frozen=True)
class KlCandidate:
    alpha: float
    method_params: dict[str, Any]
    source: str


@dataclass(frozen=True)
class TaskEvalContext:
    task: str
    loaders: Any
    classnames: list[str]
    build_cfg_task: OpenClipBuildConfig
    text_features: torch.Tensor | None
    text_features_mode: str
    tuned_ckpt_path: str


@dataclass(frozen=True)
class TaskCheckpointPayload:
    path: str
    obj: Any
    strategy: str | None
    forward_mode: str | None
    forward_mode_params: dict[str, Any] | None
    attn_meta: TaskAttentionMeta
    tuned_text_features: torch.Tensor | None


def resolve_kl_candidate(
    *,
    cfg: dict[str, Any],
    cli_alpha: float | None = None,
    cli_method_params: dict[str, Any] | None = None,
    merge_summary_path: str | None = None,
    merged_ckpt_path: str | None = None,
) -> KlCandidate:
    alpha = cli_alpha
    method_params = dict(cli_method_params) if cli_method_params is not None else None
    source_parts: list[str] = []
    if merged_ckpt_path is not None:
        source_parts.append("merged_ckpt")
    if cli_alpha is not None:
        source_parts.append("cli_alpha")
    if cli_method_params is not None:
        source_parts.append("cli_method_params")

    summary: dict[str, Any] = {}
    if merge_summary_path is not None:
        summary = load_json(merge_summary_path)
        if alpha is None and "best_alpha" in summary:
            alpha = float(summary["best_alpha"])
            source_parts.append("merge_summary_alpha")
        if method_params is None and isinstance(summary.get("best_method_params", None), dict):
            method_params = dict(summary["best_method_params"])
            source_parts.append("merge_summary_method_params")

    if alpha is None:
        if "alpha" in cfg and cfg.get("alpha") is not None:
            alpha = float(cfg["alpha"])
            source_parts.append("config_alpha")
        elif merged_ckpt_path is not None:
            alpha = float("nan")
        elif bool(cfg.get("alpha_search", False)) or cfg.get("hyperparam_search", None) is not None:
            raise ValueError(
                "KL analysis needs one selected merged candidate. Provide --alpha, --merge-summary with best_alpha, "
                "or config['alpha']; alpha-search configs without a selected candidate are ambiguous."
            )
        else:
            alpha = 1.0
            source_parts.append("default_alpha")

    if method_params is None:
        raw_method_params = cfg.get("method_params", {})
        if raw_method_params is None:
            raw_method_params = {}
        if not isinstance(raw_method_params, dict):
            raise ValueError("config['method_params'] must be a dict when provided.")
        method_params = dict(raw_method_params)
        source_parts.append("config_method_params")

    return KlCandidate(alpha=float(alpha), method_params=method_params, source="+".join(source_parts))


def _resolve_tasks(*, cfg: dict[str, Any], suite_name: str) -> list[str]:
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    suite = SUITES[suite_name]
    tasks_raw = cfg.get("tasks", "all")
    if isinstance(tasks_raw, str):
        tasks = list(suite.tasks) if tasks_raw == "all" else parse_csv(tasks_raw)
    elif isinstance(tasks_raw, (list, tuple)):
        tasks = [str(x) for x in tasks_raw]
    else:
        raise ValueError("tasks must be 'all', a CSV string, or a list.")

    allowed = set(suite.tasks)
    bad = [t for t in tasks if t not in allowed]
    if bad:
        raise ValueError(f"Unknown tasks for suite '{suite_name}': {bad}. Allowed: {sorted(allowed)}")
    return tasks


def _normalize_tuned_ckpts(raw: Any, tasks: list[str]) -> dict[str, str]:
    if isinstance(raw, dict):
        out = {str(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(tasks):
            raise ValueError("--tuned-ckpts list length must match the selected task count.")
        out = {task: str(path) for task, path in zip(tasks, raw, strict=True)}
    else:
        raise ValueError("Provide tuned checkpoints via config['tuned_ckpts'] or --tuned-ckpts.")

    missing = [task for task in tasks if task not in out]
    if missing:
        raise ValueError(f"Missing tuned checkpoint paths for tasks: {missing}")
    return out


def _load_checkpoint_payload(path: str) -> TaskCheckpointPayload:
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
    return TaskCheckpointPayload(
        path=str(resolved_path),
        obj=obj,
        strategy=strategy,
        forward_mode=forward_mode,
        forward_mode_params=forward_mode_params,
        attn_meta=attn_meta,
        tuned_text_features=tuned_text_features,
    )


def _load_full_checkpoint_state(
    *,
    payload: TaskCheckpointPayload,
    base_sd: dict[str, torch.Tensor],
    build_cfg: OpenClipBuildConfig,
    strict_load: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    obj = payload.obj
    peft_state: dict[str, torch.Tensor] | None = None
    peft_cfg_map: dict[str, Any] | None = None
    is_peft = False

    if is_peft_adapter_dir_ckpt(obj):
        peft_state, peft_cfg_map = load_peft_adapter_dir_components(obj["peft_adapter_dir"], checkpoint_path=payload.path)
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
            patched_attn=payload.attn_meta.patched_attn,
            attn_patch_cfg=payload.attn_meta.attn_patch_cfg,
        )
    else:
        sd = load_ckpt(payload.path)

    aligned = align_to_base_keys(sd, base_sd)
    if not aligned:
        raise ValueError(
            f"No tensors from tuned checkpoint aligned to base keys: {payload.path}. "
            "Check checkpoint key prefixes and model compatibility."
        )
    full = to_cpu_fp32(base_sd)
    full.update(to_cpu_fp32(aligned))
    return full, peft_state, peft_cfg_map


def _load_direct_merged_state(
    *,
    path: str,
    base_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    sd = load_ckpt(path)
    aligned = align_to_base_keys(sd, base_sd)
    if not aligned:
        raise ValueError(
            f"No tensors from merged checkpoint aligned to base keys: {path}. "
            "Check checkpoint key prefixes and model compatibility."
        )
    full = to_cpu_fp32(base_sd)
    full.update(to_cpu_fp32(aligned))
    return full


def _build_task_contexts(
    *,
    clf: OpenClipClassifier,
    cfg: dict[str, Any],
    suite_name: str,
    tasks: list[str],
    build_cfg: OpenClipBuildConfig,
    tuned_payload_by_task: dict[str, TaskCheckpointPayload],
    tuned_ckpt_by_task: dict[str, str],
) -> list[TaskEvalContext]:
    suite = SUITES[suite_name]
    text_features_source = str(cfg.get("text_features_source", "auto")).strip().lower()
    if text_features_source not in {"auto", "zero_shot", "tuned_ckpt"}:
        raise ValueError("text_features_source must be one of: auto, zero_shot, tuned_ckpt")
    use_humanized_classnames = not bool(cfg.get("no_humanize", True))

    out: list[TaskEvalContext] = []
    for task in tasks:
        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf.preprocess,
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
        payload = tuned_payload_by_task[task]
        text_features, text_features_mode = clf.resolve_eval_text_features(
            text_features_source=text_features_source,
            classnames=classnames,
            build_cfg=build_cfg_task,
            tuned_text_features=payload.tuned_text_features,
            cache_dir="src/.cache/zs_cache",
            force_rebuild_zeroshot=False,
            task_name=task,
            ckpt_path=tuned_ckpt_by_task[task],
            verbose=True,
        )
        out.append(
            TaskEvalContext(
                task=task,
                loaders=loaders,
                classnames=classnames,
                build_cfg_task=build_cfg_task,
                text_features=text_features,
                text_features_mode=text_features_mode,
                tuned_ckpt_path=tuned_ckpt_by_task[task],
            )
        )
    return out


def _select_loader(loaders: Any, split: str) -> Any:
    if split == "val":
        return loaders.val
    if split == "test":
        return loaders.test
    raise ValueError(f"Unknown split '{split}'. Expected one of: val, test.")


def _batch_kl_sum(
    *,
    ref_log_probs: torch.Tensor,
    merged_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return _batch_kl_per_sample(
        ref_log_probs=ref_log_probs,
        merged_logits=merged_logits,
        temperature=temperature,
    ).sum()


def _batch_kl_per_sample(
    *,
    ref_log_probs: torch.Tensor,
    merged_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    merged_log_probs = F.log_softmax(merged_logits / float(temperature), dim=-1)
    ref_probs = ref_log_probs.exp()
    return (ref_probs * (ref_log_probs - merged_log_probs)).sum(dim=-1)


def _prepare_text_features(clf: OpenClipClassifier, item: TaskEvalContext, device: str) -> None:
    if item.text_features is None:
        clf.build_zeroshot_text_features(
            item.classnames,
            item.build_cfg_task,
            cache_dir="src/.cache/zs_cache",
            force_rebuild=False,
        )
        return

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    feats = item.text_features.to(device=dev)
    if clf.normalize:
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
    clf._zs_text_features = feats
    clf._zs_text_fingerprint = None


@torch.no_grad()
def _collect_reference_logits(
    *,
    clf: OpenClipClassifier,
    item: TaskEvalContext,
    device: str,
    split: str,
    max_batches: int | None,
) -> tuple[list[torch.Tensor], int]:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    clf.to(dev)
    clf.eval()
    _prepare_text_features(clf, item, device)

    loader = _select_loader(item.loaders, split)
    iterator = iter(loader) if max_batches is None else islice(iter(loader), max(0, int(max_batches)))
    chunks: list[torch.Tensor] = []
    samples = 0
    for x, _y in iterator:
        x = x.to(dev, non_blocking=True)
        logits = clf(x)
        chunks.append(logits.detach().cpu())
        samples += int(logits.shape[0])
    return chunks, samples


@torch.no_grad()
def _compute_kl_against_reference(
    *,
    clf: OpenClipClassifier,
    item: TaskEvalContext,
    ref_logit_chunks: list[torch.Tensor],
    device: str,
    split: str,
    temperature: float,
    max_batches: int | None,
) -> tuple[float, int, list[float], list[float], list[float]]:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    clf.to(dev)
    clf.eval()
    _prepare_text_features(clf, item, device)

    loader = _select_loader(item.loaders, split)
    iterator = iter(loader) if max_batches is None else islice(iter(loader), max(0, int(max_batches)))
    total_kl = 0.0
    total_samples = 0
    per_sample_kl: list[float] = []
    reference_logits_flat: list[float] = []
    merged_logits_flat: list[float] = []
    for ref_logits_cpu, (x, _y) in zip(ref_logit_chunks, iterator, strict=True):
        x = x.to(dev, non_blocking=True)
        logits = clf(x)
        ref_logits = ref_logits_cpu.to(device=dev)
        if tuple(ref_logits.shape) != tuple(logits.shape):
            raise ValueError(
                f"Logit shape mismatch for task '{item.task}': reference={tuple(ref_logits.shape)} "
                f"merged={tuple(logits.shape)}"
            )
        ref_log_probs = F.log_softmax(ref_logits / float(temperature), dim=-1)
        batch_kl_values = _batch_kl_per_sample(
            ref_log_probs=ref_log_probs,
            merged_logits=logits,
            temperature=temperature,
        )
        total_kl += float(batch_kl_values.sum().item())
        total_samples += int(logits.shape[0])
        per_sample_kl.extend(float(v) for v in batch_kl_values.detach().cpu().tolist())
        reference_logits_flat.extend(float(v) for v in ref_logits_cpu.reshape(-1).tolist())
        merged_logits_flat.extend(float(v) for v in logits.detach().cpu().reshape(-1).tolist())
    return total_kl, total_samples, per_sample_kl, reference_logits_flat, merged_logits_flat


def _safe_plot_name(name: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(name))
    out = out.strip("_")
    return out or "task"


def _template_id(template: Any) -> str:
    if isinstance(template, str):
        return template
    if callable(template):
        mod = getattr(template, "__module__", "<?>")
        name = getattr(template, "__qualname__", getattr(template, "__name__", "<callable>"))
        code = getattr(template, "__code__", None)
        if code is not None:
            return f"{mod}:{name}@{code.co_filename}:{code.co_firstlineno}"
        return f"{mod}:{name}"
    return repr(template)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_signature(path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        stat = p.stat()
    except FileNotFoundError:
        return {"path": str(p), "exists": False}
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _tensor_signature(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    cpu = tensor.detach().cpu().contiguous()
    data = cpu.numpy().tobytes()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _reference_cache_metadata(
    *,
    cfg: dict[str, Any],
    item: TaskEvalContext,
    suite_name: str,
    split: str,
    temperature: float,
    max_batches: int | None,
    resolved_forward_mode: str,
) -> dict[str, Any]:
    prompt_templates = item.build_cfg_task.prompt_templates
    return {
        "version": 1,
        "kind": "vision_logit_kl_reference_logits",
        "suite": suite_name,
        "task": item.task,
        "split": split,
        "max_batches": max_batches,
        "forward_mode": resolved_forward_mode,
        "checkpoint": _file_signature(item.tuned_ckpt_path),
        "clip_model": item.build_cfg_task.model_name,
        "clip_pretrained": item.build_cfg_task.pretrained,
        "dtype": item.build_cfg_task.dtype,
        "normalize": bool(item.build_cfg_task.normalize),
        "logit_scale": float(item.build_cfg_task.logit_scale),
        "classnames": list(item.classnames),
        "prompt_template": _template_id(item.build_cfg_task.prompt_template),
        "prompt_templates": None
        if prompt_templates is None
        else [_template_id(template) for template in list(prompt_templates)],
        "text_features_mode": item.text_features_mode,
        "text_features": _tensor_signature(item.text_features),
        "batch_size": int(cfg.get("batch_size", 128)),
        "val_fraction": float(cfg.get("val_fraction", 0.1)),
        "seed": int(cfg.get("seed", 42)),
        "classnames_mode": "humanized" if not bool(cfg.get("no_humanize", True)) else "raw",
    }


def _reference_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    digest = _stable_digest(metadata)
    return cache_dir / f"{_safe_plot_name(str(metadata['task']))}_{digest[:24]}.pt"


def _load_cached_reference_logits(
    *,
    path: Path,
    metadata: dict[str, Any],
) -> tuple[list[torch.Tensor], int] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[warn] Failed to read reference logit cache {path}: {exc}")
        return None
    if not isinstance(payload, dict) or payload.get("metadata") != metadata:
        return None
    chunks = payload.get("chunks", None)
    samples = payload.get("samples", None)
    if not isinstance(chunks, list) or not all(torch.is_tensor(chunk) for chunk in chunks):
        return None
    try:
        return [chunk.detach().cpu() for chunk in chunks], int(samples)
    except Exception:
        return None


def _save_cached_reference_logits(
    *,
    path: Path,
    metadata: dict[str, Any],
    chunks: list[torch.Tensor],
    samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "metadata": metadata,
            "samples": int(samples),
            "chunks": [chunk.detach().cpu() for chunk in chunks],
        },
        str(tmp),
    )
    tmp.replace(path)


def _maybe_save_logit_distribution_plots(
    *,
    plot_dir: Path,
    per_task_values: dict[str, dict[str, list[float]]],
    bins: int,
) -> dict[str, Any]:
    if not per_task_values:
        return {"enabled": False, "reason": "no logits to plot", "per_task": {}, "avg": None, "kind": "logits"}

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Could not import matplotlib. Skipping logit distribution plots. ({exc})")
        return {
            "enabled": False,
            "reason": f"matplotlib import failed: {exc}",
            "per_task": {},
            "avg": None,
            "kind": "logits",
        }

    plot_dir.mkdir(parents=True, exist_ok=True)
    bins = max(1, int(bins))
    paths_by_task: dict[str, str] = {}

    def _save_hist(reference_values: list[float], merged_values: list[float], *, title: str, path: Path) -> None:
        ref_mean = sum(reference_values) / max(1, len(reference_values))
        merged_mean = sum(merged_values) / max(1, len(merged_values))
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.hist(
            reference_values,
            bins=bins,
            density=True,
            color="#4f81bd",
            edgecolor="white",
            alpha=0.48,
            label=f"fine-tuned logits mean={ref_mean:.4f}",
        )
        ax.hist(
            merged_values,
            bins=bins,
            density=True,
            color="#c0504d",
            edgecolor="white",
            alpha=0.48,
            label=f"merged logits mean={merged_mean:.4f}",
        )
        ax.axvline(ref_mean, color="#1f4e79", linestyle="--", linewidth=1.4)
        ax.axvline(merged_mean, color="#8b1a1a", linestyle="--", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Raw class logits")
        ax.set_ylabel("Density")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    all_reference_values: list[float] = []
    all_merged_values: list[float] = []
    for task, values in per_task_values.items():
        reference_values = values.get("reference", [])
        merged_values = values.get("merged", [])
        if not reference_values or not merged_values:
            continue
        all_reference_values.extend(reference_values)
        all_merged_values.extend(merged_values)
        path = plot_dir / f"{_safe_plot_name(task)}_logit_distribution.png"
        _save_hist(reference_values, merged_values, title=f"{task} Logit Distribution", path=path)
        paths_by_task[task] = str(path)

    avg_path: str | None = None
    if all_reference_values and all_merged_values:
        avg = plot_dir / "avg_logit_distribution.png"
        _save_hist(all_reference_values, all_merged_values, title="All Tasks Logit Distribution", path=avg)
        avg_path = str(avg)

    return {"enabled": True, "per_task": paths_by_task, "avg": avg_path, "bins": bins, "kind": "logits"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_vision_logit_kl(cfg: dict[str, Any], *, candidate: KlCandidate, output_path: Path) -> dict[str, Any]:
    t0 = time.time()
    subspace_artifact_dir = output_path.parent / f"{output_path.stem}.artifacts"
    suite_name = str(cfg.get("suite", "vision8"))
    tasks = _resolve_tasks(cfg=cfg, suite_name=suite_name)
    tuned_ckpt_by_task = _normalize_tuned_ckpts(cfg.get("tuned_ckpts", None), tasks)
    strict_load = bool(cfg.get("strict_load", False))
    peft_subspace = str(cfg.get("peft_subspace", "full"))
    merged_ckpt = cfg.get("merged_ckpt", None)
    merged_ckpt_path = None if merged_ckpt is None else str(merged_ckpt)
    split = str(cfg.get("split", "test")).strip().lower()
    if split not in {"test", "val"}:
        raise ValueError("split must be one of: test, val")
    temperature = float(cfg.get("temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    max_batches_raw = cfg.get("max_batches_per_task", None)
    max_batches = None if max_batches_raw is None else int(max_batches_raw)
    if max_batches is not None and max_batches < 0:
        raise ValueError("max_batches_per_task must be >= 0 when provided.")
    save_plots = bool(cfg.get("save_plots", True))
    plot_bins = int(cfg.get("plot_bins", 50))
    if plot_bins <= 0:
        raise ValueError("plot_bins must be > 0.")
    plot_dir_raw = cfg.get("plot_dir", None)
    plot_dir = Path(str(plot_dir_raw)) if plot_dir_raw is not None else output_path.parent / f"{output_path.stem}_plots"
    use_logit_cache = bool(cfg.get("logit_cache", True))
    recompute_logit_cache = bool(cfg.get("recompute_logit_cache", False))
    logit_cache_dir = Path(str(cfg.get("logit_cache_dir", "src/.cache/vision_logit_kl_logits")))

    build_cfg = OpenClipBuildConfig(
        model_name=cfg.get("clip_model", "ViT-B-32"),
        pretrained=cfg.get("clip_pretrained", "openai"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", None),
    )
    clf = OpenClipClassifier.build(build_cfg)

    base_ckpt = cfg.get("base_ckpt", None)
    if base_ckpt is None:
        print(f"Using open_clip {build_cfg.model_name} (pretrain={build_cfg.pretrained}) weights as base checkpoint")
        base_sd = {k: v.detach().cpu() for k, v in clf.model.state_dict().items()}
    else:
        print(f"Loading base checkpoint from {base_ckpt}")
        base_sd = load_ckpt(str(base_ckpt))
    base_sd = to_cpu_fp32(base_sd)

    tuned_payload_by_task: dict[str, TaskCheckpointPayload] = {}
    tuned_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}
    peft_state_by_task: dict[str, dict[str, torch.Tensor]] = {}
    peft_cfg_map: dict[str, Any] | None = None
    base_patched_for_attn = False

    for task in tasks:
        payload = _load_checkpoint_payload(tuned_ckpt_by_task[task])
        tuned_payload_by_task[task] = payload
        base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
            task_meta=payload.attn_meta,
            base_patched_for_attn=base_patched_for_attn,
            clf=clf,
            base_ckpt=base_ckpt,
            strict_load=strict_load,
            base_sd=base_sd,
        )
        base_sd = to_cpu_fp32(base_sd)
        aligned, peft_state, task_peft_cfg_map = _load_full_checkpoint_state(
            payload=payload,
            base_sd=base_sd,
            build_cfg=build_cfg,
            strict_load=strict_load,
        )
        tuned_sds_by_task[task] = aligned
        if peft_state is not None and task_peft_cfg_map is not None:
            peft_state_by_task[task] = peft_state
            peft_cfg_map = ensure_peft_cfg_map(peft_cfg_map, task_peft_cfg_map)
        elif peft_subspace != "full" and merged_ckpt_path is None:
            raise ValueError(f"peft_subspace='{peft_subspace}' requires PEFT checkpoints. Got: {payload.path}")

    attn_meta_tasks = [tuned_payload_by_task[t].attn_meta for t in tasks]
    if attn_meta_tasks:
        patched0 = attn_meta_tasks[0].patched_attn
        if any(meta.patched_attn != patched0 for meta in attn_meta_tasks):
            raise ValueError("Inconsistent patched_attn flags across tuned checkpoints.")
        patch_cfgs = [meta.attn_patch_cfg or {} for meta in attn_meta_tasks if meta.patched_attn]
        if patch_cfgs and any(cfg_i != patch_cfgs[0] for cfg_i in patch_cfgs[1:]):
            raise ValueError("Inconsistent attn_patch_cfg across tuned checkpoints.")

    subspace = None
    subspace_prepared = None
    peft_cfg: dict[str, Any] | None = None
    dense_tuned_sds_list: list[dict[str, torch.Tensor]] = []
    dense_base_sd_for_merge: dict[str, torch.Tensor] = {}
    merge_base_sd = to_cpu_fp32(base_sd)
    if peft_subspace != "full" and merged_ckpt_path is None:
        resolved_merge_weights = resolve_merge_weights(len(tasks), cfg.get("weights", None))
        if peft_cfg_map is None:
            raise ValueError(f"peft_subspace='{peft_subspace}' requires peft_config in checkpoints.")
        peft_cfg = get_peft_cfg(peft_cfg_map)
        subspace = get_subspace(peft_subspace)
        subspace_prepared = subspace.prepare(
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
            method_params=candidate.method_params,
            weights=resolved_merge_weights,
            artifact_dir=subspace_artifact_dir,
        )
        if getattr(subspace_prepared, "merge_weight_override", None) is not None:
            cfg["weights"] = list(subspace_prepared.merge_weight_override)
        projected_by_task = subspace.project(subspace_prepared, lora_by_task=peft_state_by_task, peft_cfg=peft_cfg)
        tuned_sds_list = [projected_by_task[t] for t in tasks]
        base_sd_for_merge = {k: torch.zeros_like(v) for k, v in tuned_sds_list[0].items()}
        lora_only_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}
        for task in tasks:
            payload = tuned_payload_by_task[task]
            lora_only_sd = materialize_peft_sd_from_adapter(
                peft_state=peft_state_by_task[task],
                base_sd=base_sd,
                build_cfg=build_cfg,
                peft_cfg=peft_cfg,
                peft_dense_state=None,
                strict_load=strict_load,
                patched_attn=payload.attn_meta.patched_attn,
                attn_patch_cfg=payload.attn_meta.attn_patch_cfg,
            )
            lora_only_aligned = align_to_base_keys(lora_only_sd, base_sd)
            if not lora_only_aligned:
                raise ValueError(
                    f"No tensors from LoRA-only checkpoint aligned to base keys: {payload.path}. "
                    "Check checkpoint key prefixes and model compatibility."
                )
            lora_only_sds_by_task[task] = to_cpu_fp32(lora_only_aligned)
        dense_base_sd_for_merge, dense_tuned_sds_list = build_dense_delta_branch(
            tasks=tasks,
            full_tuned_by_task=tuned_sds_by_task,
            lora_only_tuned_by_task=lora_only_sds_by_task,
            base_sd=merge_base_sd,
        )
    else:
        tuned_sds_list = [tuned_sds_by_task[t] for t in tasks]
        base_sd_for_merge = to_cpu_fp32(base_sd)

    needs_linear_attention = any(payload.attn_meta.linearized_attn for payload in tuned_payload_by_task.values())
    assert_qkv_patched_before_linearizing(
        needs_linear_attention=needs_linear_attention,
        base_patched_for_attn=base_patched_for_attn,
        model_state_dict=clf.model.state_dict(),
    )

    requested_forward_mode = str(cfg.get("forward_mode", "auto"))
    if requested_forward_mode == "auto":
        resolved_forward_mode = resolve_auto_forward_mode([tuned_payload_by_task[t].forward_mode for t in tasks])
    else:
        resolved_forward_mode = requested_forward_mode
    resolved_forward_mode_params = resolve_shared_forward_mode_params(
        resolved_forward_mode,
        [
            tuned_payload_by_task[t].forward_mode_params
            for t in tasks
            if tuned_payload_by_task[t].forward_mode == "linearized_ntk"
        ],
    )
    get_forward_mode(resolved_forward_mode).bind(
        clf=clf,
        base_sd=merge_base_sd,
        strict_load=strict_load,
        params=resolved_forward_mode_params,
    )
    print(f"Using forward mode: {resolved_forward_mode} params={resolved_forward_mode_params}")

    per_task = _build_task_contexts(
        clf=clf,
        cfg=cfg,
        suite_name=suite_name,
        tasks=tasks,
        build_cfg=build_cfg,
        tuned_payload_by_task=tuned_payload_by_task,
        tuned_ckpt_by_task=tuned_ckpt_by_task,
    )

    method_name = str(cfg.get("method", "task_arithmetic"))
    if merged_ckpt_path is not None:
        print(f"Loading merged checkpoint directly from {merged_ckpt_path}")
        merged_sd = _load_direct_merged_state(path=merged_ckpt_path, base_sd=merge_base_sd)
    else:
        method = get_method(method_name)
        merge_weights = cfg.get("weights", None)
        prepared = None
        dense_prepared = None
        if isinstance(method, PreparedMergeMethod):
            prepared = method.prepare(
                base=base_sd_for_merge,
                tuned=tuned_sds_list,
                weights=merge_weights,
                strict=strict_load,
                tasks=tasks,
                merge_context={
                    "kind": "vision_logit_kl",
                    "cfg": cfg,
                    "model": clf.model,
                    "classifier": clf,
                    "tasks": tasks,
                    "per_task": [item.__dict__ for item in per_task],
                    "tuned_state_by_task": tuned_sds_by_task,
                    "peft_subspace": peft_subspace,
                    "subspace_prepared": subspace_prepared,
                    "peft_state_by_task": peft_state_by_task,
                    "suite_name": suite_name,
                },
                method_params=candidate.method_params,
            )
            if dense_tuned_sds_list and dense_base_sd_for_merge:
                dense_prepared = method.prepare(
                    base=dense_base_sd_for_merge,
                    tuned=dense_tuned_sds_list,
                    weights=merge_weights,
                    strict=strict_load,
                    tasks=tasks,
                    merge_context={
                        "kind": "vision_logit_kl_dense_delta",
                        "cfg": cfg,
                        "tasks": tasks,
                        "suite_name": suite_name,
                        "peft_subspace": peft_subspace,
                    },
                    method_params=candidate.method_params,
                )

        merged_sd = build_merged_state_for_alpha(
            method=method,
            prepared=prepared,
            base_sd_for_merge=base_sd_for_merge,
            tuned_sds_list=tuned_sds_list,
            weights=merge_weights,
            method_params=candidate.method_params,
            alpha=candidate.alpha,
            peft_subspace=peft_subspace,
            subspace=subspace,
            subspace_prepared=subspace_prepared,
            peft_cfg=peft_cfg,
            peft_state_by_task=peft_state_by_task,
            tasks=tasks,
            merge_base_sd=merge_base_sd,
            dense_prepared=dense_prepared,
            dense_base_sd_for_merge=dense_base_sd_for_merge,
            dense_tuned_sds_list=dense_tuned_sds_list,
        )

    reference_logits_by_task: dict[str, list[torch.Tensor]] = {}
    reference_samples_by_task: dict[str, int] = {}
    logit_cache_info: dict[str, Any] = {
        "enabled": bool(use_logit_cache),
        "dir": str(logit_cache_dir) if use_logit_cache else None,
        "recompute": bool(recompute_logit_cache),
        "per_task": {},
    }
    for item in per_task:
        print(f"\nCollecting reference logits for {item.task}")
        cache_path: Path | None = None
        chunks: list[torch.Tensor] | None = None
        samples: int | None = None
        cache_status = "disabled"
        if use_logit_cache:
            cache_metadata = _reference_cache_metadata(
                cfg=cfg,
                item=item,
                suite_name=suite_name,
                split=split,
                temperature=temperature,
                max_batches=max_batches,
                resolved_forward_mode=resolved_forward_mode,
            )
            cache_path = _reference_cache_path(logit_cache_dir, cache_metadata)
            if not recompute_logit_cache:
                cached = _load_cached_reference_logits(path=cache_path, metadata=cache_metadata)
                if cached is not None:
                    chunks, samples = cached
                    cache_status = "hit"
                    print(f"{item.task}: [cache hit] reference logits loaded from {cache_path}")
            if chunks is None or samples is None:
                cache_status = "miss" if not recompute_logit_cache else "recomputed"

        if chunks is None or samples is None:
            load_into_model(clf.model, tuned_sds_by_task[item.task], strict=strict_load)
            chunks, samples = _collect_reference_logits(
                clf=clf,
                item=item,
                device=str(cfg.get("device", "cuda")),
                split=split,
                max_batches=max_batches,
            )
            if use_logit_cache and cache_path is not None:
                _save_cached_reference_logits(
                    path=cache_path,
                    metadata=cache_metadata,
                    chunks=chunks,
                    samples=int(samples),
                )
                print(f"{item.task}: [cache saved] reference logits saved to {cache_path}")
        if not chunks:
            raise RuntimeError(f"No batches were evaluated for task '{item.task}'.")
        reference_logits_by_task[item.task] = chunks
        reference_samples_by_task[item.task] = int(samples)
        logit_cache_info["per_task"][item.task] = {
            "status": cache_status,
            "path": str(cache_path) if cache_path is not None else None,
        }

    print("\nLoading merged model for KL pass")
    load_into_model(clf.model, merged_sd, strict=strict_load)
    del merged_sd
    if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu":
        torch.cuda.empty_cache()

    per_task_result: dict[str, dict[str, Any]] = {}
    per_task_logit_values: dict[str, dict[str, list[float]]] = {}
    total_kl = 0.0
    total_samples = 0
    for item in per_task:
        kl_sum, samples, _kl_values, reference_logit_values, merged_logit_values = _compute_kl_against_reference(
            clf=clf,
            item=item,
            ref_logit_chunks=reference_logits_by_task[item.task],
            device=str(cfg.get("device", "cuda")),
            split=split,
            temperature=temperature,
            max_batches=max_batches,
        )
        if samples != reference_samples_by_task[item.task]:
            raise RuntimeError(
                f"Sample-count mismatch for task '{item.task}': reference={reference_samples_by_task[item.task]} "
                f"merged={samples}"
            )
        mean_kl = float(kl_sum / max(1, samples))
        total_kl += float(kl_sum)
        total_samples += int(samples)
        per_task_logit_values[item.task] = {
            "reference": reference_logit_values,
            "merged": merged_logit_values,
        }
        per_task_result[item.task] = {
            "kl": mean_kl,
            "kl_sum": float(kl_sum),
            "samples": int(samples),
            "num_classes": int(len(item.classnames)),
            "tuned_checkpoint": item.tuned_ckpt_path,
            "text_features_mode": item.text_features_mode,
        }
        print(f"{item.task}: kl={mean_kl:.8f} samples={samples}")

    avg_kl = sum(float(row["kl"]) for row in per_task_result.values()) / max(1, len(per_task_result))
    sample_weighted_kl = float(total_kl / max(1, total_samples))
    plot_info: dict[str, Any]
    if save_plots:
        plot_info = _maybe_save_logit_distribution_plots(
            plot_dir=plot_dir,
            per_task_values=per_task_logit_values,
            bins=plot_bins,
        )
    else:
        plot_info = {"enabled": False, "reason": "disabled", "per_task": {}, "avg": None}
    result = {
        "suite": suite_name,
        "tasks": tasks,
        "split": split,
        "temperature": float(temperature),
        "max_batches_per_task": max_batches,
        "method": method_name,
        "alpha": None if candidate.alpha != candidate.alpha else float(candidate.alpha),
        "method_params": _json_safe(candidate.method_params),
        "candidate_source": candidate.source,
        "merged_checkpoint": merged_ckpt_path,
        "peft_subspace": peft_subspace,
        "forward_mode": resolved_forward_mode,
        "forward_mode_params": dict(resolved_forward_mode_params),
        "resolved_config": _json_safe(cfg),
        "per_task": per_task_result,
        "avg_kl": float(avg_kl),
        "sample_weighted_kl": float(sample_weighted_kl),
        "plots": plot_info,
        "logit_cache": logit_cache_info,
        "total_samples": int(total_samples),
        "runtime_seconds": float(time.time() - t0),
        "subspace_artifacts": (
            {"similarity_artifact_path": getattr(subspace_prepared, "similarity_artifact_path", None)}
            if subspace_prepared is not None
            else {}
        ),
    }
    atomic_write_json(str(output_path), result)
    print(f"\nSaved KL analysis to {output_path}")
    print(f"avg_kl={avg_kl:.8f} sample_weighted_kl={sample_weighted_kl:.8f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Measure KL(single-task tuned logits || merged logits) for vision merges")
    add_config_arg(parser)
    add_suite_arg(parser, choices=sorted(SUITES.keys()))
    add_tasks_arg(parser, help_text="Comma-separated task names, or 'all'.")
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--clip-pretrained", type=str, default=None)
    add_device_dtype_args(parser, device_default="cuda", dtype_default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-humanize", action="store_true", default=None)
    parser.add_argument("--text-features-source", choices=["auto", "zero_shot", "tuned_ckpt"], default=None)
    parser.add_argument("--forward-mode", choices=["auto", *list_forward_modes()], default="auto")
    add_merge_io_args(
        parser,
        method_choices=list_methods(),
        subspace_choices=list_subspaces(),
        tuned_help="Paths to tuned checkpoints to compare against and merge.",
        weights_help="Weights for tuned checkpoints.",
        strict_mode="store_true",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--merge-summary", type=str, default=None)
    parser.add_argument(
        "--merged-ckpt",
        type=str,
        default=None,
        help="Load this merged state_dict directly instead of reconstructing the merge from method/alpha.",
    )
    parser.add_argument("--split", choices=["test", "val"], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-batches-per-task", type=int, default=None)
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory for KL distribution PNG plots.")
    parser.add_argument("--plot-bins", type=int, default=None, help="Histogram bin count for KL distribution plots.")
    parser.add_argument("--no-plots", action="store_true", default=None, help="Disable KL distribution plot output.")
    parser.add_argument(
        "--logit-cache-dir",
        type=str,
        default=None,
        help="Directory for cached per-task reference log-prob chunks.",
    )
    parser.add_argument("--no-logit-cache", action="store_true", default=None, help="Disable reference logit cache.")
    parser.add_argument(
        "--recompute-logit-cache",
        action="store_true",
        default=None,
        help="Recompute and overwrite cached reference logits.",
    )
    parser.add_argument("--output", type=str, default="src/.cache/vision_logit_kl.json")

    args = parser.parse_args()
    method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)
    merged_ckpt_path = args.merged_ckpt if args.merged_ckpt is not None else cfg.get("merged_ckpt", None)

    candidate = resolve_kl_candidate(
        cfg=cfg,
        cli_alpha=args.alpha,
        cli_method_params=method_params_cli,
        merge_summary_path=args.merge_summary,
        merged_ckpt_path=(str(merged_ckpt_path) if merged_ckpt_path is not None else None),
    )

    cli_overrides: dict[str, Any] = {
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "no_humanize": args.no_humanize,
        "text_features_source": args.text_features_source,
        "forward_mode": args.forward_mode,
        "split": args.split,
        "temperature": args.temperature,
        "max_batches_per_task": args.max_batches_per_task,
        "plot_dir": args.plot_dir,
        "plot_bins": args.plot_bins,
        "save_plots": (False if args.no_plots else None),
        "logit_cache_dir": args.logit_cache_dir,
        "logit_cache": (False if args.no_logit_cache else None),
        "recompute_logit_cache": (True if args.recompute_logit_cache else None),
        "merged_ckpt": args.merged_ckpt,
    }
    cli_overrides = merge_non_none(cli_overrides, build_common_eval_overrides(args))
    cli_overrides = merge_non_none(
        cli_overrides,
        build_common_merge_overrides(args=args, method_params=method_params_cli, strict_as_bool=True),
    )
    cfg = merge_non_none(cfg, cli_overrides)
    if candidate.alpha == candidate.alpha:
        cfg["alpha"] = float(candidate.alpha)
    elif "alpha" in cfg:
        del cfg["alpha"]
    cfg["method_params"] = dict(candidate.method_params)
    print_config_args(
        {
            **cfg,
            "config": args.config,
            "output": str(Path(args.output)),
            "candidate_source": candidate.source,
        },
        title="Run config (eval.vision_logit_kl)",
    )

    run_vision_logit_kl(cfg, candidate=candidate, output_path=Path(args.output))


if __name__ == "__main__":
    main()
