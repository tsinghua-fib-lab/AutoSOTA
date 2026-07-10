# src/merge_and_rebase/eval/vision_merge.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.hyperparam_search import (
    SearchEvaluation,
    build_search_planner,
    describe_candidate,
    summarize_search_results,
)
from merge_and_rebase.io.peft_helpers import (
    is_peft_adapter_dir_ckpt,
    load_peft_adapter_dir_components,
    normalize_peft_adapter_dir_checkpoint,
)
from merge_and_rebase.io.utils import atomic_write_json, read_json_silent
from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_merge_io_args,
    add_postmerge_args,
    add_suite_arg,
    add_tasks_arg,
    build_common_eval_overrides,
    build_common_merge_overrides,
    build_logging_overrides,
    build_postmerge_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import (
    TaskAttentionMeta,
    acc_cache_key,
    assert_qkv_patched_before_linearizing,
    build_dense_delta_branch,
    build_merged_state_for_alpha,
    ensure_peft_cfg_map,
    eval_norm_accs_for_split,
    eval_task_top1,
    extract_checkpoint_attn_patch_info,
    extract_peft_components,
    get_peft_cfg,
    humanize,
    is_peft_checkpoint,
    load_vision_checkpoint_reference,
    materialize_peft_sd_from_adapter,
    maybe_patch_base_for_task_attn,
    stable_method_params_cache_key,
    to_cpu_fp32,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model, resolve_ckpt_path
from ..merge import subspaces as _subspaces  # noqa: F401
from ..merge.base import PreparedMergeMethod
from ..merge.methods._common import resolve_merge_weights
from ..merge.registry import get_method, list_methods  # methods are registered on import
from ..merge.subspaces.registry import get_subspace, list_subspaces
from ..models.forward_modes import (
    get_forward_mode,
    list_forward_modes,
    normalize_forward_mode_params,
    resolve_auto_forward_mode,
    resolve_shared_forward_mode_params,
)
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..postmerge import PostMergeContext, get_postmerge_method
from ..run_logging import default_summary_path, merge_logging_config, start_run
from .datasets.vision8_14_20 import SUITES
from .print_utils import pretty_print_task_accuracies, print_latex_task_rows

# Backward-compatible test hooks for utility helpers.
_acc_cache_key = acc_cache_key
_assert_qkv_patched_before_linearizing = assert_qkv_patched_before_linearizing
_extract_checkpoint_attn_patch_info = extract_checkpoint_attn_patch_info


def _resolve_zero_shot_only(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("zero_shot_only", False)) or (cfg.get("tuned_ckpts", None) is None)


def _read_cached_top1(
    cache: dict[str, Any],
    *,
    key: str,
    label: str,
    task: str,
) -> float | None:
    if key not in cache:
        return None
    try:
        value = float(cache[key]["top1"])
    except Exception:
        return None
    print(f"{task}: [cache] {label}: {value:.6f}")
    return value


def _write_cached_top1(
    cache: dict[str, Any],
    *,
    cache_path: str,
    key: str,
    top1: float,
    model_name: str,
    pretrained: str,
    task: str,
    baseline_mode: str,
    checkpoint: str,
    seconds: float,
    label: str,
) -> None:
    cache[key] = {
        "top1": float(top1),
        "model": model_name,
        "pretrained": pretrained,
        "dataset": task,
        "baseline_mode": baseline_mode,
        "baseline_checkpoint": checkpoint,
        "ts": int(time.time()),
        "seconds": float(seconds),
    }
    atomic_write_json(cache_path, cache)
    print(f"{task}: [computed] {label}: {top1:.6f} (saved to {cache_path})")


def _save_merged_state_dict_if_requested(
    merged_sd: dict[str, torch.Tensor],
    save_merged: Any,
    *,
    label: str,
) -> str | None:
    if save_merged is None:
        return None
    outp = Path(str(save_merged))
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_sd, str(outp))
    print(f"Saved {label} state_dict to {outp}")
    return str(outp)


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    run_logger = None
    p = argparse.ArgumentParser("Merge checkpoints with selectable method and evaluate on a vision suite (open_clip)")

    add_config_arg(p)
    add_suite_arg(p, choices=sorted(SUITES.keys()))
    add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

    # open_clip model
    p.add_argument("--backbone-name", type=str, default=None, choices=["openclip", "openai_clip"])
    p.add_argument("--clip-model", type=str, default=None)
    p.add_argument("--clip-pretrained", type=str, default=None)
    add_device_dtype_args(p, device_default="cuda", dtype_default=None)

    # Eval
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-humanize",
        action="store_true",
        default=None,
        help="Use raw classnames. Default is raw classnames for training/eval consistency.",
    )

    add_merge_io_args(
        p,
        method_choices=list_methods(),
        subspace_choices=list_subspaces(),
        tuned_help="Paths to tuned checkpoints to merge.",
        weights_help="Weights for tuned checkpoints.",
        strict_mode="store_true",
    )
    # Method knobs
    p.add_argument("--keep-ratio", type=float, default=None, help="Keep top |Δ| ratio (task arithmetic)")

    # Single-task accuracy cache (baseline for normalization)
    p.add_argument(
        "--single-acc-cache",
        type=str,
        default="src/.cache/single_task_acc.json",
        help="JSON cache for baseline accuracies keyed by model/pretrain/dataset/baseline mode/checkpoint.",
    )
    p.add_argument(
        "--recompute-single-acc",
        action="store_true",
        help="Ignore cached single-task accuracy and recompute it.",
    )
    p.add_argument(
        "--single-acc-zero-shot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also compute zero-shot base-model accuracy per task (reported only; not used for normalization).",
    )
    p.add_argument(
        "--text-features-source",
        type=str,
        choices=["auto", "zero_shot", "tuned_ckpt"],
        default=None,
        help=(
            "Text features for classification: "
            "'auto' (default: use tuned_text_features from checkpoint when present, else zero-shot), "
            "'zero_shot', or 'tuned_ckpt' (strict)."
        ),
    )
    p.add_argument(
        "--zero-shot-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip all merge/tuned-checkpoint logic and run only base-model zero-shot evaluation.",
    )

    # Alpha search
    add_alpha_args(
        p,
        alpha_default=None,
        alpha_min_default=0.0,
        alpha_max_default=2.0,
        alpha_step_default=0.1,
        alpha_search_default=None,
        alpha_search_help="Enable linear search over alpha.",
    )
    add_postmerge_args(p)
    p.add_argument(
        "--forward-mode",
        type=str,
        choices=["auto", *list_forward_modes()],
        default="auto",
        help="Inference forward mode. 'auto' uses linearized_ntk when all tuned checkpoints explicitly saved forward_mode='linearized_ntk'.",
    )
    add_logging_args(p)

    args = p.parse_args()
    method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")
    postmerge_cli = build_postmerge_overrides(args).get("postmerge", {})

    # Load config file if provided (JSON), then override with CLI where meaningful.
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)

    cli_overrides: dict[str, Any] = {
        "backbone_name": args.backbone_name,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "no_humanize": args.no_humanize,
        "single_acc_cache": args.single_acc_cache,
        "recompute_single_acc": bool(args.recompute_single_acc),
        "single_acc_zero_shot": args.single_acc_zero_shot,
        "text_features_source": args.text_features_source,
        "zero_shot_only": args.zero_shot_only,
        "forward_mode": args.forward_mode,
    }
    cli_overrides = merge_non_none(cli_overrides, build_common_eval_overrides(args))
    cli_overrides = merge_non_none(
        cli_overrides,
        build_common_merge_overrides(args=args, method_params=method_params_cli, strict_as_bool=True),
    )
    cfg = merge_non_none(cfg, cli_overrides)
    if postmerge_cli:
        existing_postmerge = cfg.get("postmerge", {})
        if existing_postmerge is None:
            existing_postmerge = {}
        if not isinstance(existing_postmerge, dict):
            raise ValueError("config['postmerge'] must be a dict when provided.")
        cfg["postmerge"] = merge_non_none(dict(existing_postmerge), dict(postmerge_cli))
    logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
    cfg["logging"] = logging_cfg

    suite_name = cfg.get("suite", "vision8")
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    suite = SUITES[suite_name]

    tasks_arg = cfg.get("tasks", "all")
    if tasks_arg == "all":
        tasks = list(suite.tasks)
    else:
        tasks = parse_csv(tasks_arg)
        allowed = set(suite.tasks)
        bad = [t for t in tasks if t not in allowed]
        if bad:
            raise ValueError(f"Unknown tasks for suite '{suite_name}': {bad}. Allowed: {sorted(allowed)}")
    run_summary_path = default_summary_path(
        entrypoint="eval.vision_merge",
        logging_cfg=logging_cfg,
        default_parent=(Path(str(cfg["save_merged"])).parent if cfg.get("save_merged") else None),
    )
    run_logger = start_run(
        entrypoint="eval.vision_merge",
        logging_cfg=logging_cfg,
        summary_path=run_summary_path,
        metadata={
            "config_path": args.config,
            "resolved_config": cfg,
            "suite": suite_name,
            "tasks": tasks,
            "summary_path": str(run_summary_path),
        },
    )
    subspace_artifact_dir = run_summary_path.with_name(f"{run_summary_path.stem}.artifacts")

    tuned_by_task = cfg.get("tuned_ckpts", None)
    if tuned_by_task is not None:
        tuned_by_task = {t: resolve_ckpt_path(str(p)) for t, p in tuned_by_task.items()}
    zero_shot_only = _resolve_zero_shot_only(cfg)
    if zero_shot_only:
        if tuned_by_task is not None:
            print("zero_shot_only=True: ignoring tuned_ckpts and merge method.")
        else:
            print("No tuned_ckpts provided; running zero-shot-only evaluation.")
    elif not tuned_by_task:
        raise ValueError("You must provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")
    method_params = cfg.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("config['method_params'] must be a dict when provided.")
    method_params = dict(method_params)
    strict_load = bool(cfg.get("strict_load", False))
    merge_weights = cfg.get("weights", None)
    merge_weights_raw = merge_weights

    # Build open_clip classifier (single instance used for everything to keep memory bounded)
    backbone_name = cfg.get("backbone_name", "openclip")
    dtype = cfg.get("dtype", None)
    if dtype is None and backbone_name == "openai_clip":
        # OpenAI CLIP often loads in fp16 on GPU; keep eval in fp32 unless the config
        # explicitly opts into a lower precision to avoid image/weight dtype mismatches.
        dtype = "fp32"

    build_cfg = OpenClipBuildConfig(
        loader=backbone_name,
        model_name=cfg.get("clip_model", "ViT-B-32"),
        pretrained=cfg.get("clip_pretrained", "openai"),
        device=cfg.get("device", "cuda"),
        dtype=dtype,
    )
    clf = OpenClipClassifier.build(build_cfg)

    # Base state dict (CPU) for merging
    base_ckpt = cfg.get("base_ckpt", None)
    if base_ckpt is None:
        print(
            f"Using {build_cfg.loader} {build_cfg.model_name} (pretrain={build_cfg.pretrained}) "
            "weights as base checkpoint"
        )
        base_sd = {k: v.detach().cpu() for k, v in clf.model.state_dict().items()}
    else:
        print(f"Loading base checkpoint from {base_ckpt}")
        base_sd = load_ckpt(str(base_ckpt))

    # Load, align tuned checkpoints once (CPU)
    peft_subspace = str(cfg.get("peft_subspace", "full"))

    tuned_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}
    peft_state_by_task: dict[str, dict[str, torch.Tensor]] = {}
    peft_dense_state_by_task: dict[str, dict[str, torch.Tensor]] = {}
    attn_meta_by_task: dict[str, TaskAttentionMeta] = {}
    forward_mode_by_task: dict[str, str | None] = {}
    forward_mode_params_by_task: dict[str, dict[str, Any] | None] = {}
    tuned_text_features_by_task: dict[str, torch.Tensor | None] = {}
    peft_cfg_map: dict[str, Any] | None = None
    peft_cfg: dict[str, Any] | None = None
    subspace = None
    subspace_prepared = None
    base_patched_for_attn = False
    tuned_sds_list: list[dict[str, torch.Tensor]] = []
    dense_tuned_sds_list: list[dict[str, torch.Tensor]] = []
    dense_base_sd_for_merge: dict[str, torch.Tensor] = {}
    base_sd_for_merge = to_cpu_fp32(base_sd)
    merge_base_sd = to_cpu_fp32(base_sd)
    resolved_merge_weights = resolve_merge_weights(len(tasks), merge_weights) if tasks else []

    if not zero_shot_only:
        for t in tasks:
            ckpt_ref = str(tuned_by_task[t])
            ckpt_path, obj = load_vision_checkpoint_reference(ckpt_ref=ckpt_ref)
            obj = normalize_peft_adapter_dir_checkpoint(obj, checkpoint_path=ckpt_path)
            print(f"Loaded checkpoint for task '{t}' from {ckpt_path}")
            forward_mode_by_task[t] = obj.get("forward_mode", None) if isinstance(obj, dict) else None
            forward_mode_params_by_task[t] = (
                normalize_forward_mode_params(str(forward_mode_by_task[t]), obj.get("forward_mode_params", None))
                if isinstance(obj, dict) and forward_mode_by_task[t] is not None
                else None
            )
            attn_meta_by_task[t] = extract_checkpoint_attn_patch_info(obj=obj, ckpt_path=ckpt_path)
            # Merge/eval stays stage-agnostic: it only consumes final tuned_text_features.
            # Stage-specific artifacts (e.g. tuned_prompt_context) are ignored here.
            tuned_text_features_by_task[t] = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(
                obj=obj,
                ckpt_path=ckpt_path,
            )

            is_peft = False
            state: dict[str, torch.Tensor] | None = None
            cfg_map: dict[str, Any] | None = None
            dense_state = dict(obj.get("peft_dense_state", {})) if isinstance(obj, dict) and isinstance(obj.get("peft_dense_state", {}), dict) else {}

            if is_peft_adapter_dir_ckpt(obj):
                state, cfg_map = load_peft_adapter_dir_components(obj["peft_adapter_dir"], checkpoint_path=ckpt_path)
                is_peft = True
            elif is_peft_checkpoint(obj) and isinstance(obj, dict):
                state, cfg_map = extract_peft_components(obj)
                is_peft = True

            if peft_subspace != "full":
                if not is_peft:
                    raise ValueError(f"peft_subspace='{peft_subspace}' requires PEFT checkpoints. Got: {ckpt_path}")
                assert state is not None and cfg_map is not None
                peft_cfg_map = ensure_peft_cfg_map(peft_cfg_map, cfg_map)
                peft_state_by_task[t] = state
                peft_dense_state_by_task[t] = dense_state
            else:
                base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
                    task_meta=attn_meta_by_task[t],
                    base_patched_for_attn=base_patched_for_attn,
                    clf=clf,
                    base_ckpt=base_ckpt,
                    strict_load=strict_load,
                    base_sd=base_sd,
                )
                if is_peft:
                    assert state is not None and cfg_map is not None
                    peft_cfg_map = ensure_peft_cfg_map(peft_cfg_map, cfg_map)

                    # if checkpoint is PEFT but we want full, we construct full weights now
                    sd = materialize_peft_sd_from_adapter(
                        peft_state=state,
                        base_sd=base_sd,
                        build_cfg=build_cfg,
                        peft_cfg=get_peft_cfg(cfg_map),
                        peft_dense_state=dense_state,
                        strict_load=strict_load,
                        patched_attn=attn_meta_by_task[t].patched_attn,
                        attn_patch_cfg=attn_meta_by_task[t].attn_patch_cfg,
                    )
                else:
                    sd = load_ckpt(ckpt_path)
                aligned = align_to_base_keys(sd, base_sd)
                if not aligned:
                    raise ValueError(
                        f"No tensors from tuned checkpoint aligned to base keys for task '{t}': {ckpt_path}. "
                        "Check checkpoint key prefixes and model compatibility."
                    )
                tuned_sds_by_task[t] = to_cpu_fp32(aligned)

        # Attention mode (patched + linearized-vs-softmax) must be consistent across tuned checkpoints.
        attn_meta_tasks = [t for t in tasks if t in attn_meta_by_task]
        if attn_meta_tasks:
            flag0 = attn_meta_by_task[attn_meta_tasks[0]].patched_attn
            if any(attn_meta_by_task[t].patched_attn != flag0 for t in attn_meta_tasks):
                raise ValueError("Inconsistent patched_attn flags across tuned checkpoints.")
            patch_tasks = [t for t in attn_meta_tasks if attn_meta_by_task[t].patched_attn]
            if patch_tasks:
                patch_cfg0 = attn_meta_by_task[patch_tasks[0]].attn_patch_cfg or {}
                for t in patch_tasks[1:]:
                    patch_cfgt = attn_meta_by_task[t].attn_patch_cfg or {}
                    if patch_cfgt != patch_cfg0:
                        raise ValueError("Inconsistent attn_patch_cfg across tuned checkpoints.")
                if patch_cfg0:
                    print(f"Using checkpoint attention mode: {patch_cfg0.get('attn_impl', 'softmax')}")
            # In subspace mode we still need patched attention keyspace in the base model/state dict
            # because lifted deltas target q_proj/k_proj/v_proj keys.
            base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
                task_meta=attn_meta_by_task[attn_meta_tasks[0]],
                base_patched_for_attn=base_patched_for_attn,
                clf=clf,
                base_ckpt=base_ckpt,
                strict_load=strict_load,
                base_sd=base_sd,
            )
        merge_base_sd = to_cpu_fp32(base_sd)

        if peft_subspace != "full":
            if peft_cfg_map is None:
                raise ValueError(f"peft_subspace='{peft_subspace}' requires peft_config in checkpoints.")
            peft_cfg = get_peft_cfg(peft_cfg_map)
            subspace = get_subspace(peft_subspace)
            subspace_prepared = subspace.prepare(
                lora_by_task=peft_state_by_task,
                peft_cfg=peft_cfg,
                method_params=method_params,
                weights=resolved_merge_weights,
                artifact_dir=subspace_artifact_dir,
            )
            if getattr(subspace_prepared, "merge_weight_override", None) is not None:
                merge_weights = list(subspace_prepared.merge_weight_override)
            projected_by_task = subspace.project(subspace_prepared, lora_by_task=peft_state_by_task, peft_cfg=peft_cfg)
            if not projected_by_task:
                raise ValueError("Subspace projection returned empty projected_by_task.")
            tuned_sds_list = [projected_by_task[t] for t in tasks]
            base_sd_for_merge = {k: torch.zeros_like(v) for k, v in tuned_sds_list[0].items()}
            lora_only_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}

            # Build full-space tuned checkpoints once for single-task baseline eval.
            for t in tasks:
                tuned_sd = materialize_peft_sd_from_adapter(
                    peft_state=peft_state_by_task[t],
                    base_sd=base_sd,
                    build_cfg=build_cfg,
                    peft_cfg=peft_cfg,
                    peft_dense_state=peft_dense_state_by_task.get(t, None),
                    strict_load=strict_load,
                    patched_attn=attn_meta_by_task[t].patched_attn,
                    attn_patch_cfg=attn_meta_by_task[t].attn_patch_cfg,
                )
                aligned = align_to_base_keys(tuned_sd, base_sd)
                if not aligned:
                    raise ValueError(
                        f"No tensors from tuned checkpoint aligned to base keys for task '{t}'. "
                        "Check checkpoint key prefixes and model compatibility."
                    )
                tuned_sds_by_task[t] = to_cpu_fp32(aligned)
                lora_only_sd = materialize_peft_sd_from_adapter(
                    peft_state=peft_state_by_task[t],
                    base_sd=base_sd,
                    build_cfg=build_cfg,
                    peft_cfg=peft_cfg,
                    peft_dense_state=None,
                    strict_load=strict_load,
                    patched_attn=attn_meta_by_task[t].patched_attn,
                    attn_patch_cfg=attn_meta_by_task[t].attn_patch_cfg,
                )
                lora_only_aligned = align_to_base_keys(lora_only_sd, base_sd)
                if not lora_only_aligned:
                    raise ValueError(
                        f"No tensors from LoRA-only checkpoint aligned to base keys for task '{t}'. "
                        "Check checkpoint key prefixes and model compatibility."
                    )
                lora_only_sds_by_task[t] = to_cpu_fp32(lora_only_aligned)
            base_sd_for_merge = to_cpu_fp32(base_sd_for_merge)
            dense_base_sd_for_merge, dense_tuned_sds_list = build_dense_delta_branch(
                tasks=tasks,
                full_tuned_by_task=tuned_sds_by_task,
                lora_only_tuned_by_task=lora_only_sds_by_task,
                base_sd=merge_base_sd,
            )
        else:
            tuned_sds_list = [tuned_sds_by_task[t] for t in tasks]
            base_sd_for_merge = to_cpu_fp32(base_sd)

        needs_linear_attention = any(attn_meta_by_task[t].linearized_attn for t in tasks)
        assert_qkv_patched_before_linearizing(
            needs_linear_attention=needs_linear_attention,
            base_patched_for_attn=base_patched_for_attn,
            model_state_dict=clf.model.state_dict(),
        )
        if needs_linear_attention:
            print("Verified q/k/v attention patch is active before linearized attention evaluation.")

    requested_forward_mode = str(cfg.get("forward_mode", "auto"))
    if requested_forward_mode == "auto":
        resolved_forward_mode = (
            resolve_auto_forward_mode([forward_mode_by_task.get(t) for t in tasks])
            if (not zero_shot_only) and bool(tasks)
            else "standard"
        )
    else:
        resolved_forward_mode = requested_forward_mode

    resolved_forward_mode_params = resolve_shared_forward_mode_params(
        resolved_forward_mode,
        [
            forward_mode_params_by_task.get(t)
            for t in tasks
            if forward_mode_by_task.get(t) == "linearized_ntk"
        ],
    )
    forward_mode = get_forward_mode(resolved_forward_mode)
    forward_mode.bind(
        clf=clf,
        base_sd=merge_base_sd,
        strict_load=strict_load,
        params=resolved_forward_mode_params,
    )
    print(f"Using forward mode: {resolved_forward_mode} params={resolved_forward_mode_params}")

    if not zero_shot_only:
        print("base keys:", len(base_sd_for_merge))
        print("example tuned aligned keys:", len(tuned_sds_list[0]))
        print("example intersection:", len(set(base_sd_for_merge).intersection(tuned_sds_list[0])))

    device = str(cfg.get("device", "cuda"))
    text_features_source = str(cfg.get("text_features_source", "auto")).strip().lower()
    if text_features_source not in {"auto", "zero_shot", "tuned_ckpt"}:
        raise ValueError("text_features_source must be one of: auto, zero_shot, tuned_ckpt")
    if zero_shot_only and text_features_source != "zero_shot":
        print("zero_shot_only=True: forcing text_features_source='zero_shot'.")
        text_features_source = "zero_shot"
    print(f"Text features source: {text_features_source}")
    use_humanized_classnames = not bool(cfg.get("no_humanize", True))
    classnames_mode = "humanized" if use_humanized_classnames else "raw"
    print(f"Classname mode: {classnames_mode}")

    # Pre-load datasets/loaders and compute single-task accuracies once (cached)
    single_cache_path = str(cfg.get("single_acc_cache", "src/.cache/single_task_acc.json"))
    single_cache = read_json_silent(single_cache_path)
    recompute_single = bool(cfg.get("recompute_single_acc", False))
    compute_zero_shot_acc = True if zero_shot_only else bool(cfg.get("single_acc_zero_shot", False))
    if zero_shot_only:
        print("Single-accuracy baseline mode: zero-shot only (no tuned checkpoints)")
    else:
        print("Single-accuracy baseline mode: single-task tuned (used for normalization)")
    if compute_zero_shot_acc:
        print("Zero-shot base-model accuracies will also be computed (not used for normalization).")

    per_task = []  # {task, loaders, classnames, build_cfg_task, single_acc, zero_shot_acc?}
    for task in tasks:
        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf.preprocess,
            train_preprocess=None,
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
            loader=build_cfg.loader,
            model_name=build_cfg.model_name,
            pretrained=build_cfg.pretrained,
            device=build_cfg.device,
            dtype=build_cfg.dtype,
            prompt_templates=templates,
        )

        task_text_features = None
        task_text_features_mode: str | None = None
        single_acc: float | None = None
        zero_shot_acc: float | None = None

        if not zero_shot_only:
            # Single-task baseline (used for normalization) always comes from the tuned checkpoint.
            task_text_features, task_text_features_mode = clf.resolve_eval_text_features(
                text_features_source=text_features_source,
                classnames=classnames,
                build_cfg=build_cfg_task,
                tuned_text_features=tuned_text_features_by_task.get(task, None),
                cache_dir="src/.cache/zs_cache",
                force_rebuild_zeroshot=False,
                task_name=task,
                ckpt_path=str(tuned_by_task[task]),
                verbose=True,
            )

            k = acc_cache_key(
                build_cfg.model_name,
                build_cfg.pretrained,
                task,
                chk_path=str(tuned_by_task[task]),
                baseline_mode="tuned",
                forward_mode=resolved_forward_mode,
                forward_mode_params=resolved_forward_mode_params,
                classnames_mode=classnames_mode,
                text_features_mode=task_text_features_mode,
            )
            if not recompute_single:
                single_acc = _read_cached_top1(
                    single_cache,
                    key=k,
                    label="single-task tuned acc",
                    task=task,
                )

            if single_acc is None:
                # Load tuned weights into the single model instance, evaluate, then overwrite later during merge eval.
                tuned_sd = tuned_sds_by_task[task]
                load_into_model(clf.model, tuned_sd, strict=strict_load)
                t0 = time.time()
                single_acc = eval_task_top1(
                    clf=clf,
                    loaders=loaders,
                    classnames=classnames,
                    build_cfg_task=build_cfg_task,
                    device=device,
                    split="test",
                    text_features=task_text_features,
                )
                dt = time.time() - t0

                _write_cached_top1(
                    single_cache,
                    cache_path=single_cache_path,
                    key=k,
                    top1=float(single_acc),
                    model_name=build_cfg.model_name,
                    pretrained=build_cfg.pretrained,
                    task=task,
                    baseline_mode="tuned",
                    checkpoint=str(tuned_by_task[task]),
                    seconds=float(dt),
                    label="single-task tuned acc",
                )
        else:
            task_text_features_mode = "zero_shot"

        if compute_zero_shot_acc:
            k_zs = acc_cache_key(
                build_cfg.model_name,
                build_cfg.pretrained,
                task,
                chk_path=str(base_ckpt) if base_ckpt is not None else "open_clip_pretrained",
                baseline_mode="zero_shot",
                forward_mode=resolved_forward_mode,
                forward_mode_params=resolved_forward_mode_params,
                classnames_mode=classnames_mode,
                text_features_mode="zero_shot",
            )
            if not recompute_single:
                zero_shot_acc = _read_cached_top1(
                    single_cache,
                    key=k_zs,
                    label="zero-shot acc",
                    task=task,
                )

            if zero_shot_acc is None:
                load_into_model(clf.model, merge_base_sd, strict=strict_load)
                t0 = time.time()
                zero_shot_acc = eval_task_top1(
                    clf=clf,
                    loaders=loaders,
                    classnames=classnames,
                    build_cfg_task=build_cfg_task,
                    device=device,
                    split="test",
                    text_features=None,
                )
                dt = time.time() - t0
                _write_cached_top1(
                    single_cache,
                    cache_path=single_cache_path,
                    key=k_zs,
                    top1=float(zero_shot_acc),
                    model_name=build_cfg.model_name,
                    pretrained=build_cfg.pretrained,
                    task=task,
                    baseline_mode="zero_shot",
                    checkpoint=(str(base_ckpt) if base_ckpt is not None else "open_clip_pretrained"),
                    seconds=float(dt),
                    label="zero-shot acc",
                )

        if zero_shot_only:
            if zero_shot_acc is None:
                raise RuntimeError(f"Zero-shot-only mode failed to compute zero-shot accuracy for task '{task}'.")
            single_acc = float(zero_shot_acc)

        per_task.append(
            {
                "task": task,
                "loaders": loaders,
                "classnames": classnames,
                "build_cfg_task": build_cfg_task,
                "text_features": task_text_features,
                "single_acc": float(single_acc),
                "zero_shot_acc": (float(zero_shot_acc) if zero_shot_acc is not None else None),
            }
        )

    if zero_shot_only:
        zs_accs = [float(item["single_acc"]) for item in per_task]
        print(f"Average zero-shot acc across {len(zs_accs)} tasks: {sum(zs_accs) / len(zs_accs):.6f}")
        pretty_print_task_accuracies(
            suite_name,
            "zero_shot",
            peft_subspace,
            per_task,
            zs_accs,
            [1.0] * len(zs_accs),
            single_accs=zs_accs,
            baseline_label="zero_shot",
            result_label="top1",
        )
        print_latex_task_rows(per_task, zs_accs, [1.0] * len(zs_accs))
        if run_logger is not None:
            run_logger.log_summary(
                {
                    "mode": "zero_shot_only",
                    "suite": suite_name,
                    "tasks": tasks,
                    "per_task_acc": {item["task"]: float(item["single_acc"]) for item in per_task},
                    "avg_acc": float(sum(zs_accs) / len(zs_accs)),
                }
            )
            run_logger.finish("success")
        return

    print(
        f"Average single-task tuned acc across {len(per_task)} tasks: {sum(item['single_acc'] for item in per_task) / len(per_task):.6f}"
    )
    if compute_zero_shot_acc:
        zs_vals = [float(item["zero_shot_acc"]) for item in per_task if item.get("zero_shot_acc") is not None]
        if zs_vals:
            print(f"Average zero-shot acc across {len(zs_vals)} tasks: {sum(zs_vals) / len(zs_vals):.6f}")

    # Merge method
    method = get_method(str(cfg.get("method", "task_arithmetic")))
    merge_context = {
        "kind": "vision",
        "cfg": cfg,
        "model": clf.model,
        "classifier": clf,
        "device": device,
        "strict_load": strict_load,
        "tasks": tasks,
        "per_task": per_task,
        "tuned_state_by_task": tuned_sds_by_task,
        "num_workers": int(cfg.get("num_workers", 6)),
        "seed": int(cfg.get("seed", 42)),
        "peft_subspace": peft_subspace,
        "subspace_prepared": subspace_prepared,
        "peft_state_by_task": peft_state_by_task,
        "suite_name": suite_name,
    }
    prepared = None
    search_planner = build_search_planner(cfg=cfg, base_method_params=method_params)
    subspace_state_cache: dict[str, dict[str, Any]] = {}
    prepared_cache: dict[str, Any] = {}
    
    dense_prepared_cache: dict[str, Any] = {}

    def _subspace_state_for(candidate_method_params: dict[str, Any]) -> dict[str, Any]:
        if peft_subspace == "full" or subspace is None or peft_cfg is None:
            return {
                "subspace_prepared": subspace_prepared,
                "tuned_sds_list": tuned_sds_list,
                "base_sd_for_merge": base_sd_for_merge,
                "weights": merge_weights,
            }

        cache_key = stable_method_params_cache_key(candidate_method_params)
        if cache_key in subspace_state_cache:
            return subspace_state_cache[cache_key]

        print(f"\nPreparing PEFT subspace: {peft_subspace} ({candidate_method_params})")
        candidate_subspace_prepared = subspace.prepare(
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
            method_params=candidate_method_params,
            weights=resolved_merge_weights,
            artifact_dir=subspace_artifact_dir,
        )
        candidate_weights = (
            list(candidate_subspace_prepared.merge_weight_override)
            if getattr(candidate_subspace_prepared, "merge_weight_override", None) is not None
            else merge_weights_raw
        )
        projected_by_task = subspace.project(
            candidate_subspace_prepared,
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
        )
        if not projected_by_task:
            raise ValueError("Subspace projection returned empty projected_by_task.")
        candidate_tuned_sds_list = [projected_by_task[t] for t in tasks]
        candidate_base_sd_for_merge = to_cpu_fp32({k: torch.zeros_like(v) for k, v in candidate_tuned_sds_list[0].items()})
        state = {
            "subspace_prepared": candidate_subspace_prepared,
            "tuned_sds_list": candidate_tuned_sds_list,
            "base_sd_for_merge": candidate_base_sd_for_merge,
            "weights": candidate_weights,
        }
        subspace_state_cache[cache_key] = state
        return state

    def _subspace_state_for(candidate_method_params: dict[str, Any]) -> dict[str, Any]:
        if peft_subspace == "full" or subspace is None or peft_cfg is None:
            return {
                "subspace_prepared": subspace_prepared,
                "tuned_sds_list": tuned_sds_list,
                "base_sd_for_merge": base_sd_for_merge,
                "weights": merge_weights,
            }

        cache_key = stable_method_params_cache_key(candidate_method_params)
        if cache_key in subspace_state_cache:
            return subspace_state_cache[cache_key]

        print(f"\nPreparing PEFT subspace: {peft_subspace} ({candidate_method_params})")
        candidate_subspace_prepared = subspace.prepare(
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
            method_params=candidate_method_params,
            weights=resolved_merge_weights,
            artifact_dir=subspace_artifact_dir,
        )
        candidate_weights = (
            list(candidate_subspace_prepared.merge_weight_override)
            if getattr(candidate_subspace_prepared, "merge_weight_override", None) is not None
            else merge_weights_raw
        )
        projected_by_task = subspace.project(
            candidate_subspace_prepared,
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
        )
        if not projected_by_task:
            raise ValueError("Subspace projection returned empty projected_by_task.")
        candidate_tuned_sds_list = [projected_by_task[t] for t in tasks]
        candidate_base_sd_for_merge = to_cpu_fp32({k: torch.zeros_like(v) for k, v in candidate_tuned_sds_list[0].items()})
        state = {
            "subspace_prepared": candidate_subspace_prepared,
            "tuned_sds_list": candidate_tuned_sds_list,
            "base_sd_for_merge": candidate_base_sd_for_merge,
            "weights": candidate_weights,
        }
        subspace_state_cache[cache_key] = state
        return state

    def _prepared_for(candidate_method_params: dict[str, Any]) -> Any:
        if not isinstance(method, PreparedMergeMethod):
            return None
        candidate_subspace_state = _subspace_state_for(candidate_method_params)
        cache_prepared = bool(candidate_method_params.get("cache_prepared", True))
        cache_key = stable_method_params_cache_key(candidate_method_params)
        if cache_prepared and cache_key in prepared_cache:
            return prepared_cache[cache_key]
        print(f"\nPreparing merge directions with method: {method.name} ({candidate_method_params})")
        candidate_merge_context = dict(merge_context)
        candidate_merge_context["subspace_prepared"] = candidate_subspace_state["subspace_prepared"]
        prepared_value = method.prepare(
            base=candidate_subspace_state["base_sd_for_merge"],
            tuned=candidate_subspace_state["tuned_sds_list"],
            weights=candidate_subspace_state["weights"],
            strict=strict_load,
            tasks=tasks,
            merge_context=candidate_merge_context,
            method_params=candidate_method_params,
        )
        if cache_prepared:
            prepared_cache[cache_key] = prepared_value
        return prepared_value

    def _dense_prepared_for(candidate_method_params: dict[str, Any]) -> Any:
        if not isinstance(method, PreparedMergeMethod):
            return None
        if peft_subspace == "full" or not dense_tuned_sds_list or not dense_base_sd_for_merge:
            return None
        cache_prepared = bool(candidate_method_params.get("cache_prepared", True))
        cache_key = str(sorted(candidate_method_params.items()))
        if cache_prepared and cache_key in dense_prepared_cache:
            return dense_prepared_cache[cache_key]
        prepared_value = method.prepare(
            base=dense_base_sd_for_merge,
            tuned=dense_tuned_sds_list,
            weights=merge_weights,
            strict=strict_load,
            tasks=tasks,
            merge_context={
                "kind": "vision_dense_delta",
                "cfg": cfg,
                "tasks": tasks,
                "suite_name": suite_name,
                "peft_subspace": peft_subspace,
            },
            method_params=candidate_method_params,
        )
        if cache_prepared:
            dense_prepared_cache[cache_key] = prepared_value
        return prepared_value

    if isinstance(method, PreparedMergeMethod):
        fixed_only = not search_planner.is_multi_param() and not bool(cfg.get("hyperparam_search"))
        if fixed_only:
            prepared = _prepared_for(method_params)
            if prepared is not None:
                print("Prepared merge directions will be reused across all alpha evaluations.")

    postmerge_cfg_raw = cfg.get("postmerge", None)
    if postmerge_cfg_raw is not None and not isinstance(postmerge_cfg_raw, dict):
        raise ValueError("config['postmerge'] must be a dict when provided.")
    postmerge_cfg = dict(postmerge_cfg_raw) if isinstance(postmerge_cfg_raw, dict) else {}
    postmerge_name = postmerge_cfg.get("method", None)
    if postmerge_name is not None:
        postmerge_cfg.setdefault("device", device)
        postmerge_method = get_postmerge_method(str(postmerge_name))
        print(f"\n=== Postmerge method = {postmerge_method.name} ===")
        t1 = time.time()
        postmerge_result = postmerge_method.run(
            PostMergeContext(
                kind="vision",
                model=clf.model,
                base=base_sd_for_merge,
                tuned=tuned_sds_list,
                tasks=tasks,
                weights=merge_weights,
                peft_subspace=peft_subspace,
                config=postmerge_cfg,
                resources={
                    "classifier": clf,
                    "per_task": per_task,
                    "device": device,
                },
            )
        )
        print(f"Postmerge method '{postmerge_method.name}' completed in {time.time() - t1:.2f} seconds.")

        miss, unexp = load_into_model(clf.model, postmerge_result.merged_state, strict=strict_load)
        print(f"Loaded postmerged weights. missing={miss}, unexpected={unexp}")

        merged_accs, norm_accs = eval_norm_accs_for_split(
            clf=clf,
            per_task=per_task,
            device=device,
            split="test",
            print_per_task=False,
        )
        result_method_name = f"{method.name}+{postmerge_method.name}"
        pretty_print_task_accuracies(
            suite_name,
            result_method_name,
            peft_subspace,
            per_task,
            merged_accs,
            norm_accs,
            single_accs=[item["single_acc"] for item in per_task],
        )
        print_latex_task_rows(per_task, merged_accs, norm_accs)
        saved_merged_path = _save_merged_state_dict_if_requested(
            postmerge_result.merged_state,
            cfg.get("save_merged", None),
            label="postmerged",
        )
        if run_logger is not None:
            run_logger.log_summary(
                {
                    "suite": suite_name,
                    "tasks": tasks,
                    "method": method.name,
                    "postmerge": postmerge_result.metadata,
                    "peft_subspace": peft_subspace,
                    "test_results": {
                        "per_task_acc": {item["task"]: float(merged_accs[idx]) for idx, item in enumerate(per_task)},
                        "per_task_norm_acc": {item["task"]: float(norm_accs[idx]) for idx, item in enumerate(per_task)},
                        "avg_acc": float(sum(merged_accs) / len(merged_accs)),
                        "avg_norm_acc": float(sum(norm_accs) / len(norm_accs)),
                    },
                    "saved_merged_path": saved_merged_path,
                }
            )
            run_logger.finish("success")
        return

    alpha_early_stop = bool(cfg.get("alpha_early_stop", True))

    search_results: list[SearchEvaluation] = []
    best_norm_per_task: dict[str, float] = {}
    best_alpha_per_task: dict[str, float] = {}
    best_method_params_per_task: dict[str, dict[str, Any]] = {}
    best_result: SearchEvaluation | None = None
    legacy_alpha_results: dict[float, list[float]] = {}
    legacy_alpha_results_norm: dict[float, list[float]] = {}

    while True:
        batch = search_planner.next_batch()
        if batch is None:
            break
        batch_results: list[SearchEvaluation] = []
        batch_best_score = float("-inf")
        for candidate in batch:
            print(f"\n=== Method = {method.name} - Space = {peft_subspace} - {describe_candidate(candidate)} ===")
            candidate_subspace_state = _subspace_state_for(candidate.method_params)
            candidate_prepared = prepared if prepared is not None else _prepared_for(candidate.method_params)
            candidate_dense_prepared = _dense_prepared_for(candidate.method_params)
            merged_sd = build_merged_state_for_alpha(
                method=method,
                prepared=candidate_prepared,
                base_sd_for_merge=candidate_subspace_state["base_sd_for_merge"],
                tuned_sds_list=candidate_subspace_state["tuned_sds_list"],
                weights=candidate_subspace_state["weights"],
                method_params=candidate.method_params,
                alpha=float(candidate.alpha),
                peft_subspace=peft_subspace,
                subspace=subspace,
                subspace_prepared=candidate_subspace_state["subspace_prepared"],
                peft_cfg=peft_cfg,
                peft_state_by_task=peft_state_by_task,
                tasks=tasks,
                merge_base_sd=merge_base_sd,
                dense_prepared=candidate_dense_prepared,
                dense_base_sd_for_merge=dense_base_sd_for_merge,
                dense_tuned_sds_list=dense_tuned_sds_list,
            )

            miss, unexp = load_into_model(clf.model, merged_sd, strict=strict_load)
            print(f"Loaded merged weights ({describe_candidate(candidate)}). missing={miss}, unexpected={unexp}")

            del merged_sd

            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.empty_cache()

            accs, norm_accs = eval_norm_accs_for_split(
                clf=clf,
                per_task=per_task,
                device=device,
                split="val",
                print_per_task=True,
            )
            avg_norm = sum(float(norm) for norm in norm_accs) / len(tasks)
            avg_abs = sum(float(acc) for acc in accs) / len(tasks)
            result = SearchEvaluation(
                candidate=candidate,
                score=float(avg_norm),
                avg_acc=float(avg_abs),
                avg_norm_acc=float(avg_norm),
                per_task_acc=[float(v) for v in accs],
                per_task_norm_acc=[float(v) for v in norm_accs],
            )
            batch_results.append(result)
            search_results.append(result)

            if not search_planner.is_multi_param():
                legacy_alpha_results[float(candidate.alpha)] = [float(v) for v in accs]
                legacy_alpha_results_norm[float(candidate.alpha)] = [float(v) for v in norm_accs]

            for idx, item in enumerate(per_task):
                task = str(item["task"])
                norm = float(norm_accs[idx])
                if (task not in best_norm_per_task) or (norm > best_norm_per_task[task]):
                    best_norm_per_task[task] = norm
                    best_alpha_per_task[task] = float(candidate.alpha)
                    best_method_params_per_task[task] = dict(candidate.method_params)

            print(f"{describe_candidate(candidate)}  avg_abs={avg_abs:.6f} avg_norm={avg_norm:.6f}")
            if run_logger is not None:
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(candidate.alpha),
                        "alpha/avg_acc": float(avg_abs),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "search_stage": int(candidate.stage),
                        "method_params": candidate.method_params,
                        "search_values": candidate.values,
                        "per_task_acc": {item["task"]: float(accs[idx]) for idx, item in enumerate(per_task)},
                        "per_task_norm_acc": {item["task"]: float(norm_accs[idx]) for idx, item in enumerate(per_task)},
                    },
                )

            if best_result is None or result.score > best_result.score:
                best_result = result
            if result.score > batch_best_score:
                batch_best_score = result.score
            elif len(batch) > 1 and alpha_early_stop:
                print("Avg norm did not improve for this parameter setting, stopping this alpha sweep early.")
                break

        search_planner.observe(batch_results)

    if best_result is None:
        raise RuntimeError("Search produced no evaluated candidates.")

    print("\n=== Search summary ===")
    for result in search_results:
        print(
            f"{describe_candidate(result.candidate)}  avg_abs={result.avg_acc:.6f} avg_norm={result.avg_norm_acc:.6f}"
        )
    best_alpha = float(best_result.candidate.alpha)
    best_method_params = dict(best_result.candidate.method_params)
    print(
        f"\nBest setting: {describe_candidate(best_result.candidate)} -> "
        f"avg_abs={best_result.avg_acc:.6f} avg_norm={best_result.avg_norm_acc:.6f}"
    )

    print("\nBest setting per task:")
    for t in tasks:
        if t in best_alpha_per_task:
            method_desc = best_method_params_per_task.get(t, {})
            print(
                f"{t}: alpha={best_alpha_per_task[t]:.3f} method_params={method_desc} avg_norm={best_norm_per_task[t]:.6f}"
            )

    print(f"\n(Re-running best setting ({describe_candidate(best_result.candidate)}) once to report avg_top1)")
    best_dense_prepared = _dense_prepared_for(best_method_params)
    best_subspace_state = _subspace_state_for(best_method_params)
    merged_sd = build_merged_state_for_alpha(
        method=method,
        prepared=(_prepared_for(best_method_params) if prepared is None else prepared),
        base_sd_for_merge=best_subspace_state["base_sd_for_merge"],
        tuned_sds_list=best_subspace_state["tuned_sds_list"],
        weights=best_subspace_state["weights"],
        method_params=best_method_params,
        alpha=best_alpha,
        peft_subspace=peft_subspace,
        subspace=subspace,
        subspace_prepared=best_subspace_state["subspace_prepared"],
        peft_cfg=peft_cfg,
        peft_state_by_task=peft_state_by_task,
        tasks=tasks,
        merge_base_sd=merge_base_sd,
        dense_prepared=best_dense_prepared,
        dense_base_sd_for_merge=dense_base_sd_for_merge,
        dense_tuned_sds_list=dense_tuned_sds_list,
    )
    load_into_model(clf.model, merged_sd, strict=strict_load)
    saved_merged_path = _save_merged_state_dict_if_requested(
        merged_sd,
        cfg.get("save_merged", None),
        label="best-alpha merged",
    )
    subspace_prepared = best_subspace_state["subspace_prepared"]
    del merged_sd

    merged_accs, norm_accs = eval_norm_accs_for_split(
        clf=clf,
        per_task=per_task,
        device=device,
        split="test",
        print_per_task=False,
    )

    pretty_print_task_accuracies(
        suite_name,
        method.name,
        peft_subspace,
        per_task,
        merged_accs,
        norm_accs,
        single_accs=[item["single_acc"] for item in per_task],
    )

    print_latex_task_rows(per_task, merged_accs, norm_accs)
    if run_logger is not None:
        run_logger.log_summary(
            {
                "suite": suite_name,
                "tasks": tasks,
                "method": method.name,
                "peft_subspace": peft_subspace,
                "best_alpha": float(best_alpha),
                "best_method_params": best_method_params,
                "search_strategy": search_planner.search_summary(),
                "search_results": summarize_search_results(search_results),
                "alpha_results": {str(k): [float(v) for v in vals] for k, vals in legacy_alpha_results.items()},
                "alpha_results_norm": {
                    str(k): [float(v) for v in vals] for k, vals in legacy_alpha_results_norm.items()
                },
                "best_alpha_per_task": {k: float(v) for k, v in best_alpha_per_task.items()},
                "best_method_params_per_task": best_method_params_per_task,
                "best_norm_per_task": {k: float(v) for k, v in best_norm_per_task.items()},
                "test_results": {
                    "per_task_acc": {item["task"]: float(merged_accs[idx]) for idx, item in enumerate(per_task)},
                    "per_task_norm_acc": {item["task"]: float(norm_accs[idx]) for idx, item in enumerate(per_task)},
                    "avg_acc": float(sum(merged_accs) / len(merged_accs)),
                    "avg_norm_acc": float(sum(norm_accs) / len(norm_accs)),
                },
                "saved_merged_path": saved_merged_path,
                "subspace_artifacts": (
                    {"similarity_artifact_path": getattr(subspace_prepared, "similarity_artifact_path", None)}
                    if subspace_prepared is not None
                    else {}
                ),
            }
        )
        run_logger.finish("success")


if __name__ == "__main__":
    main()
