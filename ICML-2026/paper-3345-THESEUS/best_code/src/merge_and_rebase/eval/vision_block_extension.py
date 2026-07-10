from __future__ import annotations

import argparse
import itertools
from copy import deepcopy
from typing import Any

import torch
from tqdm import tqdm

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import (
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_suite_arg,
    add_tasks_arg,
    build_logging_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import humanize, to_cpu_fp32
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..run_logging import default_summary_path, merge_logging_config, start_run
from .block_extension import resolve_block_extension_config, run_block_extension, select_loader
from .datasets.vision8_14_20 import SUITES


def _resolve_eval_loader(loaders: Any, split: str):
    if split == "val":
        return loaders.val
    if split == "test":
        return loaders.test
    raise ValueError(f"Unknown split '{split}'. Expected one of: val, test.")


def _evaluate_model_top1(
    *,
    model: torch.nn.Module,
    clf_source: OpenClipClassifier,
    loaders: Any,
    classnames: list[str],
    build_cfg_task: OpenClipBuildConfig,
    device: str,
    split: str,
    first_n_batches: int | None,
) -> float:
    eval_clf = OpenClipClassifier(
        model=model,
        tokenizer=clf_source.tokenizer,
        preprocess=clf_source.preprocess,
        normalize=clf_source.normalize,
        logit_scale=clf_source.logit_scale,
    )

    eval_loader = _resolve_eval_loader(loaders, split)
    if first_n_batches is not None:
        eval_loader = itertools.islice(iter(eval_loader), max(1, int(first_n_batches)))

    eval_clf.build_zeroshot_text_features(
        classnames, build_cfg_task, cache_dir="src/.cache/zs_cache", force_rebuild=False
    )
    return float(eval_clf.top1(eval_loader, device=device))


def _maybe_target_layers_total(
    *,
    cfg: dict[str, Any],
    source_cfg: OpenClipBuildConfig,
) -> int | None:
    target_model_name = cfg.get("target_clip_model", None)
    target_pretrained = cfg.get("target_clip_pretrained", None)
    if target_model_name is None:
        return None

    target_cfg = OpenClipBuildConfig(
        model_name=str(target_model_name),
        pretrained=str(target_pretrained) if target_pretrained is not None else source_cfg.pretrained,
        device=source_cfg.device,
        dtype=source_cfg.dtype,
    )

    clf_target_depth = OpenClipClassifier.build(target_cfg)
    try:
        return int(len(clf_target_depth.model.visual.transformer.resblocks))
    except Exception:
        return None


def main() -> None:
    run_logger = None
    p = argparse.ArgumentParser("Run block extension only and evaluate pre/post source zero-shot and FT accuracy")

    add_config_arg(p)
    add_suite_arg(p, choices=sorted(SUITES.keys()))
    add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

    p.add_argument("--source-clip-model", type=str, default=None)
    p.add_argument("--source-clip-pretrained", type=str, default=None)
    p.add_argument(
        "--target-clip-model",
        type=str,
        default=None,
        help="Optional target model used only to infer target depth. If omitted, use blocks_to_add.",
    )
    p.add_argument(
        "--target-clip-pretrained",
        type=str,
        default=None,
        help="Optional target pretrained used only to infer target depth.",
    )

    add_device_dtype_args(p, device_default=None, dtype_default=None)

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-humanize", action="store_true", default=None, help="Use raw classnames.")
    p.add_argument("--eval-split", type=str, default=None, choices=["val", "test"])
    p.add_argument("--first-n-eval-batches", type=int, default=None)

    p.add_argument("--tuned-ckpts", type=str, nargs="+", default=None)
    p.add_argument("--strict-load", action="store_true", default=None)

    p.add_argument(
        "--block-extension-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable legacy-style block extension. In this runner, defaults to enabled.",
    )
    p.add_argument(
        "--block-extension-params",
        type=str,
        default=None,
        help="JSON object for block extension kwargs.",
    )
    add_logging_args(p)

    args = p.parse_args()
    block_extension_params_cli = parse_json_object_arg(args.block_extension_params, arg_name="--block-extension-params")

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)

    cli: dict[str, Any] = {
        "source_clip_model": args.source_clip_model,
        "source_clip_pretrained": args.source_clip_pretrained,
        "target_clip_model": args.target_clip_model,
        "target_clip_pretrained": args.target_clip_pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "no_humanize": args.no_humanize,
        "suite": getattr(args, "suite", None),
        "tasks": getattr(args, "tasks", None),
        "device": args.device,
        "dtype": args.dtype,
        "tuned_ckpts": args.tuned_ckpts,
        "strict_load": args.strict_load,
        "eval_split": args.eval_split,
        "first_n_eval_batches": args.first_n_eval_batches,
        "block_extension_enabled": args.block_extension_enabled,
        "block_extension_params": block_extension_params_cli,
    }
    cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
    logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
    cfg["logging"] = logging_cfg

    if "block_extension_enabled" not in cfg:
        cfg["block_extension_enabled"] = True

    block_extension_enabled, block_extension_cfg = resolve_block_extension_config(cfg)
    if not block_extension_enabled:
        raise ValueError("Block extension runner requires block_extension_enabled=true.")

    device = str(cfg.get("device", "cuda"))
    eval_split = str(cfg.get("eval_split", "test"))
    first_n_eval_batches = cfg.get("first_n_eval_batches", block_extension_cfg.first_n_eval_batches)
    if first_n_eval_batches is not None:
        first_n_eval_batches = int(first_n_eval_batches)

    suite_name = cfg.get("suite", "vision8")
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    suite = SUITES[suite_name]

    tasks_arg = cfg.get("tasks", "all")
    if tasks_arg == "all":
        tasks = list(suite.tasks)
    else:
        tasks = parse_csv(tasks_arg)
        bad = [t for t in tasks if t not in suite.tasks]
        if bad:
            raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite.tasks)}")
    run_summary_path = default_summary_path(
        entrypoint="eval.vision_block_extension",
        logging_cfg=logging_cfg,
    )
    run_logger = start_run(
        entrypoint="eval.vision_block_extension",
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

    tuned_by_task = cfg.get("tuned_ckpts", None)
    if not tuned_by_task:
        raise ValueError("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")

    source_cfg = OpenClipBuildConfig(
        model_name=cfg.get("source_clip_model", "ViT-B-32"),
        pretrained=cfg.get("source_clip_pretrained", "openai"),
        device=device,
        dtype=cfg.get("dtype", None),
    )
    clf_source = OpenClipClassifier.build(source_cfg)
    source_base_sd = to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()})

    target_layers_total = _maybe_target_layers_total(cfg=cfg, source_cfg=source_cfg)
    print(f"Source model: {source_cfg.model_name} / {source_cfg.pretrained}")
    print(
        "Block extension config: "
        f"blocks_to_add={block_extension_cfg.blocks_to_add}, "
        f"target_layers_total={target_layers_total}, "
        f"insertion_order={block_extension_cfg.insertion_order}, "
        f"extension_density={block_extension_cfg.extension_density}, "
        f"extension_strategy={block_extension_cfg.extension_strategy}, "
        f"n_batches_act={block_extension_cfg.n_batches_act}, "
        f"calibration_split={block_extension_cfg.calibration_split}"
    )

    use_humanized_classnames = not bool(cfg.get("no_humanize", True))
    show_progress = bool(block_extension_cfg.show_progress)

    rows: list[dict[str, Any]] = []
    task_iter: Any = tasks
    if show_progress and tqdm is not None:
        task_iter = tqdm(tasks, total=len(tasks), desc="block_extension.tasks")

    for task in task_iter:
        print(f"[block_extension.runner] task={task} start")
        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

        source_loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf_source.preprocess,
            ft_epochs=1,
            split_map=split_map,
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 6)),
            pin_memory=True,
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            seed=int(cfg.get("seed", 42)),
        )

        classnames = list(source_loaders.classnames)
        if use_humanized_classnames:
            classnames = [humanize(c) for c in classnames]

        templates = get_templates(task)
        if not templates:
            raise ValueError(f"get_templates('{task}') returned empty list")

        source_build_cfg_task = OpenClipBuildConfig(
            model_name=source_cfg.model_name,
            pretrained=source_cfg.pretrained,
            device=source_cfg.device,
            dtype=source_cfg.dtype,
            prompt_templates=templates,
        )

        ckpt_path = str(tuned_by_task[task])
        tuned_sd = load_ckpt(ckpt_path)
        aligned = align_to_base_keys(tuned_sd, source_base_sd)
        if not aligned:
            raise ValueError(
                f"No tensors from tuned checkpoint aligned to source base keys for task '{task}': {ckpt_path}"
            )
        tuned_sd_cpu = to_cpu_fp32(aligned)

        source_base_model = deepcopy(clf_source.model)
        source_ft_model = deepcopy(clf_source.model)
        load_into_model(source_base_model, source_base_sd, strict=False)
        load_into_model(source_ft_model, source_base_sd, strict=False)
        load_into_model(source_ft_model, tuned_sd_cpu, strict=False)

        zero_shot_pre = _evaluate_model_top1(
            model=source_base_model,
            clf_source=clf_source,
            loaders=source_loaders,
            classnames=classnames,
            build_cfg_task=source_build_cfg_task,
            device=device,
            split=eval_split,
            first_n_batches=first_n_eval_batches,
        )
        ft_pre = _evaluate_model_top1(
            model=source_ft_model,
            clf_source=clf_source,
            loaders=source_loaders,
            classnames=classnames,
            build_cfg_task=source_build_cfg_task,
            device=device,
            split=eval_split,
            first_n_batches=first_n_eval_batches,
        )

        calibration_loader = select_loader(
            block_extension_cfg.calibration_split,
            train_loader=source_loaders.train,
            test_loader=source_loaders.test,
            val_loader=source_loaders.val,
        )
        final_depth = run_block_extension(
            source_base_model=source_base_model,
            source_ft_model=source_ft_model,
            calibration_loader=calibration_loader,
            target_layers_total=target_layers_total,
            config=block_extension_cfg,
            device=device,
        )

        zero_shot_post = _evaluate_model_top1(
            model=source_base_model,
            clf_source=clf_source,
            loaders=source_loaders,
            classnames=classnames,
            build_cfg_task=source_build_cfg_task,
            device=device,
            split=eval_split,
            first_n_batches=first_n_eval_batches,
        )
        ft_post = _evaluate_model_top1(
            model=source_ft_model,
            clf_source=clf_source,
            loaders=source_loaders,
            classnames=classnames,
            build_cfg_task=source_build_cfg_task,
            device=device,
            split=eval_split,
            first_n_batches=first_n_eval_batches,
        )

        row = {
            "task": task,
            "depth": final_depth,
            "zero_shot_pre": zero_shot_pre,
            "zero_shot_post": zero_shot_post,
            "ft_pre": ft_pre,
            "ft_post": ft_post,
        }
        rows.append(row)
        if run_logger is not None:
            run_logger.log_event(
                "task_end",
                metrics={
                    f"block_extension/{task}/zero_shot_pre": float(zero_shot_pre),
                    f"block_extension/{task}/zero_shot_post": float(zero_shot_post),
                    f"block_extension/{task}/ft_pre": float(ft_pre),
                    f"block_extension/{task}/ft_post": float(ft_post),
                    f"block_extension/{task}/depth": float(final_depth),
                },
                context=row,
            )

        print(
            f"{task}: depth={final_depth} "
            f"zero_shot {zero_shot_pre:.6f}->{zero_shot_post:.6f} "
            f"ft {ft_pre:.6f}->{ft_post:.6f}"
        )

        del source_base_model
        del source_ft_model
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()

    if not rows:
        return

    avg_zero_pre = sum(r["zero_shot_pre"] for r in rows) / len(rows)
    avg_zero_post = sum(r["zero_shot_post"] for r in rows) / len(rows)
    avg_ft_pre = sum(r["ft_pre"] for r in rows) / len(rows)
    avg_ft_post = sum(r["ft_post"] for r in rows) / len(rows)

    print("\n=== Block Extension Summary ===")
    print(f"split={eval_split} tasks={len(rows)}")
    print(f"avg zero_shot: {avg_zero_pre:.6f} -> {avg_zero_post:.6f}")
    print(f"avg ft:        {avg_ft_pre:.6f} -> {avg_ft_post:.6f}")
    if run_logger is not None:
        run_logger.log_summary(
            {
                "suite": suite_name,
                "tasks": tasks,
                "eval_split": eval_split,
                "rows": rows,
                "averages": {
                    "zero_shot_pre": float(avg_zero_pre),
                    "zero_shot_post": float(avg_zero_post),
                    "ft_pre": float(avg_ft_pre),
                    "ft_post": float(avg_ft_post),
                },
            }
        )
        run_logger.finish("success")


if __name__ == "__main__":
    main()
