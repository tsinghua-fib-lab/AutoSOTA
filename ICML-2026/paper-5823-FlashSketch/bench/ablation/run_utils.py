from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import asdict
import time

import torch

import globals as g
from bench.e2e.report import maybe_send_results, summarize_results, write_summary
from bench.common.run_shared import (
    TASK_RUNNERS,
    dataset_id,
    format_duration,
    load_dataset,
    make_rhs,
    method_config_metadata,
    oom_metrics,
    run_task,
    warmup_sketch,
)
from config_utils import apply_env_overrides
from io_utils import ensure_dir, write_json, write_parquet
from logging_utils import get_logger
from provenance import (
    ensure_clean_repo,
    get_git_state,
    get_tree_hashes,
    make_run_id,
    run_dir,
)
from sketches.registry import get_sketch_fn
from torch_utils import resolve_device


TREE_HASH_PATHS = ("bench/ablation", "bench/e2e/tasks", "sketches", "kernels", "data")


def run_ablation(config, logger_name):
    """Run ablation end-to-end benchmarks for the given config."""
    logger = get_logger(logger_name)

    prior_cuda_timers = g.ENABLE_CUDA_TIMERS
    g.ENABLE_CUDA_TIMERS = bool(config.enable_cuda_timers)

    if config.require_clean_repo:
        ensure_clean_repo()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for end-to-end benchmarks.")

    device = resolve_device()
    run_id = make_run_id()
    run_path = run_dir(run_id)
    ensure_dir(run_path)

    git_state = get_git_state()
    tree_hashes = get_tree_hashes(TREE_HASH_PATHS)

    rows = []
    dataset_entries = []
    for entry in config.datasets:
        if isinstance(entry, tuple) and len(entry) == 2:
            dataset_entries.append((entry[0], set(entry[1])))
        else:
            dataset_entries.append((entry, None))

    def _filter_methods(methods, allowed_k):
        if allowed_k is None:
            return list(methods)
        return [method for method in methods if getattr(method, "k", None) in allowed_k]

    total_iters = sum(
        len(_filter_methods(config.methods, allowed_k))
        for _, allowed_k in dataset_entries
    ) * len(config.tasks)
    progress_every = max(1, int(config.progress_every)) if total_iters else 0
    iter_idx = 0
    start_time = time.perf_counter()

    try:
        for dataset_cfg, allowed_k in dataset_entries:
            A, metadata = load_dataset(dataset_cfg, device)
            A = A.contiguous() if not A.is_contiguous() else A

            dataset_id_str = dataset_id(metadata)
            d, n = A.shape
            b = make_rhs(A, config.seed, config.b_noise)
            logger.info("Dataset %s (%dx%d)", dataset_id_str, d, n)

            method_list = _filter_methods(config.methods, allowed_k)
            for method_cfg in method_list:
                method_cfg = apply_env_overrides(method_cfg)
                sketch_fn = get_sketch_fn(method_cfg.method)
                base_row = {
                    "dataset_id": dataset_id_str,
                    "dataset": metadata["dataset"],
                    "dataset_name": metadata.get("name") or "",
                    "dataset_group": metadata.get("group") or "",
                    "d": d,
                    "n": n,
                    "method": method_cfg.method,
                    "k": method_cfg.k,
                    "seed": method_cfg.seed,
                    "dtype": method_cfg.dtype,
                    "gpu_name": torch.cuda.get_device_name(0),
                }
                base_row.update(method_config_metadata(method_cfg))

                try:
                    warmup_sketch(A, sketch_fn, method_cfg)
                except torch.OutOfMemoryError:
                    logger.warning(
                        "OOM during warmup for %s/%s k=%s seed=%s",
                        dataset_id,
                        method_cfg.method,
                        method_cfg.k,
                        method_cfg.seed,
                    )
                    torch.cuda.empty_cache()
                    for task_cfg in config.tasks:
                        row = dict(base_row)
                        row.update(oom_metrics(task_cfg))
                        rows.append(row)
                        iter_idx += 1
                    continue

                oom_skip_remaining = False
                for task_cfg in config.tasks:
                    runner = TASK_RUNNERS.get(task_cfg.task)
                    if runner is None:
                        raise ValueError(f"Unknown task: {task_cfg.task}")

                    if oom_skip_remaining:
                        metrics = oom_metrics(task_cfg)
                        status = "oom"
                    else:
                        try:
                            metrics = run_task(
                                runner, A, b, sketch_fn, method_cfg, task_cfg
                            )
                            status = metrics.get("status", "ok")
                        except torch.OutOfMemoryError:
                            logger.warning(
                                "OOM during task for %s/%s k=%s seed=%s task=%s",
                                dataset_id,
                                method_cfg.method,
                                method_cfg.k,
                                method_cfg.seed,
                                task_cfg.task,
                            )
                            torch.cuda.empty_cache()
                            metrics = oom_metrics(task_cfg)
                            status = "oom"
                            oom_skip_remaining = True

                    row = dict(base_row)
                    row.update(metrics)
                    row.setdefault("status", status)
                    rows.append(row)
                    iter_idx += 1

                    if progress_every and (
                        iter_idx % progress_every == 0 or iter_idx == total_iters
                    ):
                        elapsed = time.perf_counter() - start_time
                        rate = elapsed / max(iter_idx, 1)
                        eta = rate * max(total_iters - iter_idx, 0)
                        percent = 100.0 * iter_idx / max(total_iters, 1)
                        logger.info(
                            "Progress %d/%d (%.1f%%) elapsed=%s eta=%s last=%s/%s k=%s seed=%s task=%s",
                            iter_idx,
                            total_iters,
                            percent,
                            format_duration(elapsed),
                            format_duration(eta),
                            dataset_id_str,
                            method_cfg.method,
                            method_cfg.k,
                            method_cfg.seed,
                            task_cfg.task,
                        )
    finally:
        g.ENABLE_CUDA_TIMERS = prior_cuda_timers

    results_path = run_path / "results.parquet"
    summary_path = run_path / "summary.json"
    manifest_path = run_path / "manifest.json"

    write_parquet(results_path, rows)
    summary = summarize_results(rows)
    write_summary(summary_path, summary)

    manifest = {
        "run_id": run_id,
        "git": git_state,
        "tree_hashes": tree_hashes,
        "config": asdict(config),
        "results_path": str(results_path.relative_to(g.REPO_ROOT)),
        "summary_path": str(summary_path.relative_to(g.REPO_ROOT)),
    }
    write_json(manifest_path, manifest)

    repo_state = "dirty" if git_state["dirty"] else "clean"
    summary_text = (
        "Ablation E2E run complete.\n"
        f"Repo state: {repo_state}.\n"
        f"Run id: {run_id}\n"
        "Artifacts: none (no plots generated)."
    )

    maybe_send_results(summary_text, [], config.send_slack)
