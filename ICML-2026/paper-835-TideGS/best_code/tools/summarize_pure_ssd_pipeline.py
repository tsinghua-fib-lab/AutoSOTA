#!/usr/bin/env python3
"""Summarize pure SSD pipeline metrics from training logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List


FIELDS = [
    "run",
    "status",
    "resource_ok",
    "recommendation_rank",
    "bsz",
    "iterations_requested",
    "resident_capacity_blocks",
    "projection_max_cameras_per_chunk",
    "max_ram_gb_config",
    "checkpoint_iterations_config",
    "pure_ssd_checkpoint_mode_config",
    "resident_policy_config",
    "resident_lambda_config",
    "resident_recency_decay_config",
    "balanced_seed_fraction_config",
    "debug_max_train_cameras",
    "debug_camera_sample_mode",
    "debug_camera_sample_start",
    "total_points",
    "num_blocks",
    "start_iteration",
    "end_iteration",
    "training_complete",
    "gpu_peak_gb",
    "ram_cache_gb",
    "ram_cache_limit_gb",
    "cache_hits",
    "cache_misses",
    "hit_rate",
    "urgent_prefetch_blocks",
    "urgent_prefetch_miss_blocks",
    "urgent_prefetch_time_s",
    "future_prefetch_jobs",
    "future_prefetch_blocks",
    "future_prefetch_time_s",
    "dirty_blocks_before_shutdown",
    "dirty_blocks_after_shutdown",
    "sync_flush_blocks",
    "sync_flush_time_s",
    "storage_reads",
    "storage_writes",
    "patches_created",
    "storage_total_size_gb",
    "patch_size_gb",
    "end2end_time_s",
    "throughput_it_s",
    "ab_activations",
    "ab_sync_fallbacks",
    "delta_stream_events",
    "delta_stream_in_blocks",
    "delta_evict_blocks",
    "delta_keep_blocks",
    "zero_visible_camera_warning_lines",
    "zero_visible_camera_total",
    "resident_score_events",
    "next_camera_coverage_min",
    "next_camera_coverage_mean",
    "next_camera_total_mean",
    "kt_events",
    "kt_union_blocks_min",
    "kt_union_blocks_mean",
    "kt_union_blocks_max",
    "kt_per_camera_mean",
    "kt_overlap_ratio_mean",
    "kt_resident_coverage_mean",
    "kt_resident_coverage_min",
    "schedule_cache",
    "checkpoint_count",
    "checkpoint_total_gb",
    "snapshot_checkpoint_count",
    "incremental_checkpoint_count",
    "incremental_checkpoint_total_gb",
    "incremental_checkpoint_patch_files",
    "checkpoint_last_next_iteration",
    "resume_next_iteration",
    "cold_restart_row_ratio_pct",
    "mean_resident_streak",
    "densification_log_count",
    "errors",
]


PATTERNS = {
    "pure_check": re.compile(
        r"gaussians=([\d,]+) blocks=(\d+) .*ram_cache_limit=([0-9.]+)GB"
    ),
    "resume": re.compile(r"next_iteration=(\d+)"),
    "iter": re.compile(r"Iter (\d+)"),
    "gpu_peak": re.compile(r"Max Memory usage: ([0-9.]+) GB"),
    "checkpoint": re.compile(r"Snapshot complete: ([0-9.]+)GB .*checkpoints/(\d+)/"),
    "incremental_checkpoint": re.compile(
        r"Incremental complete: patches=(\d+) size=([0-9.]+)GB .*checkpoints/(\d+)/"
    ),
    "optimizer_churn": re.compile(
        r"cold_ratio=([0-9.]+)%.*mean_streak=([0-9.]+)"
    ),
    "patch": re.compile(r"Created patch \d+ with \d+ blocks \(([0-9.]+) MB\)"),
    "delta": re.compile(r"keep .*?=(\d+) stream .*?=(\d+) evict .*?=(\d+)"),
    "zero_visible": re.compile(r"\[WARNING\] (?:(\d+)/(\d+) cameras|Camera \d+) see[s]? no Gaussians"),
    "camera_coverage": re.compile(r"next_camera_coverage=(\d+)/(\d+)"),
    "kt_metrics": re.compile(r"\[PAPER K_T METRICS\].*"),
    "end2end": re.compile(r"end2end total_time: ([0-9.]+) s, iterations: (\d+), throughput ([0-9.]+) it/s"),
    "kv": re.compile(r"([A-Za-z_]+)=([0-9.]+)"),
}


def _load_args_json(run_dir: Path) -> Dict:
    args_path = run_dir / "args.json"
    if not args_path.is_file():
        return {}
    try:
        with open(args_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _format_checkpoint_iterations(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _float_or_none(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_summary(path: Path) -> Dict[str, object]:
    return {field: "" for field in FIELDS} | {
        "run": str(path),
        "status": "unknown",
        "resource_ok": False,
        "training_complete": False,
        "errors": 0,
        "checkpoint_count": 0,
        "checkpoint_total_gb": 0.0,
        "snapshot_checkpoint_count": 0,
        "incremental_checkpoint_count": 0,
        "incremental_checkpoint_total_gb": 0.0,
        "incremental_checkpoint_patch_files": 0,
        "densification_log_count": 0,
        "ab_activations": 0,
        "ab_sync_fallbacks": 0,
        "delta_stream_events": 0,
        "delta_stream_in_blocks": 0,
        "delta_evict_blocks": 0,
        "delta_keep_blocks": 0,
        "zero_visible_camera_warning_lines": 0,
        "zero_visible_camera_total": 0,
        "resident_score_events": 0,
        "next_camera_coverage_min": "",
        "next_camera_coverage_mean": "",
        "next_camera_total_mean": "",
        "kt_events": 0,
        "kt_union_blocks_min": "",
        "kt_union_blocks_mean": "",
        "kt_union_blocks_max": "",
        "kt_per_camera_mean": "",
        "kt_overlap_ratio_mean": "",
        "kt_resident_coverage_mean": "",
        "kt_resident_coverage_min": "",
        "patch_size_gb": 0.0,
    }


def _parse_stats_dict(line: str) -> Dict:
    start = line.find("Stats:")
    if start < 0:
        return {}
    payload = line[start + len("Stats:"):].strip()
    end = payload.rfind("}")
    if end < 0:
        return {}
    return ast.literal_eval(payload[: end + 1])


def _load_checkpoint_next_iteration(run_dir: Path, checkpoint_iter: str) -> int | str:
    manifest = run_dir / "checkpoints" / str(checkpoint_iter) / "pure_ssd_checkpoint.json"
    if not manifest.is_file():
        return ""
    try:
        with open(manifest, "r", encoding="utf-8") as handle:
            return int(json.load(handle).get("next_iteration", ""))
    except Exception:
        return ""


def _as_gb_mb(value_mb: float) -> float:
    return float(value_mb) / 1024.0


def _kv_numbers(line: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in PATTERNS["kv"].findall(line):
        number = float(value)
        values[key] = int(number) if number.is_integer() else number
    return values


def _update_iter_bounds(summary: Dict[str, object], line: str) -> None:
    match = PATTERNS["iter"].search(line)
    if not match:
        return
    iteration = int(match.group(1))
    current_start = summary.get("start_iteration")
    current_end = summary.get("end_iteration")
    summary["start_iteration"] = iteration if current_start == "" else min(int(current_start), iteration)
    summary["end_iteration"] = iteration if current_end == "" else max(int(current_end), iteration)


def summarize_log(log_path: Path) -> Dict[str, object]:
    run_dir = log_path.parent if log_path.name == "python.log" else log_path
    summary = _empty_summary(run_dir)
    args_json = _load_args_json(run_dir)
    if args_json:
        summary["bsz"] = args_json.get("bsz", "")
        summary["iterations_requested"] = args_json.get("iterations", "")
        summary["resident_capacity_blocks"] = args_json.get("paper_resident_capacity_blocks", "")
        summary["projection_max_cameras_per_chunk"] = args_json.get(
            "projection_max_cameras_per_chunk", ""
        )
        summary["max_ram_gb_config"] = args_json.get("max_ram_gb", "")
        summary["checkpoint_iterations_config"] = _format_checkpoint_iterations(
            args_json.get("checkpoint_iterations", "")
        )
        summary["pure_ssd_checkpoint_mode_config"] = args_json.get("pure_ssd_checkpoint_mode", "")
        summary["resident_policy_config"] = args_json.get("paper_resident_selection_policy", "")
        summary["resident_lambda_config"] = args_json.get("paper_resident_lambda", "")
        summary["resident_recency_decay_config"] = args_json.get(
            "paper_resident_recency_decay", ""
        )
        summary["balanced_seed_fraction_config"] = args_json.get(
            "paper_balanced_seed_fraction", ""
        )
        summary["debug_max_train_cameras"] = args_json.get("debug_max_train_cameras", "")
        summary["debug_camera_sample_mode"] = args_json.get("debug_camera_sample_mode", "")
        summary["debug_camera_sample_start"] = args_json.get("debug_camera_sample_start", "")

    camera_coverage_values: List[int] = []
    camera_total_values: List[int] = []
    kt_union_values: List[int] = []
    kt_per_camera_values: List[float] = []
    kt_overlap_values: List[float] = []
    kt_resident_coverage_values: List[float] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if any(token in line for token in ("Traceback", "RuntimeError", "CUDA out of memory", "NaN", "nan")):
                summary["errors"] = int(summary["errors"]) + 1
            if "densify_and_prune" in line or "Densification" in line:
                summary["densification_log_count"] = int(summary["densification_log_count"]) + 1
            if "see no Gaussians" in line or "sees no Gaussians" in line:
                summary["zero_visible_camera_warning_lines"] = int(summary["zero_visible_camera_warning_lines"]) + 1
                match = PATTERNS["zero_visible"].search(line)
                if match and match.group(1):
                    summary["zero_visible_camera_total"] = (
                        int(summary["zero_visible_camera_total"]) + int(match.group(1))
                    )
                else:
                    summary["zero_visible_camera_total"] = int(summary["zero_visible_camera_total"]) + 1

            _update_iter_bounds(summary, line)

            if "[PURE SSD CHECK]" in line:
                match = PATTERNS["pure_check"].search(line)
                if match:
                    summary["total_points"] = int(match.group(1).replace(",", ""))
                    summary["num_blocks"] = int(match.group(2))
                    summary["ram_cache_limit_gb"] = float(match.group(3))
            elif "[PURE SSD RESUME]" in line and "next_iteration=" in line:
                match = PATTERNS["resume"].search(line)
                if match:
                    summary["resume_next_iteration"] = int(match.group(1))
            elif "Loaded camera schedule cache" in line:
                summary["schedule_cache"] = "hit"
            elif "Camera schedule cache miss" in line:
                summary["schedule_cache"] = "miss"
            elif "Max Memory usage:" in line:
                match = PATTERNS["gpu_peak"].search(line)
                if match:
                    summary["gpu_peak_gb"] = float(match.group(1))
            elif "Training complete" in line:
                summary["training_complete"] = True
            elif "[PURE SSD CHECKPOINT] Snapshot complete" in line:
                match = PATTERNS["checkpoint"].search(line)
                if match:
                    summary["checkpoint_count"] = int(summary["checkpoint_count"]) + 1
                    summary["snapshot_checkpoint_count"] = int(summary["snapshot_checkpoint_count"]) + 1
                    summary["checkpoint_total_gb"] = float(summary["checkpoint_total_gb"]) + float(match.group(1))
                    summary["checkpoint_last_next_iteration"] = _load_checkpoint_next_iteration(
                        log_path.parent,
                        match.group(2),
                    )
            elif "[PURE SSD CHECKPOINT] Incremental complete" in line:
                match = PATTERNS["incremental_checkpoint"].search(line)
                if match:
                    patch_files = int(match.group(1))
                    size_gb = float(match.group(2))
                    summary["checkpoint_count"] = int(summary["checkpoint_count"]) + 1
                    summary["incremental_checkpoint_count"] = int(summary["incremental_checkpoint_count"]) + 1
                    summary["incremental_checkpoint_patch_files"] = (
                        int(summary["incremental_checkpoint_patch_files"]) + patch_files
                    )
                    summary["incremental_checkpoint_total_gb"] = (
                        float(summary["incremental_checkpoint_total_gb"]) + size_gb
                    )
                    summary["checkpoint_total_gb"] = float(summary["checkpoint_total_gb"]) + size_gb
                    summary["checkpoint_last_next_iteration"] = _load_checkpoint_next_iteration(
                        log_path.parent,
                        match.group(3),
                    )
            elif "[OPTIMIZER CHURN]" in line and "cold_ratio=" in line:
                match = PATTERNS["optimizer_churn"].search(line)
                if match:
                    summary["cold_restart_row_ratio_pct"] = float(match.group(1))
                    summary["mean_resident_streak"] = float(match.group(2))
            elif "end2end total_time:" in line:
                match = PATTERNS["end2end"].search(line)
                if match:
                    summary["end2end_time_s"] = float(match.group(1))
                    summary["throughput_it_s"] = float(match.group(3))
                    summary["training_complete"] = True
                    end_iteration = int(match.group(2))
                    current_end = summary.get("end_iteration")
                    summary["end_iteration"] = (
                        end_iteration if current_end == "" else max(int(current_end), end_iteration)
                    )
            elif "[PAPER WARM LAYER] Cache @" in line:
                values = _kv_numbers(line)
                if "dirty" in values:
                    summary["dirty_blocks_before_shutdown"] = values["dirty"]
                if "hits" in values:
                    summary["cache_hits"] = values["hits"]
                if "misses" in values:
                    summary["cache_misses"] = values["misses"]
                if "hit_rate" in values:
                    summary["hit_rate"] = float(values["hit_rate"]) / 100.0
                if "ram" in values:
                    summary["ram_cache_gb"] = _as_gb_mb(values["ram"])
                if "async_blocks" in values:
                    summary["future_prefetch_blocks"] = values["async_blocks"]
                if "sync_blocks" in values:
                    summary["sync_flush_blocks"] = values["sync_blocks"]
                if "sync_time" in values:
                    summary["sync_flush_time_s"] = float(values["sync_time"]) / 1000.0
            elif "[PAPER PIPELINE CACHE]" in line:
                values = _kv_numbers(line)
                if "urgent_blocks" in values:
                    summary["urgent_prefetch_blocks"] = values["urgent_blocks"]
                if "urgent_misses" in values:
                    summary["urgent_prefetch_miss_blocks"] = values["urgent_misses"]
                if "urgent_time" in values:
                    summary["urgent_prefetch_time_s"] = float(values["urgent_time"]) / 1000.0
                if "future_jobs" in values:
                    summary["future_prefetch_jobs"] = values["future_jobs"]
                if "future_blocks" in values:
                    summary["future_prefetch_blocks"] = values["future_blocks"]
                if "future_time" in values:
                    summary["future_prefetch_time_s"] = float(values["future_time"]) / 1000.0
            elif "[LogStorage] Created patch" in line:
                match = PATTERNS["patch"].search(line)
                if match:
                    summary["patch_size_gb"] = float(summary["patch_size_gb"]) + _as_gb_mb(float(match.group(1)))
            elif "[PAPER A/B BUFFER]" in line:
                if "Activated prefetched buffer" in line:
                    summary["ab_activations"] = int(summary["ab_activations"]) + 1
                if "falling back to synchronous" in line:
                    summary["ab_sync_fallbacks"] = int(summary["ab_sync_fallbacks"]) + 1
            elif "[PAPER DELTA STREAM]" in line and "stream" in line and "evict" in line:
                match = PATTERNS["delta"].search(line)
                if match:
                    summary["delta_stream_events"] = int(summary["delta_stream_events"]) + 1
                    summary["delta_keep_blocks"] = int(summary["delta_keep_blocks"]) + int(match.group(1))
                    summary["delta_stream_in_blocks"] = int(summary["delta_stream_in_blocks"]) + int(match.group(2))
                    summary["delta_evict_blocks"] = int(summary["delta_evict_blocks"]) + int(match.group(3))
            elif "[PAPER RESIDENT SCORE]" in line:
                match = PATTERNS["camera_coverage"].search(line)
                if match:
                    summary["resident_score_events"] = int(summary["resident_score_events"]) + 1
                    camera_coverage_values.append(int(match.group(1)))
                    camera_total_values.append(int(match.group(2)))
            elif "[PAPER K_T METRICS]" in line:
                values = _kv_numbers(line)
                if values:
                    summary["kt_events"] = int(summary["kt_events"]) + 1
                    if "union_blocks" in values:
                        kt_union_values.append(int(values["union_blocks"]))
                    if "per_camera_mean" in values:
                        kt_per_camera_values.append(float(values["per_camera_mean"]))
                    if "overlap_ratio" in values:
                        kt_overlap_values.append(float(values["overlap_ratio"]))
                    if "resident_coverage" in values:
                        kt_resident_coverage_values.append(float(values["resident_coverage"]))
            elif "[Pipeline] Shutdown complete. Stats:" in line:
                stats = _parse_stats_dict(line)
                cache_stats = stats.get("cache_stats", {})
                storage_stats = stats.get("storage_stats", {})
                summary["dirty_blocks_before_shutdown"] = cache_stats.get("dirty_blocks", "")
                summary["ram_cache_gb"] = _as_gb_mb(cache_stats.get("ram_usage_mb", 0.0))
                summary["cache_hits"] = cache_stats.get("cache_hits", "")
                summary["cache_misses"] = cache_stats.get("cache_misses", "")
                summary["hit_rate"] = cache_stats.get("hit_rate", "")
                summary["urgent_prefetch_blocks"] = cache_stats.get("urgent_prefetch_blocks", "")
                summary["urgent_prefetch_miss_blocks"] = cache_stats.get("urgent_prefetch_miss_blocks", "")
                summary["urgent_prefetch_time_s"] = cache_stats.get("urgent_prefetch_time", "")
                summary["future_prefetch_jobs"] = cache_stats.get("future_prefetch_jobs", "")
                summary["future_prefetch_blocks"] = cache_stats.get("future_prefetch_blocks", "")
                summary["future_prefetch_time_s"] = cache_stats.get("future_prefetch_time", "")
                summary["storage_reads"] = storage_stats.get("reads", "")
                summary["storage_writes"] = storage_stats.get("writes", "")
                summary["patches_created"] = storage_stats.get("patches_created", "")
            elif "[TieredCache] Shutdown complete. Stats:" in line:
                stats = _parse_stats_dict(line)
                summary["dirty_blocks_after_shutdown"] = stats.get("dirty_blocks", "")
                summary["sync_flush_blocks"] = stats.get("sync_flush_blocks", "")
                summary["sync_flush_time_s"] = stats.get("sync_flush_time", "")
            elif "[LogStorage] Closed. Stats:" in line:
                stats = _parse_stats_dict(line)
                summary["storage_reads"] = stats.get("reads", summary["storage_reads"])
                summary["storage_writes"] = stats.get("writes", summary["storage_writes"])
                summary["patches_created"] = stats.get("patches_created", summary["patches_created"])
                summary["storage_total_size_gb"] = _as_gb_mb(stats.get("total_size_mb", 0.0))

    if camera_coverage_values:
        summary["next_camera_coverage_min"] = min(camera_coverage_values)
        summary["next_camera_coverage_mean"] = sum(camera_coverage_values) / len(camera_coverage_values)
        summary["next_camera_total_mean"] = sum(camera_total_values) / len(camera_total_values)
    if kt_union_values:
        summary["kt_union_blocks_min"] = min(kt_union_values)
        summary["kt_union_blocks_mean"] = sum(kt_union_values) / len(kt_union_values)
        summary["kt_union_blocks_max"] = max(kt_union_values)
    if kt_per_camera_values:
        summary["kt_per_camera_mean"] = sum(kt_per_camera_values) / len(kt_per_camera_values)
    if kt_overlap_values:
        summary["kt_overlap_ratio_mean"] = sum(kt_overlap_values) / len(kt_overlap_values)
    if kt_resident_coverage_values:
        summary["kt_resident_coverage_mean"] = (
            sum(kt_resident_coverage_values) / len(kt_resident_coverage_values)
        )
        summary["kt_resident_coverage_min"] = min(kt_resident_coverage_values)

    summary["status"] = "ok" if summary["training_complete"] and int(summary["errors"]) == 0 else "failed"
    gpu_peak = _float_or_none(summary.get("gpu_peak_gb"))
    ram_cache = _float_or_none(summary.get("ram_cache_gb"))
    ram_limit = _float_or_none(summary.get("max_ram_gb_config"))
    if ram_limit is None:
        ram_limit = _float_or_none(summary.get("ram_cache_limit_gb"))
    gpu_ok = gpu_peak is None or gpu_peak <= 22.0
    ram_ok = ram_cache is None or ram_limit is None or ram_cache <= ram_limit
    no_densification = int(summary.get("densification_log_count") or 0) == 0
    checkpoint_mode = str(summary.get("pure_ssd_checkpoint_mode_config") or "").lower()
    snapshot_count = int(summary.get("snapshot_checkpoint_count") or 0)
    snapshot_ok = checkpoint_mode != "incremental" or snapshot_count == 0
    summary["resource_ok"] = bool(
        summary["status"] == "ok" and gpu_ok and ram_ok and no_densification and snapshot_ok
    )
    return summary


def _recommendation_key(row: Dict[str, object]) -> tuple:
    resource_ok = 1 if row.get("resource_ok") is True else 0
    status_ok = 1 if row.get("status") == "ok" else 0
    throughput = _float_or_none(row.get("throughput_it_s")) or -1.0
    hit_rate = _float_or_none(row.get("hit_rate")) or -1.0
    gpu_peak = _float_or_none(row.get("gpu_peak_gb")) or 999.0
    ram_cache = _float_or_none(row.get("ram_cache_gb")) or 999.0
    urgent_time = _float_or_none(row.get("urgent_prefetch_time_s")) or 999999.0
    return (resource_ok, status_ok, throughput, hit_rate, -urgent_time, -gpu_peak, -ram_cache)


def rank_recommendations(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = [dict(row) for row in sorted(rows, key=_recommendation_key, reverse=True)]
    for idx, row in enumerate(ranked, start=1):
        row["recommendation_rank"] = idx
    return ranked


def resolve_logs(paths: Iterable[str]) -> List[Path]:
    logs: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_file():
            logs.append(path)
        elif path.is_dir():
            direct = path / "python.log"
            if direct.is_file():
                logs.append(direct)
            else:
                logs.extend(sorted(path.rglob("python.log")))
        else:
            raise FileNotFoundError(path)
    return logs


def write_tsv(rows: List[Dict[str, object]], output: Path | None) -> None:
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run directories or python.log files")
    parser.add_argument("--output", default="", help="Optional output TSV path")
    parser.add_argument(
        "--recommend-output",
        default="",
        help="Optional TSV sorted by resource safety and throughput",
    )
    args = parser.parse_args()

    logs = resolve_logs(args.paths)
    if not logs:
        raise SystemExit("No python.log files found")
    rows = [summarize_log(log) for log in logs]
    recommended_rows = rank_recommendations(rows)
    write_tsv(rows, Path(args.output) if args.output else None)
    if args.recommend_output:
        write_tsv(recommended_rows, Path(args.recommend_output))


if __name__ == "__main__":
    main()
