from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import JETSON_EFFECTIVE_MEMORY_MIB, run_policies
from modern_workloads import (
    build_mixed_modern_case,
    build_smart_traffic_case,
    build_uav_vit_case,
    scaled_deadlines,
)
from rtinfer.model import Job, ModelProfile
from rtinfer.scheduler import SimulationResult


OUT = ROOT / "outputs" / "scheduling_analysis"
POLICIES = ("pantheon", "rtinfer")
KAPPA = 1.0


@dataclass(frozen=True)
class CaseRun:
    app: str
    models: dict[str, ModelProfile]
    atlas: dict
    results: dict[str, SimulationResult]


def short_app(title: str) -> str:
    return {
        "modern CNN smart traffic: YOLOv8-L 1080p + YOLOv8n": "Smart Traffic",
        "ViT UAV scene recognition: ViT-L + MobileViT-S": "UAV Ground",
        "mixed modern deployment: YOLO + ViT + edge GPT-2": "Service Robot",
    }.get(title, title)


def pct(value: float) -> float:
    return round(value * 100.0, 3)


def full_acc(job: Job, models: dict[str, ModelProfile]) -> float:
    return models[job.task.model_name].full_accuracy


def completed_jobs(result: SimulationResult) -> list[Job]:
    return [job for job in result.schedule_events if job.variant is not None and not job.missed]


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_case(builder: Callable[[], tuple]) -> CaseRun:
    title, models, atlas, tasks, duration_ms = builder()
    results = run_policies(
        models,
        atlas,
        scaled_deadlines(tasks, KAPPA),
        policies=POLICIES,
        memory_mib=JETSON_EFFECTIVE_MEMORY_MIB,
        duration_ms=duration_ms,
        bandwidth_gbps=24.0,
    )
    return CaseRun(short_app(title), models, atlas, {result.policy: result for result in results})


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_loss_rows(cases: list[CaseRun]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    job_rows: list[dict[str, object]] = []
    for case in cases:
        pantheon = case.results["pantheon"]
        for policy, result in case.results.items():
            jobs = result.schedule_events
            completed = completed_jobs(result)
            full_all = avg([full_acc(job, case.models) for job in jobs])
            full_completed = avg([full_acc(job, case.models) for job in completed])
            completed_selected = avg([job.variant.accuracy for job in completed if job.variant is not None])
            deadline_weighted = result.average_accuracy
            completed_loss = max(0.0, full_completed - completed_selected)
            total_rt_score_loss = max(0.0, full_all - deadline_weighted)
            summary_rows.append(
                {
                    "application": case.app,
                    "kappa": KAPPA,
                    "policy": policy,
                    "total_jobs": result.total_jobs,
                    "missed_jobs": result.missed_jobs,
                    "dmr": round(result.deadline_miss_rate, 6),
                    "full_depth_accuracy_avg": round(full_all, 6),
                    "completed_full_depth_accuracy_avg": round(full_completed, 6),
                    "completed_selected_accuracy": round(completed_selected, 6),
                    "deadline_weighted_accuracy": round(deadline_weighted, 6),
                    "completed_only_loss_pp": pct(completed_loss),
                    "deadline_weighted_loss_pp": pct(total_rt_score_loss),
                    "avg_exit_index_completed": round(avg([job.variant.exit_index for job in completed if job.variant is not None]), 3),
                    "avg_pruning_completed": round(avg([job.variant.pruning for job in completed if job.variant is not None]), 4),
                    "avg_load_ms": round(result.average_load_us / 1000.0, 3),
                }
            )
            for job in jobs:
                variant = job.variant
                job_rows.append(
                    {
                        "application": case.app,
                        "policy": policy,
                        "job_id": job.job_id,
                        "model": job.task.model_name,
                        "release_ms": round(job.release_us / 1000.0, 3),
                        "deadline_ms": round(job.absolute_deadline_us / 1000.0, 3),
                        "finish_ms": round((job.finish_us or job.release_us) / 1000.0, 3),
                        "missed": int(job.missed),
                        "full_depth_accuracy": round(full_acc(job, case.models), 6),
                        "selected_accuracy": round(variant.accuracy if variant and not job.missed else 0.0, 6),
                        "raw_selected_accuracy": round(variant.accuracy if variant else 0.0, 6),
                        "pruning": round(variant.pruning if variant else 0.0, 4),
                        "exit_index": variant.exit_index if variant else "",
                        "memory_mib": round(variant.memory_mib if variant else 0.0, 3),
                        "latency_ms": round(variant.latency_us / 1000.0 if variant else 0.0, 3),
                        "load_ms": round(job.load_us / 1000.0, 3),
                    }
                )
        # Keep a compact app-level line on stdout for the run log.
        pantheon_completed = completed_jobs(pantheon)
        loss = avg([full_acc(job, case.models) for job in pantheon_completed]) - avg(
            [job.variant.accuracy for job in pantheon_completed if job.variant is not None]
        )
        print(
            "pantheon_accuracy_loss,"
            f"application={case.app},jobs={pantheon.total_jobs},missed={pantheon.missed_jobs},"
            f"completed_only_loss_pp={pct(max(0.0, loss)):.2f},"
            f"deadline_weighted_acc={pct(pantheon.average_accuracy):.2f}"
        )
    return summary_rows, job_rows


def build_comparison_rows(cases: list[CaseRun]) -> list[dict[str, object]]:
    representative = [
        ("t1", "YOLOv8-L", "yolov8l_1080p"),
        ("t2", "ViT-L", "vit_l_1024"),
        ("t3", "GPT-2 KV", "gpt2_edge_kv_bound"),
    ]
    rows: list[dict[str, object]] = []
    for task_label, display_name, model_name in representative:
        full_values: list[float] = []
        raw_selected: dict[str, list[float]] = {policy: [] for policy in POLICIES}
        selected: dict[str, list[float]] = {policy: [] for policy in POLICIES}
        weighted: dict[str, list[float]] = {policy: [] for policy in POLICIES}
        for case in cases:
            if model_name not in case.models:
                continue
            for job in case.results["pantheon"].schedule_events:
                if job.task.model_name == model_name:
                    full_values.append(case.models[model_name].full_accuracy)
            for policy in POLICIES:
                for job in case.results[policy].schedule_events:
                    if job.task.model_name != model_name or job.variant is None:
                        continue
                    raw_selected[policy].append(job.variant.accuracy)
                    if not job.missed:
                        selected[policy].append(job.variant.accuracy)
                    weighted[policy].append(0.0 if job.missed else job.variant.accuracy)
        orig = avg(full_values)
        pantheon_selected = avg(raw_selected["pantheon"])
        rtinfer_selected = avg(raw_selected["rtinfer"])
        rows.append(
            {
                "task": task_label,
                "model": display_name,
                "orig_accuracy": round(orig, 6),
                "pantheon_selected_accuracy": round(pantheon_selected, 6),
                "rtinfer_selected_accuracy": round(rtinfer_selected, 6),
                "pantheon_completed_accuracy": round(avg(selected["pantheon"]), 6),
                "rtinfer_completed_accuracy": round(avg(selected["rtinfer"]), 6),
                "pantheon_deadline_weighted_accuracy": round(avg(weighted["pantheon"]), 6),
                "rtinfer_deadline_weighted_accuracy": round(avg(weighted["rtinfer"]), 6),
                "pantheon_selected_loss_pp": pct(max(0.0, orig - pantheon_selected)),
                "rtinfer_selected_loss_pp": pct(max(0.0, orig - rtinfer_selected)),
                "pantheon_completed_loss_pp": pct(max(0.0, orig - avg(selected["pantheon"]))),
                "rtinfer_completed_loss_pp": pct(max(0.0, orig - avg(selected["rtinfer"]))),
            }
        )
    return rows


def find_model_and_variant(
    cases: list[CaseRun],
    model_name: str,
    pruning: float,
    exit_index: int,
):
    for case in cases:
        if model_name not in case.models:
            continue
        model = case.models[model_name]
        for variant in case.atlas[model_name]:
            if abs(variant.pruning - pruning) < 1e-9 and variant.exit_index == exit_index:
                return model, variant
    raise KeyError(f"variant not found: model={model_name} pruning={pruning} exit={exit_index}")


def build_pc_pantheon_rows(cases: list[CaseRun]) -> list[dict[str, object]]:
    """Original Fig. 1(c)-style comparison.

    Orig. is the full-depth unpruned model. Pantheon and PC-Pantheon are
    reported as deadline-weighted application-level accuracy, so deadline
    misses contribute zero. PC-Pantheon estimates Pantheon with a simple
    pruning/compression knob added, not the full RTInfer layout/Delta-Graph
    method.
    """

    selections = [
        # task, display model, model key, Pantheon(p,e), PC-Pantheon(p,e)
        ("t1", "YOLOv8-L", "yolov8l_1080p", (0.0, 2), (0.25, 2)),
        ("t2", "ViT-L", "vit_l_1024", (0.0, 2), (0.25, 2)),
        ("t3", "GPT-2 KV", "gpt2_edge_kv_bound", (0.0, 1), (0.50, 1)),
    ]
    rows: list[dict[str, object]] = []
    for task, display, model_name, pantheon_choice, pc_choice in selections:
        model, pantheon_static_variant = find_model_and_variant(cases, model_name, *pantheon_choice)
        _, pc_variant = find_model_and_variant(cases, model_name, *pc_choice)
        orig_values: list[float] = []
        pantheon_weighted: list[float] = []
        pc_weighted: list[float] = []
        for case in cases:
            if model_name not in case.models:
                continue
            for job in case.results["pantheon"].schedule_events:
                if job.task.model_name != model_name:
                    continue
                orig_values.append(case.models[model_name].full_accuracy)
                pantheon_weighted.append(0.0 if job.missed or job.variant is None else job.variant.accuracy)
                start_us = job.start_us if job.start_us is not None else job.release_us
                # PC-Pantheon keeps Pantheon's serial/preemptive execution order
                # but uses a simple compressed variant. We scale the observed
                # Pantheon load cost by the variant size ratio; no layout packing
                # or Delta-Graph reuse is credited here.
                if job.variant is not None and job.variant.bytes_to_load > 0:
                    load_ratio = pc_variant.bytes_to_load / job.variant.bytes_to_load
                else:
                    load_ratio = 1.0
                pc_load_us = int(job.load_us * load_ratio)
                pc_finish_us = start_us + pc_variant.latency_us + pc_load_us
                pc_weighted.append(0.0 if pc_finish_us > job.absolute_deadline_us else pc_variant.accuracy)
        rows.append(
            {
                "task": task,
                "model": display,
                "orig_accuracy": round(avg(orig_values), 6),
                "pantheon_accuracy": round(avg(pantheon_weighted), 6),
                "pc_pantheon_accuracy": round(avg(pc_weighted), 6),
                "pantheon_pruning": pantheon_static_variant.pruning,
                "pantheon_exit_point": pantheon_static_variant.exit_index + 1,
                "pc_pantheon_pruning": pc_variant.pruning,
                "pc_pantheon_exit_point": pc_variant.exit_index + 1,
                "note": "Fig.1 motivation: deadline-weighted application-level accuracy; PC-Pantheon is simple pruning/compression only, not full RTInfer.",
            }
        )
    return rows


def write_markdown(summary_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Pantheon Accuracy Loss Probe",
        "",
        "This deterministic probe updates the original Fig. 1(c)-style accuracy comparison after replacing the old CNN-only setup with reviewer-aligned modern models.",
        "",
        "Definitions:",
        "- `orig_accuracy` is the full-depth, unpruned model accuracy profile used by the simulator.",
        "- `completed_selected_accuracy` excludes missed deadlines and isolates accuracy sacrificed by selecting shallow exits / pruned variants.",
        "- `deadline_weighted_accuracy` counts missed jobs as zero, matching the real-time score used in the main evaluation.",
        "",
        "## Representative Task Accuracy",
        "",
        "| task | model | Orig. | Pantheon selected | RTInfer selected | Pantheon loss | RTInfer loss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {pct(float(row['orig_accuracy'])):.1f}% | "
            f"{pct(float(row['pantheon_selected_accuracy'])):.1f}% | "
            f"{pct(float(row['rtinfer_selected_accuracy'])):.1f}% | "
            f"{float(row['pantheon_selected_loss_pp']):.1f} pp | "
            f"{float(row['rtinfer_selected_loss_pp']):.1f} pp |"
        )
    lines += [
        "",
        "## Application-Level Pantheon Sacrifice",
        "",
        "| application | DMR | Orig. avg | completed-only | deadline-weighted | completed-only loss | total RT-score loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["policy"] != "pantheon":
            continue
        lines.append(
            f"| {row['application']} | {float(row['dmr']) * 100:.1f}% | "
            f"{pct(float(row['full_depth_accuracy_avg'])):.1f}% | "
            f"{pct(float(row['completed_selected_accuracy'])):.1f}% | "
            f"{pct(float(row['deadline_weighted_accuracy'])):.1f}% | "
            f"{float(row['completed_only_loss_pp']):.1f} pp | "
            f"{float(row['deadline_weighted_loss_pp']):.1f} pp |"
        )
    lines += [
        "",
        "Interpretation: Pantheon improves timeliness by moving completed jobs to shallower exits / lighter variants. Under the modern memory-heavy setup, this already reduces completed-only accuracy; remaining missed jobs further reduce deadline-weighted accuracy.",
        "",
    ]
    (OUT / "pantheon_accuracy_loss.md").write_text("\n".join(lines))


def main() -> None:
    cases = [run_case(builder) for builder in (build_smart_traffic_case, build_uav_vit_case, build_mixed_modern_case)]
    summary_rows, job_rows = build_loss_rows(cases)
    comparison_rows = build_comparison_rows(cases)
    pc_rows = build_pc_pantheon_rows(cases)
    write_csv(OUT / "pantheon_accuracy_loss.csv", summary_rows)
    write_csv(OUT / "pantheon_accuracy_loss_jobs.csv", job_rows)
    write_csv(OUT / "modern_acc_comparison.csv", comparison_rows)
    write_csv(OUT / "fig1_pc_pantheon_acc_comparison.csv", pc_rows)
    write_markdown(summary_rows, comparison_rows)
    print(f"wrote {OUT / 'pantheon_accuracy_loss.csv'}")
    print(f"wrote {OUT / 'pantheon_accuracy_loss_jobs.csv'}")
    print(f"wrote {OUT / 'modern_acc_comparison.csv'}")
    print(f"wrote {OUT / 'fig1_pc_pantheon_acc_comparison.csv'}")
    print(f"wrote {OUT / 'pantheon_accuracy_loss.md'}")


if __name__ == "__main__":
    main()
