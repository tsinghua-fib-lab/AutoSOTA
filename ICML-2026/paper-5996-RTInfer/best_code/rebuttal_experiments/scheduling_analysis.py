from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtinfer.delta_graph import DeltaGraph
from rtinfer.layout import BufferBlock, MemoryLayoutScheduler, buffers_for_job
from rtinfer.model import Job, TaskSpec, iter_jobs
from rtinfer.scheduler import OnlineScheduler

from modern_workloads import (
    KAPPAS,
    POLICIES,
    build_mixed_modern_case,
    build_smart_traffic_case,
    build_uav_vit_case,
    scaled_deadlines,
)


OUT = ROOT / "outputs" / "scheduling_analysis"
MEMORY_MIB = 6144.0
BANDWIDTH_GBPS = 24.0


TITLE_TO_APP = {
    "modern CNN smart traffic: YOLOv8-L 1080p + YOLOv8n": "smart_traffic_yolov8",
    "ViT UAV scene recognition: ViT-L + MobileViT-S": "uav_vit_mobilevit",
    "mixed modern deployment: YOLO + ViT + edge GPT-2": "service_robot_yolo_vit_gpt2",
}


@dataclass(frozen=True)
class RunRecord:
    app: str
    title: str
    kappa: float
    policy: str
    duration_ms: int
    jobs: list[Job]
    task_indices: dict[int, int]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_all() -> list[RunRecord]:
    cases = [build_smart_traffic_case(), build_uav_vit_case(), build_mixed_modern_case()]
    records: list[RunRecord] = []
    for title, models, atlas, base_tasks, duration_ms in cases:
        app = TITLE_TO_APP[title]
        for kappa in KAPPAS:
            tasks = scaled_deadlines(base_tasks, kappa)
            task_indices = {id(task): idx for idx, task in enumerate(tasks)}
            for policy in POLICIES:
                scheduler = OnlineScheduler(models, atlas, MEMORY_MIB, DeltaGraph(bandwidth_floor_gbps=BANDWIDTH_GBPS), policy)
                result = scheduler.run(tasks, duration_ms * 1000)
                records.append(RunRecord(app, title, kappa, policy, duration_ms, result.schedule_events, task_indices))
    return records


def job_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rec in records:
        for job in sorted(rec.jobs, key=lambda item: item.job_id):
            variant = job.variant
            finish_us = job.finish_us if job.finish_us is not None else job.release_us
            start_us = job.start_us if job.start_us is not None else job.release_us
            raw_acc = variant.accuracy if variant else 0.0
            rows.append(
                {
                    "app": rec.app,
                    "kappa": rec.kappa,
                    "policy": job.variant and rec.policy or rec.policy,
                    "job_id": job.job_id,
                    "task_index": rec.task_indices.get(id(job.task), -1),
                    "model": job.task.model_name,
                    "release_ms": job.release_us / 1000.0,
                    "relative_deadline_ms": job.task.deadline_us / 1000.0,
                    "absolute_deadline_ms": job.absolute_deadline_us / 1000.0,
                    "start_ms": start_us / 1000.0,
                    "finish_ms": finish_us / 1000.0,
                    "response_ms": (finish_us - job.release_us) / 1000.0,
                    "missed": int(job.missed),
                    "variant_pruning": variant.pruning if variant else "",
                    "variant_exit": variant.exit_index if variant else "",
                    "variant_latency_ms": variant.latency_us / 1000.0 if variant else "",
                    "variant_memory_mib": variant.memory_mib if variant else "",
                    "load_ms": job.load_us / 1000.0,
                    "raw_accuracy": raw_acc,
                    "deadline_weighted_accuracy": 0.0 if job.missed else raw_acc,
                }
            )
    return rows


def window_rows(records: list[RunRecord], windows: int = 4) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rec in records:
        step = rec.duration_ms / windows
        for idx in range(windows):
            start = idx * step
            end = (idx + 1) * step
            jobs = [job for job in rec.jobs if start <= job.release_us / 1000.0 < end]
            total = len(jobs)
            missed = sum(1 for job in jobs if job.missed)
            raw_acc = [job.variant.accuracy for job in jobs if job.variant is not None and not job.missed]
            weighted = [0.0 if job.missed or job.variant is None else job.variant.accuracy for job in jobs]
            rows.append(
                {
                    "app": rec.app,
                    "kappa": rec.kappa,
                    "policy": rec.policy,
                    "window_start_ms": round(start, 3),
                    "window_end_ms": round(end, 3),
                    "jobs": total,
                    "missed": missed,
                    "dmr": missed / total if total else 0.0,
                    "completed_only_accuracy": sum(raw_acc) / len(raw_acc) if raw_acc else 0.0,
                    "deadline_weighted_accuracy": sum(weighted) / total if total else 0.0,
                }
            )
    return rows


def utilization_rows(records: list[RunRecord], interval_ms: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rec in records:
        max_active = 1
        for t in range(0, rec.duration_ms + 1, interval_ms):
            live = [job for job in rec.jobs if (job.start_us or 0) / 1000.0 <= t < (job.finish_us or 0) / 1000.0]
            max_active = max(max_active, len(live))
        for t in range(0, rec.duration_ms + 1, interval_ms):
            live = [job for job in rec.jobs if (job.start_us or 0) / 1000.0 <= t < (job.finish_us or 0) / 1000.0]
            memory = sum(job.variant.memory_mib for job in live if job.variant is not None)
            rows.append(
                {
                    "app": rec.app,
                    "kappa": rec.kappa,
                    "policy": rec.policy,
                    "time_ms": t,
                    "active_jobs": len(live),
                    "memory_mib": memory,
                    "memory_util_pct": min(100.0, memory / MEMORY_MIB * 100.0),
                    "active_job_util_proxy_pct": min(100.0, len(live) / max_active * 100.0),
                }
            )
    return rows


def preemption_proxy_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rec in records:
        completed = [job for job in rec.jobs if job.finish_us is not None and job.start_us is not None]
        interleaved = 0
        for job in completed:
            assert job.start_us is not None and job.finish_us is not None
            overlaps = [
                other
                for other in completed
                if other.job_id != job.job_id
                and other.start_us is not None
                and other.finish_us is not None
                and job.start_us < other.finish_us
                and other.start_us < job.finish_us
            ]
            if overlaps:
                interleaved += 1
        rows.append(
            {
                "app": rec.app,
                "kappa": rec.kappa,
                "policy": rec.policy,
                "jobs": len(completed),
                "interleaved_or_preempted_jobs_proxy": interleaved,
                "ratio": interleaved / len(completed) if completed else 0.0,
                "note": "Proxy for Pantheon count_preemption.py: counts jobs whose execution interval overlaps another job. It captures RTInfer concurrency, not hardware preemption.",
            }
        )
    return rows


def active_buffers(jobs: list[Job], now_us: int) -> list[BufferBlock]:
    buffers: list[BufferBlock] = []
    for job in jobs:
        if job.variant is None or job.start_us is None or job.finish_us is None:
            continue
        if job.release_us <= now_us < job.finish_us:
            buffers.extend(buffers_for_job(job.job_id, job.start_us, job.variant.latency_us, job.variant.memory_mib))
    return buffers


def scheduler_rows(records: list[RunRecord]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    latency_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    layout = MemoryLayoutScheduler(MEMORY_MIB)
    for rec in records:
        if not rec.policy.startswith("rtinfer"):
            continue
        release_times = sorted({job.release_us for job in rec.jobs})
        for now_us in release_times:
            buffers = active_buffers(rec.jobs, now_us)
            if not buffers:
                continue
            active_jobs = len({buffer.job_id for buffer in buffers})
            start = perf_counter()
            feasible = layout.place(buffers) is not None
            latency_us = (perf_counter() - start) * 1_000_000.0
            bucket = "1-4" if active_jobs <= 4 else ("5-8" if active_jobs <= 8 else "9+")
            latency_rows.append(
                {
                    "app": rec.app,
                    "kappa": rec.kappa,
                    "policy": rec.policy,
                    "time_ms": now_us / 1000.0,
                    "active_jobs": active_jobs,
                    "active_buffers": len(buffers),
                    "bucket": bucket,
                    "latency_us": latency_us,
                    "feasible": int(feasible),
                }
            )
            block_rows.append(
                {
                    "app": rec.app,
                    "kappa": rec.kappa,
                    "policy": rec.policy,
                    "time_ms": now_us / 1000.0,
                    "active_jobs": active_jobs,
                    "scheduled_blocks": len(buffers),
                    "total_buffer_mib": sum(buffer.size_mib for buffer in buffers),
                }
            )
    return latency_rows, block_rows


def cdf_rows(latency_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bucket in ("1-4", "5-8", "9+"):
        values = sorted(float(row["latency_us"]) for row in latency_rows if row["bucket"] == bucket)
        n = len(values)
        for idx, value in enumerate(values, 1):
            rows.append({"bucket": bucket, "latency_us": value, "cdf": idx / n if n else 0.0})
    return rows


def write_summary(
    records: list[RunRecord],
    preemption_rows: list[dict[str, object]],
    latency_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Scheduling Analysis Outputs",
        "",
        "This runner mirrors the Pantheon `experiments/logs/scheduling` scripts using RTInfer's deterministic modern workload simulations.",
        "",
        "## Mapping From Pantheon Scripts",
        "",
        "- `save_trace.py` -> `modern_response_trace.csv`, `modern_time_window_stats.csv`.",
        "- `save_gpu_util.py` -> `modern_utilization_trace.csv` with active-job and memory-utilization proxies.",
        "- `save_scheduling_latency_CDF.py` / `save_scheduling_latency_scatter.py` -> `modern_scheduler_latency.csv`, `modern_scheduler_latency_cdf.csv`.",
        "- `count_preemption.py` -> `modern_preemption_proxy.csv`; this is an interleaving proxy because the Python simulator does not emit Pantheon C++ `[EXEC:BLOCK]` preemption logs.",
        "- `stat_block_missing.py` -> `modern_scheduled_block_counts.csv`.",
        "",
        "## Key Checks",
        "",
    ]
    for rec in records:
        if rec.policy != "rtinfer" or abs(rec.kappa - 1.0) > 1e-6:
            continue
        total = len(rec.jobs)
        missed = sum(1 for job in rec.jobs if job.missed)
        acc = sum(0.0 if job.missed or job.variant is None else job.variant.accuracy for job in rec.jobs) / total
        lines.append(f"- `{rec.app}` κ=1.0 RTInfer: DMR={missed}/{total}={missed / total:.4f}, deadline-weighted accuracy={acc:.4f}.")
    if preemption_rows:
        worst = max(preemption_rows, key=lambda row: float(row["ratio"]))
        lines.append(f"- Highest interleaving/preemption proxy: {worst['app']} {worst['policy']} κ={worst['kappa']} ratio={float(worst['ratio']):.4f}.")
    if latency_rows:
        rtinfer_lat = [float(row["latency_us"]) for row in latency_rows if row["policy"] == "rtinfer"]
        if rtinfer_lat:
            lines.append(f"- RTInfer scheduler latency samples: n={len(rtinfer_lat)}, avg={sum(rtinfer_lat)/len(rtinfer_lat):.2f} us, max={max(rtinfer_lat):.2f} us.")
    lines.append("")
    lines.append("Note: these are deterministic scheduling simulations, not raw Pantheon C++ log parses.")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = run_all()
    response = job_rows(records)
    windows = window_rows(records)
    util = utilization_rows(records)
    preempt = preemption_proxy_rows(records)
    latency, blocks = scheduler_rows(records)
    cdf = cdf_rows(latency)
    write_csv(OUT / "modern_response_trace.csv", response)
    write_csv(OUT / "modern_time_window_stats.csv", windows)
    write_csv(OUT / "modern_utilization_trace.csv", util)
    write_csv(OUT / "modern_preemption_proxy.csv", preempt)
    write_csv(OUT / "modern_scheduler_latency.csv", latency)
    write_csv(OUT / "modern_scheduler_latency_cdf.csv", cdf)
    write_csv(OUT / "modern_scheduled_block_counts.csv", blocks)
    write_summary(records, preempt, latency)
    print(f"wrote scheduling analysis outputs to {OUT}")


if __name__ == "__main__":
    main()
