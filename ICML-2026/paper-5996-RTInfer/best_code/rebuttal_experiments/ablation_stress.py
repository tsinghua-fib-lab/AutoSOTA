from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import run_policies
from modern_workloads import (
    build_mixed_modern_case,
    build_smart_traffic_case,
    build_uav_vit_case,
    scaled_deadlines,
)


OUT = ROOT / "outputs" / "scheduling_analysis"
POLICIES = ("rtinfer", "rtinfer-wo-alc", "rtinfer-wo-ms", "rtinfer-wo-dlp")
MEMORY_MIB = 4096.0

# Component ablations are run near each application's stress boundary so the
# disabled mechanism is visible without collapsing RTInfer itself.
ABLATION_SETTINGS = {
    "Traffic": {"kappa": 0.55, "bandwidth_gbps": 0.5},
    "UAV": {"kappa": 0.30, "bandwidth_gbps": 0.7},
    "Robot": {"kappa": 0.45, "bandwidth_gbps": 1.0},
}

APP_LABELS = {
    "modern CNN smart traffic: YOLOv8-L 1080p + YOLOv8n": ("Traffic", "Smart Traffic"),
    "ViT UAV scene recognition: ViT-L + MobileViT-S": ("UAV", "UAV Ground"),
    "mixed modern deployment: YOLO + ViT + edge GPT-2": ("Robot", "Service Robot"),
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relative_accuracy_metrics(result, models) -> tuple[float, float]:
    weighted: list[float] = []
    completed: list[float] = []
    for job in result.schedule_events:
        if job.variant is None:
            weighted.append(0.0)
            continue
        original = models[job.task.model_name].full_accuracy
        relative = job.variant.accuracy / original if original else 0.0
        if job.missed:
            weighted.append(0.0)
        else:
            weighted.append(relative)
            completed.append(relative)
    deadline_weighted = sum(weighted) / len(weighted) if weighted else 0.0
    completed_only = sum(completed) / len(completed) if completed else 0.0
    return deadline_weighted, completed_only


def main() -> None:
    rows: list[dict[str, object]] = []
    for builder in (build_smart_traffic_case, build_uav_vit_case, build_mixed_modern_case):
        title, models, atlas, tasks, duration_ms = builder()
        app_short, app_name = APP_LABELS[title]
        setting = ABLATION_SETTINGS[app_short]
        kappa = float(setting["kappa"])
        bandwidth_gbps = float(setting["bandwidth_gbps"])
        results = run_policies(
            models,
            atlas,
            scaled_deadlines(tasks, kappa),
            policies=POLICIES,
            memory_mib=MEMORY_MIB,
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
        )
        for result in results:
            deadline_weighted_acc, completed_only_acc = relative_accuracy_metrics(result, models)
            rows.append(
                {
                    "application": app_name,
                    "app_short": app_short,
                    "policy": result.policy,
                    "kappa": kappa,
                    "memory_mib": MEMORY_MIB,
                    "bandwidth_gbps": bandwidth_gbps,
                    "total_jobs": result.total_jobs,
                    "missed_jobs": result.missed_jobs,
                    "dmr": round(result.deadline_miss_rate, 6),
                    "deadline_weighted_accuracy": round(deadline_weighted_acc, 6),
                    "completed_only_accuracy": round(completed_only_acc, 6),
                    "avg_latency_ms": round(result.average_latency_us / 1000.0, 3),
                    "avg_load_ms": round(result.average_load_us / 1000.0, 3),
                }
            )

    write_csv(OUT / "modern_ablation_stress.csv", rows)

    lines = [
        "# Revised Original-style Ablation Setup",
        "",
        "- Deadline scaling: Traffic `0.55`, UAV `0.30`, Robot `0.45`.",
        f"- Effective memory budget: `{MEMORY_MIB:.0f} MiB`.",
        "- H2D bandwidth floor: Traffic `0.5 GB/s`, UAV `0.7 GB/s`, Robot `1.0 GB/s`.",
        "- Purpose: follow the original component ablation definitions: fixed single-model execution without ALC, greedy non-layout placement without MS, and full reloads without DLP.",
        "",
        "| application | policy | DMR | accuracy | avg load (ms) | avg latency (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['application']} | {row['policy']} | "
            f"{float(row['dmr']) * 100:.1f}% | "
            f"{float(row['deadline_weighted_accuracy']) * 100:.1f}% | "
            f"{float(row['avg_load_ms']):.1f} | "
            f"{float(row['avg_latency_ms']):.1f} |"
        )
    (OUT / "modern_ablation_stress.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'modern_ablation_stress.csv'}")
    print(f"wrote {OUT / 'modern_ablation_stress.md'}")


if __name__ == "__main__":
    main()
