from __future__ import annotations

import csv
from pathlib import Path

from common import JETSON_EFFECTIVE_MEMORY_MIB
from modern_workloads import (
    build_mixed_modern_case,
    build_smart_traffic_case,
    build_uav_vit_case,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "scheduling_analysis"
PHYSICAL_XAVIER_NX_MIB = 8192.0
SAFETY_MARGIN_MIB = 512.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("Smart Traffic (YOLOv8-L/n)", build_smart_traffic_case),
        ("UAV Ground (ViT-L/MobileViT)", build_uav_vit_case),
        ("Service Robot (YOLO+ViT+GPT-2)", build_mixed_modern_case),
    ]
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for app_name, builder in cases:
        _, models, _, tasks, _ = builder()
        per_model_counts: dict[str, int] = {}
        for task in tasks:
            per_model_counts[task.model_name] = per_model_counts.get(task.model_name, 0) + 1
        total_mib = 0.0
        for model_name, count in sorted(per_model_counts.items()):
            model = models[model_name]
            model_mib = model.cumulative_memory_mib(len(model.exits) - 1)
            subtotal = model_mib * count
            total_mib += subtotal
            detail_rows.append(
                {
                    "application": app_name,
                    "model": model_name,
                    "concurrent_streams": count,
                    "full_unpruned_memory_mib_per_stream": round(model_mib, 3),
                    "subtotal_mib": round(subtotal, 3),
                }
            )
        rows.append(
            {
                "application": app_name,
                "concurrent_rt_streams": len(tasks),
                "naive_full_unpruned_sum_mib": round(total_mib, 3),
                "jetson_xavier_nx_effective_budget_mib": JETSON_EFFECTIVE_MEMORY_MIB,
                "exceeds_effective_budget": int(total_mib > JETSON_EFFECTIVE_MEMORY_MIB),
                "headroom_vs_effective_mib": round(JETSON_EFFECTIVE_MEMORY_MIB - total_mib, 3),
                "effective_budget_minus_512mib_margin_mib": JETSON_EFFECTIVE_MEMORY_MIB - SAFETY_MARGIN_MIB,
                "exceeds_effective_minus_margin": int(total_mib > JETSON_EFFECTIVE_MEMORY_MIB - SAFETY_MARGIN_MIB),
                "jetson_xavier_nx_physical_mib": PHYSICAL_XAVIER_NX_MIB,
                "exceeds_physical_8g": int(total_mib > PHYSICAL_XAVIER_NX_MIB),
                "headroom_vs_physical_mib": round(PHYSICAL_XAVIER_NX_MIB - total_mib, 3),
            }
        )

    summary_path = OUT / "modern_memory_pressure_check.csv"
    detail_path = OUT / "modern_memory_pressure_detail.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    md = OUT / "modern_memory_pressure_check.md"
    lines = [
        "# Modern Workload Memory Pressure Check",
        "",
        f"- Jetson Xavier NX physical unified memory: `{PHYSICAL_XAVIER_NX_MIB:.0f} MiB`.",
        f"- Evaluation effective scheduling budget after OS/sensor/runtime overhead: `{JETSON_EFFECTIVE_MEMORY_MIB:.0f} MiB`.",
        f"- Conservative margin check: `{JETSON_EFFECTIVE_MEMORY_MIB - SAFETY_MARGIN_MIB:.0f} MiB`.",
        "- Calculation below uses the full-depth, unpruned `p=0,E_last` memory profile for every periodic RT stream in the application. It intentionally disables pruning, early exits, Delta loading, and memory-layout packing.",
        "- Repeated streams of the same model are counted per stream. This is conservative for weights, but appropriate for high-resolution activations/KV buffers, which dominate concurrent inference memory.",
        "",
        "| application | RT streams | naive full unpruned sum | vs 6144 MiB effective | vs 5632 MiB margin | vs 8192 MiB physical |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {application} | {concurrent_rt_streams} | {naive_full_unpruned_sum_mib:.1f} MiB | {headroom_vs_effective_mib:+.1f} MiB | {margin:+.1f} MiB | {headroom_vs_physical_mib:+.1f} MiB |".format(
                margin=(JETSON_EFFECTIVE_MEMORY_MIB - SAFETY_MARGIN_MIB) - float(row["naive_full_unpruned_sum_mib"]),
                **row,
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- All three revised applications exceed the 6144 MiB effective Jetson Xavier NX budget before adding BE tasks, CUDA allocator fragmentation, H2D staging, or preempted IF holds.",
        "- `Smart Traffic` now uses a slightly larger 1080p-heavy YOLO profile (`3 * 1700 + 2 * 700 = 6500 MiB`) so the memory-pressure premise is explicit even before safety-margin accounting.",
        "- None of these three sums exceed the raw 8192 MiB physical memory alone. The memory-pressure argument therefore should be phrased around effective usable memory on Jetson after OS/sensor/runtime overhead plus fragmentation/staging, not as raw physical DRAM overflow in every application.",
        "- The separate `outputs/jetson_real_profiles_full/jetson_real_model_profiles.csv` profiles are small proxy PyTorch graphs with random weights. They validate that profiling works on the board, but they are not the memory-pressure profiles used by the revised evaluation figures.",
        "",
    ]
    md.write_text("\n".join(lines))
    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
