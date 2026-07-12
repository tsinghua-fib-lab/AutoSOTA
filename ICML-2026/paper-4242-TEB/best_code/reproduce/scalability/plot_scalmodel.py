# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plot scalmodel.pdf for §4.5.2 (Scalability across model architectures).

Compares v0 / v1 / EB(k̂*) on RTX PRO 6000 with the WildChat workload across:
  Llama-3.1-8B-Instruct, Mathstral-7B-v0.1,
  Qwen2.5-Coder-7B, DeepSeek-R1-Distill-Qwen-7B

INPUT (consumed):
  per-model `optimal_per_scheduler.json` files, each produced by
  reproduce/real_workloads/analyze_grid_search.py on a single-workload grid search.

REPRODUCTION:
  1. For each MODEL in
     {Llama3.1-8B, Mathstral-7B, Qwen2.5-Coder-7B, DeepSeek-R1-Distill}:
       MODEL=$MODEL ./run_grid_search.sh   (with --workloads wildchat)
       python analyze_grid_search.py outputs/RTX6000_$MODEL/
  2. python plot_scalmodel.py \
       --inputs outputs/RTX6000_Llama-3.1-8B/optimal_per_scheduler.json \
                outputs/RTX6000_Mathstral-7B/optimal_per_scheduler.json \
                ...

Pass --demo for a synthetic illustrative version using approximate paper numbers.

Output:
  scalmodel.pdf — 3 panels: RPS, mean TTFT, mean TPOT,
  bars grouped by model × scheduler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCHEDULERS = ["v0", "v1", "eb"]
SCHEDULER_LABELS = {"v0": "v0", "v1": "v1", "eb": "EB(k̂*)"}
COLORS = {"v0": "tab:gray", "v1": "tab:red", "eb": "tab:blue"}


def demo_data() -> list[tuple[str, dict]]:
    # Approximate paper §4.5.2 numbers (Section commented-out table)
    return [
        (
            "Llama-3.1-8B",
            {
                "v0": {
                    "throughput": 10.50,
                    "mean_ttft_ms": 18200,
                    "mean_tpot_ms": 245.30,
                },
                "v1": {
                    "throughput": 14.93,
                    "mean_ttft_ms": 29100,
                    "mean_tpot_ms": 358.04,
                },
                "eb": {
                    "throughput": 16.37,
                    "mean_ttft_ms": 34790,
                    "mean_tpot_ms": 188.45,
                },
            },
        ),
        (
            "Mathstral-7B",
            {
                "v0": {
                    "throughput": 11.25,
                    "mean_ttft_ms": 32400,
                    "mean_tpot_ms": 235.60,
                },
                "v1": {
                    "throughput": 12.85,
                    "mean_ttft_ms": 41900,
                    "mean_tpot_ms": 221.10,
                },
                "eb": {
                    "throughput": 14.39,
                    "mean_ttft_ms": 55260,
                    "mean_tpot_ms": 178.30,
                },
            },
        ),
        (
            "Qwen2.5-Coder-7B",
            {
                "v0": {
                    "throughput": 10.83,
                    "mean_ttft_ms": 28500,
                    "mean_tpot_ms": 175.80,
                },
                "v1": {
                    "throughput": 13.20,
                    "mean_ttft_ms": 41880,
                    "mean_tpot_ms": 168.90,
                },
                "eb": {
                    "throughput": 14.19,
                    "mean_ttft_ms": 22550,
                    "mean_tpot_ms": 125.91,
                },
            },
        ),
        (
            "DeepSeek-R1-Distill",
            {
                "v0": {
                    "throughput": 10.21,
                    "mean_ttft_ms": 35800,
                    "mean_tpot_ms": 225.40,
                },
                "v1": {
                    "throughput": 11.80,
                    "mean_ttft_ms": 44200,
                    "mean_tpot_ms": 198.70,
                },
                "eb": {
                    "throughput": 13.75,
                    "mean_ttft_ms": 42100,
                    "mean_tpot_ms": 165.54,
                },
            },
        ),
    ]


def load_inputs(paths: list[Path]) -> list[tuple[str, dict]]:
    out = []
    for path in paths:
        # Expect parent dir name like "RTX6000_<MODEL>" — extract <MODEL>
        name = path.parent.name
        model = name.split("_", 1)[1] if "_" in name else path.stem
        with open(path) as f:
            d = json.load(f)
        # Optimal JSON may have a "wildchat" key or be flat — handle both
        if "wildchat" in d:
            d = d["wildchat"]
        out.append((model, d))
    return out


def grouped_bar(ax, rows: list[tuple[str, dict]], metric: str, title: str, ylabel: str):
    models = [m for m, _ in rows]
    x = np.arange(len(models))
    width = 0.27
    for i, sched in enumerate(SCHEDULERS):
        vals = [data.get(sched, {}).get(metric, np.nan) for _, data in rows]
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=SCHEDULER_LABELS[sched],
            color=COLORS[sched],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="paths to per-model optimal_per_scheduler.json files",
    )
    p.add_argument("--demo", action="store_true")
    p.add_argument("--output", default="scalmodel.pdf")
    args = p.parse_args()

    paths = [Path(x) for x in args.inputs if Path(x).exists()]
    if args.demo or not paths:
        rows = demo_data()
        suffix = "(DEMO data — approximate paper §4.5.2)"
    else:
        rows = load_inputs(paths)
        suffix = ""

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    grouped_bar(axes[0], rows, "throughput", f"Throughput (RPS) {suffix}", "RPS")
    grouped_bar(axes[1], rows, "mean_ttft_ms", "Mean TTFT", "TTFT (ms)")
    grouped_bar(axes[2], rows, "mean_tpot_ms", "Mean TPOT", "TPOT (ms)")

    fig.suptitle(
        "Scalability across model architectures (RTX PRO 6000, WildChat)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
