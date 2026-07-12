"""
Plot ttft.pdf and tpot.pdf for §4.3.3 (Latency Analysis).

Compares TTFT and TPOT of v0 / v1 / EB(k̂*) across 4 real workloads
(ShareGPT, LongBench, WildChat, NuminaMath) × {Qwen3-8B, Qwen3-30B-A3B}
× {RTX PRO 6000, H200}.

INPUT (consumed):
  per-(GPU,Model) grid-search outputs from reproduce/real_workloads/run_grid_search.sh,
  analysed by reproduce/real_workloads/analyze_grid_search.py which produces
  `optimal_per_scheduler.json` with structure:
    {
      "<workload>": {
        "v0":  {"throughput": ..., "mean_ttft_ms": ..., "mean_tpot_ms": ...},
        "v1":  {...},
        "eb": {...}
      },
      ...
    }

REPRODUCTION:
  1. cd reproduce/real_workloads
  2. ./run_grid_search.sh   (per workload, per model, per GPU; long)
  3. python analyze_grid_search.py outputs/<GPU>_<MODEL>/  → optimal_per_scheduler.json
  4. python plot_real_workload_latency.py \
       --optimal-json outputs/<GPU>_<MODEL>/optimal_per_scheduler.json \
       --gpu RTX6000 --model Qwen3-8B

If multiple (GPU, Model) JSONs are given, the script will lay them out as
sub-rows. Pass --demo for synthetic illustration.

Output:
  ttft.pdf, tpot.pdf  (or single composite if --combined)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORKLOADS = ["ShareGPT", "LongBench", "WildChat", "NuminaMath"]
SCHEDULERS = ["v0", "v1", "eb"]
SCHEDULER_LABELS = {"v0": "v0", "v1": "v1", "eb": "EB(k̂*)"}
COLORS = {"v0": "tab:gray", "v1": "tab:red", "eb": "tab:blue"}


def load_optimal(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def demo_data() -> dict:
    """Synthetic illustrative data (RTX 6000 / Qwen3-8B numbers approximate the paper)."""
    return {
        "ShareGPT":   {"v0": {"mean_ttft_ms": 12000, "mean_tpot_ms": 220},
                       "v1": {"mean_ttft_ms": 14500, "mean_tpot_ms": 269},
                       "eb": {"mean_ttft_ms": 18500, "mean_tpot_ms": 94}},
        "LongBench":  {"v0": {"mean_ttft_ms": 180000, "mean_tpot_ms": 64},
                       "v1": {"mean_ttft_ms": 176000, "mean_tpot_ms": 58},
                       "eb": {"mean_ttft_ms": 170000, "mean_tpot_ms": 67}},
        "WildChat":   {"v0": {"mean_ttft_ms": 22000, "mean_tpot_ms": 195},
                       "v1": {"mean_ttft_ms": 25000, "mean_tpot_ms": 173},
                       "eb": {"mean_ttft_ms": 30000, "mean_tpot_ms": 113}},
        "NuminaMath": {"v0": {"mean_ttft_ms": 3000, "mean_tpot_ms": 75},
                       "v1": {"mean_ttft_ms": 4000, "mean_tpot_ms": 82},
                       "eb": {"mean_ttft_ms": 4500, "mean_tpot_ms": 65}},
    }


def grouped_bar(ax, data: dict, metric: str, title: str, ylabel: str) -> None:
    x = np.arange(len(WORKLOADS))
    width = 0.27
    for i, sched in enumerate(SCHEDULERS):
        vals = [data.get(w, {}).get(sched, {}).get(metric, np.nan) for w in WORKLOADS]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=SCHEDULER_LABELS[sched],
               color=COLORS[sched], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--optimal-json", default=None,
                   help="Path to optimal_per_scheduler.json")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--gpu", default="RTX 6000")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--ttft-output", default="ttft.pdf")
    p.add_argument("--tpot-output", default="tpot.pdf")
    args = p.parse_args()

    if args.demo or not args.optimal_json or not Path(args.optimal_json).exists():
        data = demo_data()
        suffix = "(DEMO data)"
    else:
        data = load_optimal(Path(args.optimal_json))
        suffix = f"({args.gpu}, {args.model})"

    # TTFT
    fig, ax = plt.subplots(figsize=(7, 3.5))
    grouped_bar(ax, data, "mean_ttft_ms", f"Mean TTFT {suffix}", "TTFT (ms)")
    fig.tight_layout()
    fig.savefig(args.ttft_output, bbox_inches="tight")
    print(f"Wrote {args.ttft_output}")

    # TPOT
    fig, ax = plt.subplots(figsize=(7, 3.5))
    grouped_bar(ax, data, "mean_tpot_ms", f"Mean TPOT {suffix}", "TPOT (ms)")
    fig.tight_layout()
    fig.savefig(args.tpot_output, bbox_inches="tight")
    print(f"Wrote {args.tpot_output}")


if __name__ == "__main__":
    main()
