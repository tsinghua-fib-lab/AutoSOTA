"""
Plot combined_ctx_comparison_tok1024.pdf for §4.4 (long-context comparison).

Compares v1 / EB(k̂*) / EB+ on long-context workloads (8K, 16K, 32K, 64K, 128K
input tokens at fixed output length 1024).

INPUT (consumed):
  outputs from reproduce/long_context/run_128k_all_models.sh and run_long_context_comparison.sh.
  These produce per-(context-length, scheduler) bench JSON files; expected
  intermediate aggregation is a CSV with columns:
    context_len, scheduler, throughput, mean_ttft_ms, mean_tpot_ms, mean_itl_ms

REPRODUCTION:
  1. cd reproduce/long_context
  2. ./run_long_context_comparison.sh
  3. python aggregate.py outputs/   → long_context_summary.csv
  4. python plot_long_context.py long_context_summary.csv

If --demo or no CSV is given, renders an illustrative version.

Output:
  combined_ctx_comparison_tok1024.pdf — 4 panels (throughput, TTFT, TPOT, ITL)
  vs context length, one curve per scheduler.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEDULERS = ["v1", "eb", "ebplus"]
SCHEDULER_LABELS = {"v1": "v1", "eb": "EB(k̂*)", "ebplus": "EB⁺"}
COLORS = {"v1": "tab:red", "eb": "tab:blue", "ebplus": "tab:green"}
METRICS = [
    ("throughput", "Throughput (RPS)", True),
    ("mean_ttft_ms", "Mean TTFT (ms)", False),
    ("mean_tpot_ms", "Mean TPOT (ms)", False),
    ("mean_itl_ms", "Mean ITL (ms)", False),
]


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(r)
    return rows


def demo_rows() -> list[dict]:
    ctx_lens = [8192, 16384, 32768, 65536, 131072]
    rng = np.random.default_rng(1)
    rows = []
    for c in ctx_lens:
        # rough shape: throughput falls with context; EB+ degrades least
        base_t = 12.0 * (8192 / c) ** 0.6
        rows.append({"context_len": c, "scheduler": "v1",
                     "throughput": base_t * 0.85 + rng.normal(0, 0.1),
                     "mean_ttft_ms": 4000 * (c / 8192) ** 1.3 + rng.normal(0, 50),
                     "mean_tpot_ms": 60 + 0.5 * np.log2(c / 1024),
                     "mean_itl_ms": 60 + 0.6 * np.log2(c / 1024)})
        rows.append({"context_len": c, "scheduler": "eb",
                     "throughput": base_t * 1.0 + rng.normal(0, 0.1),
                     "mean_ttft_ms": 5500 * (c / 8192) ** 1.3 + rng.normal(0, 50),
                     "mean_tpot_ms": 50 + 0.3 * np.log2(c / 1024),
                     "mean_itl_ms": 50 + 0.4 * np.log2(c / 1024)})
        rows.append({"context_len": c, "scheduler": "ebplus",
                     "throughput": base_t * 1.08 + rng.normal(0, 0.1),
                     "mean_ttft_ms": 4500 * (c / 8192) ** 1.3 + rng.normal(0, 50),
                     "mean_tpot_ms": 53 + 0.35 * np.log2(c / 1024),
                     "mean_itl_ms": 53 + 0.45 * np.log2(c / 1024)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?",
                   help="long_context_summary.csv with columns context_len, scheduler, ...")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--output", default="combined_ctx_comparison_tok1024.pdf")
    args = p.parse_args()

    if args.demo or not args.csv or not Path(args.csv).exists():
        rows = demo_rows()
        suffix = "(DEMO data)"
    else:
        rows = load_csv(Path(args.csv))
        suffix = f"({Path(args.csv).name})"

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, (mkey, mlabel, _hb) in zip(axes, METRICS):
        for sched in SCHEDULERS:
            sub = sorted(
                [r for r in rows if r.get("scheduler") == sched],
                key=lambda r: r.get("context_len", 0),
            )
            if not sub:
                continue
            xs = [r["context_len"] for r in sub]
            ys = [r.get(mkey, np.nan) for r in sub]
            ax.plot(xs, ys, marker="o", color=COLORS[sched],
                    label=SCHEDULER_LABELS[sched])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel(mlabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Long-context comparison (output=1024 tokens) {suffix}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
