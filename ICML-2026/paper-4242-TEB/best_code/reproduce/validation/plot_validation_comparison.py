#!/usr/bin/env python3
"""Overlay paper Figure 3 (§4.2) with the reproduced validation results.

For each scenario × metric, plots:
  - paper's `v1` (red line), paper's `EB(k̂*)` (blue dashed), paper's `fixed-k` sweep (green points)
  - reproduced `v1` (red, paler), reproduced `EB(k̂*)` (blue, paler)

So you can see exactly where the reproduction departs from the paper figure.

Reads:
  paper_data/validation_grid.json       (Figure 3 data, extracted from the paper PDF)
  outputs/controller_validation/<GPU>_<MODEL>/validation_summary.csv  (our run)
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = Path(__file__).resolve().parent
PAPER_DATA = PLOT_DIR / "paper_data" / "validation_grid.json"


C_SWEEP = "#2D6A4F"   # green
C_V1    = "#E63946"   # red
C_EB    = "#457B9D"   # blue


SCENARIOS = [
    ("decode_heavy",  "decode_heavy_in128_out1024",  "Decode-heavy (128/1024)"),
    ("balanced",      "balanced_in512_out512",       "Balanced (512/512)"),
    ("prefill_heavy", "prefill_heavy_in1024_out128", "Prefill-heavy (1024/128)"),
]


def load_paper() -> dict:
    with open(PAPER_DATA) as f:
        return json.load(f)


def load_reproduction(csv_path: Path) -> dict:
    """Return {scenario: {scheduler: row_dict}} indexed by name."""
    out: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            scen = row["scenario"].strip("'\"")
            sched = row["scheduler"].strip("'\"")
            row_clean = {}
            for k, v in row.items():
                try:
                    row_clean[k] = float(v)
                except (TypeError, ValueError):
                    row_clean[k] = v
            out.setdefault(scen, {})[sched] = row_clean
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reproduction-csv",
        default=str(PLOT_DIR / "outputs" / "controller_validation"
                    / "H200_Qwen3-8B" / "validation_summary.csv"),
        help="Path to our reproduced validation_summary.csv",
    )
    p.add_argument(
        "--output",
        default=str(PLOT_DIR / "validation_grid_comparison.pdf"),
    )
    args = p.parse_args()

    paper = load_paper()
    repro_path = Path(args.reproduction_csv)
    repro = load_reproduction(repro_path) if repro_path.exists() else {}

    k_values = np.array(paper["k_values"])

    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5))
    fig.subplots_adjust(hspace=0.30, wspace=0.32)

    col_titles = ["Throughput (tok/s)", "TPOT mean (ms)", "TPOT p99 (ms)"]

    for i, (scen_key, paper_key, scen_label) in enumerate(SCENARIOS):
        wd = paper["workloads"][paper_key]
        tp = wd["throughput"]
        tpot = wd["tpot"]

        # Throughput panel ----------------------------------------------------
        ax = axes[i][0]
        # Paper sweep
        ax.plot(k_values, tp["fixed_k"], "o-", color=C_SWEEP,
                label="Paper fixed-k", alpha=0.9)
        ax.axhline(tp["v1"], color=C_V1, linewidth=2,
                   label=f"Paper v1 = {tp['v1']:.0f}")
        ax.axhline(tp["ours"], color=C_EB, linestyle="--", linewidth=2,
                   label=f"Paper EB(k̂*) = {tp['ours']:.0f}")
        # Reproduction
        if scen_key in repro:
            r = repro[scen_key]
            if "v1" in r:
                ax.axhline(r["v1"]["tp_real"], color=C_V1, linewidth=1.2,
                           alpha=0.45, linestyle=":",
                           label=f"Ours v1 = {r['v1']['tp_real']:.0f}")
            if "eb" in r:
                ax.axhline(r["eb"]["tp_real"], color=C_EB, linewidth=1.2,
                           alpha=0.45, linestyle=":",
                           label=f"Ours EB(k̂*) = {r['eb']['tp_real']:.0f}")
        ax.set_ylabel(scen_label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        if i == 0:
            ax.set_title(col_titles[0])

        # TPOT mean panel -----------------------------------------------------
        ax = axes[i][1]
        ax.plot(k_values, tpot["fixed_k_mean"], "o-", color=C_SWEEP,
                label="Paper fixed-k", alpha=0.9)
        ax.axhline(tpot["v1_mean"], color=C_V1, linewidth=2,
                   label=f"Paper v1 = {tpot['v1_mean']:.1f}")
        ax.axhline(tpot["ours_mean"], color=C_EB, linestyle="--", linewidth=2,
                   label=f"Paper EB(k̂*) = {tpot['ours_mean']:.1f}")
        # No reproduction TPOT available in validation_summary.csv (only throughput)
        # — note that to compare TPOT we'd need to read each cell's bench_*.json
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        if i == 0:
            ax.set_title(col_titles[1])

        # TPOT p99 panel ------------------------------------------------------
        ax = axes[i][2]
        ax.plot(k_values, tpot["fixed_k_p99"], "o-", color=C_SWEEP,
                label="Paper fixed-k", alpha=0.9)
        ax.axhline(tpot["v1_p99"], color=C_V1, linewidth=2,
                   label=f"Paper v1 = {tpot['v1_p99']:.1f}")
        ax.axhline(tpot["ours_p99"], color=C_EB, linestyle="--", linewidth=2,
                   label=f"Paper EB(k̂*) = {tpot['ours_p99']:.1f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        if i == 0:
            ax.set_title(col_titles[2])

        if i == 2:
            for ax in axes[i]:
                ax.set_xlabel("Fixed k")

    fig.suptitle(
        "§4.2 Validation: paper Figure 3 (solid) vs. our reproduction (dotted)",
        fontsize=12,
    )

    out = Path(args.output)
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {png}")

    # Also print a numerical diff table
    print()
    print("=" * 80)
    print("Throughput comparison (paper vs ours):")
    print(f"{'Scenario':<18} {'paper v1':>10} {'paper EB':>10} {'ours v1':>10} {'ours EB':>10}")
    print("-" * 80)
    for scen_key, paper_key, label in SCENARIOS:
        ptp = paper["workloads"][paper_key]["throughput"]
        rv1 = repro.get(scen_key, {}).get("v1", {}).get("tp_real", float("nan"))
        reb = repro.get(scen_key, {}).get("eb", {}).get("tp_real", float("nan"))
        print(f"{scen_key:<18} {ptp['v1']:>10.0f} {ptp['ours']:>10.0f} "
              f"{rv1:>10.0f} {reb:>10.0f}")


if __name__ == "__main__":
    main()
