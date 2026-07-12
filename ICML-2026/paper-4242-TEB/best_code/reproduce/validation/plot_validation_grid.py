"""
Plot validation_grid.pdf for §4.2 (Model Validation).

A 3x3 grid:
  rows: scenarios = {decode_heavy, balanced, prefill_heavy}
  cols: metrics   = {throughput, TPOT mean, TPOT p99}
  green points = fixed-k sweep, red marker = v1, blue dashed = EB(k̂*)

INPUT CSV (consumed):
  validation_summary.csv, produced by
    reproduce/validation/ or synthetic_e2e/ or eb_plus/traffic/analyze_validation.py outputs/controller_validation/<GPU>_<MODEL>

  Required columns (per row = one bench result):
    scenario          (decode_heavy / balanced / prefill_heavy)
    scheduler         (v1 / eb / eb_fixed_k_<N> / ...)
    k_hat_final       (the realised switching threshold)
    tp_real           (request throughput, requests/s)
    mean_tpot_ms      (mean TPOT in ms)
    p99_tpot_ms       (p99 TPOT in ms)
    completed         (request count, for sanity)

REPRODUCTION:
  1. ./run_validation.sh                                 (generates data)
  2. python analyze_validation.py outputs/.../<GPU>_<MODEL>   (generates CSV)
  3. python plot_validation_grid.py outputs/.../<GPU>_<MODEL>/validation_summary.csv

If the CSV is missing, pass --demo to render with synthetic data.

Output:
  validation_grid.pdf
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ["decode_heavy", "balanced", "prefill_heavy"]
SCENARIO_LABELS = {
    "decode_heavy": "Decode-heavy (128/1024)",
    "balanced": "Balanced (512/512)",
    "prefill_heavy": "Prefill-heavy (1024/128)",
}
METRICS = [
    ("tp_real", "Throughput (req/s)", True),    # higher is better
    ("mean_tpot_ms", "TPOT mean (ms)", False),  # lower is better
    ("p99_tpot_ms", "TPOT p99 (ms)", False),
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
    """Synthetic data for offline demo (no real CSV yet)."""
    rng = np.random.default_rng(0)
    rows = []
    for scen in SCENARIOS:
        # Fixed-k sweep: k in 1..1024 logspace
        ks = np.unique(np.round(np.logspace(0, 10, 12, base=2)).astype(int))
        for k in ks:
            if scen == "decode_heavy":
                tp = 5.5 + 1.2 * np.exp(-((np.log2(k) - 4) ** 2) / 3) + rng.normal(0, 0.08)
                tpot = 95 + 15 * abs(np.log2(k) - 4) + rng.normal(0, 2)
            elif scen == "balanced":
                tp = 8.0 + 1.0 * np.exp(-((np.log2(k) - 5) ** 2) / 4) + rng.normal(0, 0.1)
                tpot = 80 + 12 * abs(np.log2(k) - 5) + rng.normal(0, 2)
            else:
                tp = 12.0 + 0.8 * np.exp(-((np.log2(k) - 6) ** 2) / 4) + rng.normal(0, 0.1)
                tpot = 50 + 9 * abs(np.log2(k) - 6) + rng.normal(0, 1)
            rows.append({
                "scenario": scen, "scheduler": f"eb_fixed_k_{k}",
                "k_hat_final": float(k),
                "tp_real": float(tp), "mean_tpot_ms": float(tpot),
                "p99_tpot_ms": float(tpot * 1.4),
            })
        # v1 baseline
        rows.append({
            "scenario": scen, "scheduler": "v1",
            "k_hat_final": float("nan"),
            "tp_real": 5.5 if scen == "decode_heavy" else 7.5 if scen == "balanced" else 11.5,
            "mean_tpot_ms": 110 if scen == "decode_heavy" else 95 if scen == "balanced" else 60,
            "p99_tpot_ms": 160 if scen == "decode_heavy" else 130 if scen == "balanced" else 80,
        })
        # EB(k_hat) — adaptive
        rows.append({
            "scenario": scen, "scheduler": "eb",
            "k_hat_final": 16.0 if scen == "decode_heavy" else 32.0 if scen == "balanced" else 64.0,
            "tp_real": (6.7 if scen == "decode_heavy"
                        else 9.0 if scen == "balanced" else 12.6),
            "mean_tpot_ms": 92 if scen == "decode_heavy" else 80 if scen == "balanced" else 52,
            "p99_tpot_ms": 130 if scen == "decode_heavy" else 110 if scen == "balanced" else 72,
        })
    return rows


def plot(rows: list[dict], output: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex="col")
    for i, scen in enumerate(SCENARIOS):
        for j, (mkey, mlabel, higher_better) in enumerate(METRICS):
            ax = axes[i][j]
            # Fixed-k sweep
            sweep = [r for r in rows if r["scenario"] == scen and
                     str(r["scheduler"]).startswith("eb_fixed_k_")]
            sweep.sort(key=lambda r: r["k_hat_final"])
            if sweep:
                xs = [r["k_hat_final"] for r in sweep]
                ys = [r.get(mkey, np.nan) for r in sweep]
                ax.plot(xs, ys, "o-", color="tab:green", label="Fixed-k sweep")
                # Annotate the best
                best_idx = int(np.nanargmax(ys) if higher_better else np.nanargmin(ys))
                ax.annotate(f"k*={int(xs[best_idx])}",
                            xy=(xs[best_idx], ys[best_idx]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
            # v1 (horizontal red line)
            v1_rows = [r for r in rows if r["scenario"] == scen and r["scheduler"] == "v1"]
            if v1_rows:
                v1_val = v1_rows[0].get(mkey, np.nan)
                ax.axhline(v1_val, color="tab:red", linestyle="-", linewidth=1.5,
                           label="v1 (mixed)")
            # EB(k̂*)
            eb_rows = [r for r in rows if r["scenario"] == scen and r["scheduler"] == "eb"]
            if eb_rows:
                eb_val = eb_rows[0].get(mkey, np.nan)
                ax.axhline(eb_val, color="tab:blue", linestyle="--", linewidth=1.5,
                           label="EB(k̂*)")
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(mlabel)
            if j == 0:
                ax.set_ylabel(SCENARIO_LABELS[scen])
            if i == 2:
                ax.set_xlabel("Fixed k")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Controller validation: fixed-k sweep vs EB(k̂*) {title_suffix}",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?",
                   help="validation_summary.csv (from analyze_validation.py)")
    p.add_argument("--demo", action="store_true",
                   help="Render with synthetic data (no CSV needed)")
    p.add_argument("--output", default="validation_grid.pdf")
    args = p.parse_args()

    if args.demo or not args.csv or not Path(args.csv).exists():
        rows = demo_rows()
        suffix = "(DEMO data — replace with validation_summary.csv from analyze_validation.py)"
    else:
        rows = load_csv(Path(args.csv))
        suffix = f"({Path(args.csv).parent.name})"
    plot(rows, Path(args.output), suffix)


if __name__ == "__main__":
    main()
