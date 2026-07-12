"""Plot the synthetic-workload e2e figure (§4.3.1) from analyze_e2e.py output.

Reads `optimal_per_scheduler.csv` (one row per (scheduler, scenario) at the
best-throughput (B, N)) and produces a 3-metric bar figure:
    Throughput (RPS) | TTFT mean (s) | TPOT mean (ms)
with 3 scenarios on the x-axis (decode-heavy / balanced / prefill-heavy).

Stars annotate the best scheduler per (scenario, metric).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = [
    ("decode_heavy",  "Decode-heavy\n(128/1024)"),
    ("balanced",      "Balanced\n(512/512)"),
    ("prefill_heavy", "Prefill-heavy\n(1024/128)"),
]

# Schedulers we expect to find in the CSV. v0 is optional (only present if
# its grid search was actually run).
SCHEDULERS = [("v0", "v0"), ("v1", "v1"), ("eb", "EB(k̂*)")]
COLORS = {"v0": "#7f7f7f", "v1": "#d62728", "eb": "#1f77b4"}

METRICS = [
    # (column, label, higher_is_better)
    ("request_throughput", "Throughput (RPS)", True),
    ("mean_ttft_ms",       "TTFT mean (ms)",   False),
    ("mean_tpot_ms",       "TPOT mean (ms)",   False),
]


def load_optimal(csv_path: Path) -> dict:
    """Return {(scenario, scheduler): row_dict}."""
    out: dict = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            scen = r["scenario"].strip("'\"")
            sched = r["scheduler"].strip("'\"")
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            out[(scen, sched)] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid-search-dir",
        default="outputs/e2e_grid_search",
        help="Output dir from analyze_e2e.py "
             "(contains <GPU>_<MODEL>/optimal_per_scheduler.csv)",
    )
    p.add_argument("--output", default="fig_synthetic_e2e.pdf")
    args = p.parse_args()

    base = Path(args.grid_search_dir)
    if not base.exists():
        raise SystemExit(
            f"Grid dir not found: {base}\n"
            "Generate it first with:\n"
            "  ./run_grid_search.sh\n"
            "  python analyze_e2e.py outputs/e2e_grid_search/<GPU>_<MODEL>"
        )

    # Auto-pick the <GPU>_<MODEL>/ subdir if user passed the parent dir.
    csv_path = base / "optimal_per_scheduler.csv"
    if not csv_path.exists():
        candidates = list(base.glob("*/optimal_per_scheduler.csv"))
        if not candidates:
            raise SystemExit(f"No optimal_per_scheduler.csv under {base}")
        csv_path = candidates[0]

    data = load_optimal(csv_path)
    present_schedulers = [(s, label) for s, label in SCHEDULERS
                          if any((scen, s) in data for scen, _ in SCENARIOS)]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
    x = np.arange(len(SCENARIOS))
    n_sched = len(present_schedulers)
    width = 0.8 / n_sched

    for j, (mkey, mlabel, higher_better) in enumerate(METRICS):
        ax = axes[j]
        for i, (sched, sched_label) in enumerate(present_schedulers):
            vals = []
            for scen_key, _ in SCENARIOS:
                row = data.get((scen_key, sched), {})
                vals.append(row.get(mkey, np.nan))
            offset = (i - (n_sched - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=sched_label,
                   color=COLORS.get(sched, f"C{i}"),
                   edgecolor="black", linewidth=0.4)

        # Star the best scheduler per scenario
        for k, (scen_key, _) in enumerate(SCENARIOS):
            cell_vals = [data.get((scen_key, s), {}).get(mkey, np.nan)
                         for s, _ in present_schedulers]
            if any(np.isfinite(v) for v in cell_vals):
                if higher_better:
                    best_i = int(np.nanargmax(cell_vals))
                else:
                    best_i = int(np.nanargmin(cell_vals))
                best_val = cell_vals[best_i]
                offset = (best_i - (n_sched - 1) / 2) * width
                ax.text(k + offset, best_val * 1.03, "★",
                        ha="center", va="bottom", fontsize=12)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in SCENARIOS])
        ax.set_ylabel(mlabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"§4.3.1 Synthetic workloads, H200 — best (B, N) per scheduler\n"
        f"(source: {csv_path.parent.name})",
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(args.output)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    # Also print a summary
    print()
    print(f"{'Scenario':<18}", *[f"{s:>14}" for s, _ in present_schedulers])
    for mkey, mlabel, _ in METRICS:
        print(f"  {mlabel}:")
        for scen_key, scen_label in SCENARIOS:
            row = [f"{scen_key:<18}"]
            for sched, _ in present_schedulers:
                v = data.get((scen_key, sched), {}).get(mkey, float("nan"))
                row.append(f"{v:>14.2f}")
            print("    " + "".join(row))


if __name__ == "__main__":
    main()
