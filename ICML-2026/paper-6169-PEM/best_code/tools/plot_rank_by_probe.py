#!/usr/bin/env python3
"""
Generate the per-function mean-rank figure sorted by probe statistic P.

Two panels separated by a broken axis: low-misranking (left), high-misranking (right).
Five methods: RB-PEM, Probe-and-Switch, CMA-ES, LRA-CMA-ES, UH-CMA-ES.

Output: evidence/paper_figures/figure_rank_by_probe.pdf (and .png)

Usage:
    cd "Supplementary Material"
    python3 tools/plot_rank_by_probe.py
"""

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

# Ensure tools/ is on path for project imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plot_style import apply_style, get_algo_color, add_grid, save_figure, LINEWIDTH_SECONDARY

import matplotlib.pyplot as plt
import numpy as np

apply_style("science")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
def _latest_summary(sub):
    """Most recent timestamped all-budgets summary under Results/<sub>/, else the base path."""
    base = os.path.join(BASE_DIR, "Results", sub)
    cands = sorted(glob.glob(os.path.join(base, "*", "bbob_summary_all_budgets.csv")))
    return cands[-1] if cands else os.path.join(base, "bbob_summary_all_budgets.csv")


LOGW_CSV = os.path.join(BASE_DIR, "Results", "log_weight_ablation", "bbob_summary_all_budgets.csv")
LRA_CSV = _latest_summary("lra_cmaes")
UH_CSV = _latest_summary("uh_cmaes")

# ---------------------------------------------------------------------------
# High-misranking function IDs
# ---------------------------------------------------------------------------
HIGH_MR = {108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129}

# Probe statistic values (median P, d=40)
PROBE_P = {
    123: 0.349, 114: 0.348, 111: 0.346, 129: 0.337, 126: 0.335,
    128: 0.333, 120: 0.331, 117: 0.328, 108: 0.326, 110: 0.321,
    116: 0.314, 122: 0.311, 107: 0.309, 125: 0.309, 119: 0.301,
    113: 0.298, 127: 0.153, 124: 0.105, 130: 0.078, 102: 0.042,
    101: 0.025, 121: 0.025, 109: 0.019, 105: 0.017, 104: 0.009,
    115: 0.002, 103: 0.000, 106: 0.000, 112: 0.000, 118: 0.000,
}

# ---------------------------------------------------------------------------
# Algorithm config — display order determines legend order
# ---------------------------------------------------------------------------
ALGO_ORDER = [
    ("BERW-Hetero",            "RB-PEM"),
    ("ProbeSwitch-MR(t=0.12)", "Probe-and-Switch"),
    ("CMA-ES-sep",             "CMA-ES"),
    ("LRA-CMA-ES",             "LRA-CMA-ES"),
    ("UH-CMA-ES(maxevals=30)", "UH-CMA-ES"),
]

# LRA-CMA-ES is not in the project's ALGO_COLORS; assign it here
_EXTRA_COLORS = {"LRA-CMA-ES": "#EE7733"}

MARKERS = {
    "RB-PEM": "o",
    "Probe-and-Switch": "s",
    "CMA-ES": "^",
    "LRA-CMA-ES": "D",
    "UH-CMA-ES": "v",
}

BUDGET = 100  # B=100d


# ---------------------------------------------------------------------------
# Data loading & ranking
# ---------------------------------------------------------------------------
def load_csv(path):
    algo_keys = {k for k, _ in ALGO_ORDER}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["algorithm"] not in algo_keys:
                continue
            rows.append({
                "algorithm": row["algorithm"],
                "function": int(row["function"]),
                "dimension": int(row["dimension"]),
                "instance": int(row["instance"]),
                "budget_multiplier": int(row["budget_multiplier"]),
                "best_f": float(row["best_f"]),
            })
    return rows


def compute_mean_rank_by_function(rows, budget):
    filtered = [r for r in rows if r["budget_multiplier"] == budget]
    grouped = defaultdict(list)
    for r in filtered:
        key = (r["function"], r["dimension"], r["instance"])
        grouped[key].append(r)

    algo_func_ranks = defaultdict(lambda: defaultdict(list))
    for key, items in grouped.items():
        func = key[0]
        items_sorted = sorted(items, key=lambda x: x["best_f"])
        for rank, item in enumerate(items_sorted, start=1):
            algo_func_ranks[item["algorithm"]][func].append(rank)

    result = {}
    for algo, func_ranks in algo_func_ranks.items():
        result[algo] = {f: np.mean(r) for f, r in func_ranks.items()}
    return result


# ---------------------------------------------------------------------------
# Broken-axis diagonal marks
# ---------------------------------------------------------------------------
def draw_break_marks(ax, side, size=0.015):
    """Draw diagonal break marks on the left or right edge of an axes."""
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.6)
    if side == "right":
        x = 1.0
    else:
        x = 0.0
    ax.plot((x - size, x + size), (-size, +size), **kwargs)
    ax.plot((x - size, x + size), (1 - size, 1 + size), **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    missing = [p for p in (LOGW_CSV, LRA_CSV, UH_CSV) if not os.path.exists(p)]
    if missing:
        print("Missing input data. Generate it first with:")
        print("  bash tools/run_log_weight_ablation.sh && python3 tools/merge_log_weight_results.py")
        print("  bash tools/run_lra_cmaes_baseline.sh")
        print("  bash tools/run_uh_cmaes_baseline.sh")
        for p in missing:
            print(f"  (missing: {os.path.relpath(p, BASE_DIR)})")
        sys.exit(1)

    rows = load_csv(LOGW_CSV) + load_csv(LRA_CSV) + load_csv(UH_CSV)
    print(f"Loaded {len(rows)} rows")

    mean_ranks = compute_mean_rank_by_function(rows, BUDGET)

    # Build display-name keyed ranks
    display_ranks = {}
    for raw_key, display_name in ALGO_ORDER:
        if raw_key in mean_ranks:
            display_ranks[display_name] = mean_ranks[raw_key]

    # Split functions into low/high clusters
    low_funcs = sorted(
        [f for f in PROBE_P if f not in HIGH_MR and f != 107],
        key=lambda f: PROBE_P[f],
    )
    high_funcs = sorted(
        [f for f in PROBE_P if f in HIGH_MR],
        key=lambda f: PROBE_P[f],
    )

    n_low, n_high = len(low_funcs), len(high_funcs)

    # --- Figure setup: broken axis ---
    fig, (ax_lo, ax_hi) = plt.subplots(
        1, 2, sharey=True,
        figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [n_low, n_high], "wspace": 0.06},
    )

    for ax, funcs, title in [(ax_lo, low_funcs, "Low misranking"),
                              (ax_hi, high_funcs, "High misranking")]:
        x = np.arange(len(funcs))
        for _, display_name in ALGO_ORDER:
            if display_name not in display_ranks:
                continue
            ys = [display_ranks[display_name].get(f, np.nan) for f in funcs]
            color = _EXTRA_COLORS.get(display_name, None)
            if color is None:
                # Look up by raw key
                raw_key = [k for k, d in ALGO_ORDER if d == display_name][0]
                color = get_algo_color(raw_key)
            ax.plot(
                x, ys,
                marker=MARKERS.get(display_name, "o"),
                markersize=1.8,
                linewidth=LINEWIDTH_SECONDARY,
                label=display_name,
                color=color,
                alpha=0.85,
            )

        # X-axis: function name (black) + probe value (grey, smaller)
        ax.set_xticks(x)
        ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=5.5)
        # Add probe P values as a second row of labels in grey
        for i, f in enumerate(funcs):
            ax.annotate(
                f"{PROBE_P[f]:.2f}",
                xy=(i, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=4, color="#888888",
            )
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(axis="both", length=0, which="both")
        add_grid(ax, axis="y")

    # Y-axis: rank 1 at top, 5 at bottom
    ax_lo.set_ylim(5.5, 0.5)
    ax_lo.set_ylabel("Mean rank ($1$ = best)", fontsize=8)

    # Break marks between panels
    ax_lo.spines["right"].set_visible(False)
    ax_hi.spines["left"].set_visible(False)
    ax_hi.tick_params(left=False)
    draw_break_marks(ax_lo, "right")
    draw_break_marks(ax_hi, "left")

    # Legend: upper-right of right panel, our methods first (already ordered)
    ax_hi.legend(
        fontsize=6, loc="upper right",
        framealpha=0.9, edgecolor="#cccccc",
        handlelength=1.8, handletextpad=0.4, borderpad=0.3,
    )

    # Shared x-label
    fig.text(0.5, -0.01, "Function (sorted by probe statistic $P$)", ha="center", fontsize=8)

    # --- Save ---
    out_dir = os.path.join(BASE_DIR, "evidence", "paper_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figure_rank_by_probe")
    saved = save_figure(fig, out_path, formats=["pdf", "png"], dpi=300)
    plt.close()
    for p in saved:
        print(f"Wrote: {os.path.relpath(p, BASE_DIR)}")


if __name__ == "__main__":
    main()
