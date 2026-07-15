"""
Plot subset selection comparison results.

Loads all subset_selection_{dataset}_k100.json files from the results/ folder,
aggregates BAI identification rates across datasets, and produces a mean
error-rate vs budget figure.

Usage:
    python plot_subset_selection.py [--results-dir PATH] [--out-dir PATH]

Attribution:
  - IRT coreset methods are adapted from the tinyBenchmarks repository:
    https://github.com/felipemaiapolo/tinyBenchmarks/tree/9c7e20302301ad531bfdfd9a7288e6e916bf22e9/tutorials
  - AnchorPoints method is adapted from the AnchorPoints repository:
    https://github.com/rvivek3/AnchorPoints/blob/64b6087d11176cc707ebeabfd6a5b13f8a1cfaf2/optimal_valset_validation.py
  - MetaBench method is adapted from:
    https://github.com/socialfoundations/benchmark-prediction/blob/release/benchpred/metabench.py
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ─── paths ────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent

# ─── display config ───────────────────────────────────────────────────────────
METHOD_DISPLAY = {
    "anchor":   "AnchorPt Wt.Score",
    "correct":  "Correctness Coresets",
    "irt_d2":   "IRT Coresets (D=2)",
    "irt_d5":   "IRT Coresets (D=5)",
    "irt_d10":  "IRT Coresets (D=10)",
    "irt_d15":  "IRT Coresets (D=15)",
    "smart_sr":  "SySRs",
    "metabench": "MetaBench",
}


def _irt_style(color, marker, ls="--", lw=2.0, z=7):
    return {"color": color, "marker": marker, "markersize": 7,
            "linestyle": ls, "linewidth": lw, "alpha": 0.85, "zorder": z}


METHOD_STYLES = {
    "AnchorPt Wt.Score": {
        "color": "#1E90FF", "marker": "o", "markersize": 7,
        "linestyle": "-", "linewidth": 2.5, "alpha": 0.9, "zorder": 8,
    },
    "Correctness Coresets": {
        "color": "#87CEEB", "marker": "o", "markersize": 7,
        "linestyle": "--", "linewidth": 2.0, "alpha": 0.85, "zorder": 7,
    },
    "IRT Coresets (D=2)":  _irt_style("#98FB98", "v"),
    "IRT Coresets (D=5)":  _irt_style("#228B22", "s", lw=2.5, z=8),
    "IRT Coresets (D=10)": _irt_style("#006400", "p", ls="-."),
    "IRT Coresets (D=15)": _irt_style("#556B2F", "h", ls=":"),
    "SySRs": {
        "color": "#DC143C", "marker": "*", "markersize": 9,
        "linestyle": "-", "linewidth": 3.5, "alpha": 1.0, "zorder": 10,
    },
    "MetaBench": {
        "color": "#FF8C00", "marker": "D", "markersize": 7,
        "linestyle": "-", "linewidth": 2.5, "alpha": 0.95, "zorder": 9,
    },
}

LEGEND_ORDER = [
    "SySRs",
    "MetaBench",
    "AnchorPt Wt.Score",
    "Correctness Coresets",
    "IRT Coresets (D=2)",
    "IRT Coresets (D=5)",
    "IRT Coresets (D=10)",
    "IRT Coresets (D=15)",
]


# ─── data loading ─────────────────────────────────────────────────────────────
def load_all_results(results_dir: Path):
    """Return (data, datasets) where data: method -> budget_pct -> [per-dataset means]."""
    pattern = str(results_dir / "subset_selection_*_k*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files found matching: {pattern}")

    data     = defaultdict(lambda: defaultdict(list))
    datasets = []

    for fpath in files:
        name    = Path(fpath).stem
        # extract dataset name from "subset_selection_{dataset}_k{K}"
        parts   = name.split("_k")
        ds_part = parts[0].replace("subset_selection_", "", 1)
        datasets.append(ds_part)
        with open(fpath) as f:
            d = json.load(f)
        for row in d["comparison_df"]:
            method     = row["Method"]
            budget_pct = round(float(row["Budget%"]))
            mean_rate  = float(row["Mean BAI Rate"])
            data[method][budget_pct].append(mean_rate)

    print(f"Loaded {len(files)} dataset(s): {datasets}")
    return data, datasets


def aggregate(data):
    """Compute mean ± std across datasets per method/budget."""
    agg = {}
    for method, budget_dict in data.items():
        series = []
        for budget_pct, rates in sorted(budget_dict.items()):
            series.append((budget_pct, np.mean(rates), np.std(rates)))
        agg[method] = series
    return agg


# ─── summary table ────────────────────────────────────────────────────────────
def print_summary(agg):
    all_budgets     = sorted({b for series in agg.values() for b, _, _ in series})
    methods_display = [(m, METHOD_DISPLAY.get(m, m)) for m in agg]

    header = f"{'Method':<25}" + "".join(f"  {b:>5}%" for b in all_budgets)
    print(f"\n{'='*len(header)}")
    print("Mean BAI Identification Rate (averaged across datasets)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for method, display in sorted(methods_display, key=lambda x: x[1]):
        row_dict = {b: m for b, m, _ in agg[method]}
        row = f"{display:<25}" + "".join(
            f"  {row_dict.get(b, float('nan')):>6.3f}" for b in all_budgets
        )
        print(row)

    print(f"\n{'='*len(header)}")
    print("Mean BAI Error Rate (1 - identification rate)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for method, display in sorted(methods_display, key=lambda x: x[1]):
        row_dict = {b: m for b, m, _ in agg[method]}
        row = f"{display:<25}" + "".join(
            f"  {1.0 - row_dict.get(b, float('nan')):>6.3f}" for b in all_budgets
        )
        print(row)


# ─── plotting ─────────────────────────────────────────────────────────────────
def plot_figure(agg, datasets, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    handles = {}

    for method, series in agg.items():
        if method not in METHOD_DISPLAY:
            continue
        display = METHOD_DISPLAY[method]
        style = METHOD_STYLES.get(display, {
            "color": "gray", "marker": "o", "markersize": 6,
            "linestyle": "-", "linewidth": 1.8, "alpha": 0.7, "zorder": 5,
        })
        xs = [b for b, _, _ in series]
        ys = [1.0 - m for _, m, _ in series]   # error rate

        line, = ax.plot(
            xs, ys,
            marker    = style["marker"],
            markersize = style["markersize"],
            linewidth = style["linewidth"],
            linestyle = style["linestyle"],
            color     = style["color"],
            alpha     = style["alpha"],
            zorder    = style["zorder"],
            label     = display,
        )
        handles[display] = line

    n_ds = len(datasets)
    ax.set_xlabel("Percentage of Full Evaluation Budget", fontsize=14, fontweight="bold")
    ax.set_ylabel("Identification Error Rate",            fontsize=14, fontweight="bold")
    ax.set_title(
        f"Mean Identification Error Rate Across All Datasets"
        + (f" ({n_ds} datasets)" if n_ds > 1 else f" ({datasets[0]})"),
        fontsize=16, fontweight="bold",
    )
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ordered = [(n, handles[n]) for n in LEGEND_ORDER if n in handles]
    extra   = [(n, h) for n, h in handles.items() if n not in LEGEND_ORDER]
    ordered += extra
    ax.legend(
        [h for _, h in ordered], [n for n, _ in ordered],
        loc="upper right", fontsize=10,
        framealpha=0.9, edgecolor="black",
        borderaxespad=0.5, handlelength=2.0,
        prop={"weight": "bold"},
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── entry point ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Plot subset selection comparison results.')
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Directory containing result JSON files (default: <script_dir>/results).'
    )
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory for figures (default: <results_dir>/plots).'
    )
    return parser.parse_args()


def main():
    args        = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / 'results'
    out_dir     = Path(args.out_dir)     if args.out_dir     else results_dir / 'plots'

    data, datasets = load_all_results(results_dir)
    agg            = aggregate(data)

    print_summary(agg)

    out_path = out_dir / 'subset_selection_bai_error_rate.png'
    plot_figure(agg, datasets, out_path)


if __name__ == '__main__':
    main()
