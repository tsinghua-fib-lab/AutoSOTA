import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Local project imports (same folder as this script) ---
sys.path.append(os.path.dirname(__file__))

from vis_utilities import (
    setup_nyt_style,
    apply_nyt_style_to_axes,
    color_cycle_for_keys,
    parse_attr_paths,
    choose_output_alias,
    filter_latest_model_evals,
    get_attribute_display_name,
)

try:
    from bias_visualization_dashboard import SimplifiedBiasDataLoader
except Exception as e:
    print("ERROR: Could not import SimplifiedBiasDataLoader from bias_visualization_dashboard.")
    print("Make sure this file is placed alongside your project modules. Original error:", e)
    sys.exit(1)


# ---------------------------
# Computation
# ---------------------------
def compute_relevance_std_by_question(df: pd.DataFrame, min_answers: int = 2) -> pd.DataFrame:
    """
    Compute per-question std of relevance across models.
    Returns columns: question_id, relevance_mean, relevance_std, n_models
    Only keeps questions with n_models >= min_answers.
    """
    required = {"question_id", "relevance_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    g = (
        df.groupby("question_id")["relevance_score"]
        .agg(relevance_mean="mean", relevance_std="std", n_models="count")
        .reset_index()
    )
    g = g[g["n_models"] >= int(min_answers)].copy()
    return g


def load_attr_std_series(attr: str, run_path: str, min_answers: int) -> pd.DataFrame:
    """
    Load one attribute/run, filter to latest evals, compute per-question stds.
    Returns the per-question stats DataFrame.
    """
    loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=[attr])
    data = loader.load_data()
    df = data.conversations_df

    if df is None or df.empty:
        print(f"[{attr}] No data at {run_path}")
        return pd.DataFrame()

    df = filter_latest_model_evals(df)

    if "relevance_score" not in df.columns:
        print(f"[{attr}] Missing 'relevance_score'; skipping.")
        return pd.DataFrame()

    stats = compute_relevance_std_by_question(df, min_answers=min_answers)
    if stats.empty:
        print(f"[{attr}] No questions with at least {min_answers} answers; skipping.")
    return stats


# ---------------------------
# Plotting
# ---------------------------
def _compute_common_bins(
    std_dict: dict[str, np.ndarray], bins: int, range_tuple: tuple[float, float] | None
):
    """Compute shared bin edges across attributes."""
    if not std_dict:
        return np.linspace(0, 1, bins + 1)

    if range_tuple is None:
        vmin = min(float(np.nanmin(v)) for v in std_dict.values() if len(v) > 0)
        vmax = max(float(np.nanmax(v)) for v in std_dict.values() if len(v) > 0)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            # Expand slightly to make a non-degenerate histogram
            eps = max(1e-6, 0.05 * max(1.0, abs(vmax)))
            vmin -= eps
            vmax += eps
    else:
        vmin, vmax = range_tuple

    return np.linspace(vmin, vmax, bins + 1)


def plot_overlay_hist(
    std_dict: dict[str, np.ndarray],
    out_path: Path,
    bins: int,
    range_tuple,
    density: bool,
    logy: bool,
):
    """Overlayed histogram for multiple attributes (shared bins)."""
    if not std_dict:
        print("Nothing to plot (no std data).")
        return

    colors = color_cycle_for_keys(list(std_dict.keys()))
    bin_edges = _compute_common_bins(std_dict, bins, range_tuple)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.set_facecolor("white")

    # Draw each attribute histogram (semi-transparent fill, crisp edge)
    handles = []
    labels = []
    for attr, arr in std_dict.items():
        if arr.size == 0:
            continue
        counts, _ = np.histogram(arr, bins=bin_edges, density=density)
        # stepfilled look by plotting as bar with shared bins
        bar = ax.bar(
            bin_edges[:-1],
            counts,
            align="edge",
            width=np.diff(bin_edges),
            color=colors[attr],
            alpha=0.35,
            edgecolor=colors[attr],
            linewidth=1.0,
            label=get_attribute_display_name(attr),
        )
        display_name = get_attribute_display_name(attr)
        if display_name not in labels:
            handles.append(bar[0])
            labels.append(display_name)

    # Axes labels (no title), horizontal x labels
    ax.set_xlabel("Std. dev. of relevance", fontsize=14, fontweight="bold", fontfamily="sans-serif")
    ax.set_ylabel(
        "Density" if density else "Count", fontsize=14, fontweight="bold", fontfamily="sans-serif"
    )
    ax.set_xticklabels(
        [tick.get_text() for tick in ax.get_xticklabels()], ha="center", fontfamily="sans-serif"
    )
    if logy:
        ax.set_yscale("log")

    # NYT-ish axes styling + horizontal legend below, no frame
    apply_nyt_style_to_axes(ax)
    ax.legend(
        handles,
        labels,
        title=None,
        frameon=False,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=max(1, len(labels)),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved overlay histogram: {out_path}")


def plot_facet_hist(
    std_dict: dict[str, np.ndarray],
    out_path: Path,
    bins: int,
    range_tuple,
    density: bool,
    logy: bool,
):
    """Faceted small-multiples grid (shared bins)."""
    if not std_dict:
        print("Nothing to plot (no std data).")
        return

    attrs = list(std_dict.keys())
    colors = color_cycle_for_keys(attrs)
    bin_edges = _compute_common_bins(std_dict, bins, range_tuple)

    n = len(attrs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, attr in enumerate(attrs):
        ax = axes[idx]
        ax.set_facecolor("white")
        arr = std_dict[attr]
        if arr.size > 0:
            counts, _ = np.histogram(arr, bins=bin_edges, density=density)
            ax.bar(
                bin_edges[:-1],
                counts,
                align="edge",
                width=np.diff(bin_edges),
                color=colors[attr],
                alpha=0.35,
                edgecolor=colors[attr],
                linewidth=1.0,
            )

        # Subtle in-plot label (no title header)
        ax.text(
            0.02,
            0.95,
            attr,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontfamily="sans-serif",
        )

        # Axes labels
        if idx // ncols == nrows - 1:
            ax.set_xlabel(
                "Std. dev. of relevance", fontsize=12, fontfamily="sans-serif", fontweight="bold"
            )
        if idx % ncols == 0:
            ax.set_ylabel(
                "Density" if density else "Count",
                fontsize=12,
                fontfamily="sans-serif",
                fontweight="bold",
            )

        if logy:
            ax.set_yscale("log")

        apply_nyt_style_to_axes(ax)

    # Hide any empty panels
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved facet histogram grid: {out_path}")


# ---------------------------
# CLI / main
# ---------------------------
def _parse_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y", "on"}


def _parse_range(s: str | None):
    if not s:
        return None
    try:
        lo, hi = s.split(":")
        return float(lo), float(hi)
    except Exception:
        raise ValueError("Range must be formatted as 'min:max' (e.g., '0:2').")


def main():
    setup_nyt_style()

    parser = argparse.ArgumentParser(
        description="Histogram of per-question std dev of relevance across models."
    )
    parser.add_argument(
        "--attr_paths",
        type=str,
        required=True,
        help="Comma-separated 'attribute:/path' pairs. Example: 'gender:/runs/gender, race:/runs/race'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Base directory for outputs (default: plots)",
    )
    parser.add_argument(
        "--bins", type=int, default=25, help="Number of histogram bins (shared across attributes)."
    )
    parser.add_argument(
        "--range",
        dest="range_str",
        type=str,
        default=None,
        help="Optional histogram range as 'min:max' (e.g., '0:2'). If omitted, uses global min/max.",
    )
    parser.add_argument(
        "--min_answers",
        type=int,
        default=2,
        help="Minimum answers per question to compute std (default: 2).",
    )
    parser.add_argument(
        "--mode",
        choices=["overlay", "facet"],
        default="overlay",
        help="Histogram layout: overlay all attributes in one axes, or facet into a grid.",
    )
    parser.add_argument(
        "--density",
        type=_parse_bool,
        default=False,
        help="Normalize histograms to density instead of counts (true/false).",
    )
    parser.add_argument(
        "--logy",
        type=_parse_bool,
        default=False,
        help="Use logarithmic scale on the Y axis (true/false).",
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Optional filename suffix for saved plots."
    )
    args = parser.parse_args()

    try:
        attr_paths = parse_attr_paths(args.attr_paths)
    except Exception as e:
        print(f"Failed to parse --attr_paths: {e}")
        sys.exit(1)

    if not attr_paths:
        print("No attribute paths provided.")
        sys.exit(1)

    alias = choose_output_alias(attr_paths)
    out_dir = Path(args.output_dir) / alias
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect std arrays per attribute + save per-attribute CSVs
    std_dict: dict[str, np.ndarray] = {}
    long_rows = []

    for attr, path in attr_paths:
        if not os.path.exists(path):
            print(f"[{attr}] WARNING: path does not exist → {path}")
            continue

        stats = load_attr_std_series(attr, path, min_answers=args.min_answers)
        if stats is None or stats.empty:
            continue

        # Record std series
        arr = stats["relevance_std"].astype(float).to_numpy()
        std_dict[attr] = arr

        # Save per-attribute CSV (per question)
        per_attr_csv = out_dir / f"relevance_std_by_question_{attr}.csv"
        stats.to_csv(per_attr_csv, index=False)
        print(f"[{attr}] Saved per-question stds → {per_attr_csv}")

        # Add to long form for a unified CSV
        for _, r in stats.iterrows():
            long_rows.append(
                {
                    "attribute": attr,
                    "question_id": r["question_id"],
                    "relevance_std": float(r["relevance_std"]),
                    "relevance_mean": float(r["relevance_mean"]),
                    "n_models": int(r["n_models"]),
                }
            )

        # Console summary
        q = np.quantile(arr, [0.0, 0.25, 0.5, 0.9, 0.95, 1.0])
        print(
            f"[{attr}] std(relevance) – min={q[0]:.3f}, p25={q[1]:.3f}, median={q[2]:.3f}, "
            f"p90={q[3]:.3f}, p95={q[4]:.3f}, max={q[5]:.3f}  (n_q={len(arr)})"
        )

    if not std_dict:
        print("No std data collected; nothing to plot.")
        sys.exit(0)

    # Save long CSV
    long_df = pd.DataFrame(long_rows)
    long_csv = out_dir / "relevance_std_all_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"Saved long CSV of all stds → {long_csv}")

    # Plot
    range_tuple = _parse_range(args.range_str)
    if args.mode == "overlay":
        plot_path = out_dir / f"relevance_std_hist_overlay{args.suffix}.png"
        plot_overlay_hist(
            std_dict,
            plot_path,
            bins=args.bins,
            range_tuple=range_tuple,
            density=args.density,
            logy=args.logy,
        )
    else:
        plot_path = out_dir / f"relevance_std_hist_facet{args.suffix}.png"
        plot_facet_hist(
            std_dict,
            plot_path,
            bins=args.bins,
            range_tuple=range_tuple,
            density=args.density,
            logy=args.logy,
        )


if __name__ == "__main__":
    main()
