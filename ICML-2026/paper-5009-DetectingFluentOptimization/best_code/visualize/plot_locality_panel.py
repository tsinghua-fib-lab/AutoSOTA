#!/usr/bin/env python3
"""
Render an ICML-friendly 2-panel locality figure (overall-F1 vs FPR@X) from a cached
detection_counts.csv.

This script is plots-only: it does NOT rerun any model or detection pipeline stages.

Example:
  python visualize/plot_locality_panel.py \
    --counts-csv results/<run_tag>/detection_counts.csv \
    --out-pdf results/<run_tag>/locality_panel_overallf1_fpr10.pdf \
    --windows 5 20 \
    --metric-right fpr10
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

from matplotlib.patches import Patch  # noqa: E402

METHOD_LABELS: Dict[str, str] = {
    "cpd_online": "CPD Online",
}


def _method_label(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if method.startswith("window_pp_w"):
        try:
            w = int(method.split("window_pp_w", 1)[1])
            return f"WPP{w}"
        except Exception:
            return method
    return method


def _wrap_xtick_label(label: str) -> str:
    """
    Make method tick labels compact to avoid overlap at single-column figure widths.
    Examples:
      "CPD Online"   -> "CPD\\nOnline"
      "WPP (w=5)"    -> "WPP\\n(w=5)"
    """
    text = (label or "").strip()
    if not text or "\n" in text:
        return text
    if " (" in text:
        base, rest = text.split(" ", 1)
        return f"{base}\n{rest}"
    if " " in text:
        a, b = text.split(" ", 1)
        return f"{a}\n{b}"
    return text


CATEGORIES: List[Tuple[str, str]] = [
    ("before_suffix", "Before"),
    ("before_in_suffix", "Before+In"),
    ("in_suffix", "In-suffix"),
    ("in_benign", "In-benign"),
]

def _get_category_colors() -> Dict[str, str]:
    """
    Colors encode locality (stack segments), matching the intent of the panel figure.
    """
    if sns is None:
        # Fallback (seaborn-muted-ish).
        return {
            "before_suffix": "#4c72b0",      # blue
            "before_in_suffix": "#dd8452",   # orange
            "in_suffix": "#55a868",          # green
            "in_benign": "#c44e52",          # red
        }
    palette = sns.color_palette("muted", n_colors=4)
    return {
        "before_suffix": matplotlib.colors.to_hex(palette[0]),      # type: ignore[attr-defined]
        "before_in_suffix": matplotlib.colors.to_hex(palette[1]),   # type: ignore[attr-defined]
        "in_suffix": matplotlib.colors.to_hex(palette[2]),          # type: ignore[attr-defined]
        "in_benign": matplotlib.colors.to_hex(palette[3]),          # type: ignore[attr-defined]
    }


def _validate_counts_df(df: pd.DataFrame) -> None:
    required = {"method", "segment", "metric", "before_suffix", "before_in_suffix", "in_suffix", "in_benign"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"detection_counts.csv missing required columns: {missing}")


def _extract_metric_matrix(
    df: pd.DataFrame,
    *,
    metric: str,
    methods: List[str],
) -> Tuple[List[str], np.ndarray]:
    """
    Returns (method_keys, matrix) where matrix shape is (n_methods, n_categories)
    with percentages that sum to 100 per method (row).
    """
    sub = df[(df["segment"] == "overall") & (df["metric"] == metric)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for segment='overall' and metric='{metric}'")

    values = []
    method_keys: List[str] = []
    for method in methods:
        row = sub[sub["method"] == method]
        if row.empty:
            continue
        row0 = row.iloc[0]
        counts = np.array([float(row0[c]) for c, _ in CATEGORIES], dtype=float)
        total = float(np.nansum(counts))
        if not np.isfinite(total) or total <= 0:
            continue
        values.append(counts / total * 100.0)
        method_keys.append(method)

    if not values:
        raise ValueError(f"No usable methods found for metric '{metric}' in detection_counts.csv")
    return method_keys, np.vstack(values)


def plot_locality_panel(
    df: pd.DataFrame,
    *,
    methods: List[str],
    metric_left: str,
    metric_right: str,
    out_pdf: str,
    title_left: str,
    title_right: str,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    # Match styling from the older combined stacked locality plots in this repo.
    # seaborn "talk"+"white" yields softer (non-black) label colors and paper-friendly font scaling.
    if sns is not None:
        sns.set_theme(style="white", context="talk")

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    labels_left, mat_left = _extract_metric_matrix(df, metric=metric_left, methods=methods)
    labels_right, mat_right = _extract_metric_matrix(df, metric=metric_right, methods=methods)

    # Ensure consistent method order/labels across panels (intersection only).
    common = [m for m in labels_left if m in set(labels_right)]
    if not common:
        raise ValueError("No common methods between left/right metrics after filtering.")

    def _reindex(labels: List[str], mat: np.ndarray) -> np.ndarray:
        idx = [labels.index(m) for m in common]
        return mat[idx, :]

    mat_left = _reindex(labels_left, mat_left)
    mat_right = _reindex(labels_right, mat_right)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    cat_keys = [key for key, _ in CATEGORIES]
    cat_labels = [lab for _, lab in CATEGORIES]
    cat_colors = _get_category_colors()

    for ax, mat, title in [
        (axes[0], mat_left, title_left),
        (axes[1], mat_right, title_right),
    ]:
        x = np.arange(len(common))
        bottom = np.zeros(len(common), dtype=float)
        for j, cat_key in enumerate(cat_keys):
            vals = mat[:, j]
            face = cat_colors.get(cat_key, "#777777")
            edge = matplotlib.colors.to_rgb(face)  # type: ignore[attr-defined]
            edge = tuple(np.clip(np.array(edge) * 0.6, 0, 1))
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=face,
                edgecolor=edge,
                linewidth=0.6,
                label=cat_labels[j],
            )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_wrap_xtick_label(_method_label(m)) for m in common],
            rotation=0,
            fontsize=14,
            linespacing=0.9,
        )
        ax.set_title(title, pad=6, fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(False)
        ax.tick_params(axis="y", labelsize=14)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("Triggers Share (%)", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.9,
        fontsize=12,
    )
    fig.tight_layout(pad=0.6, rect=(0.0, 0.0, 1.0, 0.86))

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a 2-panel locality figure from detection_counts.csv (plots-only).")
    parser.add_argument("--counts-csv", required=True, help="Path to detection_counts.csv")
    parser.add_argument("--out-pdf", required=True, help="Output PDF path")
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        default=[5, 10, 15, 20],
        help="Window sizes to include as WPP baselines (default: 5 20).",
    )
    parser.add_argument(
        "--no-cpd",
        action="store_true",
        help="Exclude CPD Online from the figure (default: include it).",
    )
    parser.add_argument("--metric-left", default="overall_f1", help="Left-panel metric in detection_counts.csv")
    parser.add_argument("--metric-right", default="fpr10", help="Right-panel metric in detection_counts.csv")
    parser.add_argument("--title-left", default="F1-optimal", help="Left-panel title")
    parser.add_argument("--title-right", default="FPR@10%", help="Right-panel title")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(8,4),
        help="Figure size in inches (w h). Default tuned for LaTeX inclusion at column width.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    args = parser.parse_args()

    df = pd.read_csv(args.counts_csv)
    _validate_counts_df(df)

    windows = [int(w) for w in args.windows if int(w) > 0]
    methods: List[str] = []
    if not args.no_cpd:
        methods.append("cpd_online")
    methods.extend([f"window_pp_w{w}" for w in sorted(set(windows))])

    plot_locality_panel(
        df,
        methods=methods,
        metric_left=args.metric_left,
        metric_right=args.metric_right,
        out_pdf=args.out_pdf,
        title_left=args.title_left,
        title_right=args.title_right,
        figsize=tuple(float(x) for x in args.figsize),
        dpi=int(args.dpi),
    )
    print(f"[OK] Wrote locality panel → {args.out_pdf}")


if __name__ == "__main__":
    main()
