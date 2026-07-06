"""Generate benchmark figures for the paper.

Usage:
    uv run python benchmarks/plot_paper.py benchmarks/results/softjax_benchmark_combined.csv
    uv run python benchmarks/plot_paper.py benchmarks/results/softjax_benchmark_combined.csv --figure main
    uv run python benchmarks/plot_paper.py benchmarks/results/softjax_benchmark_combined.csv --figure appendix --format png
    uv run python benchmarks/plot_paper.py benchmarks/results/softjax_benchmark_combined.csv --output-dir ../../paper/graphics/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_ORDER = [
    "hard",
    "ot",
    "softsort",
    "neuralsort",
    "fast_soft_sort",
    "smooth_sort",
    "sorting_network",
]

DISPLAY_NAMES = {
    "hard": "Hard",
    "ot": "OT",
    "softsort": "SoftSort",
    "neuralsort": "NeuralSort",
    "fast_soft_sort": "FastSoftSort",
    "smooth_sort": "SmoothSort",
    "sorting_network": "SortingNet",
}

METHOD_COLORS = {
    "hard": "#222222",
    "ot": "#00BFFF",          # sjMidBlue
    "softsort": "#368F80",     # sjDarkGreen
    "neuralsort": "#E1BE6A",   # sjMidYellow
    "fast_soft_sort": "#889FD9",  # sjMidPurple
    "smooth_sort": "#DB70D8",  # sjMidPink
    "sorting_network": "#F06247",  # sjMidRed
}

METHOD_MARKERS = {
    "hard": "o",
    "ot": "D",
    "softsort": "s",
    "neuralsort": "^",
    "fast_soft_sort": "v",
    "smooth_sort": "P",
    "sorting_network": "X",
}

MODE_ORDER = ["smooth", "c0", "c1", "c2"]

MODE_DISPLAY = {
    "smooth": "Smooth",
    "c0": "C0",
    "c1": "C1",
    "c2": "C2",
}

# Page geometry
TEXT_WIDTH = 6.75  # inches
COLUMN_WIDTH = (TEXT_WIDTH - 0.25) / 2  # 3.25 inches


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------


def configure_paper_style():
    """Set rcParams for paper figures.

    Font sizes are set so that when the figure width matches the target
    LaTeX width (column or text width), text renders at the same size as
    the paper body (10pt). No scaling needed.
    """
    plt.rcParams.update(
        {
            # Font — match paper (Times, 10pt body)
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # Lines
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linestyle": "--",
            "grid.color": "#cccccc",
            "grid.alpha": 0.7,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            # Background
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            # LaTeX-style math rendering
            "mathtext.fontset": "stix",
            # PDF
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Save
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    """Load benchmark CSV and add derived columns."""
    df = pd.read_csv(csv_path)
    # Fill empty method field for hard mode
    df["method"] = df["method"].fillna("")
    # Combined fwd+bwd time in ms
    df["total_ms"] = df["fwd_ms_mean"] + df["grad_ms_mean"]
    # Combined fwd+bwd JIT time in ms
    df["total_jit_ms"] = df["jit_fwd_ms_mean"] + df["jit_grad_ms_mean"]
    # Peak memory in KB
    df["peak_memory_kb"] = df["peak_memory_bytes"] / 1024
    return df


# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------


def format_log2_axis(ax, sizes, stride=2):
    """Format x-axis with log base-2 ticks showing $2^k$ labels.

    Args:
        stride: Show every `stride`-th power of 2 (default 2 = even powers).
    """
    ax.set_xscale("log", base=2)
    log2_sizes = np.log2(sizes).astype(int)
    # Select tick positions: every `stride`-th power, always include first and last
    tick_mask = (log2_sizes - log2_sizes[0]) % stride == 0
    tick_sizes = sizes[tick_mask]
    ax.set_xticks(tick_sizes)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"$2^{{{int(np.log2(x))}}}$" if x > 0 else ""
        )
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(sizes[0] * 0.7, sizes[-1] * 1.4)


# ---------------------------------------------------------------------------
# Core plotting helper
# ---------------------------------------------------------------------------


def _plot_method_lines(ax, df, metric_col, methods):
    """Plot one metric for all methods on a single axes.

    Hard baseline is plotted as black dashed if present in df.
    Other methods are plotted as solid colored lines with markers.
    """
    # Hard baseline
    hard = df[df["mode"] == "hard"]
    if not hard.empty:
        hg = hard.groupby("problem_size")[metric_col].mean().sort_index()
        ax.plot(
            hg.index,
            hg.values,
            color=METHOD_COLORS["hard"],
            ls="--",
            marker=METHOD_MARKERS["hard"],
            label=DISPLAY_NAMES["hard"],
            zorder=10,
        )

    # Soft methods
    for method in methods:
        if method == "hard":
            continue
        sub = df[df["method"] == method].sort_values("problem_size")
        if sub.empty:
            continue
        vals = sub.groupby("problem_size")[metric_col].mean()
        if vals.empty or (vals <= 0).all():
            continue
        ax.plot(
            vals.index,
            vals.values,
            color=METHOD_COLORS[method],
            ls="-",
            marker=METHOD_MARKERS[method],
            label=DISPLAY_NAMES[method],
            zorder=5,
        )


# ---------------------------------------------------------------------------
# Figure 1: Main text (sort, smooth mode only)
# ---------------------------------------------------------------------------


def plot_main_figure(df, output_dir, fmt="pdf", functions=None):
    """2-panel figure: Time (ms) | Memory (KB) for smooth mode."""
    func = functions[0] if functions else "sort"
    fdf = df[df["function"] == func]
    smooth = fdf[(fdf["mode"] == "smooth") | (fdf["mode"] == "hard")]

    if smooth.empty:
        print(f"  No smooth-mode data for {func}, skipping main figure.")
        return

    sizes = np.sort(smooth["problem_size"].unique())

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.6))

    # Time panel
    _plot_method_lines(ax_time, smooth, "total_ms", METHOD_ORDER)
    ax_time.set_yscale("log")
    format_log2_axis(ax_time, sizes)
    ax_time.set_title("Time (ms)")
    ax_time.set_xlabel("Array Size")

    # Memory panel
    _plot_method_lines(ax_mem, smooth, "peak_memory_kb", METHOD_ORDER)
    ax_mem.set_yscale("log")
    format_log2_axis(ax_mem, sizes)
    ax_mem.set_title("Memory (KB)")
    ax_mem.set_xlabel("Array Size")

    # Shared legend at bottom
    handles, labels = [], []
    for h, l in zip(*ax_time.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    for h, l in zip(*ax_mem.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

    # Sort legend entries by METHOD_ORDER
    pairs = sorted(
        zip(labels, handles),
        key=lambda p: METHOD_ORDER.index(
            next(k for k, v in DISPLAY_NAMES.items() if v == p[0])
        )
        if p[0] in DISPLAY_NAMES.values()
        else len(METHOD_ORDER),
    )
    if pairs:
        labels, handles = zip(*pairs)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.5,
    )

    fig.tight_layout(rect=[0, 0.22, 1, 1])
    out = output_dir / f"benchmark_{func}_smooth.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Appendix (sort, all methods x all modes)
# ---------------------------------------------------------------------------


def plot_appendix_figure(df, output_dir, fmt="pdf", functions=None):
    """4 rows (modes) x 3 columns (Time, JIT Time, Memory) = 12 subplots."""
    func = functions[0] if functions else "sort"
    fdf = df[df["function"] == func]

    modes_present = [m for m in MODE_ORDER if m in fdf["mode"].unique()]
    if not modes_present:
        print(f"  No soft-mode data for {func}, skipping appendix figure.")
        return

    hard = fdf[fdf["mode"] == "hard"]
    sizes = np.sort(fdf["problem_size"].unique())

    nrows = len(modes_present)
    metrics = [
        ("total_ms", "Time (ms)"),
        ("total_jit_ms", "JIT Time (ms)"),
        ("peak_memory_kb", "Memory (KB)"),
    ]
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(TEXT_WIDTH, 1.1 * nrows),
        squeeze=False,
        sharey="col",
        sharex=True,
    )

    for row_idx, mode in enumerate(modes_present):
        mode_df = fdf[fdf["mode"] == mode]
        combined = pd.concat([mode_df, hard], ignore_index=True)

        for col_idx, (metric_col, col_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            _plot_method_lines(ax, combined, metric_col, METHOD_ORDER)
            ax.set_yscale("log")
            format_log2_axis(ax, sizes)

            # Column titles on top row
            if row_idx == 0:
                ax.set_title(col_title)

            # Mode label on left column
            if col_idx == 0:
                ax.set_ylabel(MODE_DISPLAY[mode])

            # X-axis label on bottom row only
            if row_idx == nrows - 1:
                ax.set_xlabel("Array Size")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    # Shared legend at bottom: collect from all axes, deduplicate
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    pairs = sorted(
        zip(labels, handles),
        key=lambda p: METHOD_ORDER.index(
            next(k for k, v in DISPLAY_NAMES.items() if v == p[0])
        )
        if p[0] in DISPLAY_NAMES.values()
        else len(METHOD_ORDER),
    )
    if pairs:
        labels, handles = zip(*pairs)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.5,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = output_dir / f"benchmark_{func}_all_modes.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Elementwise (rows=functions, cols=metrics, lines=modes)
# ---------------------------------------------------------------------------

MODE_COLORS = {
    "hard": "#222222",
    "smooth": "#00BFFF",
    "c0": "#F06247",
    "c1": "#368F80",
    "c2": "#889FD9",
}

MODE_MARKERS = {
    "hard": "o",
    "smooth": "D",
    "c0": "s",
    "c1": "^",
    "c2": "v",
}


def plot_elementwise_figure(df, output_dir, fmt="pdf", functions=None):
    """Grid figure: rows=functions, cols=Time|JIT Time|Memory, lines=modes."""
    func_names = functions or sorted(df["function"].unique())
    # Only keep functions with data
    func_names = [f for f in func_names if not df[df["function"] == f].empty]
    if not func_names:
        print("  No elementwise data, skipping.")
        return

    sizes = np.sort(df["problem_size"].unique())
    modes_present = ["hard"] + [m for m in MODE_ORDER if m in df["mode"].unique()]

    nrows = len(func_names)
    metrics = [
        ("total_ms", "Time (ms)"),
        ("total_jit_ms", "JIT Time (ms)"),
        ("peak_memory_kb", "Memory (KB)"),
    ]
    fig, axes = plt.subplots(
        nrows, 3,
        figsize=(TEXT_WIDTH, 1.1 * nrows),
        squeeze=False,
        sharey="col",
        sharex=True,
    )

    for row_idx, func in enumerate(func_names):
        fdf = df[df["function"] == func]
        for col_idx, (metric_col, col_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for mode in modes_present:
                sub = fdf[fdf["mode"] == mode].sort_values("problem_size")
                if sub.empty:
                    continue
                vals = sub.groupby("problem_size")[metric_col].mean()
                if vals.empty or (vals <= 0).all():
                    continue
                is_hard = mode == "hard"
                ax.plot(
                    vals.index, vals.values,
                    color=MODE_COLORS[mode],
                    ls="--" if is_hard else "-",
                    marker=MODE_MARKERS[mode],
                    label=MODE_DISPLAY.get(mode, mode),
                    zorder=10 if is_hard else 5,
                )
            ax.set_yscale("log")
            format_log2_axis(ax, sizes)

            if row_idx == 0:
                ax.set_title(col_title)
            if col_idx == 0:
                ax.set_ylabel(func)
            if row_idx == nrows - 1:
                ax.set_xlabel("Array Size")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    # Shared legend
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
            columnspacing=1.0,
            handletextpad=0.4,
            handlelength=1.5,
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_dir / f"benchmark_elementwise_all_modes.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark figures for the paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Path to benchmark CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as CSV)",
    )
    parser.add_argument(
        "--figure",
        choices=["main", "appendix", "elementwise", "all"],
        default="all",
        help="Which figure to generate (default: all)",
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="sort",
        help="Comma-separated function names (default: sort)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        parser.error(f"CSV not found: {args.csv_path}")

    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    functions = [f.strip() for f in args.functions.split(",")]

    configure_paper_style()
    df = load_and_prepare(args.csv_path)

    # Filter to requested functions
    df = df[df["function"].isin(functions)]
    if df.empty:
        parser.error(f"No data for functions: {args.functions}")

    if args.figure in ("main", "all"):
        print("Generating main figure (smooth mode)...")
        plot_main_figure(df, output_dir, args.format, functions)

    if args.figure in ("appendix", "all"):
        print("Generating appendix figure (all modes)...")
        plot_appendix_figure(df, output_dir, args.format, functions)

    if args.figure in ("elementwise", "all"):
        print("Generating elementwise figure (smooth mode)...")
        plot_elementwise_figure(df, output_dir, args.format, functions)

    print("Done.")


if __name__ == "__main__":
    main()
