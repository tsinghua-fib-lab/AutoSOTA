"""Plot benchmark results from benchmark CSVs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "ot": "tab:blue",
    "softsort": "tab:orange",
    "neuralsort": "tab:green",
    "fast_soft_sort": "tab:red",
    "smooth_sort": "tab:brown",
    "sorting_network": "tab:purple",
    # elementwise (no method) — used as fallback
    "": "tab:gray",
}

MODE_COLORS = {
    "hard": "black",
    "smooth": "tab:blue",
    "c0": "tab:orange",
    "c1": "tab:green",
    "c2": "tab:red",
}

METHOD_LINESTYLES = {
    "fast_soft_sort": "-",
    "smooth_sort": (0, (3, 1, 1, 1)),
    "neuralsort": "--",
    "softsort": "-.",
    "ot": ":",
    "sorting_network": (0, (5, 1)),
    "": "-",
}

MODE_ORDER = ["smooth", "c0", "c1", "c2"]

# (row_label, metric_columns) — each entry is (label, mean_col, std_col | None)
# Columns of the grid: runtime, JIT, memory
# Rows of the grid: forward, grad
METRIC_GRID = {
    # col_idx: (col_title, [(row_label, mean_col, std_col), ...])
    "runtime": [
        ("forward", "fwd_ms_mean", "fwd_ms_std"),
        ("grad", "grad_ms_mean", "grad_ms_std"),
    ],
    "JIT": [
        ("forward", "jit_fwd_ms_mean", "jit_fwd_ms_std"),
        ("grad", "jit_grad_ms_mean", "jit_grad_ms_std"),
    ],
    "memory": [
        ("forward", "peak_memory_mb", None),
        ("grad", "peak_memory_mb", None),
    ],
}


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "tab:gray")


def _plot_lines(ax, data, methods, hard, mean_col, std_col, log_y=True):
    """Plot method lines with optional std shading and hard baseline."""
    # Hard baseline
    if not hard.empty:
        hg = hard.groupby("problem_size")[mean_col].mean()
        if (hg > 0).any():
            ax.plot(
                hg.index,
                hg.values,
                color="black",
                ls="--",
                lw=1.5,
                label="hard",
                zorder=5,
            )

    for method in methods:
        sub = data[data["method"] == method].sort_values("problem_size")
        if sub.empty or (sub[mean_col] <= 0).all():
            continue
        label = method if method else "default"
        color = _method_color(method)
        ax.plot(
            sub["problem_size"],
            sub[mean_col],
            color=color,
            lw=1.5,
            label=label,
        )
        if std_col is not None:
            ax.fill_between(
                sub["problem_size"],
                sub[mean_col] - sub[std_col],
                sub[mean_col] + sub[std_col],
                color=color,
                alpha=0.15,
            )

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")


# ---------------------------------------------------------------------------
# 1. Scaling plots — one figure per (function, mode)
# ---------------------------------------------------------------------------


def _mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "tab:gray")


def _method_ls(method: str) -> str:
    return METHOD_LINESTYLES.get(method, "-")


MODE_LINESTYLES = {
    "hard": "--",
    "smooth": "-",
    "c0": "-.",
    "c1": ":",
    "c2": (0, (3, 1, 1, 1)),  # dash-dot-dot
}

MODE_MARKERS = {
    "hard": "x",
    "smooth": None,
    "c0": None,
    "c1": None,
    "c2": None,
}


def _mode_ls(mode: str) -> str:
    return MODE_LINESTYLES.get(mode, "-")


def _plot_all_modes(ax, fdf, mean_col, std_col, log_y=True):
    """Plot all (mode, method) combinations on a single axes.

    Colour encodes method, line style encodes mode.
    """
    hard = fdf[fdf["mode"] == "hard"]
    if not hard.empty:
        hg = hard.groupby("problem_size")[mean_col].mean()
        if (hg > 0).any():
            ax.plot(
                hg.index,
                hg.values,
                color="black",
                ls="--",
                lw=2,
                marker="x",
                markersize=5,
                label="hard",
                zorder=5,
            )

    modes_present = [m for m in MODE_ORDER if m in fdf["mode"].unique()]
    all_methods = sorted(fdf[fdf["mode"] != "hard"]["method"].unique())
    single_method = len(all_methods) <= 1

    for mode in modes_present:
        mdf = fdf[fdf["mode"] == mode]
        methods = sorted(mdf["method"].unique())
        for method in methods:
            sub = mdf[mdf["method"] == method].sort_values("problem_size")
            if sub.empty or (sub[mean_col] <= 0).all():
                continue
            method_label = method if method else "default"
            label = method_label if single_method else f"{method_label} / {mode}"
            color = _method_color(method)
            ls = _mode_ls(mode)
            ax.plot(
                sub["problem_size"],
                sub[mean_col],
                color=color,
                ls=ls,
                lw=2,
                label=label,
            )
            if std_col is not None:
                ax.fill_between(
                    sub["problem_size"],
                    sub[mean_col] - sub[std_col],
                    sub[mean_col] + sub[std_col],
                    color=color,
                    alpha=0.10,
                )

    ax.set_xscale("log", base=10)
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.grid(False)


def plot_scaling(df: pd.DataFrame, output_dir: Path, fmt: str):
    """One figure per function, all modes overlaid.

    Colour = mode, line style = method.
    """
    if "peak_memory_mb" not in df.columns:
        df = df.copy()
        df["peak_memory_mb"] = df["peak_memory_bytes"] / 1e6
    has_jit = (df["jit_fwd_ms_mean"] > 0).any()
    has_memory = (df["peak_memory_mb"] > 0).any()

    col_specs = [("runtime (ms)", METRIC_GRID["runtime"])]
    if has_jit:
        col_specs.append(("JIT (ms)", METRIC_GRID["JIT"]))
    if has_memory:
        col_specs.append(("memory (MB)", METRIC_GRID["memory"]))

    nrows = 2  # forward, grad
    ncols = len(col_specs)

    for func_name, fdf in df.groupby("function"):
        modes_present = [m for m in MODE_ORDER if m in fdf["mode"].unique()]
        if not modes_present and "hard" not in fdf["mode"].unique():
            continue

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False,
            sharex=True,
        )

        for col_idx, (col_title, row_specs) in enumerate(col_specs):
            for row_idx, (row_label, mean_col, std_col) in enumerate(row_specs):
                ax = axes[row_idx, col_idx]
                _plot_all_modes(
                    ax,
                    fdf,
                    mean_col,
                    std_col,
                    log_y=True,
                )

                if row_idx == 0:
                    ax.set_title(col_title, fontsize=13, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=12)
                if row_idx == nrows - 1:
                    ax.set_xlabel("problem size", fontsize=12)
                ax.tick_params(labelsize=10)

        # Shared legend — deduplicate and group by method
        handles, labels = [], []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        METHOD_ORDER = ["hard", "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"]

        def _legend_sort_key(label):
            method = label.split(" / ")[0] if " / " in label else label
            try:
                return METHOD_ORDER.index(method)
            except ValueError:
                return len(METHOD_ORDER)

        pairs = sorted(zip(labels, handles), key=lambda p: _legend_sort_key(p[0]))
        labels, handles = zip(*pairs) if pairs else ([], [])

        fig.suptitle(func_name, fontsize=15, fontweight="bold")
        n_legend_rows = (len(labels) + 4) // 5
        legend_height = 0.05 * n_legend_rows
        fig.tight_layout(rect=[0, legend_height, 1, 0.96], h_pad=1.0, w_pad=1.0)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=5,
                fontsize=10,
                frameon=True,
                bbox_to_anchor=(0.5, 0.0),
            )
        out = output_dir / f"{func_name}_scaling.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 1b. Combined scaling plot — all functions in one figure
# ---------------------------------------------------------------------------

FUNCTION_ORDER = ["sort", "top_k", "quantile", "max"]


def plot_scaling_combined(df: pd.DataFrame, output_dir: Path, fmt: str):
    """One figure with all functions stacked vertically, shared legend."""
    from matplotlib.gridspec import GridSpec

    if "peak_memory_mb" not in df.columns:
        df = df.copy()
        df["peak_memory_mb"] = df["peak_memory_bytes"] / 1e6
    has_jit = "jit_fwd_ms_mean" in df.columns and (df["jit_fwd_ms_mean"] > 0).any()
    has_memory = (df["peak_memory_mb"] > 0).any()

    col_specs = [("runtime (ms)", METRIC_GRID["runtime"])]
    if has_jit:
        col_specs.append(("JIT (ms)", METRIC_GRID["JIT"]))
    if has_memory:
        col_specs.append(("memory (MB)", METRIC_GRID["memory"]))

    funcs = [f for f in FUNCTION_ORDER if f in df["function"].unique()]
    funcs += sorted(set(df["function"].unique()) - set(FUNCTION_ORDER))

    ncols = len(col_specs)
    nfuncs = len(funcs)

    # Build height_ratios: [1, 1, spacer, 1, 1, spacer, ...]
    spacer = 0.15
    height_ratios = []
    for i in range(nfuncs):
        height_ratios.extend([1, 1])
        if i < nfuncs - 1:
            height_ratios.append(spacer)
    total_gs_rows = len(height_ratios)

    fig = plt.figure(figsize=(5 * ncols, 2.2 * nfuncs * 2))
    gs = GridSpec(
        total_gs_rows, ncols, figure=fig,
        height_ratios=height_ratios, hspace=0.3, wspace=0.3,
    )

    # Create axes, skipping spacer rows
    axes = np.empty((nfuncs * 2, ncols), dtype=object)
    first_ax = None
    for func_idx in range(nfuncs):
        gs_row_base = func_idx * 3  # 2 data rows + 1 spacer (except last)
        for phase_idx in range(2):
            gs_row = gs_row_base + phase_idx
            for col_idx in range(ncols):
                ax = fig.add_subplot(
                    gs[gs_row, col_idx],
                    sharex=first_ax,
                )
                if first_ax is None:
                    first_ax = ax
                axes[func_idx * 2 + phase_idx, col_idx] = ax

    for func_idx, func_name in enumerate(funcs):
        fdf = df[df["function"] == func_name]
        modes_present = [m for m in MODE_ORDER if m in fdf["mode"].unique()]
        if not modes_present and "hard" not in fdf["mode"].unique():
            continue

        for col_idx, (col_title, row_specs) in enumerate(col_specs):
            for phase_idx, (row_label, mean_col, std_col) in enumerate(row_specs):
                ax_row = func_idx * 2 + phase_idx
                ax = axes[ax_row, col_idx]
                _plot_all_modes(ax, fdf, mean_col, std_col, log_y=True)

                # Column titles on top row only
                if ax_row == 0:
                    ax.set_title(col_title, fontsize=13, fontweight="bold")
                # Function + phase label on left column
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{func_name}\n{row_label}", fontsize=11, fontweight="bold",
                    )
                # x-label on bottom row only
                if ax_row == nfuncs * 2 - 1:
                    ax.set_xlabel("problem size", fontsize=12)
                ax.tick_params(labelsize=9)

    # Shared legend — deduplicate across all axes
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    _METHOD_ORDER = [
        "hard", "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network",
    ]

    def _legend_sort_key(label):
        method = label.split(" / ")[0] if " / " in label else label
        try:
            return _METHOD_ORDER.index(method)
        except ValueError:
            return len(_METHOD_ORDER)

    pairs = sorted(zip(labels, handles), key=lambda p: _legend_sort_key(p[0]))
    labels, handles = zip(*pairs) if pairs else ([], [])

    if handles:
        # Place legend just below the bottom axes row
        bottom_y = min(ax.get_position().y0 for ax in axes[-1, :])
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, bottom_y - 0.01),
        )
    out = output_dir / f"combined_scaling.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 2. Method comparison bar charts — one figure per category
# ---------------------------------------------------------------------------


def plot_method_comparison(df: pd.DataFrame, output_dir: Path, fmt: str):
    if "peak_memory_mb" not in df.columns:
        df = df.copy()
        df["peak_memory_mb"] = df["peak_memory_bytes"] / 1e6
    has_jit = (df["jit_fwd_ms_mean"] > 0).any()
    has_memory = (df["peak_memory_mb"] > 0).any()

    col_specs = [("runtime (ms)", METRIC_GRID["runtime"])]
    if has_jit:
        col_specs.append(("JIT (ms)", METRIC_GRID["JIT"]))
    if has_memory:
        col_specs.append(("memory (MB)", METRIC_GRID["memory"]))

    for category, cdf in df.groupby("category"):
        funcs = sorted(cdf["function"].unique())
        modes_present = [m for m in MODE_ORDER if m in cdf["mode"].unique()]
        if not modes_present:
            continue

        # rows = functions × 2 (fwd/grad), columns = metric types
        nrows = len(funcs) * 2
        ncols = len(col_specs)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 1.5 * nrows),
            squeeze=False,
        )

        for func_idx, func_name in enumerate(funcs):
            fdf = cdf[cdf["function"] == func_name]
            largest_size = fdf["problem_size"].max()
            fdf = fdf[fdf["problem_size"] == largest_size]

            for col_idx, (_col_title, row_specs) in enumerate(col_specs):
                for phase_idx, (row_label, mean_col, _std_col) in enumerate(
                    row_specs
                ):
                    ax_row = func_idx * 2 + phase_idx
                    ax = axes[ax_row, col_idx]

                    methods = sorted(
                        fdf[fdf["mode"] != "hard"]["method"].unique()
                    )
                    if not methods:
                        methods = [""]
                    n_methods = len(methods)
                    x = np.arange(len(modes_present))
                    width = 0.8 / max(n_methods, 1)

                    for m_idx, method in enumerate(methods):
                        vals = []
                        for mode in modes_present:
                            row = fdf[
                                (fdf["mode"] == mode) & (fdf["method"] == method)
                            ]
                            vals.append(
                                row[mean_col].values[0] if len(row) else 0
                            )
                        offset = (m_idx - (n_methods - 1) / 2) * width
                        label = method if method else "default"
                        ax.bar(
                            x + offset,
                            vals,
                            width * 0.9,
                            label=label,
                            color=_method_color(method),
                        )

                    hard_row = fdf[fdf["mode"] == "hard"]
                    if not hard_row.empty:
                        hard_val = hard_row[mean_col].values[0]
                        if hard_val > 0:
                            ax.axhline(
                                hard_val,
                                color="black",
                                ls="--",
                                lw=1,
                                label="hard",
                            )

                    ax.set_xticks(x)
                    ax.set_xticklabels(modes_present, fontsize=9)
                    if ax_row == 0:
                        ax.set_title(_col_title, fontsize=11)
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"{func_name}\n{row_label}", fontsize=9
                        )

        # Shared legend
        handles, labels = [], []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(labels),
                fontsize=9,
                frameon=True,
                bbox_to_anchor=(0.5, 1.02),
            )

        fig.suptitle(
            f"{category} — method comparison (largest size)", fontsize=13, y=1.06
        )
        fig.tight_layout()
        out = output_dir / f"{category}_method_comparison.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from benchmark CSVs."
    )
    parser.add_argument("csv_path", type=Path, help="Path to benchmark CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as CSV)",
    )
    parser.add_argument(
        "--functions",
        type=str,
        default=None,
        help="Comma-separated function names to filter",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate a single combined scaling plot with all functions stacked",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if args.functions:
        names = {n.strip() for n in args.functions.split(",")}
        df = df[df["function"].isin(names)]
        if df.empty:
            parser.error(f"No data for functions: {args.functions}")

    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    print("Generating scaling plots...")
    plot_scaling(df, output_dir, args.format)

    if args.combined and df["function"].nunique() > 1:
        print("Generating combined scaling plot...")
        plot_scaling_combined(df, output_dir, args.format)

    print("Generating method comparison plots...")
    plot_method_comparison(df, output_dir, args.format)

    print("Done.")


if __name__ == "__main__":
    main()
