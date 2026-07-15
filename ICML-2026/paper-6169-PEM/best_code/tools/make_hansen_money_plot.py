#!/usr/bin/env python3
"""
Make a paper-ready "Money Plot" for the Hansen fixed-budget test.

- x-axis: total function evaluations (log)
- y-axis: median best-so-far *noise-free* delta (f - fopt) (log)

Key features:
- paper-facing display names
- depth bar on the right side with connection lines showing performance-depth correlation

Input:
  CSVs produced by `tools/extract_coco_traces.py`, e.g.
    <csv_dir>/trace_noisefree_f110_d40.csv

Output:
  <output_prefix>.png and <output_prefix>.pdf
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator, NullFormatter
import numpy as np


def _plain_log_formatter(x, pos):
    """Format log scale ticks as plain numbers (1, 10, 100) instead of 10^0, 10^1, 10^2."""
    if x <= 0:
        return ""
    # For integer powers of 10, show as plain integers
    if x >= 1:
        rounded = int(round(x))
        if abs(x - rounded) < 0.01 * x:  # within 1% tolerance
            return f"{rounded}"
        return f"{x:.1f}"
    else:
        # For values < 1, show decimal
        if x >= 0.1:
            return f"{x:.1f}"
        elif x >= 0.01:
            return f"{x:.2f}"
        else:
            return f"{x:.0e}"

from _project import BASE_DIR, repo_relpath
from plot_style import (
    apply_style,
    get_algo_color,
    get_algo_linewidth,
    add_grid,
    save_figure,
    WIDTHS,
)

from utils.display_names import get_display_name, get_algorithm_depth


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_csv_header(header: list[str]) -> dict[str, dict[str, int]]:
    """
    Return mapping:
      algo -> {"median": idx, "q25": idx, "q75": idx}
    """
    out: dict[str, dict[str, int]] = {}
    for idx, name in enumerate(header):
        m = re.match(r"^(.*):(median|q25|q75)$", str(name).strip())
        if not m:
            continue
        algo = str(m.group(1)).strip()
        key = str(m.group(2)).strip()
        out.setdefault(algo, {})[key] = int(idx)
    return out


@dataclass(frozen=True)
class Curve:
    x_evals: np.ndarray
    y_median: np.ndarray
    y_q25: np.ndarray | None
    y_q75: np.ndarray | None


def safe_positive(y: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    return np.maximum(np.asarray(y, dtype=float), float(eps))


def read_curve_csv(path: str, *, dim: int, algo: str) -> Curve:
    import csv

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"Empty CSV: {path}")

    header = rows[0]
    col_map = parse_csv_header(header)
    if str(algo) not in col_map or "median" not in col_map[str(algo)]:
        raise KeyError(f"Algorithm {algo!r} not found in {path}. Available: {sorted(col_map.keys())}")

    # Column 0 is evals_per_dim.
    x_per_dim = np.asarray([float(r[0]) for r in rows[1:] if r and r[0] != ""], dtype=float)
    x_evals = x_per_dim * float(dim)

    idx_med = col_map[str(algo)]["median"]
    y_med = np.asarray([float(r[idx_med]) for r in rows[1:] if len(r) > idx_med and r[idx_med] != ""], dtype=float)

    y_q25 = None
    y_q75 = None
    if "q25" in col_map[str(algo)] and "q75" in col_map[str(algo)]:
        idx_q25 = col_map[str(algo)]["q25"]
        idx_q75 = col_map[str(algo)]["q75"]
        y_q25 = np.asarray([float(r[idx_q25]) for r in rows[1:] if len(r) > idx_q25 and r[idx_q25] != ""], dtype=float)
        y_q75 = np.asarray([float(r[idx_q75]) for r in rows[1:] if len(r) > idx_q75 and r[idx_q75] != ""], dtype=float)

    if len(x_evals) != len(y_med):
        raise ValueError(f"Length mismatch in {path}: x={len(x_evals)} y={len(y_med)}")
    if y_q25 is not None and len(y_q25) != len(y_med):
        raise ValueError(f"Length mismatch (q25) in {path}")
    if y_q75 is not None and len(y_q75) != len(y_med):
        raise ValueError(f"Length mismatch (q75) in {path}")

    return Curve(x_evals=x_evals, y_median=y_med, y_q25=y_q25, y_q75=y_q75)


def get_depth_for_algorithm(algo: str, budget: int, dim: int) -> int | None:
    """
    Get the depth (number of generations) for an algorithm under a fixed budget.

    Prefer the central registry, fall back to analytical formulas where possible.
    """
    depth = get_algorithm_depth(algo, budget=budget)
    if depth is not None:
        return depth

    # Population size for CMA-ES: lambda = 4 + floor(3*ln(d))
    lambda_ = 4 + int(3 * math.log(dim))

    # Resampling variants
    if "Resample(k=" in algo:
        m = re.search(r"Resample\(k=(\d+)\)", algo)
        if m:
            k = int(m.group(1))
            return budget // (k * lambda_)

    # CMA-ES baseline
    if algo in ("CMA-ES", "CMA-ES-sep", "Sep-CMA-ES"):
        return budget // lambda_

    return None


def draw_depth_bar(ax_bar, depths: dict[str, int], colors: dict[str, str], y_lim: tuple[float, float]):
    """
    Draw a vertical depth gradient bar with markers for each algorithm.
    Inverted: high depth at bottom, low depth at top (to correlate with performance).
    """
    depth_vals = list(depths.values())
    depth_min = min(depth_vals)
    depth_max = max(depth_vals)

    # Add padding to depth range
    depth_range = depth_max - depth_min
    depth_min_padded = max(0, depth_min - depth_range * 0.1)
    depth_max_padded = depth_max + depth_range * 0.1

    # Create gradient background (darker at top = high depth after inversion)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("depth", ["#f0f0f0", "#4a90d9"])

    ax_bar.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[0, 1, depth_min_padded, depth_max_padded],
    )

    # Invert the y-axis so high depth is at bottom
    ax_bar.invert_yaxis()

    ax_bar.set_xlim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_label_position("right")
    ax_bar.set_ylabel("Depth (generations)", fontsize=7, rotation=270, labelpad=12)
    ax_bar.tick_params(axis="y", labelsize=6)

    # Use plain integer formatting, no scientific notation
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax_bar.yaxis.set_major_formatter(formatter)

    # Remove spines except right
    for spine in ["top", "bottom", "left"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["right"].set_linewidth(0.5)

    return depth_min_padded, depth_max_padded


def main() -> None:
    apply_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", required=True, help="Directory containing trace_noisefree_f*_d*.csv files.")
    parser.add_argument("--functions", required=True, help="Function ids, e.g. '108,110,114,120'.")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument(
        "--algorithms",
        default="CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),UH-CMA-ES(maxevals=30),BERW-Hetero",
        help="Comma-separated algorithms to plot (legend order).",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--output-prefix", required=True, help="Output prefix (writes .png and .pdf).")
    parser.add_argument("--no-iqr", action="store_true", help="Disable IQR shading.")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth bar and connections.")
    parser.add_argument("--budget", type=int, default=None, help="Total budget (default: 100*dim).")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    csv_dir = os.path.abspath(str(args.csv_dir))
    dim = int(args.dim)
    functions = parse_int_list(str(args.functions))
    algorithms = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    if not functions or not algorithms:
        raise SystemExit("Empty functions/algorithms.")

    budget = int(args.budget) if args.budget is not None else 100 * dim

    color_map = {alg: get_algo_color(alg) for alg in algorithms}
    linewidth_map = {alg: get_algo_linewidth(alg) for alg in algorithms}

    # Compute depths for all algorithms
    algo_depths = {}
    for alg in algorithms:
        d = get_depth_for_algorithm(alg, budget, dim)
        if d is not None:
            algo_depths[alg] = d

    # Create figure with GridSpec: 2x2 main plots, each with a small depth bar on right
    n_funcs = len(functions)
    n_rows = 2
    n_cols = 2

    # Width ratios: main plot vs depth bar
    main_width = 10
    bar_width = 1 if not args.no_depth else 0

    fig = plt.figure(figsize=(WIDTHS["double"] * 1.1, WIDTHS["double"] * 0.7))

    # Create outer grid for the 2x2 layout
    outer_gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        wspace=0.35,
        hspace=0.4,
        left=0.08,
        right=0.92,
        top=0.88,
        bottom=0.08,
    )

    handles = []
    labels = []
    all_axes: list[tuple[int, plt.Axes]] = []  # Store (idx, ax) for post-processing

    for idx, func_id in enumerate(functions):
        if idx >= n_rows * n_cols:
            break

        row = idx // n_cols
        col = idx % n_cols

        # Create inner grid for this subplot (main + bar)
        if not args.no_depth:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2,
                subplot_spec=outer_gs[row, col],
                width_ratios=[main_width, bar_width],
                wspace=0.02,
            )
            ax = fig.add_subplot(inner_gs[0])
            ax_bar = fig.add_subplot(inner_gs[1])
        else:
            ax = fig.add_subplot(outer_gs[row, col])
            ax_bar = None

        csv_path = os.path.join(csv_dir, f"trace_noisefree_f{int(func_id)}_d{int(dim)}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing: {csv_path}")

        # Store endpoint info for connection lines
        endpoint_info: list[tuple[str, float, float, int]] = []  # (algo, x_end, y_end, depth)

        for alg in algorithms:
            curve = read_curve_csv(csv_path, dim=int(dim), algo=str(alg))
            x = safe_positive(curve.x_evals)
            y = safe_positive(curve.y_median)
            (line,) = ax.plot(x, y, linewidth=linewidth_map[alg], color=color_map[alg], label=alg)

            if len(x) > 0 and len(y) > 0 and alg in algo_depths:
                endpoint_info.append((alg, float(x[-1]), float(y[-1]), algo_depths[alg]))

            if idx == 0:
                handles.append(line)
                labels.append(get_display_name(alg))

            if not bool(args.no_iqr) and curve.y_q25 is not None and curve.y_q75 is not None:
                y_lo = safe_positive(curve.y_q25)
                y_hi = safe_positive(curve.y_q75)
                ax.fill_between(x, y_lo, y_hi, color=color_map[alg], alpha=0.16, linewidth=0.0)

        ax.set_xscale("log")
        ax.set_yscale("log")
        add_grid(ax, which="both", alpha=0.2)

        f_idx = int(func_id) - 100 if int(func_id) >= 101 else int(func_id)
        ax.set_title(f"bbob-noisy f{f_idx}", fontsize=9)
        ax.set_xlabel("Function evaluations", fontsize=7)
        ax.set_ylabel(r"$f(\mathbf{x}_{\mathrm{best}}) - f^*$", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)

        # Store for post-processing
        all_axes.append((idx, ax))

        # Draw depth bar and connection lines
        if not args.no_depth and ax_bar is not None and endpoint_info:
            y_vals = [ep[2] for ep in endpoint_info]
            y_lim = (min(y_vals), max(y_vals))

            depth_min, depth_max = draw_depth_bar(ax_bar, algo_depths, color_map, y_lim)

            # Draw connection lines from curve endpoints to depth bar
            for alg, x_end, y_end, depth in endpoint_info:
                # Use ConnectionPatch to draw lines across axes
                # Start point: curve endpoint in main plot (data coords)
                # End point: depth position on bar (data coords)
                con = ConnectionPatch(
                    xyA=(x_end, y_end),  # start in main plot
                    xyB=(0.0, depth),    # end on depth bar (left edge)
                    coordsA="data",
                    coordsB="data",
                    axesA=ax,
                    axesB=ax_bar,
                    color=color_map[alg],
                    alpha=0.7,
                    lw=1.0,
                    clip_on=False,
                )
                fig.add_artist(con)

                # Draw marker on depth bar
                ax_bar.scatter(
                    [0.5], [depth],
                    s=30,
                    c=color_map[alg],
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=10,
                    clip_on=False,
                )

    # Post-process y-axis formatting for specific subplots
    for idx, ax in all_axes:
        if idx == 3:
            # For f25 (values around 1-6), use plain numbers
            # Use 1 and 8 as ticks (8 is closer to actual data range than 10)
            ticks = [1, 8]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks])
            # Disable offset text (the ×10^n label)
            ax.yaxis.get_offset_text().set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.tick_params(axis="y", labelsize=6)
            # Also clear any minor ticks that might have labels
            ax.set_yticks([], minor=True)
        elif idx == 1:
            # For f13 (top-right), only show ticks within reasonable range of data
            ymin, ymax = ax.get_ylim()
            import math
            # Use ceiling for min to avoid showing ticks far below data
            log_min = math.ceil(math.log10(max(ymin, 1e-10)))
            log_max = math.ceil(math.log10(max(ymax, 1e-10)))
            ticks = [10**i for i in range(log_min, log_max + 1)]
            ax.set_yticks(ticks)
            ax.tick_params(axis="y", labelsize=6)

    # Add figure title only if explicitly provided
    if args.title:
        fig.suptitle(str(args.title), y=0.98, fontsize=11)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(algorithms), 5),
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.98),
    )

    out_prefix = os.path.abspath(str(args.output_prefix))
    save_figure(fig, out_prefix)
    plt.close(fig)

    print("Wrote:", repo_relpath(out_prefix + ".png"))
    print("Wrote:", repo_relpath(out_prefix + ".pdf"))


if __name__ == "__main__":
    main()
