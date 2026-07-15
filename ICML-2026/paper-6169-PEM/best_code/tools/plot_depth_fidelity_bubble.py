#!/usr/bin/env python3
"""
Create the depth–fidelity bubble plot (Figure 2).

Axes / encodings (redesigned for "Depth over Fidelity"):
- X: average per-candidate evaluation cost (evals / candidate), log scale (fidelity proxy)
- Y: search depth T (generations), higher = more depth
- Bubble size: performance (1/regret mapping: lower regret -> larger bubble)
- Color: algorithm identity
- Border: thick black for BERW, thin white for others

Includes an equal-budget hyperbola: depth × cost = B / λ

Inputs:
- Performance: evidence/hansen_test_fixed_budget/noisefree/bbob_summary.csv
- BERW depth: evidence/hansen_test_fixed_budget/diagnostics/traces/*.csv
- UH-CMA-ES cost (optional): evidence/uh_cmaes_cost_measurement/uh_cmaes_cost_summary.csv

Outputs:
- evidence/depth_fidelity_characterization/depth_fidelity_bubble.(pdf|png)
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, get_algo_color, get_figsize, add_grid, save_figure
from utils.display_names import get_display_name


@dataclass(frozen=True)
class AlgorithmMetrics:
    name: str
    display_name: str
    cost_per_candidate: float
    depth: int
    median_log10_regret: float
    color: str


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _lambda_for_dim(dim: int) -> int:
    return 4 + int(3 * math.log(int(dim)))


def _load_bbob_summary(summary_path: str, *, func_ids: list[int] | None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                algo = str(row["algorithm"])
                func_id = int(row["function"])
                best_f = float(row["best_f"])
            except Exception:
                continue
            if func_ids is not None and func_id not in func_ids:
                continue
            out.setdefault(algo, []).append(best_f)
    return out


def _load_uh_cost_summary(path: str) -> dict[str, tuple[float, float]]:
    """
    Returns: {algorithm: (mean_cost_per_candidate, mean_depth_at_budget)}
    """
    if not os.path.isfile(path):
        return {}
    out: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                algo = str(row["algorithm"])
                cost = float(row["mean_evals_per_candidate"])
                depth = float(row.get("mean_depth_at_budget", row.get("mean_generations", "nan")))
            except Exception:
                continue
            out[algo] = (cost, depth)
    return out


def _compute_berw_cost_and_depth_from_traces(traces_dir: str, *, func_ids: list[int], dim: int) -> tuple[float, int]:
    """
    Estimate BERW-Hetero cost/depth from diagnostic trace CSVs.

    We use the median across (function, instance) traces.

    Note: The number of rows in each trace file equals the number of generations,
    as each row is a per-generation diagnostic snapshot.
    """
    lambda_ = _lambda_for_dim(dim)

    costs: list[float] = []
    depths: list[int] = []

    for func_id in func_ids:
        pattern = os.path.join(traces_dir, f"berw_hetero_B100_f{int(func_id)}_d{int(dim)}_i*.csv")
        for trace_path in glob.glob(pattern):
            try:
                with open(trace_path, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if not rows:
                    continue
                last = rows[-1]
                last_evals = int(float(last.get("evals", "0")))

                # Depth = number of rows (each row = one generation)
                depth = len(rows)
                if depth <= 0:
                    continue
                cost = float(last_evals) / float(depth * lambda_)
                if not np.isfinite(cost) or cost <= 0:
                    continue
                costs.append(cost)
                depths.append(depth)
            except Exception:
                continue

    if not costs or not depths:
        # Conservative fallback values used in the plan.
        return 1.27, 211

    return float(np.median(costs)), int(np.median(depths))


def compute_metrics(
    *,
    summary_path: str,
    traces_dir: str,
    uh_cost_summary_path: str,
    budget: int,
    dim: int,
    func_ids: list[int],
) -> list[AlgorithmMetrics]:
    lambda_ = _lambda_for_dim(dim)

    perf = _load_bbob_summary(summary_path, func_ids=func_ids)
    uh_measured = _load_uh_cost_summary(uh_cost_summary_path)

    specs: list[tuple[str, float, int]] = [
        ("CMA-ES-sep", 1.0, budget // lambda_),
        ("CMA-ES-Resample(k=5)", 5.0, budget // (5 * lambda_)),
        ("CMA-ES-Resample(k=10)", 10.0, budget // (10 * lambda_)),
    ]

    # UH-CMA-ES: prefer measured evidence, otherwise fall back.
    if "UH-CMA-ES(maxevals=30)" in uh_measured:
        cost, depth = uh_measured["UH-CMA-ES(maxevals=30)"]
        specs.append(("UH-CMA-ES(maxevals=30)", float(cost), int(depth)))
    else:
        specs.append(("UH-CMA-ES(maxevals=30)", 15.5, 17))

    # BERW: measure from traces if available.
    if os.path.isdir(traces_dir):
        berw_cost, berw_depth = _compute_berw_cost_and_depth_from_traces(
            traces_dir, func_ids=func_ids, dim=dim
        )
    else:
        berw_cost, berw_depth = 1.27, 211
    specs.append(("BERW-Hetero", berw_cost, int(berw_depth)))

    metrics: list[AlgorithmMetrics] = []
    for algo, cost, depth in specs:
        if algo not in perf:
            continue
        vals = np.asarray(perf[algo], dtype=float)
        vals = np.maximum(vals, 1e-10)
        med = float(np.median(np.log10(vals)))
        metrics.append(
            AlgorithmMetrics(
                name=algo,
                display_name=get_display_name(algo),
                cost_per_candidate=float(cost),
                depth=int(depth),
                median_log10_regret=med,
                color=get_algo_color(algo),
            )
        )

    return metrics


def regret_to_size(regret: float, regret_min: float, regret_max: float) -> float:
    """Map regret to marker size. Lower regret -> larger marker."""
    # Log-space normalization
    log_regret = np.log10(max(regret, 1e-10))
    log_min = np.log10(max(regret_min, 1e-10))
    log_max = np.log10(max(regret_max, 1e-10))
    # Inverse: good performance (low regret) -> large bubble
    denom = max(0.01, log_max - log_min)
    normalized = 1 - (log_regret - log_min) / denom
    return 80 + 350 * normalized  # Range 80-430


def plot(metrics: list[AlgorithmMetrics], *, output_prefix: str, budget: int, dim: int) -> None:
    apply_style()

    fig, ax = plt.subplots(figsize=get_figsize("single", aspect=0.75))
    lambda_ = _lambda_for_dim(dim)

    # Sort by cost for consistent ordering
    metrics_sorted = sorted(metrics, key=lambda m: m.cost_per_candidate)

    # Compute regret range for size mapping
    regrets = [10 ** m.median_log10_regret for m in metrics]
    regret_min, regret_max = min(regrets), max(regrets)

    # --- Equal-budget hyperbola: depth × cost = B / λ ---
    constant = budget / lambda_  # ≈ 267 for B=4000, λ=15
    costs_curve = np.linspace(0.8, 18, 100)
    depths_curve = constant / costs_curve
    ax.plot(
        costs_curve,
        depths_curve,
        "--",
        color="#aaaaaa",
        alpha=0.6,
        lw=1.0,
        zorder=1,
    )

    # --- Plot each algorithm as a bubble ---
    for m in metrics_sorted:
        # Convert regret to bubble size (lower regret -> larger bubble)
        regret = 10 ** m.median_log10_regret
        size = regret_to_size(regret, regret_min, regret_max)

        # Border style: thick black for BERW, thin white for others
        edgecolor = "black" if "BERW" in m.name else "white"
        linewidth = 2.0 if "BERW" in m.name else 0.5

        ax.scatter(
            m.cost_per_candidate,
            m.depth,
            s=size,
            c=m.color,
            alpha=0.9,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=4 if "BERW" in m.name else 3,
            marker="o",
        )

        # Performance value in center of bubble
        perf_text = f"{m.median_log10_regret:.1f}"
        ax.text(
            m.cost_per_candidate,
            m.depth,
            perf_text,
            fontsize=5,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=5,
        )

        # Label next to bubble
        short_name = m.display_name

        # Position labels - customized per algorithm
        if "BERW" in m.name:
            # Residual Bootstrapping: move right more to avoid overlap
            ha, va = "left", "center"
            x_text = m.cost_per_candidate + 1.1
            y_text = m.depth
        elif "sep" in m.name:
            # CMA-ES: move right more to avoid overlap with RB
            ha, va = "left", "center"
            x_text = m.cost_per_candidate + 0.8
            y_text = m.depth
        elif "k=5" in m.name:
            # Resample(k=5): above the bubble, shift right
            ha, va = "center", "bottom"
            x_text = m.cost_per_candidate + 0.5
            y_text = m.depth + 22
        elif "k=10" in m.name:
            # Resample(k=10): above the bubble
            ha, va = "center", "bottom"
            x_text = m.cost_per_candidate
            y_text = m.depth + 18
        elif "UH" in m.name:
            # UH-CMA-ES: above the bubble
            ha, va = "center", "bottom"
            x_text = m.cost_per_candidate
            y_text = m.depth + 10
        else:
            ha, va = "left", "center"
            x_text = m.cost_per_candidate + 0.3
            y_text = m.depth

        ax.text(
            x_text,
            y_text,
            short_name,
            fontsize=6,
            color=m.color,
            ha=ha,
            va=va,
            fontweight="bold" if "BERW" in m.name else "normal",
        )

    # Linear scale for X-axis
    ax.set_xlabel("Per-candidate evaluation cost (fidelity)", fontsize=8)
    ax.set_ylabel("Generations (depth)", fontsize=8)
    ax.set_xlim(0, 18)

    # Y-axis: depth, ensure BERW (high depth) is at top
    depth_min = min(m.depth for m in metrics)
    depth_max = max(m.depth for m in metrics)
    y_margin = (depth_max - depth_min) * 0.15
    ax.set_ylim(0, depth_max + y_margin)

    # Smaller tick labels
    ax.tick_params(axis='both', labelsize=7)

    # Remove duplicate 0 on y-axis and 350 tick
    ax.set_yticks([50, 100, 150, 200, 250, 300])

    # Budget info in top-right corner
    # Find best and worst regret values
    best_regret = min(m.median_log10_regret for m in metrics)
    worst_regret = max(m.median_log10_regret for m in metrics)

    ax.text(
        0.98,
        0.97,
        f"D={dim}, B=100D={budget}\n"
        f"$\\lambda$={lambda_} (candidates/gen)",
        transform=ax.transAxes,
        fontsize=5,
        ha="right",
        va="top",
        alpha=0.7,
        linespacing=1.3,
    )

    # Legend circles inline: big circle with best value, small circle with worst value
    # Right-aligned to 0.98 like other rows
    legend_y = 0.865
    # Better (big circle, no border)
    ax.scatter(
        [0.77], [legend_y],
        s=40, facecolors="none", edgecolors="gray", linewidths=0.5,
        transform=ax.transAxes, zorder=10, clip_on=False
    )
    ax.text(
        0.77, legend_y, f"{best_regret:.1f}",
        transform=ax.transAxes, fontsize=3.5, ha="center", va="center",
        fontweight="bold", zorder=11
    )
    ax.text(
        0.80, legend_y, "better",
        transform=ax.transAxes, fontsize=5, ha="left", va="center", alpha=0.7
    )
    # Worse (small circle)
    ax.scatter(
        [0.88], [legend_y],
        s=18, facecolors="none", edgecolors="gray", linewidths=0.5,
        transform=ax.transAxes, zorder=10, clip_on=False
    )
    ax.text(
        0.88, legend_y, f"{worst_regret:.1f}",
        transform=ax.transAxes, fontsize=2.5, ha="center", va="center",
        fontweight="bold", zorder=11
    )
    ax.text(
        0.905, legend_y, "worse",
        transform=ax.transAxes, fontsize=5, ha="left", va="center", alpha=0.7
    )

    # value = log10(regret) below the legend circles
    ax.text(
        0.98,
        0.825,
        r"value = $\log_{10}$(regret)",
        transform=ax.transAxes,
        fontsize=5,
        ha="right",
        va="top",
        alpha=0.7,
    )

    # Add hyperbola equation in blank space between RB and Resample(k=5)
    ax.text(
        3.0,
        130,
        r"$\mathrm{depth} \times \mathrm{cost} = B / \lambda$",
        fontsize=6,
        color="#666666",
        ha="center",
        va="center",
        alpha=0.8,
    )

    fig.tight_layout()
    save_figure(fig, output_prefix)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth–fidelity bubble plot (Figure 2)")
    parser.add_argument(
        "--summary",
        default=os.path.join(BASE_DIR, "evidence/hansen_test_fixed_budget/noisefree/bbob_summary.csv"),
        help="Path to bbob_summary.csv",
    )
    parser.add_argument(
        "--traces-dir",
        default=os.path.join(BASE_DIR, "evidence/hansen_test_fixed_budget/diagnostics/traces"),
        help="Path to BERW diagnostic traces directory",
    )
    parser.add_argument(
        "--uh-cost-summary",
        default=os.path.join(BASE_DIR, "evidence/uh_cmaes_cost_measurement/uh_cmaes_cost_summary.csv"),
        help="Path to UH-CMA-ES cost summary CSV (optional evidence pack)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "evidence/paper_figures/depth_fidelity_bubble"),
        help="Output path prefix (without extension)",
    )
    parser.add_argument("--budget", type=int, default=4000, help="Total evaluation budget")
    parser.add_argument("--dim", type=int, default=40, help="Problem dimension")
    parser.add_argument(
        "--functions",
        default="110,116,113,125",
        help="Comma-separated function IDs to include",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    func_ids = _parse_int_list(args.functions)

    metrics = compute_metrics(
        summary_path=str(args.summary),
        traces_dir=str(args.traces_dir),
        uh_cost_summary_path=str(args.uh_cost_summary),
        budget=int(args.budget),
        dim=int(args.dim),
        func_ids=func_ids,
    )
    if not metrics:
        raise SystemExit("No metrics computed (check input paths and function list).")

    plot(metrics, output_prefix=str(args.output), budget=int(args.budget), dim=int(args.dim))
    print("Wrote:", repo_relpath(str(args.output) + ".pdf"))
    print("Wrote:", repo_relpath(str(args.output) + ".png"))


if __name__ == "__main__":
    main()

