#!/usr/bin/env python3
"""Generate LaTeX report from SBC/Coverage diagnostics results.

Reads metrics JSON files produced by run_sbc_coverage.py and generates
a standalone LaTeX report with tables and figure inclusions.

Usage:
    python scripts/generate_sbc_report.py \
        --results_dir results/comparison_benchmark/rebuttal_plots/sbc_coverage \
        --tasks high_dim_gaussian \
        --seeds 33 43 53 \
        --ncals 10 50 200 1000
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TASK_DISPLAY = {
    "high_dim_gaussian": "High-Dimensional Gaussian",
    "pendulum": "Pendulum",
    "wind_tunnel": "Wind Tunnel",
    "light_tunnel": "Light Tunnel",
}

METHOD_ORDER = ["NPE", "FMPE", "FMCPE", "MF-NPE", "RoPE", "DPE"]


def load_metrics(results_dir: Path, task: str, seeds: list[int], ncals: list[int]):
    """Load all metrics JSONs for a task, grouped by ncal.

    Returns:
        {ncal: {method: {"acauc": [vals], "mce": [vals], "coverage": {alpha: [vals]}}}}
    """
    data = {}
    for ncal in ncals:
        data[ncal] = defaultdict(lambda: {"acauc": [], "mce": [], "coverage": defaultdict(list)})
        for seed in seeds:
            path = results_dir / task / f"metrics_seed{seed}_ncal{ncal}.json"
            if not path.exists():
                print(f"  Warning: missing {path}")
                continue
            with open(path) as f:
                metrics = json.load(f)
            for method, m in metrics.items():
                data[ncal][method]["acauc"].append(m["acauc"])
                data[ncal][method]["mce"].append(m["mce"])
                for alpha, cov in m["empirical_coverage"].items():
                    data[ncal][method]["coverage"][alpha].append(cov)
    return data


def fmt(vals, bold_best=False, lower_is_better=True):
    """Format mean ± std, optionally bolding."""
    if not vals:
        return "---"
    mean = np.mean(vals)
    if len(vals) > 1:
        std = np.std(vals, ddof=1)
        s = f"{mean:.3f} $\\pm$ {std:.3f}"
    else:
        s = f"{mean:.3f}"
    if bold_best:
        s = f"\\textbf{{{s}}}"
    return s


def generate_acauc_mce_table(data, task: str):
    """Generate LaTeX table of ACAUC and MCE across ncals."""
    ncals = sorted(data.keys())
    methods = [m for m in METHOD_ORDER if any(m in data[n] for n in ncals)]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{ACAUC and MCE for " + TASK_DISPLAY.get(task, task) + r" across calibration set sizes. " +
                 r"ACAUC: 0 = perfect calibration, ${>}0$ = overconfident, ${<}0$ = underconfident. " +
                 r"MCE: mean absolute calibration error (lower is better). " +
                 r"Best MCE per $N_\text{cal}$ in \textbf{bold}.}")
    lines.append(r"\label{tab:acauc_mce_" + task + "}")

    # Two sub-tables: ACAUC and MCE
    ncal_cols = " ".join(["c"] * len(ncals))
    lines.append(r"\begin{tabular}{l " + ncal_cols + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join([f"$N_{{\\text{{cal}}}}={n}$" for n in ncals]) + r" \\"
    lines.append(header)

    # ACAUC section
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(ncals) + 1) + r"}{c}{\textit{ACAUC} ($\downarrow$ closer to 0 is better)} \\")
    lines.append(r"\midrule")

    for method in methods:
        # Find best |ACAUC| per ncal
        best_abs_acauc = {}
        for ncal in ncals:
            abs_vals = {}
            for m in methods:
                if m in data[ncal] and data[ncal][m]["acauc"]:
                    abs_vals[m] = abs(np.mean(data[ncal][m]["acauc"]))
            if abs_vals:
                best_abs_acauc[ncal] = min(abs_vals, key=abs_vals.get)

        cells = []
        for ncal in ncals:
            if method in data[ncal] and data[ncal][method]["acauc"]:
                is_best = best_abs_acauc.get(ncal) == method
                cells.append(fmt(data[ncal][method]["acauc"], bold_best=is_best))
            else:
                cells.append("---")
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")

    # MCE section
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(ncals) + 1) + r"}{c}{\textit{MCE} ($\downarrow$ lower is better)} \\")
    lines.append(r"\midrule")

    for method in methods:
        best_mce = {}
        for ncal in ncals:
            mce_vals = {}
            for m in methods:
                if m in data[ncal] and data[ncal][m]["mce"]:
                    mce_vals[m] = np.mean(data[ncal][m]["mce"])
            if mce_vals:
                best_mce[ncal] = min(mce_vals, key=mce_vals.get)

        cells = []
        for ncal in ncals:
            if method in data[ncal] and data[ncal][method]["mce"]:
                is_best = best_mce.get(ncal) == method
                cells.append(fmt(data[ncal][method]["mce"], bold_best=is_best))
            else:
                cells.append("---")
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_coverage_table(data, task: str, ncal: int):
    """Generate LaTeX table of empirical coverage at each alpha level for a given ncal."""
    methods = [m for m in METHOD_ORDER if m in data[ncal]]
    if not methods:
        return ""

    alphas = sorted(data[ncal][methods[0]]["coverage"].keys(), key=float)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Empirical coverage for " + TASK_DISPLAY.get(task, task) +
                 f", $N_{{\\text{{cal}}}}={ncal}$. " +
                 r"Each cell shows the empirical coverage at the given expected level $\alpha$. " +
                 r"Perfect calibration: empirical $= \alpha$.}")
    lines.append(r"\label{tab:coverage_" + task + f"_ncal{ncal}" + "}")

    alpha_cols = " ".join(["c"] * len(alphas))
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l " + alpha_cols + "}")
    lines.append(r"\toprule")
    header = r"Method & " + " & ".join([f"${float(a):.2f}$" for a in alphas]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in methods:
        cells = []
        for alpha in alphas:
            vals = data[ncal][method]["coverage"].get(alpha, [])
            if vals:
                mean = np.mean(vals)
                expected = float(alpha)
                # Color code: close to expected = good
                diff = abs(mean - expected)
                if diff < 0.05:
                    cells.append(f"\\textbf{{{mean:.2f}}}")
                else:
                    cells.append(f"{mean:.2f}")
            else:
                cells.append("---")
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_figure_includes(results_dir: Path, task: str, seeds: list[int], ncals: list[int]):
    """Generate LaTeX figure includes for SBC and coverage plots."""
    lines = []

    # SBC histograms
    for ncal in ncals:
        # Use first seed for the representative plot, or find which exist
        for seed in seeds:
            sbc_path = results_dir / task / f"sbc_seed{seed}_ncal{ncal}.pdf"
            if sbc_path.exists():
                rel_path = sbc_path.relative_to(results_dir)
                lines.append(r"\begin{figure}[htbp]")
                lines.append(r"\centering")
                lines.append(r"\includegraphics[width=\textwidth]{" + str(rel_path) + "}")
                lines.append(r"\caption{SBC rank histograms for " + TASK_DISPLAY.get(task, task) +
                             f", $N_{{\\text{{cal}}}}={ncal}$, seed={seed}. " +
                             r"Uniform histograms indicate well-calibrated posteriors. " +
                             r"Gray band: 95\% confidence interval under uniformity.}")
                lines.append(r"\label{fig:sbc_" + task + f"_ncal{ncal}_seed{seed}" + "}")
                lines.append(r"\end{figure}")
                lines.append("")
                break  # Only include one seed per ncal

    # Coverage curves
    for ncal in ncals:
        for seed in seeds:
            cov_path = results_dir / task / f"coverage_seed{seed}_ncal{ncal}.pdf"
            if cov_path.exists():
                rel_path = cov_path.relative_to(results_dir)
                lines.append(r"\begin{figure}[htbp]")
                lines.append(r"\centering")
                lines.append(r"\includegraphics[width=0.7\textwidth]{" + str(rel_path) + "}")
                lines.append(r"\caption{Expected vs.\ empirical coverage for " + TASK_DISPLAY.get(task, task) +
                             f", $N_{{\\text{{cal}}}}={ncal}$, seed={seed}. " +
                             r"Methods close to the diagonal are better calibrated.}")
                lines.append(r"\label{fig:coverage_" + task + f"_ncal{ncal}_seed{seed}" + "}")
                lines.append(r"\end{figure}")
                lines.append("")
                break

    return "\n".join(lines)


def generate_report(results_dir: Path, tasks: list[str], seeds: list[int], ncals: list[int],
                    output_path: Path):
    """Generate the full LaTeX report."""
    sections = []

    for task in tasks:
        data = load_metrics(results_dir, task, seeds, ncals)
        task_display = TASK_DISPLAY.get(task, task)

        sections.append(f"\\section{{{task_display}}}")
        sections.append("")

        # Summary table
        sections.append("\\subsection{ACAUC and MCE Summary}")
        sections.append(generate_acauc_mce_table(data, task))
        sections.append("")

        # Coverage tables for key ncals
        sections.append("\\subsection{Empirical Coverage Tables}")
        for ncal in ncals:
            table = generate_coverage_table(data, task, ncal)
            if table:
                sections.append(table)
                sections.append("")

        # Figures
        sections.append("\\subsection{SBC Rank Histograms}")
        sections.append(generate_figure_includes(results_dir, task, seeds, ncals))
        sections.append(r"\clearpage")
        sections.append("")

    body = "\n".join(sections)

    latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{float}

\title{SBC and Coverage Diagnostics Report\\
\large FMCPE Rebuttal — Experiment 3}
\author{Automated Report}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report presents Simulation-Based Calibration (SBC) rank histograms and
empirical coverage curves for all posterior estimation methods on the benchmark
tasks. The diagnostics use the true Data-Generating Process (DGP), i.e.,
observations $y$ (not the misspecified $x$), to evaluate whether each method
produces well-calibrated posteriors under the correct model.

\textbf{Key metrics:}
\begin{itemize}
    \item \textbf{ACAUC} (Area-weighted Calibration Under the Curve): 0 indicates
    perfect calibration; positive values indicate overconfidence (credible intervals
    too narrow); negative values indicate underconfidence (intervals too wide).
    \item \textbf{MCE} (Mean Calibration Error): average absolute deviation between
    expected and empirical coverage levels. Lower is better.
\end{itemize}
\end{abstract}

""" + body + r"""
\end{document}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX SBC/Coverage report")
    parser.add_argument("--results_dir", required=True, help="Path to sbc_coverage results")
    parser.add_argument("--tasks", nargs="+", default=["high_dim_gaussian"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 43, 53])
    parser.add_argument("--ncals", nargs="+", type=int, default=[10, 50, 200, 1000])
    parser.add_argument("--output", default=None, help="Output .tex path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.output is None:
        output_path = results_dir / "report_sbc_coverage.tex"
    else:
        output_path = Path(args.output)

    generate_report(results_dir, args.tasks, args.seeds, args.ncals, output_path)


if __name__ == "__main__":
    main()
