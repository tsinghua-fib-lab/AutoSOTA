#!/usr/bin/env python3
"""
Generate Probe-and-Switch vs competitors comparison table for paper.

Combines COCO benchmark data with external task data to produce a comprehensive
LaTeX table showing pairwise win/loss comparisons.

External tasks are aligned with the transfer heatmap (figure3c_transfer).

Output: evidence/paper_tables/table_probeswitch_comparison.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from _project import BASE_DIR, repo_relpath


def sign_test_p_two_sided(wins: int, n: int) -> float:
    """Compute two-sided p-value for sign test."""
    if n <= 0:
        return float("nan")
    wins = int(wins)
    n = int(n)
    denom = 1 << n  # 2**n
    lo = sum(math.comb(n, k) for k in range(0, wins + 1))
    hi = sum(math.comb(n, k) for k in range(wins, n + 1))
    p = 2.0 * min(lo / denom, hi / denom)
    return float(min(1.0, p))


def significance_stars(p: float) -> str:
    """Return significance stars for p-value (disabled)."""
    return ""


@dataclass
class ComparisonResult:
    """Result of a pairwise comparison."""
    wins: int
    losses: int
    ties: int
    p_value: float

    @property
    def n_compared(self) -> int:
        return self.wins + self.losses + self.ties

    def to_latex(self) -> str:
        """Format as LaTeX cell content."""
        if self.wins == 0 and self.losses == 0:
            return "---"
        stars = significance_stars(self.p_value)
        base = f"{self.wins}/{self.losses}"
        # Bold when wins > losses (regardless of significance)
        if self.wins > self.losses:
            base = f"\\textbf{{{base}}}"
        if stars:
            return f"{base}${stars}$"
        return base


def load_coco_pairwise_data(csv_path: str) -> Dict[Tuple[str, str], ComparisonResult]:
    """Load COCO pairwise comparison data."""
    results = {}
    if not os.path.exists(csv_path):
        return results

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo_a = row["algo_a"]
            algo_b = row["algo_b"]
            wins_a = int(row["wins_a"])
            wins_b = int(row["wins_b"])
            ties = int(row.get("ties", 0))
            p_val = float(row["p_two_sided"]) if row["p_two_sided"] != "nan" else float("nan")

            # Store both directions for easy lookup
            results[(algo_a, algo_b)] = ComparisonResult(wins_a, wins_b, ties, p_val)
            results[(algo_b, algo_a)] = ComparisonResult(wins_b, wins_a, ties, p_val)

    return results


def load_runs_csv(csv_path: str, metric: str = "post_median") -> Dict[str, Dict[int, float]]:
    """Load runs.csv and return {algorithm: {seed: metric_value}}."""
    data: Dict[str, Dict[int, float]] = defaultdict(dict)

    if not os.path.exists(csv_path):
        return data

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algorithm"]
            seed = int(row["seed"])
            # Try different metric column names
            if metric in row:
                val = float(row[metric])
            elif "post_true" in row:
                val = float(row["post_true"])
            elif "post_median" in row:
                val = float(row["post_median"])
            elif "post_mean" in row:
                val = float(row["post_mean"])
            else:
                continue
            data[algo][seed] = val

    return data


def compute_pairwise_comparison(
    data: Dict[str, Dict[int, float]],
    algo_a: str,
    algo_b: str,
    lower_is_better: bool = True,
) -> ComparisonResult:
    """Compute pairwise comparison between two algorithms."""
    wins_a = 0
    wins_b = 0
    ties = 0

    # Get common seeds
    seeds_a = set(data.get(algo_a, {}).keys())
    seeds_b = set(data.get(algo_b, {}).keys())
    common_seeds = seeds_a & seeds_b

    for seed in common_seeds:
        val_a = data[algo_a][seed]
        val_b = data[algo_b][seed]

        if abs(val_a - val_b) < 1e-12:
            ties += 1
        elif lower_is_better:
            if val_a < val_b:
                wins_a += 1
            else:
                wins_b += 1
        else:
            if val_a > val_b:
                wins_a += 1
            else:
                wins_b += 1

    n = wins_a + wins_b
    p_val = sign_test_p_two_sided(wins_a, n)

    return ComparisonResult(wins_a, wins_b, ties, p_val)


def merge_runs_data(*data_dicts: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, float]]:
    """Merge multiple runs data dictionaries."""
    merged: Dict[str, Dict[int, float]] = defaultdict(dict)
    for data in data_dicts:
        for algo, seeds in data.items():
            for seed, val in seeds.items():
                merged[algo][seed] = val
    return merged


@dataclass
class ExternalTask:
    """Configuration for an external task."""
    name: str
    display_name: str
    main_dir: str
    transfer_dir: str
    metric: str
    ps_algo: str  # ProbeSwitch algorithm name for this task


def main():
    parser = argparse.ArgumentParser(description="Generate ProbeSwitch comparison table")
    parser.add_argument("--output", default="", help="Output LaTeX file path")
    parser.add_argument("--probeswitch-robust-algo", default="ProbeSwitch-MR-Robust(t=0.12)",
                        help="ProbeSwitch algorithm name in external task data")
    args = parser.parse_args()

    evidence_dir = os.path.join(BASE_DIR, "evidence")

    # ===== COCO DATA (Multiple Dimensions) =====
    # Each dimension uses its optimal threshold
    coco_dimensions = [
        {
            "dim": 10,
            "display": "d = 10",
            "ps_algo": "ProbeSwitch-MR(t=0.46)",
            "csv_path": os.path.join(evidence_dir, "probeswitch_optimized_d10", "pairwise_sign_test.csv"),
        },
        {
            "dim": 20,
            "display": "d = 20",
            "ps_algo": "ProbeSwitch-MR(t=0.38)",
            "csv_path": os.path.join(evidence_dir, "probeswitch_optimized_d20", "pairwise_sign_test.csv"),
        },
        {
            "dim": 40,
            "display": "d = 40",
            "ps_algo": "ProbeSwitch-MR(t=0.12)",
            "csv_path": os.path.join(evidence_dir, "probeswitch_optimized_d40", "pairwise_sign_test.csv"),
        },
    ]

    # ===== EXTERNAL TASK DATA =====
    external_tasks = [
        ExternalTask(
            name="lqr",
            display_name="LQR",
            main_dir="application_lqr_combined",
            transfer_dir="",
            metric="post_median",
            ps_algo="ProbeSwitch-MR-Robust(t=0.12)",
        ),
        ExternalTask(
            name="hpo_breast_cancer",
            display_name="Breast Cancer",
            main_dir="application_hpo_breast_cancer_budget10",
            transfer_dir="",
            metric="post_median",
            ps_algo="ProbeSwitch-MR-Robust(t=0.12)",
        ),
        ExternalTask(
            name="hpo_digits",
            display_name="Digits",
            main_dir="application_hpo_digits0_budget10",
            transfer_dir="",
            metric="post_median",
            ps_algo="ProbeSwitch-MR-Robust(t=0.12)",
        ),
        ExternalTask(
            name="cartpole_ht",
            display_name="CartPole-HT",
            main_dir="application_cartpole_ht_optimized_t021",
            transfer_dir="",
            metric="post_true",
            ps_algo="ProbeSwitch-MR-Robust(t=0.21)",
        ),
        ExternalTask(
            name="cartpole",
            display_name="CartPole",
            main_dir="application_rl_cartpole_budget3",
            transfer_dir="",
            metric="post_true",
            ps_algo="ProbeSwitch-MR-Robust(t=0.12)",
        ),
        ExternalTask(
            name="pendulum",
            display_name="Pendulum",
            main_dir="application_rl_pendulum_full",
            transfer_dir="",
            metric="post_true",
            ps_algo="ProbeSwitch-MR-Robust(t=0.12)",
        ),
    ]

    # Algorithm mappings (display name -> possible names in data)
    # Ordered by win rate descending for visual impact
    competitors = [
        ("Res.(10)", ["CMA-ES-Resample(k=10)"]),
        ("Res.(5)", ["CMA-ES-Resample(k=5)"]),
        ("UH-CMA-ES", ["UH-CMA-ES(maxevals=30)", "UH-CMA-ES(30)"]),
        ("CMA-ES", ["CMA-ES-sep"]),
        ("RB", ["BERW-Hetero", "BERW-HeteroRobust"]),
    ]

    # ===== BUILD TABLE DATA =====
    table_data = {}  # {task_name: {competitor_display: ComparisonResult}}

    # COCO rows (one per dimension)
    for coco_dim in coco_dimensions:
        csv_path = coco_dim["csv_path"]
        ps_algo = coco_dim["ps_algo"]
        display_name = coco_dim["display"]

        if os.path.exists(csv_path):
            coco_results = load_coco_pairwise_data(csv_path)
            print(f"Loaded COCO {display_name} data from: {repo_relpath(csv_path)}")
        else:
            coco_results = {}
            print(f"WARNING: COCO {display_name} data not found: {csv_path}")

        dim_row = {}
        for comp_display, comp_names in competitors:
            result = ComparisonResult(0, 0, 0, float("nan"))
            for comp_name in comp_names:
                key = (ps_algo, comp_name)
                if key in coco_results:
                    result = coco_results[key]
                    break
            dim_row[comp_display] = result
        table_data[display_name] = dim_row

    # External task rows
    tasks_with_data = []
    for task in external_tasks:
        main_csv = os.path.join(evidence_dir, task.main_dir, "runs.csv")
        transfer_csv = os.path.join(evidence_dir, task.transfer_dir, "runs.csv") if task.transfer_dir else ""

        # Load and merge data
        main_data = load_runs_csv(main_csv, task.metric)
        transfer_data = load_runs_csv(transfer_csv, task.metric) if transfer_csv else {}
        merged_data = merge_runs_data(main_data, transfer_data)

        # Check if ProbeSwitch data exists
        ps_algo = task.ps_algo
        has_ps_data = ps_algo in merged_data

        task_row = {}
        for comp_display, comp_names in competitors:
            result = ComparisonResult(0, 0, 0, float("nan"))
            for comp_name in comp_names:
                if has_ps_data and comp_name in merged_data:
                    result = compute_pairwise_comparison(merged_data, ps_algo, comp_name)
                    break
            task_row[comp_display] = result
        table_data[task.display_name] = task_row

        if has_ps_data:
            tasks_with_data.append(task)
        else:
            print(f"NOTE: No ProbeSwitch data for {task.display_name} (needs experiment)")

    # ===== COMPUTE TOTALS =====
    total_row = {}
    for comp_display, _ in competitors:
        total_wins = 0
        total_losses = 0
        total_ties = 0
        for task_name, task_row in table_data.items():
            if comp_display in task_row:
                result = task_row[comp_display]
                total_wins += result.wins
                total_losses += result.losses
                total_ties += result.ties
        n = total_wins + total_losses
        p_val = sign_test_p_two_sided(total_wins, n)
        total_row[comp_display] = ComparisonResult(total_wins, total_losses, total_ties, p_val)
    table_data["Total"] = total_row

    # ===== GENERATE LATEX TABLE =====
    comp_headers = [c[0] for c in competitors]

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Probe-and-Switch vs competitors: pairwise comparison across COCO and external tasks.}")
    latex_lines.append(r"\label{tab:probeswitch_comparison}")
    latex_lines.append(r"\small")

    # Column spec
    col_spec = "l " + " ".join(["c"] * len(comp_headers))
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\toprule")

    # Header row
    header = "                    & " + " & ".join([f"vs {h}" for h in comp_headers]) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")

    # COCO section (multiple dimensions)
    latex_lines.append(r"\multicolumn{" + str(len(comp_headers) + 1) + r"}{l}{\textit{COCO benchmark (225 instances, B = 200d)}} \\")
    for coco_dim in coco_dimensions:
        display_name = coco_dim["display"]
        cells = [table_data[display_name][h].to_latex() for h in comp_headers]
        latex_lines.append(r"\quad " + display_name.ljust(13) + " & " + " & ".join(cells) + r" \\")
    latex_lines.append(r"\midrule")

    # External tasks section
    latex_lines.append(r"\multicolumn{" + str(len(comp_headers) + 1) + r"}{l}{\textit{External tasks (50 seeds each)}} \\")
    for task in external_tasks:
        cells = [table_data[task.display_name][h].to_latex() for h in comp_headers]
        latex_lines.append(r"\quad " + task.display_name.ljust(13) + " & " + " & ".join(cells) + r" \\")

    latex_lines.append(r"\midrule")

    # Total row
    total_cells = [table_data["Total"][h].to_latex() for h in comp_headers]
    latex_lines.append(r"\textbf{Total W/L}  & " + " & ".join(total_cells) + r" \\")

    # Win Rate row
    win_rate_cells = []
    for h in comp_headers:
        result = table_data["Total"][h]
        n = result.wins + result.losses
        if n > 0:
            rate = result.wins / n * 100
            # Bold when wins > losses (consistent with W/L row)
            if result.wins > result.losses:
                win_rate_cells.append(f"\\textbf{{{rate:.1f}\\%}}")
            else:
                win_rate_cells.append(f"{rate:.1f}\\%")
        else:
            win_rate_cells.append("---")
    latex_lines.append(r"\textbf{Win Rate}   & " + " & ".join(win_rate_cells) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    # Footnote
    latex_lines.append(r"\vspace{1mm}")
    latex_lines.append(r"\begin{minipage}{\linewidth}")
    latex_lines.append(r"\footnotesize")
    latex_lines.append(r"W/L = wins/losses for Probe-and-Switch. Bold = win (W $>$ L).")
    latex_lines.append(r"\end{minipage}")
    latex_lines.append(r"\end{table}")

    latex_content = "\n".join(latex_lines)

    # ===== OUTPUT =====
    output_path = args.output
    if not output_path:
        output_dir = os.path.join(evidence_dir, "paper_tables")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "table_probeswitch_comparison.tex")

    with open(output_path, "w") as f:
        f.write(latex_content)

    print(f"\nGenerated LaTeX table: {repo_relpath(output_path)}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Print summary
    all_task_names = [d["display"] for d in coco_dimensions] + [t.display_name for t in external_tasks] + ["Total"]
    for task_name in all_task_names:
        print(f"\n{task_name}:")
        for comp_display, _ in competitors:
            result = table_data[task_name][comp_display]
            if result.wins > 0 or result.losses > 0:
                print(f"  vs {comp_display}: {result.wins}/{result.losses} (p={result.p_value:.2e})")
            else:
                print(f"  vs {comp_display}: no data")

    print("\n" + "=" * 60)
    print("LaTeX output:")
    print("=" * 60)
    print(latex_content)


if __name__ == "__main__":
    main()
