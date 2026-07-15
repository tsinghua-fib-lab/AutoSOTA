#!/usr/bin/env python3
"""
Generate LaTeX table for Appendix A13: Complete performance on high-misranking functions.

This script:
1. Loads bbob_summary.csv
2. Filters to 4 key algorithms: BERW-Hetero, CMA-ES-sep, ProbeSwitch-MR(t=0.12), UH-CMA-ES(maxevals=30)
3. Computes median regret per (function, algorithm) across 15 instances
4. Converts to log10 scale
5. Generates LaTeX table with bold for best per row
6. Computes pairwise win/loss statistics

Usage:
    python make_high_misranking_summary_table.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load the bbob_summary.csv file."""
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "evidence" / "hansen_test_fixed_budget" / "noisefree" / "bbob_summary.csv"
    return pd.read_csv(csv_path)


def compute_summary_table(df):
    """Compute median log10 regret per (function, algorithm)."""
    # Filter to the 4 key algorithms
    algorithms = [
        "BERW-Hetero",
        "CMA-ES-sep",
        "ProbeSwitch-MR(t=0.12)",
        "UH-CMA-ES(maxevals=30)"
    ]
    df_filtered = df[df["algorithm"].isin(algorithms)].copy()

    # The 15 high-misranking functions
    functions = [108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129]
    df_filtered = df_filtered[df_filtered["function"].isin(functions)]

    # Compute median best_f per (function, algorithm)
    summary = df_filtered.groupby(["function", "algorithm"])["best_f"].median().reset_index()
    summary["log10_regret"] = np.log10(summary["best_f"])

    # Pivot to wide format
    pivot = summary.pivot(index="function", columns="algorithm", values="log10_regret")
    pivot = pivot[algorithms]  # Ensure column order

    return pivot, df_filtered


def compute_pairwise_wins(df_filtered):
    """Compute pairwise win statistics based on median performance."""
    algorithms = [
        "BERW-Hetero",
        "CMA-ES-sep",
        "ProbeSwitch-MR(t=0.12)",
        "UH-CMA-ES(maxevals=30)"
    ]
    functions = [108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129]

    # Compute median per (function, algorithm)
    summary = df_filtered.groupby(["function", "algorithm"])["best_f"].median().reset_index()
    pivot = summary.pivot(index="function", columns="algorithm", values="best_f")

    results = {}

    # BERW vs UH-CMA-ES
    berw_wins = (pivot["BERW-Hetero"] < pivot["UH-CMA-ES(maxevals=30)"]).sum()
    results["BERW vs UH-CMA-ES"] = (berw_wins, len(functions))

    # BERW vs CMA-ES-sep
    berw_wins_cma = (pivot["BERW-Hetero"] < pivot["CMA-ES-sep"]).sum()
    results["BERW vs CMA-ES"] = (berw_wins_cma, len(functions))

    # ProbeSwitch vs UH-CMA-ES
    ps_wins_uh = (pivot["ProbeSwitch-MR(t=0.12)"] < pivot["UH-CMA-ES(maxevals=30)"]).sum()
    results["ProbeSwitch vs UH-CMA-ES"] = (ps_wins_uh, len(functions))

    # ProbeSwitch vs CMA-ES-sep
    ps_wins_cma = (pivot["ProbeSwitch-MR(t=0.12)"] < pivot["CMA-ES-sep"]).sum()
    results["ProbeSwitch vs CMA-ES"] = (ps_wins_cma, len(functions))

    # ProbeSwitch vs BERW
    ps_wins_berw = (pivot["ProbeSwitch-MR(t=0.12)"] < pivot["BERW-Hetero"]).sum()
    results["ProbeSwitch vs BERW"] = (ps_wins_berw, len(functions))

    return results


def generate_latex_table(pivot):
    """Generate LaTeX table code."""
    # Column headers (short names)
    col_names = {
        "BERW-Hetero": "Res.~Boot.",
        "CMA-ES-sep": "CMA-ES",
        "ProbeSwitch-MR(t=0.12)": "ProbeSwitch",
        "UH-CMA-ES(maxevals=30)": "UH-CMA-ES"
    }

    lines = []
    lines.append(r"\begin{tabular}{l cccc}")
    lines.append(r"\toprule")
    lines.append(r"Function & " + " & ".join(col_names.values()) + r" \\")
    lines.append(r"\midrule")

    for func in pivot.index:
        row_vals = pivot.loc[func]
        min_val = row_vals.min()

        formatted = []
        for alg in pivot.columns:
            val = row_vals[alg]
            if np.isclose(val, min_val, atol=0.005):  # Bold if best (within 0.005)
                formatted.append(r"\textbf{" + f"{val:.2f}" + r"}")
            else:
                formatted.append(f"{val:.2f}")

        lines.append(f"f{func} & " + " & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def main():
    print("Loading data...")
    df = load_data()

    print("Computing summary table...")
    pivot, df_filtered = compute_summary_table(df)

    print("\n" + "="*60)
    print("MEDIAN LOG10 REGRET TABLE")
    print("="*60)
    print(pivot.round(2).to_string())

    print("\n" + "="*60)
    print("PAIRWISE WIN STATISTICS")
    print("="*60)
    wins = compute_pairwise_wins(df_filtered)
    for comparison, (w, total) in wins.items():
        pct = 100 * w / total
        print(f"{comparison}: {w}/{total} ({pct:.0f}%)")

    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    latex = generate_latex_table(pivot)
    print(latex)

    # Also save to file
    output_path = Path(__file__).parent / "a13_high_misranking_table.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"\nTable saved to: {output_path}")


if __name__ == "__main__":
    main()
