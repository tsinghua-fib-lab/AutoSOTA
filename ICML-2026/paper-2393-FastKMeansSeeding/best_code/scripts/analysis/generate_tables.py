#!/usr/bin/env python3
"""
Generate LaTeX tables from benchmark results.

Usage:
    python scripts/analysis/generate_tables.py <results_csv> [options]
    python scripts/analysis/generate_tables.py results/comparison_*.csv --output tables/

Examples:
    python scripts/analysis/generate_tables.py results/comparison_mnist.csv
    python scripts/analysis/generate_tables.py results/*.csv --format markdown
    python scripts/analysis/generate_tables.py results/*.csv --speedup qkmeans
    python scripts/analysis/generate_tables.py results/*.csv --style compact
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Algorithm display names for LaTeX
LATEX_NAMES = {
    'kmeanspp': r'$k$-means++',
    'afkmc2': r'AFKMC$^2$',
    'prone': r'PRONE',
    'pronecoreset': r'PRONECoreset',
    'fastcoreset': r'FastCoreset',
    'rejectionlsh': r'RejectionLSH',
    'qkmeans': r'QKMEANS',
    'qkmeans_anns': r'QKMEANS-ANNS',
}

# Algorithm order for tables
ALGO_ORDER = [
    'kmeanspp', 'afkmc2', 'prone', 'pronecoreset',
    'fastcoreset', 'rejectionlsh', 'qkmeans'
]


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess results CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize column names
    if 'seeding_time_ms' in df.columns:
        df['time_ms'] = df['seeding_time_ms']
    if 'seeding_cost' in df.columns:
        df['cost'] = df['seeding_cost']

    return df


def format_time(ms: float, latex: bool = True) -> str:
    """Format time value for display."""
    if ms < 1:
        return f"{ms*1000:.0f}"
    elif ms < 10:
        return f"{ms:.1f}"
    elif ms < 1000:
        return f"{ms:.0f}"
    else:
        return f"{ms/1000:.1f}s" if latex else f"{ms:.0f}"


def format_cost_scaled(cost: float, scale: float) -> str:
    """Format cost value with given scale."""
    scaled = cost / scale
    if scaled >= 100:
        return f"{scaled:.0f}"
    elif scaled >= 10:
        return f"{scaled:.1f}"
    else:
        return f"{scaled:.2f}"


def format_cost(cost: float) -> str:
    """Format cost value for display (auto-scale)."""
    if cost >= 1e12:
        return f"{cost/1e12:.2f}T"
    elif cost >= 1e9:
        return f"{cost/1e9:.2f}B"
    elif cost >= 1e6:
        return f"{cost/1e6:.2f}M"
    elif cost >= 1e3:
        return f"{cost/1e3:.2f}K"
    else:
        return f"{cost:.2f}"


def determine_cost_scale(df: pd.DataFrame) -> tuple:
    """Determine appropriate scale for cost values."""
    max_cost = df['cost'].max()
    if max_cost >= 1e12:
        return 1e12, r"$\times 10^{12}$"
    elif max_cost >= 1e11:
        return 1e11, r"$\times 10^{11}$"
    elif max_cost >= 1e9:
        return 1e9, r"$\times 10^{9}$"
    elif max_cost >= 1e6:
        return 1e6, r"$\times 10^{6}$"
    else:
        return 1, ""


def generate_latex_table(df: pd.DataFrame, dataset_name: str = None,
                         highlight_best: bool = True, style: str = "full") -> str:
    """Generate LaTeX table from benchmark results.

    Args:
        df: DataFrame with columns: method, k, cost, time_ms
        dataset_name: Name for caption
        highlight_best: Bold best values per k
        style: "full" (cost & time), "time-only", "cost-only", "compact"
    """
    k_values = sorted(df['k'].unique())
    methods = [m for m in ALGO_ORDER if m in df['method'].unique()]

    # Determine cost scale
    cost_scale, cost_scale_str = determine_cost_scale(df)

    # Find best values for each k
    best_cost = {}
    best_time = {}
    for k in k_values:
        k_df = df[df['k'] == k]
        best_cost[k] = k_df['cost'].min()
        best_time[k] = k_df['time_ms'].min()

    if style == "compact":
        # Compact style: Time (ms) | Cost columns with cmidrule
        n_k = len(k_values)
        col_spec = 'l' + 'rr' * n_k

        latex = r"""\begin{table}[htbp]
\centering
\small
"""
        if dataset_name:
            latex += f"\\caption{{Seeding benchmark on {dataset_name}. Time in ms, cost {cost_scale_str}.}}\n"
            latex += f"\\label{{tab:benchmark_{dataset_name.lower().replace(' ', '_')}}}\n"

        latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex += r"\toprule" + "\n"

        # Header row 1: k values with multicolumn
        latex += r"\textbf{Algorithm}"
        for k in k_values:
            latex += f" & \\multicolumn{{2}}{{c}}{{$k={k}$}}"
        latex += r" \\" + "\n"

        # cmidrule for each k
        for i, k in enumerate(k_values):
            col_start = 2 + i * 2
            col_end = col_start + 1
            latex += f"\\cmidrule(lr){{{col_start}-{col_end}}} "
        latex += "\n"

        # Header row 2: Time/Cost
        latex += ""
        for _ in k_values:
            latex += r" & Time & Cost"
        latex += r" \\" + "\n"
        latex += r"\midrule" + "\n"

        # Data rows
        for method in methods:
            method_df = df[df['method'] == method]
            latex += LATEX_NAMES.get(method, method)

            for k in k_values:
                k_row = method_df[method_df['k'] == k]
                if len(k_row) > 0:
                    cost = k_row['cost'].values[0]
                    time = k_row['time_ms'].values[0]

                    time_str = format_time(time)
                    cost_str = format_cost_scaled(cost, cost_scale)

                    # Highlight best values
                    if highlight_best:
                        if abs(time - best_time[k]) < 0.01:
                            time_str = r"\textbf{" + time_str + "}"
                        if abs(cost - best_cost[k]) < 1:
                            cost_str = r"\textbf{" + cost_str + "}"

                    latex += f" & {time_str} & {cost_str}"
                else:
                    latex += " & -- & --"

            latex += r" \\" + "\n"

        latex += r"\bottomrule" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"

    else:
        # Full style (original)
        col_spec = 'l' + 'rr' * len(k_values)

        latex = r"""\begin{table}[htbp]
\centering
\small
"""
        if dataset_name:
            latex += f"\\caption{{Benchmark results on {dataset_name}. Cost {cost_scale_str}.}}\n"
            latex += f"\\label{{tab:benchmark_{dataset_name.lower().replace(' ', '_')}}}\n"

        latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex += r"\toprule" + "\n"

        # Header row 1: k values
        latex += r"\multirow{2}{*}{\textbf{Method}}"
        for k in k_values:
            latex += f" & \\multicolumn{{2}}{{c}}{{$k={k}$}}"
        latex += r" \\" + "\n"

        # Header row 2: Cost/Time
        latex += ""
        for _ in k_values:
            latex += r" & Time & Cost"
        latex += r" \\" + "\n"
        latex += r"\midrule" + "\n"

        # Data rows
        for method in methods:
            method_df = df[df['method'] == method]
            latex += LATEX_NAMES.get(method, method)

            for k in k_values:
                k_row = method_df[method_df['k'] == k]
                if len(k_row) > 0:
                    cost = k_row['cost'].values[0]
                    time = k_row['time_ms'].values[0]

                    time_str = format_time(time)
                    cost_str = format_cost_scaled(cost, cost_scale)

                    # Highlight best values
                    if highlight_best:
                        if abs(time - best_time[k]) < 0.01:
                            time_str = r"\textbf{" + time_str + "}"
                        if abs(cost - best_cost[k]) < 1:
                            cost_str = r"\textbf{" + cost_str + "}"

                    latex += f" & {time_str} & {cost_str}"
                else:
                    latex += " & -- & --"

            latex += r" \\" + "\n"

        latex += r"\bottomrule" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"

    return latex


def generate_speedup_table(df: pd.DataFrame, baseline: str = 'kmeanspp',
                           target: str = 'qkmeans') -> str:
    """Generate speedup comparison table."""
    k_values = sorted(df['k'].unique())

    latex = r"""\begin{table}[htbp]
\centering
\caption{Speedup of """ + LATEX_NAMES.get(target, target) + r""" over baselines}
\begin{tabular}{l""" + 'r' * len(k_values) + r"""}
\toprule
\textbf{Baseline}"""

    for k in k_values:
        latex += f" & $k={k}$"
    latex += r" \\" + "\n"
    latex += r"\midrule" + "\n"

    # Get target times
    target_df = df[df['method'] == target]

    for method in ALGO_ORDER:
        if method == target or method not in df['method'].unique():
            continue

        method_df = df[df['method'] == method]
        latex += LATEX_NAMES.get(method, method)

        for k in k_values:
            target_time = target_df[target_df['k'] == k]['time_ms'].values
            method_time = method_df[method_df['k'] == k]['time_ms'].values

            if len(target_time) > 0 and len(method_time) > 0:
                speedup = method_time[0] / target_time[0]
                latex += f" & {speedup:.1f}$\\times$"
            else:
                latex += " & --"

        latex += r" \\" + "\n"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    return latex


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate Markdown table from benchmark results."""
    k_values = sorted(df['k'].unique())
    methods = [m for m in ALGO_ORDER if m in df['method'].unique()]

    # Header
    header = "| Method |"
    separator = "|--------|"
    for k in k_values:
        header += f" k={k} (Cost) | k={k} (Time) |"
        separator += "-------------|-------------|"
    header += "\n"
    separator += "\n"

    # Data rows
    rows = ""
    for method in methods:
        method_df = df[df['method'] == method]
        row = f"| {method} |"

        for k in k_values:
            k_row = method_df[method_df['k'] == k]
            if len(k_row) > 0:
                cost = format_cost(k_row['cost'].values[0])
                time = format_time(k_row['time_ms'].values[0])
                row += f" {cost} | {time} |"
            else:
                row += " -- | -- |"

        rows += row + "\n"

    return header + separator + rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate tables from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analysis/generate_tables.py results/comparison_mnist.csv
  python scripts/analysis/generate_tables.py results/*.csv --style compact
  python scripts/analysis/generate_tables.py results/*.csv --speedup qkmeans
  python scripts/analysis/generate_tables.py results/*.csv --format markdown
        """
    )
    parser.add_argument("csv_files", nargs="+", help="CSV result files")
    parser.add_argument("--format", choices=["latex", "markdown"],
                        default="latex", help="Output format (default: latex)")
    parser.add_argument("--style", choices=["full", "compact"],
                        default="compact", help="Table style (default: compact)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory or file")
    parser.add_argument("--speedup", type=str, default=None,
                        help="Generate speedup table with this method as target")
    parser.add_argument("--no-highlight", action="store_true",
                        help="Don't highlight best values")

    args = parser.parse_args()

    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"ERROR: File not found: {csv_path}")
            continue

        print(f"Processing: {csv_path}")
        df = load_results(csv_path)

        # Get dataset name
        dataset_name = df['dataset'].iloc[0] if 'dataset' in df.columns else csv_path.stem

        # Generate table
        if args.speedup:
            table = generate_speedup_table(df, target=args.speedup)
        elif args.format == "markdown":
            table = generate_markdown_table(df)
        else:
            table = generate_latex_table(df, dataset_name, not args.no_highlight, args.style)

        # Output
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                ext = ".md" if args.format == "markdown" else ".tex"
                output_path = output_path / f"{csv_path.stem}_table{ext}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table)
            print(f"Saved: {output_path}")
        else:
            print("\n" + table)


if __name__ == "__main__":
    main()
