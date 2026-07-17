#!/usr/bin/env python
"""
Summarize P6 synthetic control experiment results.

Produces:
- results/p6_all_runs.csv: Full results table
- results/p6_summary.csv: Aggregated results (mean ± std over seeds)
- results/p6_summary_latex.tex: LaTeX table for paper
- Prints markdown table to stdout

Usage: python scripts/summarize_p6_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS

METHODS = ['classification', 'regression']
WEIGHTINGS = ['none', 'absdelta']


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename."""
    name = filepath.stem
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
        }
    return None


def load_results(results_dir: str = 'results/synthetic') -> pd.DataFrame:
    """Load P6 synthetic result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    if not results_path.exists():
        return pd.DataFrame()

    for csv_file in results_path.glob('logreg_*_s*.csv'):
        meta = parse_result_filename(csv_file)
        if meta is None:
            continue

        df = pd.read_csv(csv_file)

        # Get test metrics
        if 'split' in df.columns:
            test_row = df[df['split'] == 'test'].iloc[0]
        else:
            test_row = df.iloc[0]

        row = {
            'dataset': 'synthetic',
            **meta,
        }

        # Extract metrics based on method
        if meta['method'] == 'classification':
            row['accuracy'] = test_row.get('test_accuracy', test_row.get('accuracy'))
            row['weighted_accuracy'] = test_row.get('test_weighted_accuracy', test_row.get('weighted_accuracy'))
            row['expected_cost'] = test_row.get('test_expected_cost', test_row.get('expected_cost'))
        else:  # regression
            row['sign_accuracy'] = test_row.get('test_sign_accuracy', test_row.get('sign_accuracy'))
            row['weighted_sign_accuracy'] = test_row.get('test_weighted_sign_accuracy', test_row.get('weighted_sign_accuracy'))
            row['mae'] = test_row.get('test_mae', test_row.get('mae'))
            row['weighted_mae'] = test_row.get('test_weighted_mae', test_row.get('weighted_mae'))

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds, computing mean ± std."""
    group_cols = ['dataset', 'model', 'method', 'weighting']
    metric_cols = [c for c in df.columns if c not in group_cols + ['seed']]

    agg_rows = []
    for name, group in df.groupby(group_cols):
        row = dict(zip(group_cols, name))
        row['n_seeds'] = len(group)

        for metric in metric_cols:
            if metric in group.columns and group[metric].notna().any():
                vals = group[metric].dropna()
                row[f'{metric}_mean'] = vals.mean()
                row[f'{metric}_std'] = vals.std()

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format as 'mean ± std'."""
    if pd.isna(mean):
        return '-'
    if pd.isna(std) or std == 0:
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f} ± {std:.{decimals}f}'


def generate_markdown_table(agg_df: pd.DataFrame) -> str:
    """Generate markdown table for P6 results."""
    lines = []

    # Classification results
    clf_df = agg_df[agg_df['method'] == 'classification']
    if not clf_df.empty:
        lines.append('## P6: Classification Results\n')
        lines.append('| Weighting | Accuracy | Weighted Acc | Expected Cost |')
        lines.append('|-----------|----------|--------------|---------------|')

        for weighting in WEIGHTINGS:
            row = clf_df[clf_df['weighting'] == weighting]
            if row.empty:
                continue
            row = row.iloc[0]

            acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
            wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
            cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
            lines.append(f"| {weighting} | {acc} | {wacc} | {cost} |")

    # Regression results
    reg_df = agg_df[agg_df['method'] == 'regression']
    if not reg_df.empty:
        lines.append('\n## P6: Regression Results\n')
        lines.append('| Weighting | Sign Acc | Weighted Sign Acc | MAE | Weighted MAE |')
        lines.append('|-----------|----------|-------------------|-----|--------------|')

        for weighting in WEIGHTINGS:
            row = reg_df[reg_df['weighting'] == weighting]
            if row.empty:
                continue
            row = row.iloc[0]

            sacc = format_mean_std(row.get('sign_accuracy_mean'), row.get('sign_accuracy_std'))
            wsacc = format_mean_std(row.get('weighted_sign_accuracy_mean'), row.get('weighted_sign_accuracy_std'))
            mae = format_mean_std(row.get('mae_mean'), row.get('mae_std'))
            wmae = format_mean_std(row.get('weighted_mae_mean'), row.get('weighted_mae_std'))
            lines.append(f"| {weighting} | {sacc} | {wsacc} | {mae} | {wmae} |")

    return '\n'.join(lines)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{P6 Synthetic Control Results (mean $\pm$ std over 3 seeds)}')
    lines.append(r'\label{tab:p6-synthetic}')
    lines.append(r'\begin{tabular}{llcccc}')
    lines.append(r'\toprule')
    lines.append(r'Method & Weighting & Accuracy/Sign Acc & Weighted Acc & Cost/MAE & Weighted MAE \\')
    lines.append(r'\midrule')

    for method in METHODS:
        for i, weighting in enumerate(WEIGHTINGS):
            row = agg_df[(agg_df['method'] == method) & (agg_df['weighting'] == weighting)]
            if row.empty:
                continue
            row = row.iloc[0]

            if method == 'classification':
                acc = row.get('accuracy_mean', np.nan)
                wacc = row.get('weighted_accuracy_mean', np.nan)
                cost = row.get('expected_cost_mean', np.nan)
                wmae = np.nan
            else:
                acc = row.get('sign_accuracy_mean', np.nan)
                wacc = row.get('weighted_sign_accuracy_mean', np.nan)
                cost = row.get('mae_mean', np.nan)
                wmae = row.get('weighted_mae_mean', np.nan)

            acc_str = f'{acc:.4f}' if pd.notna(acc) else '-'
            wacc_str = f'{wacc:.4f}' if pd.notna(wacc) else '-'
            cost_str = f'{cost:.4f}' if pd.notna(cost) else '-'
            wmae_str = f'{wmae:.4f}' if pd.notna(wmae) else '-'

            method_label = method if i == 0 else ''
            lines.append(f'{method_label} & {weighting} & {acc_str} & {wacc_str} & {cost_str} & {wmae_str} \\\\')

        lines.append(r'\midrule')

    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    print('Loading P6 results...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P6 results found!')
        return

    # Expected: 2 methods × 2 weightings × N_SEEDS
    expected = len(METHODS) * len(WEIGHTINGS) * N_SEEDS
    if len(df) != expected:
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print(f'  Methods: {sorted(df["method"].unique())}')
    print(f'  Weightings: {sorted(df["weighting"].unique())}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    output_dir = Path('results')
    df.to_csv(output_dir / 'p6_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p6_summary.csv', index=False)
    print(f'  Saved: results/p6_all_runs.csv')
    print(f'  Saved: results/p6_summary.csv')

    latex = generate_latex_table(agg_df)
    with open(output_dir / 'p6_summary_latex.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved: results/p6_summary_latex.tex')

    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)


if __name__ == '__main__':
    main()
