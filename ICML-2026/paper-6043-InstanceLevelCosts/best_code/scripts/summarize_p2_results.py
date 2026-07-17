#!/usr/bin/env python
"""
Summarize P2 sample size scaling experiment results.

Produces:
- results/p2_all_runs.csv: Full results table
- results/p2_summary.csv: Aggregated results (mean ± std over seeds)
- results/p2_summary_latex.tex: LaTeX table for paper
- Prints markdown table to stdout

Usage: python scripts/summarize_p2_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS

# P2 specific filters - only baseline scaling experiments
P2_WEIGHTINGS = ['none', 'absdelta']
P2_MODELS = ['tfidf', 'roberta']


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename."""
    # Pattern: {model}_{method}_{weighting}_s{seed}_n{sample_size}.csv
    name = filepath.stem
    match = re.match(r'(\w+)_(\w+)_(\w+)_s(\d+)_n(\d+)', name)
    if not match:
        return None
    return {
        'model': match.group(1),
        'method': match.group(2),
        'weighting': match.group(3),
        'seed': int(match.group(4)),
        'sample_size': int(match.group(5)),
    }


def format_n(n: int) -> str:
    """Format sample size as Nk."""
    return f"{n//1000}k" if n >= 1000 else str(n)


def load_results(results_dir: str = 'results/jigsaw') -> pd.DataFrame:
    """Load all P2 result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    for csv_file in results_path.glob('*_n*.csv'):
        meta = parse_result_filename(csv_file)
        if meta is None:
            continue

        # Filter to P2 weightings only (exclude P4's alpha_balanced)
        if meta['weighting'] not in P2_WEIGHTINGS:
            continue

        # Filter to P2 models only (exclude P5's roberta_finetune)
        if meta['model'] not in P2_MODELS:
            continue

        df = pd.read_csv(csv_file)

        # Handle different CSV structures
        if 'split' in df.columns:
            # Filter by split == 'test'
            test_df = df[df['split'] == 'test']
            if test_df.empty:
                print(f'  WARNING: No test split in {csv_file}')
                continue
            test_row = test_df.iloc[0]
        else:
            # Assume single row per file with test metrics in columns prefixed with test_
            test_row = df.iloc[0]

        row = {
            'dataset': 'jigsaw',
            **meta,
        }

        # Extract relevant metrics based on method
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
    # Group by everything except seed
    group_cols = ['dataset', 'model', 'method', 'weighting', 'sample_size']
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


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format as 'mean ± std'."""
    if pd.isna(mean):
        return '-'
    if pd.isna(std) or std == 0:
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f} ± {std:.{decimals}f}'


def generate_markdown_table(agg_df: pd.DataFrame) -> str:
    """Generate markdown table for scaling results."""
    lines = []
    sample_sizes = sorted(agg_df['sample_size'].unique())

    # Classification results
    clf_df = agg_df[agg_df['method'] == 'classification'].copy()
    if not clf_df.empty:
        lines.append('## Classification Scaling Results\n')

        for model in ['tfidf', 'roberta']:
            model_df = clf_df[clf_df['model'] == model]
            if model_df.empty:
                continue

            lines.append(f'### {model.upper()}\n')
            lines.append('| N | Weighting | Accuracy | Weighted Acc | Expected Cost |')
            lines.append('|---|-----------|----------|--------------|---------------|')

            for n in sample_sizes:
                for weighting in ['none', 'absdelta']:
                    row = model_df[(model_df['sample_size'] == n) & (model_df['weighting'] == weighting)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
                    wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
                    cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
                    lines.append(f"| {format_n(n)} | {weighting} | {acc} | {wacc} | {cost} |")

            lines.append('')

    # Regression results
    reg_df = agg_df[agg_df['method'] == 'regression'].copy()
    if not reg_df.empty:
        lines.append('## Regression Scaling Results\n')

        for model in ['tfidf', 'roberta']:
            model_df = reg_df[reg_df['model'] == model]
            if model_df.empty:
                continue

            lines.append(f'### {model.upper()}\n')
            lines.append('| N | Sign Acc | Weighted Sign Acc | MAE | Weighted MAE |')
            lines.append('|---|----------|-------------------|-----|--------------|')

            for n in sample_sizes:
                row = model_df[model_df['sample_size'] == n]
                if row.empty:
                    continue
                row = row.iloc[0]
                sacc = format_mean_std(row.get('sign_accuracy_mean'), row.get('sign_accuracy_std'))
                wsacc = format_mean_std(row.get('weighted_sign_accuracy_mean'), row.get('weighted_sign_accuracy_std'))
                mae = format_mean_std(row.get('mae_mean'), row.get('mae_std'), decimals=2)
                wmae = format_mean_std(row.get('weighted_mae_mean'), row.get('weighted_mae_std'), decimals=2)
                lines.append(f"| {format_n(n)} | {sacc} | {wsacc} | {mae} | {wmae} |")

            lines.append('')

    return '\n'.join(lines)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper - classification accuracy and weighted accuracy by sample size."""
    lines = []
    sample_sizes = sorted(agg_df['sample_size'].unique())

    clf_df = agg_df[agg_df['method'] == 'classification'].copy()
    if clf_df.empty:
        return ''

    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{P2 Sample Size Scaling: Classification Results (mean $\pm$ std over 3 seeds)}')
    lines.append(r'\label{tab:p2-scaling}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{ll' + 'c' * len(sample_sizes) + '}')
    lines.append(r'\toprule')

    # Header row with sample sizes
    header = r'Model & Weighting & ' + ' & '.join([f'N={format_n(n)}' for n in sample_sizes]) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{' + str(len(sample_sizes) + 2) + r'}{c}{\textit{Accuracy}} \\')
    lines.append(r'\midrule')

    # Accuracy rows
    for model in ['tfidf', 'roberta']:
        for weighting in ['none', 'absdelta']:
            model_df = clf_df[(clf_df['model'] == model) & (clf_df['weighting'] == weighting)]
            if model_df.empty:
                continue

            row_vals = []
            for n in sample_sizes:
                row = model_df[model_df['sample_size'] == n]
                if row.empty:
                    row_vals.append('-')
                else:
                    row = row.iloc[0]
                    acc = row.get('accuracy_mean', np.nan)
                    std = row.get('accuracy_std', np.nan)
                    if pd.notna(acc):
                        if pd.notna(std) and std > 0:
                            row_vals.append(f'{acc:.3f}')
                        else:
                            row_vals.append(f'{acc:.3f}')
                    else:
                        row_vals.append('-')

            label = f'{model} & {weighting}'
            lines.append(f"{label} & {' & '.join(row_vals)} \\\\")

    lines.append(r'\midrule')
    lines.append(r'\multicolumn{' + str(len(sample_sizes) + 2) + r'}{c}{\textit{Weighted Accuracy}} \\')
    lines.append(r'\midrule')

    # Weighted accuracy rows
    for model in ['tfidf', 'roberta']:
        for weighting in ['none', 'absdelta']:
            model_df = clf_df[(clf_df['model'] == model) & (clf_df['weighting'] == weighting)]
            if model_df.empty:
                continue

            row_vals = []
            for n in sample_sizes:
                row = model_df[model_df['sample_size'] == n]
                if row.empty:
                    row_vals.append('-')
                else:
                    row = row.iloc[0]
                    wacc = row.get('weighted_accuracy_mean', np.nan)
                    std = row.get('weighted_accuracy_std', np.nan)
                    if pd.notna(wacc):
                        if pd.notna(std) and std > 0:
                            row_vals.append(f'{wacc:.3f}')
                        else:
                            row_vals.append(f'{wacc:.3f}')
                    else:
                        row_vals.append('-')

            label = f'{model} & {weighting}'
            lines.append(f"{label} & {' & '.join(row_vals)} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_weighting_effect_table(agg_df: pd.DataFrame) -> str:
    """Generate table showing weighting effect (absdelta - none) by sample size."""
    lines = []
    sample_sizes = sorted(agg_df['sample_size'].unique())

    clf_df = agg_df[agg_df['method'] == 'classification'].copy()
    if clf_df.empty:
        return ''

    lines.append('\n## Weighting Effect (absdelta - none)\n')
    lines.append('| Model | Metric | ' + ' | '.join([f'N={format_n(n)}' for n in sample_sizes]) + ' |')
    lines.append('|-------|--------|' + '|'.join(['--------'] * len(sample_sizes)) + '|')

    for model in ['tfidf', 'roberta']:
        model_df = clf_df[clf_df['model'] == model]
        if model_df.empty:
            continue

        for metric, label in [('accuracy', 'Accuracy'), ('weighted_accuracy', 'Weighted Acc')]:
            row_vals = []
            for n in sample_sizes:
                none_row = model_df[(model_df['sample_size'] == n) & (model_df['weighting'] == 'none')]
                absdelta_row = model_df[(model_df['sample_size'] == n) & (model_df['weighting'] == 'absdelta')]

                if none_row.empty or absdelta_row.empty:
                    row_vals.append('-')
                else:
                    none_val = none_row.iloc[0].get(f'{metric}_mean', np.nan)
                    absdelta_val = absdelta_row.iloc[0].get(f'{metric}_mean', np.nan)
                    if pd.notna(none_val) and pd.notna(absdelta_val):
                        diff = absdelta_val - none_val
                        row_vals.append(f'{diff:+.4f}')
                    else:
                        row_vals.append('-')

            lines.append(f'| {model} | {label} | ' + ' | '.join(row_vals) + ' |')

    return '\n'.join(lines)


def main():
    print('Loading P2 results...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P2 results found!')
        return

    # Sanity check
    # P2: 2 models × 7 sizes × N_SEEDS × (2 classification weightings + 1 regression)
    n_models = len(P2_MODELS)
    n_sizes = 7  # 1k, 2k, 5k, 10k, 20k, 50k, 100k
    n_method_weightings = len(P2_WEIGHTINGS) + 1  # classification has both weightings, regression has none only
    expected = n_models * n_sizes * N_SEEDS * n_method_weightings
    if len(df) != expected:
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print(f'  Models: {sorted(df["model"].unique())}')
    print(f'  Sample sizes: {sorted(df["sample_size"].unique())}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    # Save full results
    output_dir = Path('results')
    df.to_csv(output_dir / 'p2_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p2_summary.csv', index=False)
    print(f'  Saved: results/p2_all_runs.csv')
    print(f'  Saved: results/p2_summary.csv')

    # Generate and save LaTeX
    latex = generate_latex_table(agg_df)
    with open(output_dir / 'p2_summary_latex.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved: results/p2_summary_latex.tex')

    # Print markdown tables
    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)

    # Print weighting effect analysis
    effect_table = generate_weighting_effect_table(agg_df)
    print(effect_table)


if __name__ == '__main__':
    main()
