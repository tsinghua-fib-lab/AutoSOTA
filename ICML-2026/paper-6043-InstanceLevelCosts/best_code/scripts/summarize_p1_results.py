#!/usr/bin/env python
"""
Summarize P1 experiment results across seeds.

P1: Baseline experiments comparing classification vs regression
    with none and absdelta weighting on full datasets.

Dataset-model mapping:
- jigsaw: tfidf (full dataset)
- turkey: resnet50 (full dataset)
- nhanes: histgbm (full dataset)
- inaturalist: resnet50 (full dataset)

Produces:
- results/p1_summary.csv: Full results table
- results/p1_summary_latex.tex: LaTeX table for paper
- Prints markdown table to stdout

Usage: python scripts/summarize_p1_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS


# P1: Models per dataset (full dataset, no sample_size suffix)
# For jigsaw, include both tfidf and roberta
DATASET_MODEL_MAP = {
    'jigsaw': ['tfidf', 'roberta'],
    'turkey': ['resnet50'],
    'nhanes': ['histgbm'],
    'inaturalist': ['resnet50'],
}

# P1 weightings: baseline comparison
WEIGHTINGS = ['none', 'absdelta']

# P1 methods
METHODS = ['classification', 'regression']


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename."""
    # Pattern: {model}_{method}_{weighting}_s{seed}.csv (no _n{sample_size} suffix)
    name = filepath.stem
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)$', name)
    if not match:
        return None
    return {
        'model': match.group(1),
        'method': match.group(2),
        'weighting': match.group(3),
        'seed': int(match.group(4)),
    }


def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """Load all P1 result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    for dataset, expected_models in DATASET_MODEL_MAP.items():
        dataset_dir = results_path / dataset
        if not dataset_dir.exists():
            continue

        for csv_file in dataset_dir.glob('*_s*.csv'):
            meta = parse_result_filename(csv_file)
            if meta is None:
                continue

            # Only include expected models for this dataset
            if meta['model'] not in expected_models:
                continue

            # Only include P1 weightings
            if meta['weighting'] not in WEIGHTINGS:
                continue

            # Only include P1 methods
            if meta['method'] not in METHODS:
                continue

            df = pd.read_csv(csv_file)
            # Get test metrics (last row or row with split='test')
            if 'split' in df.columns:
                test_row = df[df['split'] == 'test'].iloc[0]
            else:
                # Assume single row per file with test metrics
                test_row = df.iloc[0]

            row = {
                'dataset': dataset,
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


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format as 'mean ± std'."""
    if pd.isna(mean):
        return '-'
    if pd.isna(std) or std == 0:
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f} ± {std:.{decimals}f}'


def generate_markdown_table(agg_df: pd.DataFrame) -> str:
    """Generate markdown table for classification results."""
    lines = []

    # Classification results
    clf_df = agg_df[agg_df['method'] == 'classification'].copy()
    if not clf_df.empty:
        lines.append('## Classification Results\n')
        lines.append('| Dataset | Weighting | Accuracy | Weighted Acc | Expected Cost |')
        lines.append('|---------|-----------|----------|--------------|---------------|')

        for _, row in clf_df.sort_values(['dataset', 'weighting']).iterrows():
            acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
            wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
            cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
            lines.append(f"| {row['dataset']} | {row['weighting']} | {acc} | {wacc} | {cost} |")

    # Regression results
    reg_df = agg_df[agg_df['method'] == 'regression'].copy()
    if not reg_df.empty:
        lines.append('\n## Regression Results\n')
        lines.append('| Dataset | Sign Acc | Weighted Sign Acc | MAE | Weighted MAE |')
        lines.append('|---------|----------|-------------------|-----|--------------|')

        for _, row in reg_df.sort_values(['dataset']).iterrows():
            sacc = format_mean_std(row.get('sign_accuracy_mean'), row.get('sign_accuracy_std'))
            wsacc = format_mean_std(row.get('weighted_sign_accuracy_mean'), row.get('weighted_sign_accuracy_std'))
            mae = format_mean_std(row.get('mae_mean'), row.get('mae_std'), decimals=2)
            wmae = format_mean_std(row.get('weighted_mae_mean'), row.get('weighted_mae_std'), decimals=2)
            lines.append(f"| {row['dataset']} | {sacc} | {wsacc} | {mae} | {wmae} |")

    return '\n'.join(lines)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    lines = []

    # Classification table
    clf_df = agg_df[agg_df['method'] == 'classification'].copy()
    if not clf_df.empty:
        lines.append(r'\begin{table}[h]')
        lines.append(r'\centering')
        lines.append(r'\caption{P1 Classification Results (mean $\pm$ std over 3 seeds)}')
        lines.append(r'\label{tab:p1-classification}')
        lines.append(r'\begin{tabular}{llccc}')
        lines.append(r'\toprule')
        lines.append(r'Dataset & Weighting & Accuracy & Weighted Acc & Expected Cost \\')
        lines.append(r'\midrule')

        for _, row in clf_df.sort_values(['dataset', 'weighting']).iterrows():
            acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
            wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
            cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
            lines.append(f"{row['dataset']} & {row['weighting']} & {acc} & {wacc} & {cost} \\\\")

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    print('Loading P1 results...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P1 results found!')
        return

    # Expected: (dataset-model combos) × (2 classification weightings + 1 regression weighting) × N_SEEDS
    # Regression only has 'none' weighting
    n_dataset_model_combos = sum(len(models) for models in DATASET_MODEL_MAP.values())
    expected = n_dataset_model_combos * (len(WEIGHTINGS) + 1) * N_SEEDS
    if len(df) != expected:
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print(f'  Datasets: {sorted(df["dataset"].unique())}')
    print(f'  Weightings: {sorted(df["weighting"].unique())}')
    print(f'  Methods: {sorted(df["method"].unique())}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    # Save full results
    output_dir = Path('results')
    df.to_csv(output_dir / 'p1_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p1_summary.csv', index=False)
    print(f'  Saved: results/p1_all_runs.csv')
    print(f'  Saved: results/p1_summary.csv')

    # Generate and save LaTeX
    latex = generate_latex_table(agg_df)
    with open(output_dir / 'p1_summary_latex.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved: results/p1_summary_latex.tex')

    # Print markdown table
    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)


if __name__ == '__main__':
    main()
