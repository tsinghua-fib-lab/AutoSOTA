#!/usr/bin/env python
"""
Summarize P4 alpha-balanced weighting experiment results.

Loads P1 results (none, absdelta) and P4 results (alpha_balanced) to compare.

Produces:
- results/p4_all_runs.csv: Full results table
- results/p4_summary.csv: Aggregated results (mean ± std over seeds)
- results/p4_summary_latex.tex: LaTeX table for paper
- Prints markdown table to stdout

Usage: python scripts/summarize_p4_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS

# Dataset to model mapping
DATASET_MODEL_MAP = {
    'jigsaw': ['tfidf', 'roberta'],
    'turkey': ['resnet50'],
    'nhanes': ['histgbm'],
    'inaturalist': ['resnet50'],
}

# Weightings to compare
WEIGHTINGS = ['none', 'absdelta', 'alpha_balanced']


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename.

    Pattern: {model}_{method}_{weighting}_s{seed}.csv
    Pattern: {model}_{method}_{weighting}_s{seed}_n{sample_size}.csv
    """
    name = filepath.stem
    # Try with sample_size suffix first
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)_n(\d+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'sample_size': int(match.group(5)),
        }
    # Try without sample_size suffix
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'sample_size': None,
        }
    return None


def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """Load P1 and P4 classification result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    for dataset, models in DATASET_MODEL_MAP.items():
        dataset_dir = results_path / dataset
        if not dataset_dir.exists():
            continue

        # Load classification files with none, absdelta, or alpha_balanced weighting
        for csv_file in dataset_dir.glob('*_classification_*_s*.csv'):
            meta = parse_result_filename(csv_file)
            if meta is None:
                continue

            # Only include P1/P4 weightings
            if meta['weighting'] not in WEIGHTINGS:
                continue

            # Only include expected models (skip files with sample_size suffix)
            if meta['model'] not in models or meta['sample_size'] is not None:
                continue

            df = pd.read_csv(csv_file)

            # Get test metrics
            if 'split' in df.columns:
                test_row = df[df['split'] == 'test'].iloc[0]
            else:
                test_row = df.iloc[0]

            row = {
                'dataset': dataset,
                **meta,
            }

            # Extract classification metrics
            row['accuracy'] = test_row.get('test_accuracy', test_row.get('accuracy'))
            row['weighted_accuracy'] = test_row.get('test_weighted_accuracy', test_row.get('weighted_accuracy'))
            row['expected_cost'] = test_row.get('test_expected_cost', test_row.get('expected_cost'))

            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds, computing mean ± std."""
    group_cols = ['dataset', 'model', 'method', 'weighting']
    group_cols = [c for c in group_cols if c in df.columns]
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
    """Generate markdown table for P4 results."""
    lines = []
    lines.append('## P4: Alpha-Balanced Weighting Results\n')
    lines.append('| Dataset | Model | Weighting | Accuracy | Weighted Acc | Expected Cost |')
    lines.append('|---------|-------|-----------|----------|--------------|---------------|')

    for dataset, models in DATASET_MODEL_MAP.items():
        for model in models:
            model_df = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]
            if model_df.empty:
                continue

            for weighting in WEIGHTINGS:
                row = model_df[model_df['weighting'] == weighting]
                if row.empty:
                    continue
                row = row.iloc[0]

                acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
                wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
                cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
                lines.append(f"| {dataset} | {model} | {weighting} | {acc} | {wacc} | {cost} |")

    return '\n'.join(lines)


def generate_effect_table(agg_df: pd.DataFrame) -> str:
    """Generate table showing alpha_balanced effect vs baselines."""
    lines = []
    lines.append('\n## Alpha-Balanced Effect (vs Baselines)\n')
    lines.append('| Dataset | Model | vs none Δ Cost | vs absdelta Δ Cost |')
    lines.append('|---------|-------|----------------|-------------------|')

    for dataset, models in DATASET_MODEL_MAP.items():
        for model in models:
            model_df = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]
            if model_df.empty:
                continue

            # Get baseline costs
            none_row = model_df[model_df['weighting'] == 'none']
            absdelta_row = model_df[model_df['weighting'] == 'absdelta']
            alpha_row = model_df[model_df['weighting'] == 'alpha_balanced']

            if alpha_row.empty:
                continue

            alpha_cost = alpha_row.iloc[0].get('expected_cost_mean', np.nan)

            if not none_row.empty:
                none_cost = none_row.iloc[0].get('expected_cost_mean', np.nan)
                vs_none = f'{alpha_cost - none_cost:+.4f}' if pd.notna(alpha_cost) and pd.notna(none_cost) else '-'
            else:
                vs_none = '-'

            if not absdelta_row.empty:
                absdelta_cost = absdelta_row.iloc[0].get('expected_cost_mean', np.nan)
                vs_absdelta = f'{alpha_cost - absdelta_cost:+.4f}' if pd.notna(alpha_cost) and pd.notna(absdelta_cost) else '-'
            else:
                vs_absdelta = '-'

            lines.append(f'| {dataset} | {model} | {vs_none} | {vs_absdelta} |')

    return '\n'.join(lines)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{P4 Alpha-Balanced Weighting Results (mean $\pm$ std over 3 seeds)}')
    lines.append(r'\label{tab:p4-alpha-balanced}')
    lines.append(r'\begin{tabular}{lllccc}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Model & Weighting & Accuracy & Weighted Acc & Expected Cost \\')
    lines.append(r'\midrule')

    for dataset, models in DATASET_MODEL_MAP.items():
        first_in_dataset = True
        for model in models:
            model_df = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]
            if model_df.empty:
                continue

            first_in_model = True
            for weighting in WEIGHTINGS:
                row = model_df[model_df['weighting'] == weighting]
                if row.empty:
                    continue
                row = row.iloc[0]

                acc = row.get('accuracy_mean', np.nan)
                wacc = row.get('weighted_accuracy_mean', np.nan)
                cost = row.get('expected_cost_mean', np.nan)

                acc_str = f'{acc:.3f}' if pd.notna(acc) else '-'
                wacc_str = f'{wacc:.3f}' if pd.notna(wacc) else '-'
                cost_str = f'{cost:.3f}' if pd.notna(cost) else '-'

                dataset_label = dataset if first_in_dataset else ''
                model_label = model if first_in_model else ''
                lines.append(f'{dataset_label} & {model_label} & {weighting} & {acc_str} & {wacc_str} & {cost_str} \\\\')
                first_in_dataset = False
                first_in_model = False

        lines.append(r'\midrule')

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    print('Loading P4 results (with P1 baselines)...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P4 results found!')
        return

    # Expected: 5 model configs × 3 weightings × N_SEEDS
    expected = sum(len(models) for models in DATASET_MODEL_MAP.values()) * len(WEIGHTINGS) * N_SEEDS
    if len(df) != expected:
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print(f'  Datasets: {sorted(df["dataset"].unique())}')
    print(f'  Weightings: {sorted(df["weighting"].unique())}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    # Save full results
    output_dir = Path('results')
    df.to_csv(output_dir / 'p4_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p4_summary.csv', index=False)
    print(f'  Saved: results/p4_all_runs.csv')
    print(f'  Saved: results/p4_summary.csv')

    # Generate and save LaTeX
    latex = generate_latex_table(agg_df)
    with open(output_dir / 'p4_summary_latex.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved: results/p4_summary_latex.tex')

    # Print markdown tables
    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)

    # Print effect table
    effect_table = generate_effect_table(agg_df)
    print(effect_table)


if __name__ == '__main__':
    main()