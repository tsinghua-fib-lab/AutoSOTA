#!/usr/bin/env python
"""
Summarize P5 end-to-end fine-tuning experiment results.

Datasets and models:
- jigsaw: roberta_finetune
- turkey: resnet_finetune
- inaturalist: resnet_finetune

Produces:
- results/p5_all_runs.csv: Full results table
- results/p5_summary.csv: Aggregated results (mean ± std over seeds)
- results/p5_summary_latex.tex: LaTeX table for paper
- Prints markdown table to stdout

Usage: python scripts/summarize_p5_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS

# Weightings to compare
WEIGHTINGS = ['none', 'absdelta', 'alpha_balanced']

# Dataset -> model mapping for P5
DATASET_MODEL_MAP = {
    'jigsaw': 'roberta_finetune',
    'turkey': 'resnet_finetune',
    'inaturalist': 'resnet_finetune',
}


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename.

    Pattern: {model}_{method}_{weighting}_s{seed}[_n{sample_size}].csv
    """
    name = filepath.stem
    # Try with sample_size suffix
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)_n(\d+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'sample_size': int(match.group(5)),
        }
    # Without sample_size suffix
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
    """Load P5 fine-tuning result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    for dataset, model in DATASET_MODEL_MAP.items():
        dataset_path = results_path / dataset
        if not dataset_path.exists():
            continue

        # Load fine-tuning classification files
        pattern = f'{model}_classification_*_s*.csv'
        for csv_file in dataset_path.glob(pattern):
            meta = parse_result_filename(csv_file)
            if meta is None:
                continue

            # Only include P5 weightings
            if meta['weighting'] not in WEIGHTINGS:
                continue

            # Verify model matches expected
            if meta['model'] != model:
                continue

            # Skip files with sample_size suffix (use full dataset only)
            if meta['sample_size'] is not None:
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
    metric_cols = [c for c in df.columns if c not in group_cols + ['seed', 'sample_size']]

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
    """Generate markdown table for P5 results."""
    lines = []
    lines.append('## P5: End-to-End Fine-tuning Results\n')
    lines.append('| Dataset | Model | Weighting | Accuracy | Weighted Acc | Expected Cost |')
    lines.append('|---------|-------|-----------|----------|--------------|---------------|')

    for dataset in DATASET_MODEL_MAP.keys():
        model = DATASET_MODEL_MAP[dataset]
        dataset_df = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]

        for weighting in WEIGHTINGS:
            row = dataset_df[dataset_df['weighting'] == weighting]
            if row.empty:
                continue
            row = row.iloc[0]

            acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
            wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
            cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
            lines.append(f"| {dataset} | {model} | {weighting} | {acc} | {wacc} | {cost} |")

    return '\n'.join(lines)


def generate_effect_table(agg_df: pd.DataFrame) -> str:
    """Generate table showing weighting effects vs baseline."""
    lines = []
    lines.append('\n## Weighting Effects (vs none)\n')
    lines.append('| Dataset | Weighting | Δ Cost (vs none) |')
    lines.append('|---------|-----------|------------------|')

    for dataset in DATASET_MODEL_MAP.keys():
        model = DATASET_MODEL_MAP[dataset]
        dataset_df = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]

        none_row = dataset_df[dataset_df['weighting'] == 'none']
        if none_row.empty:
            continue

        none_cost = none_row.iloc[0].get('expected_cost_mean', np.nan)

        for weighting in ['absdelta', 'alpha_balanced']:
            row = dataset_df[dataset_df['weighting'] == weighting]
            if row.empty:
                continue
            row = row.iloc[0]

            cost = row.get('expected_cost_mean', np.nan)
            if pd.notna(cost) and pd.notna(none_cost):
                delta = f'{cost - none_cost:+.4f}'
            else:
                delta = '-'

            lines.append(f'| {dataset} | {weighting} | {delta} |')

    return '\n'.join(lines)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{P5 End-to-End Fine-tuning Results (mean $\pm$ std over 3 seeds)}')
    lines.append(r'\label{tab:p5-finetune}')
    lines.append(r'\begin{tabular}{llccc}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Weighting & Accuracy & Weighted Acc & Expected Cost \\')
    lines.append(r'\midrule')

    for dataset in DATASET_MODEL_MAP.keys():
        model = DATASET_MODEL_MAP[dataset]
        dataset_df = agg_df[(agg_df['dataset'] == dataset) & (agg_df['model'] == model)]

        for i, weighting in enumerate(WEIGHTINGS):
            row = dataset_df[dataset_df['weighting'] == weighting]
            if row.empty:
                continue
            row = row.iloc[0]

            acc = row.get('accuracy_mean', np.nan)
            wacc = row.get('weighted_accuracy_mean', np.nan)
            cost = row.get('expected_cost_mean', np.nan)

            acc_str = f'{acc:.3f}' if pd.notna(acc) else '-'
            wacc_str = f'{wacc:.3f}' if pd.notna(wacc) else '-'
            cost_str = f'{cost:.3f}' if pd.notna(cost) else '-'

            dataset_label = dataset if i == 0 else ''
            lines.append(f'{dataset_label} & {weighting} & {acc_str} & {wacc_str} & {cost_str} \\\\')

        lines.append(r'\midrule')

    # Remove last midrule and replace with bottomrule
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    print('Loading P5 results...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P5 results found!')
        return

    # Expected: 3 datasets × 3 weightings × N_SEEDS
    expected = len(DATASET_MODEL_MAP) * len(WEIGHTINGS) * N_SEEDS
    if len(df) != expected:
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print(f'  Datasets: {sorted(df["dataset"].unique())}')
    print(f'  Models: {sorted(df["model"].unique())}')
    print(f'  Weightings: {sorted(df["weighting"].unique())}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    # Save full results
    output_dir = Path('results')
    df.to_csv(output_dir / 'p5_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p5_summary.csv', index=False)
    print(f'  Saved: results/p5_all_runs.csv')
    print(f'  Saved: results/p5_summary.csv')

    # Generate and save LaTeX
    latex = generate_latex_table(agg_df)
    with open(output_dir / 'p5_summary_latex.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved: results/p5_summary_latex.tex')

    # Print markdown tables
    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)

    # Print effect table
    effect_table = generate_effect_table(agg_df)
    print(effect_table)


if __name__ == '__main__':
    main()
