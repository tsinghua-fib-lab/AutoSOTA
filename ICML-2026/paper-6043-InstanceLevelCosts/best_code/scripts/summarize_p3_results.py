#!/usr/bin/env python
"""
Summarize P3 sampling strategy experiment results.

Loads P1 classification results as "U" (uniform) baseline,
and P3 results for P_up, Tdown50, Tdown30 strategies.

Produces:
- results/p3_all_runs.csv: Full results table
- results/p3_summary.csv: Aggregated results (mean ± std over seeds)
- results/p3_summary_latex.tex: LaTeX table for paper
- Prints markdown table to stdout

Usage: python scripts/summarize_p3_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS

# Dataset to model mapping
# For jigsaw, include both tfidf and roberta to show result holds across model capacity
DATASET_MODEL_MAP = {
    'jigsaw': ['tfidf', 'roberta'],
    'turkey': ['resnet50'],
    'nhanes': ['histgbm'],
    'inaturalist': ['resnet50'],
}

# Valid P3 strategies (used in regex pattern)
P3_STRATEGIES = ['P_up', 'Tdown70', 'Tdown50', 'Tdown30']
ALL_STRATEGIES = ['U'] + P3_STRATEGIES  # U = P1 baseline


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename.

    Handles both P1 and P3 filename patterns:
    - P1: {model}_{method}_{weighting}_s{seed}.csv
    - P3: {model}_{method}_{weighting}_s{seed}_{strategy}.csv

    Excludes P2 files ({model}_{method}_{weighting}_s{seed}_n{sample_size}.csv)
    """
    name = filepath.stem

    # Build regex pattern from P3_STRATEGIES to avoid matching P2's n{digits}
    strategy_pattern = '|'.join(P3_STRATEGIES)
    p3_pattern = rf'(\w+)_(\w+)_(\w+)_s(\d+)_({strategy_pattern})$'

    # Try P3 pattern first (with strategy suffix)
    match = re.match(p3_pattern, name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'strategy': match.group(5),
        }

    # Try P1 pattern (no strategy = uniform baseline)
    match = re.match(r'(\w+)_(\w+)_(\w+)_s(\d+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'strategy': 'U',  # P1 = uniform baseline
        }

    return None


def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """Load all P1 (as U baseline) and P3 result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    for dataset in ['jigsaw', 'turkey', 'nhanes', 'inaturalist']:
        dataset_dir = results_path / dataset
        if not dataset_dir.exists():
            continue

        # Load all files with 'none' weighting (both classification and regression)
        for csv_file in dataset_dir.glob('*_none_s*.csv'):
            meta = parse_result_filename(csv_file)
            if meta is None:
                continue

            # Skip if not 'none' weighting
            if meta['weighting'] != 'none':
                continue

            # Only include expected models for this dataset (exclude P5's finetune models)
            expected_models = DATASET_MODEL_MAP.get(dataset, [])
            if meta['model'] not in expected_models:
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
    group_cols = ['dataset', 'model', 'method', 'weighting', 'strategy']
    # Filter to columns that exist in df
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
    """Generate markdown table for P3 results."""
    lines = []

    # Classification results
    clf_df = agg_df[agg_df['method'] == 'classification']
    if not clf_df.empty:
        lines.append('## P3: Classification Results\n')
        lines.append('| Dataset | Strategy | Accuracy | Weighted Acc | Expected Cost |')
        lines.append('|---------|----------|----------|--------------|---------------|')

        for dataset in ['jigsaw', 'turkey', 'nhanes', 'inaturalist']:
            dataset_df = clf_df[clf_df['dataset'] == dataset]
            if dataset_df.empty:
                continue

            for strategy in ALL_STRATEGIES:
                row = dataset_df[dataset_df['strategy'] == strategy]
                if row.empty:
                    continue
                row = row.iloc[0]

                acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
                wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))
                cost = format_mean_std(row.get('expected_cost_mean'), row.get('expected_cost_std'))
                lines.append(f"| {dataset} | {strategy} | {acc} | {wacc} | {cost} |")

    # Regression results
    reg_df = agg_df[agg_df['method'] == 'regression']
    if not reg_df.empty:
        lines.append('\n## P3: Regression Results\n')
        lines.append('| Dataset | Strategy | Sign Acc | Weighted Sign Acc | MAE | Weighted MAE |')
        lines.append('|---------|----------|----------|-------------------|-----|--------------|')

        for dataset in ['jigsaw', 'turkey', 'nhanes', 'inaturalist']:
            dataset_df = reg_df[reg_df['dataset'] == dataset]
            if dataset_df.empty:
                continue

            for strategy in ALL_STRATEGIES:
                row = dataset_df[dataset_df['strategy'] == strategy]
                if row.empty:
                    continue
                row = row.iloc[0]

                sacc = format_mean_std(row.get('sign_accuracy_mean'), row.get('sign_accuracy_std'))
                wsacc = format_mean_std(row.get('weighted_sign_accuracy_mean'), row.get('weighted_sign_accuracy_std'))
                mae = format_mean_std(row.get('mae_mean'), row.get('mae_std'), decimals=2)
                wmae = format_mean_std(row.get('weighted_mae_mean'), row.get('weighted_mae_std'), decimals=2)
                lines.append(f"| {dataset} | {strategy} | {sacc} | {wsacc} | {mae} | {wmae} |")

    return '\n'.join(lines)


def generate_strategy_effect_table(agg_df: pd.DataFrame) -> str:
    """Generate table showing strategy effect (strategy - U) by dataset."""
    lines = []

    # Classification effect table
    clf_df = agg_df[agg_df['method'] == 'classification']
    if not clf_df.empty:
        lines.append('\n## Classification Strategy Effect (vs Uniform Baseline)\n')
        lines.append('| Dataset | Strategy | Δ Accuracy | Δ Weighted Acc | Δ Expected Cost |')
        lines.append('|---------|----------|------------|----------------|-----------------|')

        for dataset in ['jigsaw', 'turkey', 'nhanes', 'inaturalist']:
            dataset_df = clf_df[clf_df['dataset'] == dataset]
            if dataset_df.empty:
                continue

            # Get baseline (U)
            u_row = dataset_df[dataset_df['strategy'] == 'U']
            if u_row.empty:
                continue
            u_row = u_row.iloc[0]
            u_acc = u_row.get('accuracy_mean', np.nan)
            u_wacc = u_row.get('weighted_accuracy_mean', np.nan)
            u_cost = u_row.get('expected_cost_mean', np.nan)

            for strategy in P3_STRATEGIES:
                row = dataset_df[dataset_df['strategy'] == strategy]
                if row.empty:
                    continue
                row = row.iloc[0]

                acc = row.get('accuracy_mean', np.nan)
                wacc = row.get('weighted_accuracy_mean', np.nan)
                cost = row.get('expected_cost_mean', np.nan)

                acc_diff = f'{acc - u_acc:+.4f}' if pd.notna(acc) and pd.notna(u_acc) else '-'
                wacc_diff = f'{wacc - u_wacc:+.4f}' if pd.notna(wacc) and pd.notna(u_wacc) else '-'
                cost_diff = f'{cost - u_cost:+.4f}' if pd.notna(cost) and pd.notna(u_cost) else '-'

                lines.append(f'| {dataset} | {strategy} | {acc_diff} | {wacc_diff} | {cost_diff} |')

    # Regression effect table
    reg_df = agg_df[agg_df['method'] == 'regression']
    if not reg_df.empty:
        lines.append('\n## Regression Strategy Effect (vs Uniform Baseline)\n')
        lines.append('| Dataset | Strategy | Δ Sign Acc | Δ Weighted Sign Acc | Δ MAE | Δ Weighted MAE |')
        lines.append('|---------|----------|------------|---------------------|-------|----------------|')

        for dataset in ['jigsaw', 'turkey', 'nhanes', 'inaturalist']:
            dataset_df = reg_df[reg_df['dataset'] == dataset]
            if dataset_df.empty:
                continue

            # Get baseline (U)
            u_row = dataset_df[dataset_df['strategy'] == 'U']
            if u_row.empty:
                continue
            u_row = u_row.iloc[0]
            u_sacc = u_row.get('sign_accuracy_mean', np.nan)
            u_wsacc = u_row.get('weighted_sign_accuracy_mean', np.nan)
            u_mae = u_row.get('mae_mean', np.nan)
            u_wmae = u_row.get('weighted_mae_mean', np.nan)

            for strategy in P3_STRATEGIES:
                row = dataset_df[dataset_df['strategy'] == strategy]
                if row.empty:
                    continue
                row = row.iloc[0]

                sacc = row.get('sign_accuracy_mean', np.nan)
                wsacc = row.get('weighted_sign_accuracy_mean', np.nan)
                mae = row.get('mae_mean', np.nan)
                wmae = row.get('weighted_mae_mean', np.nan)

                sacc_diff = f'{sacc - u_sacc:+.4f}' if pd.notna(sacc) and pd.notna(u_sacc) else '-'
                wsacc_diff = f'{wsacc - u_wsacc:+.4f}' if pd.notna(wsacc) and pd.notna(u_wsacc) else '-'
                mae_diff = f'{mae - u_mae:+.2f}' if pd.notna(mae) and pd.notna(u_mae) else '-'
                wmae_diff = f'{wmae - u_wmae:+.2f}' if pd.notna(wmae) and pd.notna(u_wmae) else '-'

                lines.append(f'| {dataset} | {strategy} | {sacc_diff} | {wsacc_diff} | {mae_diff} | {wmae_diff} |')

    return '\n'.join(lines)


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper (classification only)."""
    clf_df = agg_df[agg_df['method'] == 'classification']
    if clf_df.empty:
        return ''

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{P3 Sampling Strategy Results (mean $\pm$ std over 3 seeds)}')
    lines.append(r'\label{tab:p3-sampling}')
    lines.append(r'\begin{tabular}{llccc}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Strategy & Accuracy & Weighted Acc & Expected Cost \\')
    lines.append(r'\midrule')

    for dataset in ['jigsaw', 'turkey', 'nhanes', 'inaturalist']:
        dataset_df = clf_df[clf_df['dataset'] == dataset]
        if dataset_df.empty:
            continue

        for i, strategy in enumerate(ALL_STRATEGIES):
            row = dataset_df[dataset_df['strategy'] == strategy]
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
            lines.append(f'{dataset_label} & {strategy} & {acc_str} & {wacc_str} & {cost_str} \\\\')

        lines.append(r'\midrule')

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def main():
    print('Loading P3 results (P1 as U baseline)...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P3 results found!')
        return

    # Sanity check: (dataset-model combos) × 5 strategies × N_SEEDS × 2 methods
    n_methods = df['method'].nunique() if 'method' in df.columns else 1
    n_dataset_model_combos = sum(len(models) for models in DATASET_MODEL_MAP.values())
    expected = n_dataset_model_combos * len(ALL_STRATEGIES) * N_SEEDS * n_methods
    if len(df) != expected:
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print(f'  Datasets: {sorted(df["dataset"].unique())}')
    print(f'  Methods: {sorted(df["method"].unique())}')
    print(f'  Strategies: {sorted(df["strategy"].unique())}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    # Save full results
    output_dir = Path('results')
    df.to_csv(output_dir / 'p3_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p3_summary.csv', index=False)
    print(f'  Saved: results/p3_all_runs.csv')
    print(f'  Saved: results/p3_summary.csv')

    # Generate and save LaTeX
    latex = generate_latex_table(agg_df)
    with open(output_dir / 'p3_summary_latex.tex', 'w') as f:
        f.write(latex)
    print(f'  Saved: results/p3_summary_latex.tex')

    # Print markdown tables
    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)

    # Print strategy effect table
    effect_table = generate_strategy_effect_table(agg_df)
    print(effect_table)


if __name__ == '__main__':
    main()
