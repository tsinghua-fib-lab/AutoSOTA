#!/usr/bin/env python
"""
Summarize P6 sampling strategy results on synthetic data.

Compares sampling strategies (U, P_up, Tdown30/50/70) on synthetic data
where costs are predictable from features.

Produces:
- results/p6_sampling_all_runs.csv: Full results table
- results/p6_sampling_summary.csv: Aggregated results (mean ± std over seeds)
- Prints markdown table to stdout

Usage: python scripts/summarize_p6_sampling.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from config import N_SEEDS

STRATEGIES = ['U', 'P_up', 'Tdown30', 'Tdown50', 'Tdown70']


def parse_result_filename(filepath: Path) -> dict:
    """Extract metadata from result filename."""
    name = filepath.stem

    # Try pattern with strategy suffix
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)_(\w+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'strategy': match.group(5),
        }

    # Try pattern without strategy (baseline = U)
    match = re.match(r'(\w+)_(classification|regression)_(\w+)_s(\d+)$', name)
    if match:
        return {
            'model': match.group(1),
            'method': match.group(2),
            'weighting': match.group(3),
            'seed': int(match.group(4)),
            'strategy': 'U',  # Uniform baseline
        }

    return None


def load_results(results_dir: str = 'results/synthetic') -> pd.DataFrame:
    """Load P6 synthetic result CSVs into a single DataFrame."""
    results_path = Path(results_dir)
    rows = []

    if not results_path.exists():
        return pd.DataFrame()

    for csv_file in results_path.glob('logreg_classification_none_s*.csv'):
        meta = parse_result_filename(csv_file)
        if meta is None:
            continue

        # Only include classification with no weighting (to isolate sampling effect)
        if meta['method'] != 'classification' or meta['weighting'] != 'none':
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
            'accuracy': test_row.get('test_accuracy', test_row.get('accuracy')),
            'weighted_accuracy': test_row.get('test_weighted_accuracy', test_row.get('weighted_accuracy')),
            'expected_cost': test_row.get('test_expected_cost', test_row.get('expected_cost')),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds, computing mean ± std."""
    group_cols = ['dataset', 'model', 'method', 'weighting', 'strategy']
    metric_cols = ['accuracy', 'weighted_accuracy', 'expected_cost']

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
    """Generate markdown table for P6 sampling results."""
    lines = []
    lines.append('## P6 Sampling: Synthetic Data Classification Results\n')
    lines.append('| Strategy | Accuracy | Weighted Acc (1-NEC) | NEC |')
    lines.append('|----------|----------|----------------------|-----|')

    for strategy in STRATEGIES:
        row = agg_df[agg_df['strategy'] == strategy]
        if row.empty:
            lines.append(f"| {strategy} | - | - | - |")
            continue
        row = row.iloc[0]

        acc = format_mean_std(row.get('accuracy_mean'), row.get('accuracy_std'))
        wacc = format_mean_std(row.get('weighted_accuracy_mean'), row.get('weighted_accuracy_std'))

        # Compute NEC = 1 - weighted_accuracy
        nec_mean = 1 - row.get('weighted_accuracy_mean', np.nan)
        nec_std = row.get('weighted_accuracy_std', np.nan)
        nec = format_mean_std(nec_mean, nec_std)

        lines.append(f"| {strategy} | {acc} | {wacc} | {nec} |")

    # Add comparison summary
    lines.append('\n### Comparison to Uniform Baseline\n')

    u_row = agg_df[agg_df['strategy'] == 'U']
    if not u_row.empty:
        u_nec = 1 - u_row.iloc[0].get('weighted_accuracy_mean', np.nan)

        for strategy in ['P_up', 'Tdown30', 'Tdown50', 'Tdown70']:
            s_row = agg_df[agg_df['strategy'] == strategy]
            if s_row.empty:
                continue
            s_nec = 1 - s_row.iloc[0].get('weighted_accuracy_mean', np.nan)

            if pd.notna(u_nec) and pd.notna(s_nec) and u_nec > 0:
                rel_change = (u_nec - s_nec) / u_nec * 100
                lines.append(f"- {strategy} vs U: NEC {rel_change:+.1f}% relative change")

    return '\n'.join(lines)


def main():
    print('Loading P6 sampling results...')
    df = load_results()
    print(f'  Found {len(df)} result files')

    if len(df) == 0:
        print('  No P6 sampling results found!')
        print('  Run: bash scripts/run_p6_sampling.sh')
        return

    # Expected: 5 strategies × N_SEEDS
    expected = len(STRATEGIES) * N_SEEDS
    actual_strategies = sorted(df['strategy'].unique())
    print(f'  Strategies: {actual_strategies}')
    print(f'  Seeds: {sorted(df["seed"].unique())}')

    if len(df) < expected:
        missing = set(STRATEGIES) - set(actual_strategies)
        if missing:
            print(f'  WARNING: Missing strategies: {missing}')
        print(f'  WARNING: Expected {expected} runs, found {len(df)}')

    print('\nAggregating across seeds...')
    agg_df = aggregate_by_seed(df)

    output_dir = Path('results')
    df.to_csv(output_dir / 'p6_sampling_all_runs.csv', index=False)
    agg_df.to_csv(output_dir / 'p6_sampling_summary.csv', index=False)
    print(f'  Saved: results/p6_sampling_all_runs.csv')
    print(f'  Saved: results/p6_sampling_summary.csv')

    print('\n' + '=' * 60)
    markdown = generate_markdown_table(agg_df)
    print(markdown)


if __name__ == '__main__':
    main()
