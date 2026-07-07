"""
Generate summary tables from multiple experiment configs combined.

Combines processed results from multiple configs and creates summary tables
with options to filter by dataset performance.

Usage:
    python gen_combined_tables.py 5 6 --size_threshold 0.05 [--filter good] [--datasets <pattern>] [--no-rejection-fraction]

Example:
    python gen_combined_tables.py 5 6 --size_threshold 0.05 --filter good
    python gen_combined_tables.py 5 6 --size_threshold 0.05 --filter all
    python gen_combined_tables.py 5 6 --size_threshold 0.05 --datasets "veterans*"
    python gen_combined_tables.py 5 6 --size_threshold 0.05 --no-rejection-fraction  # Skip rejection fraction computation
"""

import sys
import os
sys.path.append(os.path.abspath("../src"))

import argparse
import yaml
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from glob import glob

# Suppress all warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from data.load import load_data
from utils.metrics import rej_frac
from evaluation.run_experiment import get_job_list
from config.constants import COL_NAMES, RANDOM_EPE


REJ_THRESHOLDS = [0.01, 0.05, 0.10]


def create_dataset_label(job):
    """Create human-readable dataset label from job info."""
    dataset = job['dataset']
    adjust_names = [COL_NAMES[dataset][i] for i in job['adjust_cols']]
    subgp_names = [COL_NAMES[dataset][i] for i in job['subgp_cols']]
    return f"{dataset}-{adjust_names}-{subgp_names}"


def select_best_regions(df, size_threshold, selection_metric='train_epe'):
    """
    For each (method, seed), select the best region based on selection_metric.
    """
    selected_rows = []

    for (method, seed), group in df.groupby(['method', 'seed']):
        # Filter by size threshold
        filtered = group[group['train_size'] >= size_threshold]

        if len(filtered) == 0:
            continue

        # Select best region (lowest selection_metric)
        best_idx = filtered[selection_metric].idxmin()
        best_row = filtered.loc[best_idx].copy()

        selected_rows.append(best_row)

    return pd.DataFrame(selected_rows)


def compute_rejection_fractions_for_row(row, job, config_dir='../experiments/configs', n_jobs=-1):
    """Compute rejection fractions for a single selected region."""
    try:
        # Load data for this job
        X_adjust, X_subgp, Y, X_adjust_test, X_subgp_test, Y_test, _, _ = load_data(
            job['dataset'],
            job['adjust_cols'],
            job['subgp_cols'],
            job['seed'],
            n=job.get('dataset_hyper', None)
        )

        R = row['R']
        beta = row['beta']

        # Compute train rejection fractions
        train_rej = rej_frac(X_adjust, X_subgp, Y, R, beta, REJ_THRESHOLDS, n_jobs=n_jobs)

        # Compute test rejection fractions
        test_rej = rej_frac(X_adjust_test, X_subgp_test, Y_test, R, beta, REJ_THRESHOLDS, n_jobs=n_jobs)

        return {
            'train_rej_01': train_rej[0],
            'train_rej_05': train_rej[1],
            'train_rej_10': train_rej[2],
            'test_rej_01': test_rej[0],
            'test_rej_05': test_rej[1],
            'test_rej_10': test_rej[2]
        }
    except Exception:
        return {
            'train_rej_01': np.nan,
            'train_rej_05': np.nan,
            'train_rej_10': np.nan,
            'test_rej_01': np.nan,
            'test_rej_05': np.nan,
            'test_rej_10': np.nan
        }


def add_rejection_fractions(selected_df, job_list, config_name, config_dir='../experiments/configs', n_jobs=-1):
    """Add rejection fractions to selected regions."""
    # Create mapping from (dataset, adjust_cols, subgp_cols, seed, method) -> job
    job_map = {}
    for job in job_list:
        key = (
            job['dataset'],
            tuple(job['adjust_cols']),
            tuple(job['subgp_cols']),
            job['seed'],
            job['method']
        )
        job_map[key] = job

    # Compute rejection fractions for each selected region
    results = []
    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc=f"Computing rejection fractions (config {config_name})"):
        key = (
            row.get('dataset', None),
            tuple(row['adjust_cols']),
            tuple(row['subgp_cols']),
            row['seed'],
            row['method']
        )

        # Find matching job
        job = None
        if key in job_map:
            job = job_map[key]
        else:
            # Try without dataset
            for j in job_list:
                if (tuple(j['adjust_cols']) == key[1] and
                    tuple(j['subgp_cols']) == key[2] and
                    j['seed'] == key[3] and
                    j['method'] == key[4]):
                    job = j
                    break

        if job is None:
            rej_dict = {
                'train_rej_01': np.nan,
                'train_rej_05': np.nan,
                'train_rej_10': np.nan,
                'test_rej_01': np.nan,
                'test_rej_05': np.nan,
                'test_rej_10': np.nan
            }
        else:
            rej_dict = compute_rejection_fractions_for_row(row, job, config_dir, n_jobs)

        # Add rejection fractions to row
        new_row = row.copy()
        for k, v in rej_dict.items():
            new_row[k] = v
        results.append(new_row)

    return pd.DataFrame(results)


def compute_summary_stats(df, methods, compute_rejection=True):
    """Compute mean and SEM for test metrics grouped by method."""
    metrics = [('test_epe', 'Test EPE')]

    if compute_rejection:
        metrics.extend([
            ('test_rej_01', 'Test Rej@1%'),
            ('test_rej_05', 'Test Rej@5%'),
            ('test_rej_10', 'Test Rej@10%'),
        ])

    metrics.extend([
        ('test_c_ind', 'Test C-Index'),
        ('test_pll', 'Test PLL'),
        ('test_size', 'Test Size'),
        ('test_precision', 'Test Precision'),
        ('test_recall', 'Test Recall'),
        ('test_f1', 'Test F1 Score'),
        ('test_iou', 'Test IoU')
    ])

    summary_data = []

    for metric_col, metric_name in metrics:
        row_data = {'Metric': metric_name}

        for method in methods:
            if metric_col in df.columns:
                method_data = df[df['method'] == method][metric_col]

                if len(method_data) == 0:
                    row_data[method] = 'N/A'
                else:
                    mean_val = method_data.mean()
                    sem_val = method_data.sem()
                    row_data[method] = f"{mean_val:.3f} ({sem_val:.3f})"

        summary_data.append(row_data)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Metric')

    return summary_df


def extract_mean_from_string(s):
    """Extract mean value from 'mean (sem)' string."""
    try:
        return float(s.split('(')[0].strip())
    except:
        return np.nan


def filter_good_datasets(summary_tables):
    """Filter datasets where ddgroup outperforms base, random, and RANDOM_EPE."""
    good_datasets = []

    for dataset_label, df in summary_tables.items():
        # Check if Test EPE row exists and required methods exist
        if 'Test EPE' not in df.index:
            continue

        test_epe_row = df.loc['Test EPE']

        # Extract mean values
        ddgroup_epe = extract_mean_from_string(test_epe_row.get('ddgroup', 'nan'))
        base_epe = extract_mean_from_string(test_epe_row.get('base', 'nan'))
        random_epe = extract_mean_from_string(test_epe_row.get('random', 'nan'))

        # Check if ddgroup is better than both base and random, and lower than RANDOM_EPE
        if (not np.isnan(ddgroup_epe) and
            not np.isnan(base_epe) and
            not np.isnan(random_epe)):

            if (ddgroup_epe < base_epe and
                ddgroup_epe < random_epe and
                ddgroup_epe < RANDOM_EPE):
                good_datasets.append(dataset_label)

    return good_datasets


def combine_configs(config_names, results_dir='../results'):
    """
    Combine processed results from multiple configs.
    Only includes datasets that appear in ALL configs.

    Returns dictionary: {dataset_label: combined_dataframe}
    """
    # First pass: collect all datasets from each config
    datasets_by_config = {}

    for config_name in config_names:
        processed_dir = f"{results_dir}/{config_name}/processed"
        processed_files = glob(f"{processed_dir}/*.pkl")

        datasets_by_config[config_name] = {}
        for processed_file in processed_files:
            dataset_label = os.path.basename(processed_file).replace('.pkl', '')
            datasets_by_config[config_name][dataset_label] = processed_file

    # Find datasets that appear in ALL configs
    if not datasets_by_config:
        return {}

    # Get intersection of dataset labels across all configs
    common_datasets = set(datasets_by_config[config_names[0]].keys())
    for config_name in config_names[1:]:
        common_datasets = common_datasets.intersection(set(datasets_by_config[config_name].keys()))

    print(f"Found {len(common_datasets)} datasets common to all {len(config_names)} configs")

    # Second pass: combine data only for common datasets
    combined_data = {}

    for dataset_label in common_datasets:
        df_list = []
        for config_name in config_names:
            processed_file = datasets_by_config[config_name][dataset_label]
            df = pd.read_pickle(processed_file)
            df_list.append(df)

        combined_data[dataset_label] = pd.concat(df_list, ignore_index=True)

    return combined_data


def process_combined_configs(config_names, size_threshold, selection_metric='train_epe',
                             dataset_pattern='*', filter_mode='all',
                             config_dir='../experiments/configs', results_dir='../results', n_jobs=-1,
                             compute_rejection=True):
    """Process combined results from multiple configs and generate summary tables."""

    print(f"Generating combined summary tables for configs: {', '.join(config_names)}")
    print(f"Size threshold: {size_threshold}")
    print(f"Selection metric: {selection_metric}")
    print(f"Dataset pattern: {dataset_pattern}")
    print(f"Filter mode: {filter_mode}")
    print(f"Compute rejection fractions: {compute_rejection}")
    print(f"RANDOM_EPE: {RANDOM_EPE:.3f}\n")

    # Load all job lists
    all_job_lists = {}
    for config_name in config_names:
        all_job_lists[config_name] = get_job_list(config_name, config_dir=config_dir)

    # Combine processed results
    print("Combining processed results from all configs...")
    combined_datasets = combine_configs(config_names, results_dir)

    # Filter by dataset pattern
    if dataset_pattern != '*':
        from fnmatch import fnmatch
        combined_datasets = {k: v for k, v in combined_datasets.items() if fnmatch(k, dataset_pattern)}

    print(f"Found {len(combined_datasets)} unique datasets\n")

    if len(combined_datasets) == 0:
        print("No datasets found! Run process_results.py for all configs first.")
        return

    # Process each dataset
    summary_tables = {}

    for dataset_label, df in combined_datasets.items():
        print(f"Processing: {dataset_label}")

        # Select best regions
        selected_df = select_best_regions(df, size_threshold, selection_metric)

        if len(selected_df) == 0:
            print(f"  No regions found with train_size >= {size_threshold}")
            continue

        # Add rejection fractions if requested
        if compute_rejection:
            # Need to figure out which config each row came from
            # For simplicity, try all job lists
            all_selected = []
            for config_name, job_list in all_job_lists.items():
                config_rows = []
                for _, row in selected_df.iterrows():
                    # Check if this row's method exists in this config's job list
                    methods_in_config = set(job['method'] for job in job_list)
                    if row['method'] in methods_in_config:
                        config_rows.append(row)

                if config_rows:
                    config_selected_df = pd.DataFrame(config_rows)
                    config_selected_df = add_rejection_fractions(config_selected_df, job_list, config_name, config_dir, n_jobs)
                    all_selected.append(config_selected_df)

            if not all_selected:
                print(f"  Could not match rows to configs")
                continue

            selected_df = pd.concat(all_selected, ignore_index=True)

            # Drop duplicates based on specific columns (avoiding list columns)
            duplicate_check_cols = ['method', 'seed']
            if 'subgroup_id' in selected_df.columns:
                duplicate_check_cols.append('subgroup_id')
            selected_df = selected_df.drop_duplicates(subset=duplicate_check_cols)

        # Get unique methods
        methods = sorted(selected_df['method'].unique())

        # Compute summary statistics
        summary_df = compute_summary_stats(selected_df, methods, compute_rejection)
        summary_tables[dataset_label] = summary_df

    # Apply filter
    if filter_mode == 'good':
        good_datasets = filter_good_datasets(summary_tables)
        summary_tables = {k: v for k, v in summary_tables.items() if k in good_datasets}
        print(f"\nFiltered to {len(summary_tables)} datasets where ddgroup outperforms baselines\n")
    elif filter_mode == 'all':
        print(f"\nShowing all {len(summary_tables)} datasets\n")

    # Save tables
    output_suffix = '_'.join(config_names)
    output_dir = f"{results_dir}/combined_{output_suffix}/tables/size_{size_threshold:.2f}"
    os.makedirs(output_dir, exist_ok=True)

    for dataset_label, summary_df in summary_tables.items():
        # Save in multiple formats
        summary_df.to_pickle(f"{output_dir}/{dataset_label}.pkl")
        summary_df.to_csv(f"{output_dir}/{dataset_label}.csv")
        summary_df.to_latex(f"{output_dir}/{dataset_label}.tex")

        print(f"Saved: {dataset_label}")

    print(f"\nSummary tables saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate combined summary tables from multiple configs')
    parser.add_argument('config_names', type=str, nargs='+', help='Names of configs to combine (e.g., 5 6)')
    parser.add_argument('--size_threshold', type=float, default=0.1,
                       help='Minimum train_size threshold (default: 0.1)')
    parser.add_argument('--selection_metric', type=str, default='train_epe',
                       help='Metric to use for region selection (default: train_epe)')
    parser.add_argument('--datasets', type=str, default='*',
                       help='Glob pattern for dataset files (default: * = all)')
    parser.add_argument('--filter', type=str, choices=['all', 'good'], default='all',
                       help='Filter mode: "all" shows all datasets, "good" shows only datasets where ddgroup outperforms baselines')
    parser.add_argument('--no-rejection-fraction', action='store_true',
                       help='Skip computation of rejection fractions (faster, excludes rej metrics from output)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs for rejection fraction computation (default: -1)')
    parser.add_argument('--config_dir', type=str, default='../experiments/configs',
                       help='Directory containing config files')
    parser.add_argument('--results_dir', type=str, default='../results',
                       help='Directory containing results')

    args = parser.parse_args()

    process_combined_configs(
        args.config_names,
        args.size_threshold,
        args.selection_metric,
        args.datasets,
        args.filter,
        args.config_dir,
        args.results_dir,
        args.n_jobs,
        compute_rejection=not args.no_rejection_fraction
    )


if __name__ == '__main__':
    main()
