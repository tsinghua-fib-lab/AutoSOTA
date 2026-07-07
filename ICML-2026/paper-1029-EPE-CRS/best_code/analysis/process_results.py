"""
Process raw experiment results to compute additional metrics and reorganize data.

This script:
1. Reads raw results from results/{config_name}/raw/
2. Computes rejection fractions at thresholds 1%, 5%, 10%
3. Removes unnecessary columns (subgroup_id, epe_p, c_p)
4. Consolidates method hyperparameters into a single 'hyperparams' column
5. Creates human-readable dataset names
6. Saves processed results to results/{config_name}/processed/{dataset_label}.pkl

Usage:
    python process_results.py <config_name> [--n_jobs <n>]

Example:
    python process_results.py 5 --n_jobs 8
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
from collections import defaultdict

# Suppress all warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')  # Suppress numpy warnings (divide by zero, invalid value, etc.)
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings

from evaluation.run_experiment import get_job_list
from config.constants import COL_NAMES


# Known hyperparameter columns for each method
METHOD_HYPERPARAM_COLS = {
    'ddgroup': ['core_size', 'rejection_threshold'],
    'c_ind_ddgroup': ['core_size', 'rejection_threshold'],
    'pl_ddgroup': ['core_size', 'rejection_threshold'],
    'no_exp_ddgroup': ['core_size'],
    'cox_tree': ['max_depth', 'min_samples_leaf', 'max_splits_per_feature', 'num_subgroups'],
    'prim': ['peeling_frac', 'min_support_size', 'num_subgroups'],
    'survival_tree': ['max_depth', 'min_samples_leaf', 'num_subgroups'],
    'random': ['num_subgroups', 'seeds'],
    'base': [],
    'ost': ['regularization', 'minimum_captured_points', 'depth_budget', 'num_subgroups'],
}

# Columns to remove from final output
COLS_TO_DROP = ['subgroup_id', 'train_epe_p', 'train_c_p', 'test_epe_p', 'test_c_p']

# Rejection fraction thresholds
REJ_THRESHOLDS = [0.01, 0.05, 0.10]


def create_dataset_label(job):
    """Create human-readable dataset label from job info."""
    dataset = job['dataset']
    adjust_names = [COL_NAMES[dataset][i] for i in job['adjust_cols']]
    subgp_names = [COL_NAMES[dataset][i] for i in job['subgp_cols']]
    return f"{dataset}-{adjust_names}-{subgp_names}"


def extract_hyperparams(row, method):
    """Extract method-specific hyperparameters into a dictionary."""
    hyperparam_cols = METHOD_HYPERPARAM_COLS.get(method, [])
    hyperparams = {}
    for col in hyperparam_cols:
        if col in row.index:
            hyperparams[col] = row[col]
    return hyperparams


def process_result_row(row, method, job):
    """Process a single result row to add hyperparams and job metadata."""
    # Extract hyperparameters
    hyperparams = extract_hyperparams(row, method)

    # Create new row with all data
    new_row = row.copy()
    new_row['hyperparams'] = hyperparams

    # Add job metadata that's needed downstream (for backward compatibility with old runs)
    if 'adjust_cols' not in new_row.index:
        new_row['adjust_cols'] = job['adjust_cols']
    if 'subgp_cols' not in new_row.index:
        new_row['subgp_cols'] = job['subgp_cols']

    return new_row


def process_config(config_name, config_dir='../experiments/configs', results_dir='../results'):
    """Process all results for a given config."""

    print(f"Processing config: {config_name}")

    # Load config and generate job list
    config_path = f"{config_dir}/{config_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    job_list = get_job_list(config_name, config_dir=config_dir)
    print(f"Total jobs in config: {len(job_list)}")

    # Group results by dataset label
    dataset_results = defaultdict(list)

    print("\nCollecting and processing results...")
    for job_idx, job in enumerate(tqdm(job_list)):
        dataset_label = create_dataset_label(job)

        # Load raw results
        results_path = f"{results_dir}/{config_name}/raw/{job_idx}_results.pkl"

        try:
            raw_df = pd.read_pickle(results_path)
        except FileNotFoundError:
            continue
        except Exception:
            continue

        # Process each row in the results
        method = job['method']
        processed_rows = []

        for _, row in raw_df.iterrows():
            try:
                processed_row = process_result_row(row, method, job)
                processed_rows.append(processed_row)
            except Exception:
                continue

        if processed_rows:
            processed_df = pd.DataFrame(processed_rows)
            processed_df['method'] = method
            dataset_results[dataset_label].append(processed_df)

    # Combine and save results by dataset label
    print("\nSaving processed results...")
    os.makedirs(f"{results_dir}/{config_name}/processed", exist_ok=True)

    for dataset_label, df_list in dataset_results.items():
        if not df_list:
            continue

        # Combine all results for this dataset
        combined_df = pd.concat(df_list, ignore_index=True)

        # Drop unnecessary columns
        cols_to_drop_present = [col for col in COLS_TO_DROP if col in combined_df.columns]
        combined_df = combined_df.drop(columns=cols_to_drop_present)

        # Drop individual hyperparameter columns (now in 'hyperparams' dict)
        all_hyperparam_cols = set()
        for cols in METHOD_HYPERPARAM_COLS.values():
            all_hyperparam_cols.update(cols)
        hyperparam_cols_present = [col for col in all_hyperparam_cols if col in combined_df.columns]
        combined_df = combined_df.drop(columns=hyperparam_cols_present)

        # Reorder columns for better readability
        column_order = ['method', 'seed', 'subgp_cols', 'adjust_cols', 'hyperparams', 'R', 'beta']
        metric_cols = [col for col in combined_df.columns if col.startswith(('train_', 'test_', 'f1'))]
        other_cols = [col for col in combined_df.columns if col not in column_order + metric_cols]

        final_column_order = column_order + sorted(metric_cols) + other_cols
        final_column_order = [col for col in final_column_order if col in combined_df.columns]

        combined_df = combined_df[final_column_order]

        # Save
        output_path = f"{results_dir}/{config_name}/processed/{dataset_label}.pkl"
        pd.to_pickle(combined_df, output_path)
        print(f"  Saved {len(combined_df)} rows to {output_path}")

    print(f"\nProcessing complete! Processed {len(dataset_results)} unique dataset configurations.")


def main():
    parser = argparse.ArgumentParser(description='Process experiment results')
    parser.add_argument('config_name', type=str, help='Name of config to process (e.g., "5" for configs/5.yaml)')
    parser.add_argument('--config_dir', type=str, default='../experiments/configs', help='Directory containing config files')
    parser.add_argument('--results_dir', type=str, default='../results', help='Directory containing results')

    args = parser.parse_args()

    process_config(args.config_name, args.config_dir, args.results_dir)


if __name__ == '__main__':
    main()
