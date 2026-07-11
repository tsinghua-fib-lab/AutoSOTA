import os
import sys

# Get the directory one level above the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import pandas as pd
import argparse
import gc
import numpy as np
import jax
from run_metrics import main as metrics
from maskedit.datasets.dataset import dataset
from sample import sample

def split_combinations(df, N=3, min_count=200, random_state=None):

    # Prepare the dataset to filter by component combos
    presence_mask = df.notna()

    pattern = presence_mask.apply(lambda row: tuple(row), axis=1)
    df['component_pattern'] = pattern

    grouped = (
        df.groupby('component_pattern')
        .apply(lambda g: g.index.tolist())
        .reset_index(name='row_indices')
    )
    grouped['count'] = grouped['row_indices'].apply(len)

    # print(grouped)

    colnames = df.columns[:-1]  # exclude the added pattern column

    grouped['present_components'] = grouped['component_pattern'].apply(
        lambda p: [col for col, has in zip(colnames, p) if has]
    )

    # Sort by complexity and sample N with varying number of components
    grouped['n_components'] = grouped['component_pattern'].apply(sum)

    # Keep only topologies with enough members
    eligible = grouped[grouped['count'] >= min_count].copy()

    print(f"\n--- Eligible Combinations (Count >= {min_count}) ---")
    # Only print the relevant columns for clarity
    print(eligible[['n_components', 'count']].to_string(index=False))
    print("--------------------------------------------------\n")

    if eligible.empty:
        raise ValueError("No combinations have at least min_count members.")
    
    eligible = eligible.sort_values('n_components')

    selected = eligible.sample(n=N, random_state=random_state).reset_index(drop=True)

    # Build sub-datasets
    datasets = {}
    for i, row in selected.iterrows():
        pattern_tuple = row['component_pattern']
        row_indices = row['row_indices']

        # Include all components (keep NaNs for missing ones, but remove added column)
        subset = df.loc[row_indices, df.columns[:-1]]

        datasets["combo_"+str(i)+"_n"+str(row['n_components'])+"_count"+str(row['count'])] = subset

    return datasets, selected


def main(settings):

    parser = argparse.ArgumentParser(description="Compute MMD, C2ST, marginal C2ST metrics, and MSE accuracy.")

    parser.add_argument(
        "--data",
        default="../training_data/maskedit/data/train_set.csv",
        help="Path to the SUAVE data CSV file."
    )

    parser.add_argument(
        "--test",
        default="../training_data/maskedit/data/test_set.csv",
        help="Path to the SUAVE data CSV file."
    )

    parser.add_argument(
        "--output",
        default="../output/metrics",
        help="Output file name for metrics."
    )

    parser.add_argument(
        "--indices",
        default="./training_data/maskedit/data/train_indices.pkl",
        help="Path to the indices pickle file."
    )

    parser.add_argument(
        "--test_indices",
        default="./training_data/maskedit/data/test_indices.pkl",
        help="Path to the test indices pickle file."
    )

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)

    # Choose 4 components of increasing complexity with at least 200 data points each
    datasets, summary = split_combinations(data, N=10, min_count=250, random_state=42)

    plot_output_dir = args.output + "_plots"

    # View summary of what was chosen
    print(summary[['n_components', 'count']])

    # For each dataset
    ind_mask = np.r_[0:71, 82:data.shape[0]]
    ii = 0
    samples = []
    all_results = {}
    for name, d in datasets.items():
        print(name+": shape "+str(d.shape))

        # Sample the corresponding topologies
        samples.append(sample(settings,d))

        # Test metrics in normalized space
        datastore = samples[ii]

        # Likelihood
        res_data = datastore.main.norm[:,72:82,0]
        res_samples = datastore.sets["samples"].norm[:,72:82,0]
        des_data = datastore.main.norm[:,ind_mask,0]
        des_samples = datastore.sets["samples"].norm[:,ind_mask,0]
        
        result_dict = metrics(res_data, res_samples,
                42, args.output+"_"+name+".txt")

        all_results[name] = result_dict
        
        ii += 1
        # GPU memory cleanup between topologies to prevent OOM
        jax.clear_caches()
        gc.collect()
    print("Creating summary DataFrame...")
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    
    # 5. Define the final CSV output path
    #    (Using a single summary filename is better practice)
    output_csv_path = args.output + "_metrics_summary.csv"
    
    # 6. Save the complete DataFrame to one CSV file
    results_df.to_csv(output_csv_path)
    
    print(f"Successfully saved metrics summary to: {output_csv_path}")
    print("\n--- Metrics Summary ---")
    print(results_df)


if __name__ == "__main__":
    args = {
        # Environment / paths
        "cuda_visible_devices": "0",
        "sys_path_parent": "..",

        # Data 
        "project_root": "../training_data/maskedit/data/",
        "data_csv": "train_set.csv",
        "indices_pkl": "train_indices.pkl",

        # Model restore
        "checkpoint_dir": "../training_data/maskedit/model/MaskeditModel",
        "train_args_json": None,
        "multicomponent": False,

        # Sampling
        "sample_steps": 500,
        "seed": 0,

        # Fallback model-shape flags
        "nlayers": 16,
        "dim_value": 64,
        "dim_id": 64,
        "dim_condition": 32,
        "num_heads": 2,
        "time_dim": 64,
        "split": 0.8,
        "dropout_rate": 0.1,

        # Diffusion params
        "sigma": 2.5,
    }

    main(args)