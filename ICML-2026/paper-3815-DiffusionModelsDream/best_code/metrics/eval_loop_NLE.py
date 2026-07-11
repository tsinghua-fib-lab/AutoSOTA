import os
import sys

# Get the directory one level above the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import pandas as pd
import argparse
import numpy as np
from run_metrics import main as metrics
from run_NLE import train_and_sample
import hashlib

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

    # print(grouped[['present_components', 'row_indices', 'count']])

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

        pattern_str = "".join(["1" if b else "0" for b in pattern_tuple])
        # MD5 hash (shortened)
        tid = hashlib.md5(pattern_str.encode()).hexdigest()[:8]

        datasets["combo_"+tid+"_n"+str(row['n_components'])+"_count"+str(row['count'])] = subset

    return datasets, selected

def match_combinations(
    df,
    selected,
    strict=True
):
    """
    Match component topologies from `selected` (output of split_combinations)
    against a new dataframe `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Second dataset to match against.
    selected : pd.DataFrame
        The `selected` output from split_combinations.
    strict : bool, default=True
        If True, raise an error when a topology has zero matches.
        If False, return empty subsets for unmatched topologies.

    Returns
    -------
    datasets : dict[str, pd.DataFrame]
        Dictionary of matched sub-datasets keyed by topology name.
    summary : pd.DataFrame
        Per-topology match counts.
    """

    # Recompute component patterns for df (DO NOT mutate input)
    presence_mask = df.notna()
    patterns = presence_mask.apply(lambda row: tuple(row), axis=1)

    colnames = df.columns
    datasets = {}
    summary_rows = []

    for i, row in selected.iterrows():
        pattern = row["component_pattern"]
        n_components = row["n_components"]

        match_idx = patterns[patterns == pattern].index
        subset = df.loc[match_idx, colnames]
        pattern_str = "".join(["1" if b else "0" for b in pattern])
        # MD5 hash (shortened)
        tid = hashlib.md5(pattern_str.encode()).hexdigest()[:8]

        key = f"combo_{tid}_n{n_components}_count{len(subset)}"
        datasets[key] = subset

        summary_rows.append({
            "combo": key,
            "n_components": n_components,
            "matched_count": len(subset)
        })

        if strict and len(subset) == 0:
            raise ValueError(
                f"No matches found for topology {i} "
                f"(n_components={n_components})"
            )

    summary = pd.DataFrame(summary_rows)
    return datasets, summary


def main():

    parser = argparse.ArgumentParser(description="Compute MMD, C2ST, marginal C2ST metrics, and MSE accuracy.")

    parser.add_argument(
        "--data",
        default="./training_data/maskedit/data/train_set.csv",
        help="Path to the SUAVE data CSV file."
    )

    parser.add_argument(
        "--test",
        default="./training_data/maskedit/data/test_set.csv",
        help="Path to the SUAVE data CSV file."
    )

    parser.add_argument(
        "--output",
        default="simformer_NLE_metrics_test",
        help="Output file name for metrics."
    )

    parser.add_argument(
        "--indices",
        default="./training_data/maskedit/data/test_indices_01.pkl",
        help="Path to the indices pickle file."
    )

    parser.add_argument(
        "--test_indices",
        default="./training_data/maskedit/data/test_indices_23.pkl",
        help="Path to the test indices pickle file."
    )

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)
    test_set = pd.read_csv(args.test)

    # Choose 4 components of increasing complexity with at least 200 data points each
    test_data, summary = split_combinations(test_set, N=10, min_count=250, random_state=42)

    datasets, main_summary = match_combinations(data,summary)

    plot_output_dir = args.output + "_plots"

    # View summary of what was chosen
    print(summary[['n_components', 'count']])

    # For each dataset
    sim_size = datasets[next(iter(datasets))].shape[1]
    ind_mask = np.r_[0:71, 82:sim_size]
    ii = 0
    samples = []
    all_results = {}

    for name, d in datasets.items():

        key = name[:14]
        full_key = next((k for k in test_data if k.startswith(key)), None)
        t = test_data[full_key]

        print(f"{name}: train {d.shape}, test {t.shape}")

        # Sample the corresponding topologies
        nan_cols = d.columns[d.isna().all()]
        nan_idx = d.columns.get_indexer(nan_cols)
        ind_mask_clean = np.setdiff1d(ind_mask, nan_idx)
        d_np = d.to_numpy()

        # Do the same with the test dataset
        t_np = t.to_numpy()

        # Normalize the datasets
        train_theta = d_np[:,ind_mask_clean]
        train_x = d_np[:,72:82]

        test_theta = t_np[:,ind_mask_clean]
        test_x = t_np[:,72:82]

        mu_x = train_x.mean(axis=0)
        std_x = np.std(train_x,axis=0)
        mu_theta = train_theta.mean(axis=0)
        std_theta = np.std(train_theta,axis=0)

        train_theta_norm = (train_theta-mu_theta)/std_theta
        train_x_norm = (train_x-mu_x)/std_x

        test_theta_norm = (test_theta-mu_theta)/std_theta
        test_x_norm = (test_x-mu_x)/std_x
        
        samples.append(train_and_sample(train_theta_norm,train_x_norm,Nsamples=test_theta_norm.shape[0]))
        
        result_dict = metrics(test_x_norm, samples[-1],
                42, args.output+"_"+name+".txt")

        all_results[name] = result_dict
        
        ii += 1
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
    print(summarize_metrics([results_df]))


if __name__ == "__main__":
    main()