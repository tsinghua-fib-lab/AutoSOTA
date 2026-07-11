#!/usr/bin/env python3
"""
Preprocess the CellTag/Biddy reprogramming dataset for LCL training.
Matches the paper's preprocessing from biddy_data_scanpy_final.ipynb
and biddy_train_test_split_final.ipynb.
"""
import scanpy as sc
import anndata as ad
import numpy as np
import argparse
import os

def preprocess(raw_h5ad_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading raw data from {raw_h5ad_path}...")
    adata = ad.read_h5ad(raw_h5ad_path)
    print(f"Raw data shape: {adata.shape}")
    print(f"Raw obs columns: {list(adata.obs.columns)}")

    # --- Step 1: Filter cells without CellTagD0_48k barcode ---
    nan_rows = adata.obs['CellTagD0_48k'].isna()
    print(f"Cells without CellTagD0_48k barcode: {sum(nan_rows)}")
    adata_filter = adata[~nan_rows].copy()
    print(f"After filtering out clone id nans: {adata_filter.shape}")

    # --- Step 2: Filter cells without cell_type ---
    if adata_filter.obs['cell_type'].isna().any():
        adata_filter = adata_filter[adata_filter.obs['cell_type'].notnull()].copy()
        print(f"After filtering out cell type nans: {adata_filter.shape}")

    # --- Step 3: Filter lineages with < 5 cells ---
    value_counts = adata_filter.obs['CellTagD0_48k'].value_counts()
    valid_tags = value_counts[value_counts >= 5].index
    adata_filter = adata_filter[adata_filter.obs['CellTagD0_48k'].isin(valid_tags)].copy()
    print(f"Number of lineages with >= 5 cells: {len(valid_tags)}")
    print(f"After min-lineage filter: {adata_filter.n_obs}")

    # --- Step 4: Set clone_id ---
    adata_filter.obs["clone_id"] = adata_filter.obs["CellTagD0_48k"]
    print(f"Number of unique lineages: {len(adata_filter.obs['clone_id'].unique())}")

    # --- Step 5: Basic gene/cell filtering ---
    sc.pp.filter_cells(adata_filter, min_genes=200)
    sc.pp.filter_genes(adata_filter, min_cells=3)

    # --- Step 6: Log-normalization ---
    sc.pp.normalize_total(adata_filter, target_sum=1e4)
    sc.pp.log1p(adata_filter)

    # --- Step 7: Select top 2000 HVG ---
    sc.pp.highly_variable_genes(adata_filter, n_top_genes=2000)
    adata_filter = adata_filter[:, adata_filter.var.highly_variable]
    print(f"After HVG selection: {adata_filter.shape}")

    # Save preprocessed full data
    preprocessed_path = os.path.join(output_dir, "biddy_6534_2000_norm_log.h5ad")
    adata_filter.write(preprocessed_path)
    print(f"Saved preprocessed data to {preprocessed_path}")

    # --- Step 8: Train/Test Split ---
    # Lineages with >=10 cells: 10% random test, rest train
    # Lineages with <10 cells: all train
    clone_id_counts = adata_filter.obs['clone_id'].value_counts()

    test_indices = []
    train_indices = []

    np.random.seed(42)  # Match paper's train_test_seed

    for clone_id, count in clone_id_counts.items():
        clone_indices = adata_filter.obs[adata_filter.obs['clone_id'] == clone_id].index

        if count >= 10:
            test_size = int(np.ceil(0.1 * count))
            test_clone_indices = np.random.choice(clone_indices, size=test_size, replace=False)
            test_indices.extend(test_clone_indices)
            train_clone_indices = list(set(clone_indices) - set(test_clone_indices))
            train_indices.extend(train_clone_indices)
        else:
            train_indices.extend(clone_indices)

    adata_train = adata_filter[train_indices, :].copy()
    adata_test = adata_filter[test_indices, :].copy()

    print(f"Train shape: {adata_train.shape}, Test shape: {adata_test.shape}")
    print(f"Train lineages: {len(adata_train.obs['clone_id'].unique())}")
    print(f"Test lineages: {len(adata_test.obs['clone_id'].unique())}")

    train_path = os.path.join(output_dir, "Biddy_train.h5ad")
    test_path = os.path.join(output_dir, "Biddy_test.h5ad")

    adata_train.write(train_path)
    adata_test.write(test_path)

    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_h5ad", required=True, help="Path to raw reprogramming_morris.h5ad")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed files")
    args = parser.parse_args()
    preprocess(args.raw_h5ad, args.output_dir)
