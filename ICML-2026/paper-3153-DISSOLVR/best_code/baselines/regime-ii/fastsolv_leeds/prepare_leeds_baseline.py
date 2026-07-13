# prepare_leeds_baseline.py
"""
Prepare data for FastSolv Leeds baseline evaluation:
1. Merge BigSolDB 1.0 train + test to create full training set
2. Generate fastprop features for Leeds test data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem


def get_descs(src_df: pd.DataFrame, solute_col='solute_smiles', solvent_col='solvent_smiles'):
    """Calculate molecular descriptors for solutes and solvents (Vectorized)"""
    # 1. Get unique SMILES and calculate descriptors once
    unique_smiles = np.unique(np.hstack((
        src_df[solvent_col].unique(), 
        src_df[solute_col].unique()
    )))
    
    print(f"Calculating descriptors for {len(unique_smiles)} unique molecules...")
    descs = get_descriptors(
        False, 
        ALL_2D, 
        [Chem.MolFromSmiles(smi) for smi in unique_smiles]
    ).to_numpy(dtype=np.float32)
    
    # 2. Create a lookup DataFrame
    desc_df = pd.DataFrame(descs, columns=ALL_2D)
    desc_df['smiles_key'] = unique_smiles
    
    # 3. Merge descriptors for Solutes
    print("Mapping solute descriptors...")
    solute_desc_cols = {d: "solute_" + d for d in ALL_2D}
    solute_lookup = desc_df.rename(columns=solute_desc_cols)
    
    merged_df = src_df.merge(
        solute_lookup, 
        left_on=solute_col, 
        right_on='smiles_key', 
        how='left'
    ).drop(columns=['smiles_key'])
    
    # 4. Merge descriptors for Solvents
    print("Mapping solvent descriptors...")
    solvent_desc_cols = {d: "solvent_" + d for d in ALL_2D}
    solvent_lookup = desc_df.rename(columns=solvent_desc_cols)
    
    merged_df = merged_df.merge(
        solvent_lookup, 
        left_on=solvent_col, 
        right_on='smiles_key', 
        how='left'
    ).drop(columns=['smiles_key'])
    
    return merged_df


def merge_bigsoldb_train_test():
    """Merge BigSolDB 1.0 train and test CSVs into a single training set"""
    train_path = "../fastsolv 1.0/data/baseline_fair/baseline_train.csv"
    test_path = "../fastsolv 1.0/data/baseline_fair/baseline_test.csv"
    
    print("Loading BigSolDB 1.0 train and test sets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"  Train: {len(df_train)} samples")
    print(f"  Test: {len(df_test)} samples")
    
    # Merge
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Combined: {len(df_full)} samples")
    
    # Save as new train file
    output_path = "data/bigsoldb_full_train.csv"
    Path("data").mkdir(parents=True, exist_ok=True)
    df_full.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return df_full


def prepare_leeds_test():
    """Generate fastprop features for Leeds test.csv"""
    input_path = "data/test.csv"
    output_path = "data/leeds_test_features.csv"
    
    print("\nLoading Leeds test data...")
    df = pd.read_csv(input_path)
    print(f"  Original columns: {list(df.columns)}")
    print(f"  Samples: {len(df)}")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "Solute": "solute_smiles",
        "Solvent": "solvent_smiles", 
        "Temperature": "temperature",
        "LogS": "logS"
    })
    
    # Add source column (required by training script)
    df["source"] = "leeds"
    
    # Check that all SMILES are valid
    print("Validating SMILES...")
    valid_mask = df["solute_smiles"].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    valid_mask &= df["solvent_smiles"].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"  Warning: Removing {invalid_count} invalid SMILES entries")
        df = df[valid_mask].reset_index(drop=True)
    
    print(f"  Valid samples: {len(df)}")
    
    # Generate fastprop features
    print("\nGenerating fastprop features...")
    df_features = get_descs(df)
    
    # Save
    df_features.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"  Columns: {len(df_features.columns)}")
    print(f"  Samples: {len(df_features)}")
    
    return df_features


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing FastSolv Leeds Baseline Data")
    print("=" * 60)
    
    # Step 1: Merge BigSolDB 1.0 train + test
    print("\n[Step 1] Merging BigSolDB 1.0 datasets...")
    merge_bigsoldb_train_test()
    
    # Step 2: Generate features for Leeds test
    print("\n[Step 2] Generating features for Leeds test...")
    prepare_leeds_test()
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("\nGenerated files:")
    print("  - data/bigsoldb_full_train.csv  (training set)")
    print("  - data/leeds_test_features.csv  (test set with features)")
    print("=" * 60)
