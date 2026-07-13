#!/usr/bin/env python3
"""
Generate Features for ALL Datasets

Scans all three datasets (AqSolDB, CombiSolu, BigSolDB) and generates
a unified feature store containing ALL unique molecules.

This must be run BEFORE the benchmark script.

Directory structure required:
    all_datasets/
        AqSolDB/
            train.csv
            test.csv
        CombiSolu/
            train.csv
            test.csv
        BigSolDB/
            train.csv
            test.csv

Outputs:
    feature_store/
        solute_raw.parquet
        solvent_raw.parquet
        solute_council.parquet
        solvent_council.parquet
        council_extractor.joblib
"""

import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import warnings
from rdkit import RDLogger

from featurizer import MoleculeFeaturizer
from council import CouncilExtractor

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ["bigsol1.0", "bigsol2.0", "leeds"]
ALL_DATASETS_DIR = "all_datasets"
STORE_DIR = "feature_store"

# Output files
RAW_SOLUTE_FILE = os.path.join(STORE_DIR, "solute_raw.parquet")
RAW_SOLVENT_FILE = os.path.join(STORE_DIR, "solvent_raw.parquet")
SOLUTE_COUNCIL_FILE = os.path.join(STORE_DIR, "solute_council.parquet")
SOLVENT_COUNCIL_FILE = os.path.join(STORE_DIR, "solvent_council.parquet")
EXTRACTOR_FILE = os.path.join(STORE_DIR, "council_extractor.joblib")

# For fitting the scaler, use top N solvents from the FIRST dataset
# This ensures consistent scaling across all datasets
REFERENCE_DATASET = "bigsol1.0"  # Change if needed
TOP_N_SOLVENTS_FOR_FIT = 128


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def collect_all_molecules():
    """
    Scan all datasets and collect unique solute and solvent SMILES.
    
    Returns:
        unique_solutes (list): All unique solute SMILES
        unique_solvents (list): All unique solvent SMILES
        all_dataframes (dict): Dictionary of {dataset_name: {'train': df, 'test': df}}
    """
    print("="*60)
    print("SCANNING ALL DATASETS")
    print("="*60)
    
    all_solutes = set()
    all_solvents = set()
    all_dataframes = {}
    
    for dataset_name in DATASETS:
        train_path = os.path.join(ALL_DATASETS_DIR, dataset_name, "train.csv")
        test_path = os.path.join(ALL_DATASETS_DIR, dataset_name, "test.csv")
        
        if not os.path.exists(train_path):
            print(f"WARNING: {train_path} not found. Skipping {dataset_name}.")
            continue
        if not os.path.exists(test_path):
            print(f"WARNING: {test_path} not found. Skipping {dataset_name}.")
            continue
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        all_dataframes[dataset_name] = {'train': df_train, 'test': df_test}
        
        # Collect unique molecules
        dataset_solutes = set(df_train['Solute'].unique()) | set(df_test['Solute'].unique())
        dataset_solvents = set(df_train['Solvent'].unique()) | set(df_test['Solvent'].unique())
        
        all_solutes.update(dataset_solutes)
        all_solvents.update(dataset_solvents)
        
        print(f"\n{dataset_name}:")
        print(f"  Train: {len(df_train)} samples")
        print(f"  Test:  {len(df_test)} samples")
        print(f"  Unique solutes:  {len(dataset_solutes)}")
        print(f"  Unique solvents: {len(dataset_solvents)}")
    
    print(f"\n{'='*60}")
    print("TOTAL UNIQUE MOLECULES ACROSS ALL DATASETS:")
    print(f"  Solutes:  {len(all_solutes)}")
    print(f"  Solvents: {len(all_solvents)}")
    print(f"{'='*60}\n")
    
    return list(all_solutes), list(all_solvents), all_dataframes


def generate_raw_features(unique_solutes, unique_solvents):
    """
    Generate raw molecular features for all unique molecules.
    
    Returns:
        df_sol_raw (pd.DataFrame): Raw features for solutes
        df_solv_raw (pd.DataFrame): Raw features for solvents
    """
    featurizer = MoleculeFeaturizer()
    
    # Generate solute features
    if os.path.exists(RAW_SOLUTE_FILE):
        print(f"Loading existing solute raw features from {RAW_SOLUTE_FILE}...")
        df_sol_raw = pd.read_parquet(RAW_SOLUTE_FILE)
        
        # Check if we need to add new molecules
        existing_solutes = set(df_sol_raw['SMILES_KEY'].values)
        new_solutes = [s for s in unique_solutes if s not in existing_solutes]
        
        if new_solutes:
            print(f"  Found {len(new_solutes)} new solutes. Featurizing...")
            df_new = featurizer.transform(new_solutes)
            df_new['SMILES_KEY'] = new_solutes
            df_sol_raw = pd.concat([df_sol_raw, df_new], ignore_index=True)
            df_sol_raw.to_parquet(RAW_SOLUTE_FILE, index=False)
            print(f"  Updated solute features saved.")
        else:
            print(f"  All solutes already featurized.")
    else:
        print(f"Generating solute raw features ({len(unique_solutes)} molecules)...")
        df_sol_raw = featurizer.transform(unique_solutes)
        df_sol_raw['SMILES_KEY'] = unique_solutes
        df_sol_raw.to_parquet(RAW_SOLUTE_FILE, index=False)
        print(f"  Saved to {RAW_SOLUTE_FILE}")
    
    # Generate solvent features
    if os.path.exists(RAW_SOLVENT_FILE):
        print(f"\nLoading existing solvent raw features from {RAW_SOLVENT_FILE}...")
        df_solv_raw = pd.read_parquet(RAW_SOLVENT_FILE)
        
        # Check if we need to add new molecules
        existing_solvents = set(df_solv_raw['SMILES_KEY'].values)
        new_solvents = [s for s in unique_solvents if s not in existing_solvents]
        
        if new_solvents:
            print(f"  Found {len(new_solvents)} new solvents. Featurizing...")
            df_new = featurizer.transform(new_solvents)
            df_new['SMILES_KEY'] = new_solvents
            df_solv_raw = pd.concat([df_solv_raw, df_new], ignore_index=True)
            df_solv_raw.to_parquet(RAW_SOLVENT_FILE, index=False)
            print(f"  Updated solvent features saved.")
        else:
            print(f"  All solvents already featurized.")
    else:
        print(f"\nGenerating solvent raw features ({len(unique_solvents)} molecules)...")
        df_solv_raw = featurizer.transform(unique_solvents)
        df_solv_raw['SMILES_KEY'] = unique_solvents
        df_solv_raw.to_parquet(RAW_SOLVENT_FILE, index=False)
        print(f"  Saved to {RAW_SOLVENT_FILE}")
    
    return df_sol_raw, df_solv_raw


def fit_council_extractor(all_dataframes, df_sol_raw, df_solv_raw):
    """
    Fit the council extractor (scaler) on a representative training distribution.
    
    Uses top N solvents from the reference dataset to fit the scaler,
    ensuring consistent scaling across all datasets.
    
    Returns:
        extractor (CouncilExtractor): Fitted extractor
    """
    print(f"\n{'='*60}")
    print("FITTING COUNCIL EXTRACTOR (SCALER)")
    print(f"{'='*60}")
    
    # Use reference dataset for fitting
    if REFERENCE_DATASET not in all_dataframes:
        raise ValueError(f"Reference dataset '{REFERENCE_DATASET}' not found!")
    
    df_train = all_dataframes[REFERENCE_DATASET]['train']
    
    print(f"Using {REFERENCE_DATASET} as reference for fitting scaler...")
    
    # Get top N solvents from reference dataset
    solvent_counts = df_train['Solvent'].value_counts()
    train_solvents_list = solvent_counts.head(TOP_N_SOLVENTS_FOR_FIT).index.tolist()
    
    # Filter to rows with these solvents
    train_df_rows = df_train[df_train['Solvent'].isin(train_solvents_list)]
    train_solutes = train_df_rows['Solute'].unique()
    train_solvents = train_df_rows['Solvent'].unique()
    
    print(f"  Using top {len(train_solvents)} solvents")
    print(f"  Training solutes: {len(train_solutes)}")
    print(f"  Training solvents: {len(train_solvents)}")
    
    # Get features for training molecules
    train_sol_feats = df_sol_raw[df_sol_raw['SMILES_KEY'].isin(train_solutes)].drop(columns=['SMILES_KEY'])
    train_solv_feats = df_solv_raw[df_solv_raw['SMILES_KEY'].isin(train_solvents)].drop(columns=['SMILES_KEY'])
    
    # Combine for fitting
    training_data_for_scaling = pd.concat([train_sol_feats, train_solv_feats], axis=0)
    
    print(f"  Training distribution size: {len(training_data_for_scaling)} samples")
    
    # Fit extractor
    extractor = CouncilExtractor()
    extractor.fit(training_data_for_scaling)
    
    # Save
    joblib.dump(extractor, EXTRACTOR_FILE)
    print(f"  Saved fitted extractor to {EXTRACTOR_FILE}")
    
    return extractor


def generate_council_features(df_sol_raw, df_solv_raw, extractor):
    """
    Transform all molecules to council features using fitted extractor.
    
    Returns:
        df_sol_council (pd.DataFrame): Council features for solutes
        df_solv_council (pd.DataFrame): Council features for solvents
    """
    print(f"\n{'='*60}")
    print("GENERATING COUNCIL FEATURES FOR ALL MOLECULES")
    print(f"{'='*60}")
    
    # Transform solutes
    print(f"Transforming {len(df_sol_raw)} solutes...")
    df_sol_council = extractor.transform(df_sol_raw.drop(columns=['SMILES_KEY']))
    df_sol_council['SMILES_KEY'] = df_sol_raw['SMILES_KEY'].values
    df_sol_council.to_parquet(SOLUTE_COUNCIL_FILE, index=False)
    print(f"  Saved to {SOLUTE_COUNCIL_FILE}")
    
    # Transform solvents
    print(f"\nTransforming {len(df_solv_raw)} solvents...")
    df_solv_council = extractor.transform(df_solv_raw.drop(columns=['SMILES_KEY']))
    df_solv_council['SMILES_KEY'] = df_solv_raw['SMILES_KEY'].values
    df_solv_council.to_parquet(SOLVENT_COUNCIL_FILE, index=False)
    print(f"  Saved to {SOLVENT_COUNCIL_FILE}")
    
    return df_sol_council, df_solv_council


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main feature generation pipeline for all datasets.
    """
    print("\n" + "="*60)
    print("UNIFIED FEATURE GENERATION FOR ALL DATASETS")
    print("="*60)
    print(f"Datasets to process: {', '.join(DATASETS)}")
    print(f"Output directory: {STORE_DIR}")
    print("="*60 + "\n")
    
    # Create output directory
    ensure_dir(STORE_DIR)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    RDLogger.DisableLog("rdApp.*")
    
    # Step 1: Collect all unique molecules from all datasets
    unique_solutes, unique_solvents, all_dataframes = collect_all_molecules()
    
    if not all_dataframes:
        print("ERROR: No datasets found! Check your directory structure.")
        return
    
    # Step 2: Generate raw features for all unique molecules
    print(f"\n{'='*60}")
    print("STEP 1: GENERATING RAW FEATURES")
    print(f"{'='*60}")
    df_sol_raw, df_solv_raw = generate_raw_features(unique_solutes, unique_solvents)
    
    # Step 3: Fit council extractor on reference dataset
    print(f"\n{'='*60}")
    print("STEP 2: FITTING COUNCIL EXTRACTOR")
    print(f"{'='*60}")
    
    if os.path.exists(EXTRACTOR_FILE):
        print(f"Loading existing extractor from {EXTRACTOR_FILE}...")
        extractor = joblib.load(EXTRACTOR_FILE)
    else:
        extractor = fit_council_extractor(all_dataframes, df_sol_raw, df_solv_raw)
    
    # Step 4: Transform all molecules to council features
    print(f"\n{'='*60}")
    print("STEP 3: TRANSFORMING TO COUNCIL FEATURES")
    print(f"{'='*60}")
    df_sol_council, df_solv_council = generate_council_features(df_sol_raw, df_solv_raw, extractor)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FEATURE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated features for:")
    print(f"  Solutes:  {len(df_sol_raw)} molecules")
    print(f"  Solvents: {len(df_solv_raw)} molecules")
    print(f"\nOutput files:")
    print(f"  {RAW_SOLUTE_FILE}")
    print(f"  {RAW_SOLVENT_FILE}")
    print(f"  {SOLUTE_COUNCIL_FILE}")
    print(f"  {SOLVENT_COUNCIL_FILE}")
    print(f"  {EXTRACTOR_FILE}")
    print(f"\nReady for benchmark script!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()