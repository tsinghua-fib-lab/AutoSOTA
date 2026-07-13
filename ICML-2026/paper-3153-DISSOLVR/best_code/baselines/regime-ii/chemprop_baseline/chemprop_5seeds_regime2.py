# chemprop_5seeds_regime2.py
"""
ChemProp Baseline - Regime II (Multi-Solvent) - 5 Seed Evaluation
For running on Colab with GPU

Uses ChemProp 2.x with:
- Dual SMILES: Solute + Solvent
- Temperature as additional descriptor
Reports RMSE ± std and R² across 5 seeds with timing.
"""

import os
import subprocess
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]
EPOCHS = 60

DATASETS = {
    "BigSol1.0": ("../all_datasets/bigsol1.0/train.csv", "../all_datasets/bigsol1.0/test.csv"),
    "BigSol2.0": ("../all_datasets/bigsol2.0/train.csv", "../all_datasets/bigsol2.0/test.csv"),
    "Leeds": ("../all_datasets/leeds/train.csv", "../all_datasets/leeds/test.csv"),
}

SOLUTE_COL = "Solute"
SOLVENT_COL = "Solvent"
TEMP_COL = "Temperature"
TARGET_COL = "LogS"


def filter_valid_smiles(df):
    """Filter out rows with invalid SMILES (either Solute or Solvent)"""
    valid_mask = (
        df[SOLUTE_COL].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None) &
        df[SOLVENT_COL].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"    Filtered {invalid_count} invalid SMILES")
    return df[valid_mask].reset_index(drop=True)


def prepare_data_files(train_path, test_path, checkpoint_dir):
    """
    Prepare clean data files with Temperature included for ChemProp.
    ChemProp 2.x uses --descriptors-columns to specify extra columns from main file.
    """
    # Load and filter
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = filter_valid_smiles(train_df)
    test_df = filter_valid_smiles(test_df)
    
    # Create CSVs with Solute, Solvent, Temperature, LogS
    # ChemProp will use --descriptors-columns Temperature to pick it up
    train_clean = train_df[[SOLUTE_COL, SOLVENT_COL, TEMP_COL, TARGET_COL]]
    test_clean = test_df[[SOLUTE_COL, SOLVENT_COL, TEMP_COL, TARGET_COL]]
    
    train_csv = f"{checkpoint_dir}/train_clean.csv"
    test_csv = f"{checkpoint_dir}/test_clean.csv"
    train_clean.to_csv(train_csv, index=False)
    test_clean.to_csv(test_csv, index=False)
    
    print(f"    Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_csv, test_csv, test_df


def train_and_evaluate(train_csv, test_csv, test_df, seed, checkpoint_dir):
    """Train ChemProp model with dual SMILES + Temperature and return test RMSE, R²"""
    
    # Train using CLI with dual SMILES and temperature as extra descriptor column
    train_cmd = [
        "chemprop", "train",
        "--data-path", train_csv,
        "--task-type", "regression",
        "--smiles-columns", SOLUTE_COL, SOLVENT_COL,  # Dual SMILES!
        "--target-columns", TARGET_COL,
        "--descriptors-columns", TEMP_COL,  # Temperature from same file!
        "--output-dir", checkpoint_dir,
        "--epochs", str(EPOCHS),
        "--data-seed", str(seed),
        "--pytorch-seed", str(seed),
    ]
    
    print(f"    Training with seed {seed}...")
    result = subprocess.run(train_cmd)  # Verbose - shows progress bars!
    
    if result.returncode != 0:
        print(f"    ERROR in training (exit code {result.returncode})")
        return None, None
    
    # Find the model file
    import glob
    model_patterns = [
        f"{checkpoint_dir}/model_0/best.pt",
        f"{checkpoint_dir}/best.pt",
        f"{checkpoint_dir}/**/best*.pt",
    ]
    
    model_path = None
    for pattern in model_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            model_path = matches[0]
            break
    
    if model_path is None:
        all_files = glob.glob(f"{checkpoint_dir}/**/*", recursive=True)
        print(f"    ERROR: No model found. Files: {all_files[:10]}")
        return None, None
    
    # Predict on test set
    pred_output = f"{checkpoint_dir}/predictions.csv"
    predict_cmd = [
        "chemprop", "predict",
        "--test-path", test_csv,
        "--model-path", model_path,
        "--smiles-columns", SOLUTE_COL, SOLVENT_COL,
        "--descriptors-columns", TEMP_COL,
        "--preds-path", pred_output,
    ]
    
    print(f"    Predicting on test set...")
    result = subprocess.run(predict_cmd)  # Verbose
    
    if result.returncode != 0:
        print(f"    ERROR in prediction (exit code {result.returncode})")
        return None, None
    
    # Load predictions and calculate metrics
    pred_df = pd.read_csv(pred_output)
    
    y_true = test_df[TARGET_COL].values
    y_pred = pred_df[TARGET_COL].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return rmse, r2


def evaluate_dataset(name, train_path, test_path):
    """Evaluate on a dataset across 5 seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    if not os.path.exists(train_path):
        print(f"  ERROR: Train file not found: {train_path}")
        return None
    if not os.path.exists(test_path):
        print(f"  ERROR: Test file not found: {test_path}")
        return None
    
    results = []
    seed_times = []
    
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        seed_start = time.time()
        
        # Create checkpoint dir for this seed
        checkpoint_dir = f"chemprop_checkpoints/{name.lower().replace(' ', '_')}_seed_{seed}"
        
        # Clean up old checkpoint
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare data files
        train_csv, test_csv, test_df = \
            prepare_data_files(train_path, test_path, checkpoint_dir)
        
        # Train and evaluate
        rmse, r2 = train_and_evaluate(
            train_csv, test_csv, test_df, seed, checkpoint_dir
        )
        
        if rmse is not None:
            results.append((rmse, r2))
            print(f"    RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        seed_elapsed = time.time() - seed_start
        seed_times.append(seed_elapsed)
        print(f"    Time: {seed_elapsed:.1f}s")
    
    if not results:
        return None
    
    return {
        "name": name,
        "rmse_mean": np.mean([r[0] for r in results]),
        "rmse_std": np.std([r[0] for r in results]),
        "r2_mean": np.mean([r[1] for r in results]),
        "r2_std": np.std([r[1] for r in results]),
        "avg_time": np.mean(seed_times),
        "total_time": np.sum(seed_times)
    }


def main():
    print("=" * 60)
    print("ChemProp Baseline - Regime II (Multi-Solvent)")
    print("5 Seed Evaluation with Solute + Solvent + Temperature")
    print("=" * 60)
    
    # Check if chemprop is installed
    try:
        import chemprop
        print(f"ChemProp version: {chemprop.__version__}")
    except ImportError:
        print("ERROR: ChemProp not installed. Install with: pip install chemprop")
        return
    
    all_results = []
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        if result:
            all_results.append(result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - ChemProp Regime II Baseline")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s | Total: {r['total_time']:.1f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
