# chemprop_5seeds.py
"""
ChemProp Baseline - 5 Seed Evaluation
For running on Colab with GPU

Uses ChemProp 2.x CLI (the new API).
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
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}

SMILES_COL = "SMILES"
TARGET_COL = "LogS"


def filter_valid_smiles(df, smiles_col="SMILES"):
    """Filter out rows with invalid SMILES that RDKit can't parse"""
    valid_mask = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"    Filtered {invalid_count} invalid SMILES")
    return df[valid_mask].reset_index(drop=True)


def train_and_evaluate(train_path, test_path, seed, checkpoint_dir):
    """Train ChemProp model and return test RMSE, R²"""
    
    # Clean up old checkpoint if exists
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Preprocess: filter invalid SMILES and save to temp files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = filter_valid_smiles(train_df, SMILES_COL)
    test_df = filter_valid_smiles(test_df, SMILES_COL)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    temp_train = f"{checkpoint_dir}/train_clean.csv"
    temp_test = f"{checkpoint_dir}/test_clean.csv"
    train_df.to_csv(temp_train, index=False)
    test_df.to_csv(temp_test, index=False)
    
    # Train using CLI
    train_cmd = [
        "chemprop", "train",
        "--data-path", temp_train,
        "--task-type", "regression",
        "--smiles-columns", SMILES_COL,
        "--target-columns", TARGET_COL,
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
    
    # Find the model file (ChemProp 2.x saves as model_0/best.pt or similar)
    import glob
    model_patterns = [
        f"{checkpoint_dir}/model_0/best.pt",
        f"{checkpoint_dir}/model_0/checkpoints/best.ckpt",
        f"{checkpoint_dir}/best.pt",
        f"{checkpoint_dir}/**/best*.pt",
        f"{checkpoint_dir}/**/best*.ckpt",
    ]
    
    model_path = None
    for pattern in model_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            model_path = matches[0]
            break
    
    if model_path is None:
        # List what files exist for debugging
        all_files = glob.glob(f"{checkpoint_dir}/**/*", recursive=True)
        print(f"    ERROR: No model found. Files in checkpoint: {all_files[:10]}")
        return None, None
    
    print(f"    Model saved at: {model_path}")
    
    # Predict on test set
    pred_output = f"{checkpoint_dir}/predictions.csv"
    predict_cmd = [
        "chemprop", "predict",
        "--test-path", temp_test,
        "--model-path", model_path,
        "--smiles-columns", SMILES_COL,
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
    y_pred = pred_df[TARGET_COL].values  # ChemProp outputs same column name
    
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
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    results = []
    seed_times = []
    
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        seed_start = time.time()
        
        checkpoint_dir = f"chemprop_checkpoints/{name.lower()}_seed_{seed}"
        rmse, r2 = train_and_evaluate(train_path, test_path, seed, checkpoint_dir)
        
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
    print("ChemProp Baseline - 5 Seed Evaluation")
    print("Using ChemProp 2.x CLI")
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
    print("FINAL RESULTS - ChemProp Baseline")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s | Total: {r['total_time']:.1f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
