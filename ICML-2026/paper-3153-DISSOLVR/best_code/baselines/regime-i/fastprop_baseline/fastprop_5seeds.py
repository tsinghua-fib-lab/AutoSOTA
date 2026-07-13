# fastprop_5seeds.py
"""
FastProp Baseline - 5 Seed Evaluation
For running on Colab with GPU

FastProp uses Mordred descriptors + simple FNN.
Training is FAST (~11s per seed after descriptor caching).
Reports RMSE ± std and R² across 5 seeds with timing.
"""

import os
import subprocess
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import shutil

# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]
N_EPOCHS = 100
BATCH_SIZE = 256

DATASETS = {
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}

SMILES_COL = "SMILES"
TARGET_COL = "LogS"


def train_and_evaluate(train_path, test_path, seed, output_dir):
    """Train FastProp model and return test RMSE, R²"""
    
    # Clean up old output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Train using CLI
    train_cmd = [
        "fastprop", "train",
        "--input-file", train_path,
        "--smiles-column", SMILES_COL,
        "--target-columns", TARGET_COL,
        "--output-directory", output_dir,
        "--problem-type", "regression",
        "--number-epochs", str(N_EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--random-seed", str(seed),
    ]
    
    print(f"    Training with seed {seed}...")
    result = subprocess.run(train_cmd)  # Verbose - shows progress!
    
    if result.returncode != 0:
        print(f"    ERROR in training (exit code {result.returncode})")
        return None, None
    
    # Find the checkpoint directory
    import glob
    checkpoint_dirs = glob.glob(f"{output_dir}/fastprop_*/checkpoints")
    if not checkpoint_dirs:
        print(f"    ERROR: No checkpoint found in {output_dir}")
        return None, None
    
    checkpoint_dir = checkpoint_dirs[0]
    
    # Create test SMILES file for prediction
    test_df = pd.read_csv(test_path)
    test_smiles_file = f"{output_dir}/test_smiles.txt"
    test_df[SMILES_COL].to_csv(test_smiles_file, index=False, header=False)
    
    # Run prediction
    pred_output = f"{output_dir}/predictions.csv"
    predict_cmd = [
        "fastprop", "predict",
        "-sf", test_smiles_file,
        "-ds", "all",
        "-o", pred_output,
        checkpoint_dir,
    ]
    
    print(f"    Predicting on test set...")
    result = subprocess.run(predict_cmd)  # Verbose
    
    if result.returncode != 0:
        print(f"    ERROR in prediction (exit code {result.returncode})")
        return None, None
    
    # Load predictions and calculate metrics
    pred_df = pd.read_csv(pred_output)
    
    y_true = test_df[TARGET_COL].values
    y_pred = pred_df["task_0"].values  # FastProp outputs as task_0
    
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
        
        output_dir = f"fastprop_outputs/{name.lower()}_seed_{seed}"
        rmse, r2 = train_and_evaluate(train_path, test_path, seed, output_dir)
        
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
    print("FastProp Baseline - 5 Seed Evaluation")
    print("Mordred Descriptors + FNN")
    print("=" * 60)
    
    all_results = []
    total_start = time.time()
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        if result:
            all_results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - FastProp Baseline")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s | Total: {r['total_time']:.1f}s")
    
    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
