# soltrannet_5seeds.py
"""
SolTranNet Training - 5 Seed Evaluation

Train the lightweight SolTranNet model (~3,393 params) from scratch.
Uses: d_model=8, h=2, N=8 attention layers

Run:
    cd baselines/regime-i/SolTranNet_paper
    python soltrannet_5seeds.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from tqdm import tqdm

# Add current dir for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_data_from_smiles, construct_loader
from transformer import make_model

# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]

DATASETS = {
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}

# Training parameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10

SMILES_COL = "SMILES"
TARGET_COL = "LogS"


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_and_evaluate(train_df, test_df, seed, device):
    """Train SolTranNet and evaluate on test set"""
    set_seed(seed)
    
    # Prepare data
    train_smiles = train_df[SMILES_COL].tolist()
    train_y = train_df[TARGET_COL].values.astype(np.float32)
    
    test_smiles = test_df[SMILES_COL].tolist()
    test_y = test_df[TARGET_COL].values.astype(np.float32)
    
    # Load and featurize molecules
    print("    Featurizing molecules...")
    train_x = load_data_from_smiles(train_smiles, add_dummy_node=True)
    test_x = load_data_from_smiles(test_smiles, add_dummy_node=True)
    
    # Create data loaders
    train_loader = construct_loader(train_x, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = construct_loader(test_x, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model with default small architecture (3,393 params)
    model = make_model()  # d_model=8, h=2, N=8 by default
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model parameters: {param_count}")
    
    # Loss and optimizer
    criterion = nn.SmoothL1Loss()  # Huber loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_test_rmse = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(EPOCHS), desc="    Epochs", position=0):
        model.train()
        train_losses = []
        
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"      Batch", total=len(train_loader), leave=False, position=1):
            adj_matrix, node_features, smiles_list, indices = batch
            adj_matrix = adj_matrix.to(device)
            node_features = node_features.to(device)
            
            # Get targets for this batch
            batch_y = torch.FloatTensor([train_y[i] for i in indices]).to(device).unsqueeze(1)
            
            # Forward
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            pred = model(node_features, batch_mask, adj_matrix, None)
            
            loss = criterion(pred, batch_y)
            train_losses.append(loss.item())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    adj_matrix, node_features, smiles_list, indices = batch
                    adj_matrix = adj_matrix.to(device)
                    node_features = node_features.to(device)
                    
                    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
                    pred = model(node_features, batch_mask, adj_matrix, None)
                    preds.extend(pred.cpu().numpy().flatten())
            
            test_rmse = np.sqrt(np.mean((np.array(preds) - test_y[:len(preds)])**2))
            
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOP_PATIENCE // 10:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            adj_matrix, node_features, smiles_list, indices = batch
            adj_matrix = adj_matrix.to(device)
            node_features = node_features.to(device)
            
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            pred = model(node_features, batch_mask, adj_matrix, None)
            preds.extend(pred.cpu().numpy().flatten())
    
    y_pred = np.array(preds)
    y_true = test_y[:len(y_pred)]
    
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    return rmse, r2


def evaluate_dataset(name, train_path, test_path, device):
    """Evaluate on a dataset across 5 seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    results = []
    times = []
    
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        start_time = time.time()
        
        rmse, r2 = train_and_evaluate(train_df, test_df, seed, device)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        results.append((rmse, r2))
        
        print(f"    RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"    Time: {elapsed:.1f}s")
    
    return {
        "name": name,
        "rmse_mean": np.mean([r[0] for r in results]),
        "rmse_std": np.std([r[0] for r in results]),
        "r2_mean": np.mean([r[1] for r in results]),
        "r2_std": np.std([r[1] for r in results]),
        "avg_time": np.mean(times),
        "total_time": np.sum(times)
    }


def main():
    print("=" * 60)
    print("SolTranNet Baseline - 5 Seed Evaluation")
    print("Lightweight Transformer (~3,393 params)")
    print("=" * 60)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    all_results = []
    total_start = time.time()
    
    for name, (train_path, test_path) in DATASETS.items():
        if not os.path.exists(train_path):
            print(f"Skipping {name}: file not found")
            continue
        result = evaluate_dataset(name, train_path, test_path, device)
        all_results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - SolTranNet Baseline")
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
