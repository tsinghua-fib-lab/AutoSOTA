# benchmark_inference_rilood.py
"""
RILOOD Inference Speed Benchmark
Trains single model (seed 123), measures inference speed at batch_size=256
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# Import from rilood.py
from rilood import (
    RILOOD, SolubilityDataset, collate_fn, 
    ATOM_FEAT_DIM, BOND_FEAT_DIM, DEVICE
)

# Config
SEED = 123
BATCH_SIZE = 256
NUM_EPOCHS = 30
PATIENCE = 10


def train_model(train_df, solvent_map):
    """Train single RILOOD model."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 5% validation split
    train_df_shuffled = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = int(len(train_df_shuffled) * 0.05)
    val_df = train_df_shuffled[:val_size]
    train_df_split = train_df_shuffled[val_size:]
    
    train_ds = SolubilityDataset(train_df_split, solvent_map)
    val_ds = SolubilityDataset(val_df, solvent_map)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    
    model = RILOOD(num_solvents=len(solvent_map)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5)
    
    alpha, beta = 1e-3, 1e-4
    best_val_rmse = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for b in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            _, L = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                         b['env'].to(DEVICE), targets=b['target'].to(DEVICE))
            loss = L['reg'] + alpha * L['vae'] + beta * L['mi']
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for b in val_loader:
                p, _ = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                            b['env'].to(DEVICE), training=False)
                val_preds.extend(p.cpu().numpy())
                val_targets.extend(b['target'].numpy())
        
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"Epoch {epoch+1} | Val RMSE: {val_rmse:.4f} | Best: {best_val_rmse:.4f}")
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    return model


def benchmark_inference(model, test_df, solvent_map):
    """Benchmark RILOOD inference at batch_size=256."""
    test_ds = SolubilityDataset(test_df, solvent_map)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=collate_fn, num_workers=4)
    
    model.eval()
    use_cuda = DEVICE.type == 'cuda'
    
    print(f"\nDevice: {DEVICE} | Batch size: {BATCH_SIZE} | Test samples: {len(test_ds)}")
    
    # Warmup
    with torch.no_grad():
        for b in test_loader:
            _ = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                     b['env'].to(DEVICE), training=False)
            break
    if use_cuda:
        torch.cuda.synchronize()
    
    # Benchmark (10 runs)
    times = []
    for run in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            all_preds = []
            for b in test_loader:
                p, _ = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                            b['env'].to(DEVICE), training=False)
                all_preds.append(p)
            preds = torch.cat(all_preds)
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    num_samples = len(test_ds)
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    samples_per_sec = num_samples / np.mean(times)
    
    print(f"\n{'='*60}")
    print(f"RILOOD INFERENCE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {samples_per_sec:,.0f} samples/sec")
    print(f"Latency: {avg_time/num_samples*1000:.2f} µs/sample")
    print(f"{'='*60}")
    
    return samples_per_sec


def main():
    print("=" * 60)
    print(f"RILOOD Inference Benchmark | Seed {SEED}")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    
    all_solvents = set(train_df['Solvent'].unique()) | set(test_df['Solvent'].unique())
    solvent_map = {s: i for i, s in enumerate(all_solvents)}
    
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Solvents: {len(solvent_map)}")
    
    print("\n1. Training RILOOD model...")
    model = train_model(train_df, solvent_map)
    
    print("\n2. Benchmarking inference...")
    benchmark_inference(model, test_df, solvent_map)


if __name__ == "__main__":
    main()
