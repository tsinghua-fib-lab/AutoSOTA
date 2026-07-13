# benchmark_inference.py
"""
FastSolv 1.0 Inference Speed Benchmark
Trains 4-replicate ensemble (seed 123), measures inference speed at batch_size=256
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
import pandas as pd
import time
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import ALL_2D
from sklearn.model_selection import GroupShuffleSplit

from model_architecture import fastpropSolubility, SolubilityDataset

# Config
SEED = 123
NUM_REPLICATES = 4
BATCH_SIZE = 256
TRAIN_FILE = "data/baseline_fair/baseline_train.csv"
TEST_FILE = "data/baseline_fair/baseline_test.csv"
SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]


def calculate_gradients(df_group):
    unique_points = df_group.groupby("scaled_temperature")["scaled_logS"].mean().reset_index()
    if len(unique_points) < 2:
        return [np.nan] * len(df_group)
    unique_points = unique_points.sort_values("scaled_temperature")
    grads_unique = np.gradient(unique_points["scaled_logS"], unique_points["scaled_temperature"])
    grad_map = dict(zip(unique_points["scaled_temperature"], grads_unique))
    return [grad_map.get(t, np.nan) if (np.isfinite(grad_map.get(t, np.nan)) and 0 < grad_map.get(t, np.nan) < 1.0) else np.nan 
            for t in df_group["scaled_temperature"]]


def train_ensemble(df_train):
    """Train 4-replicate ensemble."""
    seed_everything(SEED)
    
    train_solute_arr = torch.tensor(df_train[SOLUTE_COLUMNS].values, dtype=torch.float32)
    train_solvent_arr = torch.tensor(df_train[SOLVENT_COLUMNS].values, dtype=torch.float32)
    train_temp_arr = torch.tensor(df_train["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    train_logS_arr = torch.tensor(df_train["logS"].values.reshape(-1, 1), dtype=torch.float32)

    s_solute_train, sol_u_mean, sol_u_var = standard_scale(train_solute_arr)
    s_solvent_train, sol_v_mean, sol_v_var = standard_scale(train_solvent_arr)
    s_temp_train, t_mean, t_var = standard_scale(train_temp_arr)
    s_logS_train, y_mean, y_var = standard_scale(train_logS_arr)

    # Gradients
    grad_df = df_train[["solute_smiles", "solvent_smiles", "source"]].copy()
    grad_df["scaled_logS"] = s_logS_train.flatten().numpy()
    grad_df["scaled_temperature"] = s_temp_train.flatten().numpy()
    grad_df["idx"] = np.arange(len(df_train))
    
    all_grads = np.full(len(df_train), np.nan)
    for _, group in grad_df.groupby(["source", "solvent_smiles", "solute_smiles"]):
        all_grads[group["idx"].values] = calculate_gradients(group)
    train_grads = torch.tensor(all_grads.astype(np.float32)).unsqueeze(-1)

    # Train ensemble
    models = []
    scaling_params = (sol_u_mean, sol_u_var, sol_v_mean, sol_v_var, t_mean, t_var, y_mean, y_var)
    splitter = GroupShuffleSplit(n_splits=NUM_REPLICATES, test_size=0.05, random_state=SEED)
    
    for rep, (tr_idx, val_idx) in enumerate(splitter.split(np.arange(len(df_train)), groups=df_train["solute_smiles"].values)):
        print(f"  Training replicate {rep+1}/{NUM_REPLICATES}")
        
        train_ds = SolubilityDataset(s_solute_train[tr_idx], s_solvent_train[tr_idx], s_temp_train[tr_idx], s_logS_train[tr_idx], train_grads[tr_idx])
        val_ds = SolubilityDataset(s_solute_train[val_idx], s_solvent_train[val_idx], s_temp_train[val_idx], s_logS_train[val_idx], train_grads[val_idx])

        model = fastpropSolubility(
            num_layers=2, hidden_size=3000, num_features=len(ALL_2D),
            activation_fxn="leakyrelu", input_activation="clamp3",    
            target_means=y_mean, target_vars=y_var,
            solute_means=sol_u_mean, solute_vars=sol_u_var,
            solvent_means=sol_v_mean, solvent_vars=sol_v_var,
            temperature_means=t_mean, temperature_vars=t_var
        )

        Trainer(
            max_epochs=100, accelerator="auto", precision=32, 
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            callbacks=[EarlyStopping(monitor="validation_mse_scaled_loss", patience=20)]
        ).fit(model, fastpropDataLoader(train_ds, batch_size=BATCH_SIZE), fastpropDataLoader(val_ds, batch_size=1024))
        
        models.append(model)
    
    return models, scaling_params


def benchmark_inference(models, df_test, scaling_params):
    """Benchmark ensemble inference at batch_size=256."""
    sol_u_mean, sol_u_var, sol_v_mean, sol_v_var, t_mean, t_var, y_mean, y_var = scaling_params
    
    test_solute = torch.tensor(df_test[SOLUTE_COLUMNS].values, dtype=torch.float32)
    test_solvent = torch.tensor(df_test[SOLVENT_COLUMNS].values, dtype=torch.float32)
    test_temp = torch.tensor(df_test["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    test_logS = torch.tensor(df_test["logS"].values.reshape(-1, 1), dtype=torch.float32)
    
    test_ds = SolubilityDataset(test_solute, test_solvent, test_temp, test_logS, torch.zeros_like(test_logS))
    test_loader = fastpropDataLoader(test_ds, batch_size=BATCH_SIZE)
    
    for m in models:
        m.eval()
    
    # Use Lightning Trainer for prediction (handles batch format correctly)
    pred_trainer = Trainer(accelerator="auto", logger=False, enable_progress_bar=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    
    print(f"\nDevice: {device} | Batch size: {BATCH_SIZE} | Test samples: {len(df_test)}")
    
    # Warmup run
    for m in models:
        _ = pred_trainer.predict(m, test_loader)
    if use_cuda:
        torch.cuda.synchronize()
    
    # Benchmark (10 runs)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        all_preds = []
        for m in models:
            p = torch.cat(pred_trainer.predict(m, test_loader))
            all_preds.append(p.numpy())
        ensemble_pred = np.mean(all_preds, axis=0)
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    num_samples = len(df_test)
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    samples_per_sec = num_samples / np.mean(times)
    
    print(f"\n{'='*60}")
    print(f"INFERENCE BENCHMARK RESULTS (Ensemble of {NUM_REPLICATES})")
    print(f"{'='*60}")
    print(f"Time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {samples_per_sec:,.0f} samples/sec")
    print(f"Latency: {avg_time/num_samples*1000:.2f} µs/sample")
    print(f"{'='*60}")
    
    return samples_per_sec


def main():
    print("=" * 60)
    print(f"FastSolv 1.0 Inference Benchmark | Seed {SEED} | {NUM_REPLICATES} replicates")
    print("=" * 60)
    
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")
    
    print("\n1. Training ensemble...")
    models, scaling_params = train_ensemble(df_train)
    
    print("\n2. Benchmarking inference...")
    benchmark_inference(models, df_test, scaling_params)


if __name__ == "__main__":
    main()
