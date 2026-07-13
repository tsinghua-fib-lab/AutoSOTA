# train_fastsolv_5seeds.py
"""
FastSolv 2.0 evaluation over 5 seeds on BigSolDB 2.0 dataset
Reports RMSE ± std and R² for comparison with other baselines
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import ALL_2D

# Import model definition
from model_architecture import fastpropSolubility, SolubilityDataset

# Config
SEEDS = [42, 101, 123, 456, 789]
NUM_REPLICATES = 4  # ensemble size per seed
TRAIN_FILE = "data/baseline_bigsol2/baseline_train.csv"
TEST_FILE = "data/baseline_bigsol2/baseline_test.csv"
SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]


def calculate_gradients(df_group):
    """Robust gradient calculation for Sobolev training."""
    unique_points = df_group.groupby("scaled_temperature")["scaled_logS"].mean().reset_index()
    
    if len(unique_points) < 2:
        return [np.nan] * len(df_group)

    unique_points = unique_points.sort_values("scaled_temperature")
    grads_unique = np.gradient(unique_points["scaled_logS"], unique_points["scaled_temperature"])
    
    grad_map = dict(zip(unique_points["scaled_temperature"], grads_unique))
    
    final_grads = []
    for t in df_group["scaled_temperature"]:
        g = grad_map.get(t, np.nan)
        if np.isfinite(g) and np.abs(g) < 1.0 and g > 0.0:
            final_grads.append(g)
        else:
            final_grads.append(np.nan)
            
    return final_grads


def train_one_seed(seed, df_train, df_test, test_var):
    """Train FastSolv for one seed and return test RMSE."""
    seed_everything(seed)
    
    # 1. Prepare Tensors
    train_solute_arr = torch.tensor(df_train[SOLUTE_COLUMNS].values, dtype=torch.float32)
    test_solute_arr = torch.tensor(df_test[SOLUTE_COLUMNS].values, dtype=torch.float32)
    train_solvent_arr = torch.tensor(df_train[SOLVENT_COLUMNS].values, dtype=torch.float32)
    test_solvent_arr = torch.tensor(df_test[SOLVENT_COLUMNS].values, dtype=torch.float32)
    train_temp_arr = torch.tensor(df_train["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    test_temp_arr = torch.tensor(df_test["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    train_logS_arr = torch.tensor(df_train["logS"].values.reshape(-1, 1), dtype=torch.float32)
    test_logS_arr = torch.tensor(df_test["logS"].values.reshape(-1, 1), dtype=torch.float32)

    # 2. Scale Training Data
    s_solute_train, sol_u_mean, sol_u_var = standard_scale(train_solute_arr)
    s_solvent_train, sol_v_mean, sol_v_var = standard_scale(train_solvent_arr)
    s_temp_train, t_mean, t_var = standard_scale(train_temp_arr)
    s_logS_train, y_mean, y_var = standard_scale(train_logS_arr)

    # 3. Calculate Gradients
    grad_df = df_train[["solute_smiles", "solvent_smiles", "source"]].copy()
    grad_df["scaled_logS"] = s_logS_train.flatten().numpy()
    grad_df["scaled_temperature"] = s_temp_train.flatten().numpy()
    grad_df["idx"] = np.arange(len(df_train))
    
    grad_groups = grad_df.groupby(["source", "solvent_smiles", "solute_smiles"])
    all_grads = np.full(len(df_train), np.nan)
    
    for name, group in grad_groups:
        indices = group["idx"].values
        g_values = calculate_gradients(group)
        all_grads[indices] = g_values

    train_grads = torch.tensor(all_grads.astype(np.float32)).unsqueeze(-1)

    models = []
    
    # 4. Train ensemble with ShuffleSplit (consistent with original)
    splitter = ShuffleSplit(n_splits=NUM_REPLICATES, test_size=0.05, random_state=seed)
    indices = np.arange(len(df_train))

    for rep, (tr_idx, val_idx) in enumerate(splitter.split(indices)):
        print(f"  Seed {seed} | Replicate {rep+1}/{NUM_REPLICATES}")
        
        train_ds = SolubilityDataset(
            s_solute_train[tr_idx].clone().detach(), 
            s_solvent_train[tr_idx].clone().detach(),
            s_temp_train[tr_idx].clone().detach(), 
            s_logS_train[tr_idx].clone().detach(), 
            train_grads[tr_idx].clone().detach()
        )
        val_ds = SolubilityDataset(
            s_solute_train[val_idx].clone().detach(), 
            s_solvent_train[val_idx].clone().detach(),
            s_temp_train[val_idx].clone().detach(), 
            s_logS_train[val_idx].clone().detach(), 
            train_grads[val_idx].clone().detach()
        )

        model = fastpropSolubility(
            num_layers=2, 
            hidden_size=3000, 
            num_features=len(ALL_2D),
            activation_fxn="leakyrelu",   
            input_activation="clamp3",    
            target_means=y_mean, target_vars=y_var,
            solute_means=sol_u_mean, solute_vars=sol_u_var,
            solvent_means=sol_v_mean, solvent_vars=sol_v_var,
            temperature_means=t_mean, temperature_vars=t_var
        )

        trainer = Trainer(
            max_epochs=100, accelerator="auto", precision=32, 
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            callbacks=[EarlyStopping(monitor="validation_mse_scaled_loss", patience=20)]
        )
        
        trainer.fit(model, fastpropDataLoader(train_ds, batch_size=256), 
                   fastpropDataLoader(val_ds, batch_size=1024))
        models.append(model)
        
        if rep + 1 >= NUM_REPLICATES:
            break

    # 5. Ensemble Evaluation
    test_ds = SolubilityDataset(
        test_solute_arr.clone().detach(), 
        test_solvent_arr.clone().detach(), 
        test_temp_arr.clone().detach(), 
        test_logS_arr.clone().detach(), 
        torch.zeros_like(test_logS_arr)
    )
    test_loader = fastpropDataLoader(test_ds, batch_size=256)
    
    all_preds = []
    pred_trainer = Trainer(accelerator="auto", logger=False, enable_progress_bar=False)
    
    for m in models:
        m.eval()
        p = torch.cat(pred_trainer.predict(m, test_loader))
        all_preds.append(p.numpy())
    
    final_preds = np.mean(all_preds, axis=0)
    rmse = np.sqrt(mean_squared_error(df_test["logS"], final_preds))
    r2 = 1 - (rmse**2 / test_var)
    
    return rmse, r2


def main():
    print("=" * 60)
    print("FastSolv 2.0 - 5 Seed Evaluation on BigSolDB 2.0")
    print("=" * 60)
    
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    
    test_var = df_test["logS"].var()
    test_std = df_test["logS"].std()
    print(f"Train set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples, Var: {test_var:.4f}, Std: {test_std:.4f}")
    
    results = []
    times = []
    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"SEED {seed}")
        print(f"{'='*40}")
        start_time = time.time()
        rmse, r2 = train_one_seed(seed, df_train, df_test, test_var)
        elapsed = time.time() - start_time
        results.append((rmse, r2))
        times.append(elapsed)
        print(f"Seed {seed} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Time: {elapsed:.1f}s")
    
    rmses = [r[0] for r in results]
    r2s = [r[1] for r in results]
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS - FastSolv 2.0 on BigSolDB 2.0")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"RMSEs: {[f'{r:.4f}' for r in rmses]}")
    print(f"R²s: {[f'{r:.4f}' for r in r2s]}")
    print(f"\n>>> RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f} <<<")
    print(f">>> R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f} <<<")
    print(f">>> Avg Time/Seed: {np.mean(times):.1f}s <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
