import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
# FIX 1: Faithful Validation Split (GroupShuffleSplit)
from sklearn.model_selection import GroupShuffleSplit 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import ALL_2D

# Import model definition
from model_architecture import fastpropSolubility, SolubilityDataset

# Config
RANDOM_SEED = 42
NUM_REPLICATES = 4
TRAIN_FILE = "data/baseline_fair/baseline_train.csv"
TEST_FILE = "data/baseline_fair/baseline_test.csv"
SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]

def calculate_gradients(df_group):
    """
    FIX 2: Robust gradient calculation. 
    Aggregates duplicates to prevent 'divide by zero' errors in np.gradient.
    """
    unique_points = df_group.groupby("scaled_temperature")["scaled_logS"].mean().reset_index()
    
    if len(unique_points) < 2:
        return [np.nan] * len(df_group)

    unique_points = unique_points.sort_values("scaled_temperature")
    grads_unique = np.gradient(unique_points["scaled_logS"], unique_points["scaled_temperature"])
    
    # Map back to original dataframe size
    grad_map = dict(zip(unique_points["scaled_temperature"], grads_unique))
    
    final_grads = []
    for t in df_group["scaled_temperature"]:
        g = grad_map.get(t, np.nan)
        if np.isfinite(g) and np.abs(g) < 1.0 and g > 0.0:
            final_grads.append(g)
        else:
            final_grads.append(np.nan)
            
    return final_grads

def run_benchmarking():
    output_dir = Path("output/my_custom_split_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(RANDOM_SEED)

    print("Loading data...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    # 1. Prepare Tensors
    train_solute_arr = torch.tensor(df_train[SOLUTE_COLUMNS].values, dtype=torch.float32)
    test_solute_arr  = torch.tensor(df_test[SOLUTE_COLUMNS].values, dtype=torch.float32)
    train_solvent_arr = torch.tensor(df_train[SOLVENT_COLUMNS].values, dtype=torch.float32)
    test_solvent_arr  = torch.tensor(df_test[SOLVENT_COLUMNS].values, dtype=torch.float32)
    train_temp_arr = torch.tensor(df_train["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    test_temp_arr  = torch.tensor(df_test["temperature"].values.reshape(-1, 1), dtype=torch.float32)
    train_logS_arr = torch.tensor(df_train["logS"].values.reshape(-1, 1), dtype=torch.float32)
    test_logS_arr  = torch.tensor(df_test["logS"].values.reshape(-1, 1), dtype=torch.float32)

    # 2. Scale Training Data (Manual Scaling for Training Loop)
    # Note: We do NOT scale test data here for the final prediction step, 
    # but we DO need scaled test data for the gradients if we were training on it (which we aren't).
    print("Scaling training features...")
    s_solute_train, sol_u_mean, sol_u_var = standard_scale(train_solute_arr)
    s_solvent_train, sol_v_mean, sol_v_var = standard_scale(train_solvent_arr)
    s_temp_train, t_mean, t_var = standard_scale(train_temp_arr)
    s_logS_train, y_mean, y_var = standard_scale(train_logS_arr)

    # 3. Calculate Gradients (Uses Scaled Data)
    print("Calculating Sobolev gradients...")
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
    print(f"Gradients computed. Valid gradients: {np.isfinite(all_grads).sum()}/{len(all_grads)}")

    models = []
    
    # 4. Stratified Split (FIX 1: Faithful to Paper)
    # Forces the model to learn structure, not just interpolate temperature curves.
    splitter = GroupShuffleSplit(n_splits=NUM_REPLICATES, test_size=0.05, random_state=RANDOM_SEED)
    groups = df_train["solute_smiles"].values
    indices = np.arange(len(df_train))

    for rep, (tr_idx, val_idx) in enumerate(splitter.split(indices, groups=groups)):
        print(f"\nTraining Replicate {rep+1}/{NUM_REPLICATES}...")
        
        # Validation sanity check: No solute leakage
        tr_solutes = set(df_train.iloc[tr_idx]["solute_smiles"])
        val_solutes = set(df_train.iloc[val_idx]["solute_smiles"])
        assert len(tr_solutes.intersection(val_solutes)) == 0, "Solute leakage detected!"

        # Train/Val datasets use SCALED data (because forward() doesn't scale)
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

        trainer = Trainer(max_epochs=100, accelerator="auto", precision=32, 
                         logger=False, enable_checkpointing=False,
                         callbacks=[EarlyStopping(monitor="validation_mse_scaled_loss", patience=20)])
        
        trainer.fit(model, fastpropDataLoader(train_ds, batch_size=256), fastpropDataLoader(val_ds, batch_size=1024))
        models.append(model)
        
        if rep + 1 >= NUM_REPLICATES:
            break

    # 5. Ensemble Evaluation (FIX 3: Avoid Double Scaling)
    print("\nRunning Ensemble Evaluation on Test Set...")
    
    # PASS RAW TENSORS HERE. predict_step() will scale them internally.
    test_ds = SolubilityDataset(
        test_solute_arr.clone().detach(), 
        test_solvent_arr.clone().detach(), 
        test_temp_arr.clone().detach(), 
        test_logS_arr.clone().detach(), 
        torch.zeros_like(test_logS_arr)
    )
    test_loader = fastpropDataLoader(test_ds, batch_size=256)
    
    all_preds = []
    # Use a bare trainer for prediction
    pred_trainer = Trainer(accelerator="auto", logger=False, enable_progress_bar=False)
    
    for m in models:
        m.eval()
        # The trainer.predict calls the model's predict_step(), which handles scaling.
        p = torch.cat(pred_trainer.predict(m, test_loader))
        all_preds.append(p.numpy())
    
    final_preds = np.mean(all_preds, axis=0)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df_test["logS"], final_preds))
    print(f"\n{'='*30}\nFINAL BASELINE RMSE ON YOUR SPLIT: {rmse:.4f}\n{'='*30}")

if __name__ == "__main__":
    run_benchmarking()
