#!/usr/bin/env python3
import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from tqdm.auto import tqdm
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ["bigsol1.0", "bigsol2.0", "leeds"]
ALL_DATASETS_DIR = "all_datasets"
FEATURE_STORE_DIR = "feature_store"
OUTPUT_DIR = "benchmark_results"
SEEDS = [42, 101, 123, 456, 789]
TRANSFORMER_SEED = 42

# Transformer Hyperparameters
COUNCIL_SIZE = 24
EMBED_DIM = 32
NUM_HEADS = 4
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class InteractionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_proj = nn.Parameter(torch.randn(COUNCIL_SIZE, EMBED_DIM))
        self.feat_bias = nn.Parameter(torch.zeros(COUNCIL_SIZE, EMBED_DIM))
        self.type_emb = nn.Parameter(torch.randn(1, COUNCIL_SIZE, EMBED_DIM))
        self.cross_attn = nn.MultiheadAttention(EMBED_DIM, num_heads=NUM_HEADS, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(COUNCIL_SIZE * EMBED_DIM, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1)
        )

    def forward(self, sol, solv):
        sol_emb = (sol.unsqueeze(-1) * self.feat_proj) + self.feat_bias + self.type_emb
        solv_emb = (solv.unsqueeze(-1) * self.feat_proj) + self.feat_bias + self.type_emb
        enriched, attn = self.cross_attn(query=sol_emb, key=solv_emb, value=solv_emb)
        return self.head(enriched.reshape(enriched.size(0), -1)), enriched.reshape(enriched.size(0), -1), attn


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset_features(dataset_name):
    """Load council features for a specific dataset."""
    train_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "train.csv")
    test_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "test.csv")
    
    sol_store = pd.read_parquet(
        os.path.join(FEATURE_STORE_DIR, "solute_council.parquet")
    ).set_index("SMILES_KEY")
    
    solv_store = pd.read_parquet(
        os.path.join(FEATURE_STORE_DIR, "solvent_council.parquet")
    ).set_index("SMILES_KEY")
    
    sol_raw = pd.read_parquet(
        os.path.join(FEATURE_STORE_DIR, "solute_raw.parquet")
    ).set_index("SMILES_KEY")
    
    solv_raw = pd.read_parquet(
        os.path.join(FEATURE_STORE_DIR, "solvent_raw.parquet")
    ).set_index("SMILES_KEY")
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    return {
        'train': df_train,
        'test': df_test,
        'sol_council': sol_store,
        'solv_council': solv_store,
        'sol_raw': sol_raw,
        'solv_raw': solv_raw
    }


def generate_transformer_embeddings(model, X_sol, X_solv, batch_size=512):
    """Generate embeddings using trained transformer."""
    model.eval()
    embed_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_sol), batch_size):
            b_sol = torch.tensor(X_sol[i:i+batch_size]).to(DEVICE)
            b_solv = torch.tensor(X_solv[i:i+batch_size]).to(DEVICE)
            _, feats, _ = model(b_sol, b_solv)
            embed_list.append(feats.cpu().numpy())
    
    return np.vstack(embed_list)


def build_baseline_features(df, embeddings, sol_raw, solv_raw):
    """Build feature matrix for baseline models (same as CatBoost)."""
    # Raw features
    X_raw = np.hstack([
        sol_raw.loc[df['Solute']].values,
        solv_raw.loc[df['Solvent']].values
    ])
    
    # Thermodynamic features
    Tm = sol_raw.loc[df['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T = df['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df['Temperature'].values).reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)
    
    # Signed-modulus interactions (24 channels)
    X_reshaped = embeddings.reshape(embeddings.shape[0], 24, 32)
    X_modulus = np.linalg.norm(X_reshaped, axis=2)
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_modulus) * T_inv
    
    # Stack all features
    return np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])


# ============================================================================
# TRANSFORMER TRAINING
# ============================================================================

def train_transformer(dataset_name, data):
    """Train transformer using K-fold OOF to prevent data leakage.
    
    Returns OOF embeddings for training data and a final model for test inference.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING TRANSFORMER (OOF): {dataset_name}")
    print(f"{'='*60}")
    
    from sklearn.model_selection import KFold
    
    torch.manual_seed(TRANSFORMER_SEED)
    np.random.seed(TRANSFORMER_SEED)
    
    df_train = data['train']
    sol_council = data['sol_council']
    solv_council = data['solv_council']
    
    # Prepare data
    X_sol = sol_council.loc[df_train['Solute']].values.astype(np.float32)
    X_solv = solv_council.loc[df_train['Solvent']].values.astype(np.float32)
    y = df_train['LogS'].values.astype(np.float32)
    
    # K-Fold OOF Training
    N_FOLDS = 5
    oof_embeddings = np.zeros((len(df_train), COUNCIL_SIZE * EMBED_DIM))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=TRANSFORMER_SEED)
    
    print(f"  Starting {N_FOLDS}-Fold OOF Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sol)):
        print(f"  Fold {fold + 1}/{N_FOLDS}...", end=' ')
        
        # Initialize fold model
        model = InteractionTransformer().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        # Create DataLoader for this fold
        dataset = TensorDataset(
            torch.tensor(X_sol[train_idx]).to(DEVICE),
            torch.tensor(X_solv[train_idx]).to(DEVICE),
            torch.tensor(y[train_idx]).view(-1, 1).to(DEVICE)
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train on fold
        model.train()
        for epoch in range(EPOCHS):
            for b_sol, b_solv, b_y in loader:
                optimizer.zero_grad()
                pred, _, _ = model(b_sol, b_solv)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()
        
        # Generate OOF embeddings for validation set
        model.eval()
        with torch.no_grad():
            _, feats, _ = model(
                torch.tensor(X_sol[val_idx]).to(DEVICE),
                torch.tensor(X_solv[val_idx]).to(DEVICE)
            )
            oof_embeddings[val_idx] = feats.cpu().numpy()
        
        print("Done")
    
    print("  OOF embeddings generated (no data leakage)")
    
    # Train final model on ALL data for test inference
    print("  Training final model on full training set...")
    final_model = InteractionTransformer().to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(
        torch.tensor(X_sol).to(DEVICE),
        torch.tensor(X_solv).to(DEVICE),
        torch.tensor(y).view(-1, 1).to(DEVICE)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model.train()
    for epoch in range(EPOCHS):
        for b_sol, b_solv, b_y in loader:
            optimizer.zero_grad()
            pred, _, _ = final_model(b_sol, b_solv)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()
    
    # Save final model
    output_dir = os.path.join(OUTPUT_DIR, dataset_name, "transformers")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"transformer_seed{TRANSFORMER_SEED}.pth")
    torch.save(final_model.state_dict(), model_path)
    print(f"  Saved: {model_path}")
    
    return final_model, oof_embeddings



# ============================================================================
# BASELINE MODELS
# ============================================================================

def get_baseline_models(seed, num_features=None):
    """Return dictionary of baseline models.
    
    Args:
        seed: Random seed for reproducibility
        num_features: Total number of features (needed for monotone constraints)
    """
    # Monotone constraints for temperature features (last 3 features)
    # Feature order: [X_raw, X_interact, Tm, T_red, T, T_inv]
    # Constraints: T_red (+1), T (+1), T_inv (-1)
    monotone_constraints = None
    if num_features is not None:
        monotone_constraints = [0] * num_features
        monotone_constraints[-3] = 1   # T_red: increasing
        monotone_constraints[-2] = 1   # T: increasing
        monotone_constraints[-1] = -1  # T_inv: decreasing
    
    return {
        'DecisionTree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=seed
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            monotone_constraints=monotone_constraints,
            random_state=seed,
            verbose=-1,
            early_stopping_rounds=100  # Match CatBoost
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            monotone_constraints=tuple(monotone_constraints) if monotone_constraints else None,
            early_stopping_rounds=100,  # Match CatBoost
            random_state=seed,
            verbosity=0
        ),
        'ANN': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed
        )
    }


def train_baselines(dataset_name, data, transformer_model, oof_embeddings_train):
    """Train all baseline models on all seeds using OOF embeddings."""
    print(f"\n{'='*60}")
    print(f"TRAINING BASELINES: {dataset_name}")
    print(f"{'='*60}")
    
    df_train = data['train']
    df_test = data['test']
    sol_council = data['sol_council']
    solv_council = data['solv_council']
    sol_raw = data['sol_raw']
    solv_raw = data['solv_raw']
    
    # Use OOF embeddings for training data (no data leakage)
    print("  Using OOF embeddings for training data...")
    embeddings_train = oof_embeddings_train
    
    # Generate embeddings for test data using final model
    print("  Generating embeddings for test data...")
    X_sol_test = sol_council.loc[df_test['Solute']].values.astype(np.float32)
    X_solv_test = solv_council.loc[df_test['Solvent']].values.astype(np.float32)
    embeddings_test = generate_transformer_embeddings(transformer_model, X_sol_test, X_solv_test)
    
    # Build feature matrices
    print("  Building feature matrices...")
    X_train = build_baseline_features(df_train, embeddings_train, sol_raw, solv_raw)
    X_test = build_baseline_features(df_test, embeddings_test, sol_raw, solv_raw)
    y_train = df_train['LogS'].values
    y_test = df_test['LogS'].values
    
    print(f"  Feature shape: {X_train.shape}")
    print(f"  Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    # Train each model on each seed
    num_features = X_train.shape[1]
    results = {model_name: {'rmse': [], 'r2': []} for model_name in get_baseline_models(42, num_features).keys()}
    
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        
        # Create 5% validation split for early stopping (same as CatBoost)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.05, random_state=seed
        )
        
        models = get_baseline_models(seed, num_features)
        
        for model_name, model in models.items():
            print(f"    Training {model_name}...", end=' ')
            
            # Train with validation set for models that support it
            if model_name == 'LightGBM':
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)]
                )
            elif model_name == 'XGBoost':
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=0  # Silence XGBoost output
                )
            else:
                # DecisionTree, RandomForest, ANN train on full training set
                model.fit(X_train, y_train)
            
            # Predict on test set
            preds = model.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            results[model_name]['rmse'].append(rmse)
            results[model_name]['r2'].append(r2)
            
            print(f"RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # Save model
            output_dir = os.path.join(OUTPUT_DIR, dataset_name, "baselines", model_name)
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f"model_seed{seed}.joblib")
            joblib.dump(model, model_path)
    
    return results


# ============================================================================
# RESULTS REPORTING
# ============================================================================

def save_results(all_results):
    """Save aggregated results to JSON and CSV."""
    # Create summary DataFrame
    summary_data = []
    
    for dataset_name, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            rmse_mean = np.mean(metrics['rmse'])
            rmse_std = np.std(metrics['rmse'])
            r2_mean = np.mean(metrics['r2'])
            r2_std = np.std(metrics['r2'])
            
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'RMSE_mean': rmse_mean,
                'RMSE_std': rmse_std,
                'R2_mean': r2_mean,
                'R2_std': r2_std,
                'Seeds': len(metrics['rmse'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full results to: {json_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (Mean ± Std)")
    print("="*80)
    
    for dataset_name in DATASETS:
        print(f"\n{dataset_name}:")
        print("-"*80)
        dataset_df = summary_df[summary_df['Dataset'] == dataset_name].sort_values('RMSE_mean')
        for _, row in dataset_df.iterrows():
            print(f"  {row['Model']:<15} RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}  "
                  f"R²: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("MULTI-DATASET BENCHMARK")
    print("="*80)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Baseline models: 5 (DT, RF, LightGBM, XGBoost, ANN)")
    print(f"Seeds per model: {len(SEEDS)}")
    print(f"Total models to train: {3 + (5 * 3 * len(SEEDS))} "
          f"(3 transformers + 75 baselines)")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = {}
    
    for dataset_name in DATASETS:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*80}")
        
        # Load data
        print(f"\nLoading {dataset_name} data...")
        data = load_dataset_features(dataset_name)
        print(f"  Train: {len(data['train'])} samples")
        print(f"  Test: {len(data['test'])} samples")
        
        # Train transformer with OOF
        transformer_model, oof_embeddings = train_transformer(dataset_name, data)
        
        # Train baselines using OOF embeddings
        results = train_baselines(dataset_name, data, transformer_model, oof_embeddings)
        all_results[dataset_name] = results
    
    # Save and display results
    save_results(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"All models saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()