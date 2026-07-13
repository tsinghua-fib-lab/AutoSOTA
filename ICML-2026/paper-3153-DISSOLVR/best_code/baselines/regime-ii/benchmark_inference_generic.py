#!/usr/bin/env python3
"""
Benchmark inference speed for generic baseline methods.
Measures pairs/second for each model type.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "bigsol1.0"  # Use first dataset for benchmarking
ALL_DATASETS_DIR = "all_datasets"
FEATURE_STORE_DIR = "feature_store"
OUTPUT_DIR = "benchmark_results"
SEED = 42
NUM_INFERENCE_SAMPLES = 1000  # Number of samples to use for inference benchmarking
NUM_WARMUP = 100  # Warmup iterations

# Transformer config
COUNCIL_SIZE = 24
EMBED_DIM = 32
NUM_HEADS = 4

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

def load_test_data():
    """Load test data and features."""
    test_file = os.path.join(ALL_DATASETS_DIR, DATASET, "test.csv")

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

    df_test = pd.read_csv(test_file)

    # Sample for benchmarking
    df_test = df_test.head(NUM_INFERENCE_SAMPLES)

    return {
        'test': df_test,
        'sol_council': sol_store,
        'solv_council': solv_store,
        'sol_raw': sol_raw,
        'solv_raw': solv_raw
    }


def generate_transformer_embeddings(model, X_sol, X_solv, device):
    """Generate embeddings using transformer."""
    model.eval()
    with torch.no_grad():
        sol_tensor = torch.tensor(X_sol, dtype=torch.float32).to(device)
        solv_tensor = torch.tensor(X_solv, dtype=torch.float32).to(device)
        _, feats, _ = model(sol_tensor, solv_tensor)
        return feats.cpu().numpy()


def build_baseline_features(df, embeddings, sol_raw, solv_raw):
    """Build feature matrix for baseline models."""
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

    # Signed-modulus interactions
    X_reshaped = embeddings.reshape(embeddings.shape[0], 24, 32)
    X_modulus = np.linalg.norm(X_reshaped, axis=2)
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_modulus) * T_inv

    return np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_transformer_inference(model, X_sol, X_solv, device):
    """Benchmark transformer inference speed."""
    model.eval()
    model = model.to(device)

    sol_tensor = torch.tensor(X_sol, dtype=torch.float32).to(device)
    solv_tensor = torch.tensor(X_solv, dtype=torch.float32).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _, _, _ = model(sol_tensor, solv_tensor)

    # Benchmark
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _, _, _ = model(sol_tensor, solv_tensor)
        if device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
        elapsed = time.time() - start

    pairs_per_second = (len(X_sol) * 10) / elapsed
    return pairs_per_second


def benchmark_sklearn_model(model, X):
    """Benchmark sklearn model inference speed."""
    # Warmup
    for _ in range(10):
        _ = model.predict(X[:NUM_WARMUP])

    # Benchmark
    start = time.time()
    _ = model.predict(X)
    elapsed = time.time() - start

    pairs_per_second = len(X) / elapsed
    return pairs_per_second


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("INFERENCE BENCHMARK - GENERIC METHODS")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Inference samples: {NUM_INFERENCE_SAMPLES}")
    print(f"Device: {DEVICE}")
    print("="*80)

    # Load data
    print("\nLoading test data...")
    data = load_test_data()
    df_test = data['test']
    sol_council = data['sol_council']
    solv_council = data['solv_council']
    sol_raw = data['sol_raw']
    solv_raw = data['solv_raw']

    print(f"Test samples: {len(df_test)}")

    # Prepare transformer inputs
    X_sol = sol_council.loc[df_test['Solute']].values.astype(np.float32)
    X_solv = solv_council.loc[df_test['Solvent']].values.astype(np.float32)

    results = []

    # ========================================================================
    # BENCHMARK TRANSFORMER
    # ========================================================================
    print("\n" + "="*80)
    print("TRANSFORMER")
    print("="*80)

    # Load transformer model
    transformer_path = os.path.join(OUTPUT_DIR, DATASET, "transformers", f"transformer_seed{SEED}.pth")

    if os.path.exists(transformer_path):
        print(f"Loading transformer from: {transformer_path}")
        transformer_model = InteractionTransformer()
        transformer_model.load_state_dict(torch.load(transformer_path, map_location='cpu'))
        transformer_model = transformer_model.to(DEVICE)

        pairs_per_sec = benchmark_transformer_inference(transformer_model, X_sol, X_solv, DEVICE)
        device_type = "GPU" if DEVICE.type in ['cuda', 'mps'] else "CPU"

        print(f"Inference speed: {pairs_per_sec:.2f} pairs/second")
        print(f"Device: {device_type} ({DEVICE})")

        results.append({
            'Model': 'Transformer',
            'Pairs_per_Second': pairs_per_sec,
            'Device': device_type,
            'Device_Detail': str(DEVICE)
        })

        # Generate embeddings for baseline models
        print("\nGenerating embeddings for baseline models...")
        embeddings_test = generate_transformer_embeddings(transformer_model, X_sol, X_solv, DEVICE)
        X_test = build_baseline_features(df_test, embeddings_test, sol_raw, solv_raw)
        print(f"Feature matrix shape: {X_test.shape}")
    else:
        print(f"Transformer model not found at: {transformer_path}")
        print("Please train models first using baselining_generic_methods.py")
        return

    # ========================================================================
    # BENCHMARK BASELINE MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)

    model_names = ['DecisionTree', 'RandomForest', 'LightGBM', 'XGBoost', 'ANN']

    for model_name in model_names:
        print(f"\n{model_name}:")
        print("-"*80)

        model_path = os.path.join(OUTPUT_DIR, DATASET, "baselines", model_name, f"model_seed{SEED}.joblib")

        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = joblib.load(model_path)

            # Check if model can use GPU
            device_type = "CPU"
            device_detail = "CPU (sklearn/lightgbm/xgboost)"

            if model_name == "LightGBM":
                device_detail = "CPU (LightGBM - could use GPU with device='gpu')"
            elif model_name == "XGBoost":
                device_detail = "CPU (XGBoost - could use GPU with tree_method='gpu_hist')"

            pairs_per_sec = benchmark_sklearn_model(model, X_test)

            print(f"Inference speed: {pairs_per_sec:.2f} pairs/second")
            print(f"Device: {device_type}")

            results.append({
                'Model': model_name,
                'Pairs_per_Second': pairs_per_sec,
                'Device': device_type,
                'Device_Detail': device_detail
            })
        else:
            print(f"Model not found at: {model_path}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Pairs_per_Second', ascending=False)

    print("\n" + results_df.to_string(index=False))

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "inference_benchmark_generic.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print device summary
    print("\n" + "="*80)
    print("DEVICE USAGE SUMMARY")
    print("="*80)

    for _, row in results_df.iterrows():
        print(f"{row['Model']:<15} - {row['Device']:<10} - {row['Pairs_per_Second']:.2f} pairs/sec")


if __name__ == "__main__":
    main()
