#!/usr/bin/env python3
"""Quick inference speed benchmark for all models."""

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

# Config
DATASET = "bigsol1.0"
ALL_DATASETS_DIR = "all_datasets"
FEATURE_STORE_DIR = "feature_store"
OUTPUT_DIR = "benchmark_results"
SEED = 42
N_SAMPLES = 1000  # Test samples
N_ITERS = 5  # Inference iterations

COUNCIL_SIZE = 24
EMBED_DIM = 32
NUM_HEADS = 4

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                     "cuda" if torch.cuda.is_available() else "cpu")

# Transformer model
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


def load_test_data():
    """Load test data and features."""
    test_file = os.path.join(ALL_DATASETS_DIR, DATASET, "test.csv")
    sol_store = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "solute_council.parquet")).set_index("SMILES_KEY")
    solv_store = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "solvent_council.parquet")).set_index("SMILES_KEY")
    sol_raw = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "solute_raw.parquet")).set_index("SMILES_KEY")
    solv_raw = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "solvent_raw.parquet")).set_index("SMILES_KEY")

    df_test = pd.read_csv(test_file).head(N_SAMPLES)

    return {
        'test': df_test,
        'sol_council': sol_store,
        'solv_council': solv_store,
        'sol_raw': sol_raw,
        'solv_raw': solv_raw
    }


def build_baseline_features(df, embeddings, sol_raw, solv_raw):
    """Build feature matrix."""
    X_raw = np.hstack([
        sol_raw.loc[df['Solute']].values,
        solv_raw.loc[df['Solvent']].values
    ])

    Tm = sol_raw.loc[df['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T = df['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df['Temperature'].values).reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)

    X_reshaped = embeddings.reshape(embeddings.shape[0], 24, 32)
    X_modulus = np.linalg.norm(X_reshaped, axis=2)
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_modulus) * T_inv

    return np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])


def benchmark_transformer(model, X_sol, X_solv):
    """Benchmark transformer inference."""
    model.eval()
    model = model.to(DEVICE)

    sol_t = torch.tensor(X_sol, dtype=torch.float32).to(DEVICE)
    solv_t = torch.tensor(X_solv, dtype=torch.float32).to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _, _, _ = model(sol_t, solv_t)

    # Benchmark
    with torch.no_grad():
        start = time.time()
        for _ in range(N_ITERS):
            _, _, _ = model(sol_t, solv_t)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        elif DEVICE.type == 'mps':
            torch.mps.synchronize()

        elapsed = time.time() - start

    return (len(X_sol) * N_ITERS) / elapsed


def benchmark_sklearn(model, X):
    """Benchmark sklearn model."""
    # Warmup
    for _ in range(3):
        _ = model.predict(X[:100])

    # Benchmark
    start = time.time()
    for _ in range(N_ITERS):
        _ = model.predict(X)
    elapsed = time.time() - start

    return (len(X) * N_ITERS) / elapsed


def main():
    print("="*80)
    print("QUICK INFERENCE BENCHMARK")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Test samples: {N_SAMPLES}")
    print(f"Iterations: {N_ITERS}")
    print(f"Device: {DEVICE}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data = load_test_data()
    df_test = data['test']

    X_sol = data['sol_council'].loc[df_test['Solute']].values.astype(np.float32)
    X_solv = data['solv_council'].loc[df_test['Solvent']].values.astype(np.float32)

    results = []

    # ==== TRANSFORMER ====
    print("\nBenchmarking Transformer...")
    transformer_path = os.path.join(OUTPUT_DIR, DATASET, "transformers", f"transformer_seed{SEED}.pth")

    if os.path.exists(transformer_path):
        model = InteractionTransformer()
        model.load_state_dict(torch.load(transformer_path, map_location='cpu'))

        pairs_per_sec = benchmark_transformer(model, X_sol, X_solv)
        device_type = "GPU" if DEVICE.type in ['cuda', 'mps'] else "CPU"

        print(f"  {pairs_per_sec:.0f} pairs/sec - {device_type}")
        results.append({
            'Model': 'Transformer',
            'Pairs_per_Second': f"{pairs_per_sec:.0f}",
            'Device': device_type
        })

        # Generate embeddings for baseline models
        print("\nGenerating embeddings for baselines...")
        model.eval()
        with torch.no_grad():
            sol_t = torch.tensor(X_sol).to(DEVICE)
            solv_t = torch.tensor(X_solv).to(DEVICE)
            _, feats, _ = model(sol_t, solv_t)
            embeddings = feats.cpu().numpy()

        X_test = build_baseline_features(df_test, embeddings, data['sol_raw'], data['solv_raw'])
    else:
        print(f"  NOT FOUND: {transformer_path}")
        return

    # ==== BASELINE MODELS ====
    print("\nBenchmarking Baseline Models...")

    models = [
        ('DecisionTree', 'CPU (sklearn)'),
        ('RandomForest', 'CPU (sklearn)'),
        ('LightGBM', 'CPU (can use GPU)'),
        ('XGBoost', 'CPU (can use GPU)'),
        ('ANN', 'CPU (sklearn)')
    ]

    for model_name, device_info in models:
        model_path = os.path.join(OUTPUT_DIR, DATASET, "baselines", model_name, f"model_seed{SEED}.joblib")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            pairs_per_sec = benchmark_sklearn(model, X_test)

            print(f"  {model_name:<15} {pairs_per_sec:.0f} pairs/sec - {device_info}")
            results.append({
                'Model': model_name,
                'Pairs_per_Second': f"{pairs_per_sec:.0f}",
                'Device': device_info
            })
        else:
            print(f"  {model_name}: NOT FOUND")

    # Save results
    print("\n" + "="*80)
    print("RESULTS - GENERIC METHODS")
    print("="*80)

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    csv_path = "inference_speed_generic.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
