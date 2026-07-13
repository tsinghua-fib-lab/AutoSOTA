#!/usr/bin/env python3
"""
CPU-Only Timing Benchmark for Regime-I Methods
Runs non-GPU methods that work on any machine.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ============================================================================
# CONFIG
# ============================================================================

SEED = 123
TRAIN_PATH = "all_datasets/aqsoldb/train.csv"
TEST_PATH = "all_datasets/aqsoldb/test.csv"
SMILES_COL = "SMILES"
TARGET_COL = "LogS"

RESULTS = []

def log_result(method_name, train_time, inference_time=None, notes=""):
    """Log timing result."""
    n_test = pd.read_csv(TEST_PATH).shape[0]
    RESULTS.append({
        "Method": method_name,
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "ms/sample": (inference_time * 1000 / n_test) if inference_time else None,
        "Device": "CPU",
        "Notes": notes
    })
    inf_str = f", Inference: {inference_time:.3f}s" if inference_time else ""
    print(f"  ✓ {method_name}: Train={train_time:.2f}s{inf_str}")

# ============================================================================
# METHODS
# ============================================================================

def time_gse():
    """GSE: Closed-form solubility prediction."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    test_df = pd.read_csv(TEST_PATH)
    
    start = time.time()
    for smiles in test_df[SMILES_COL]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logP = Descriptors.MolLogP(mol)
            _ = -logP + 0.5  # Simplified GSE
    elapsed = time.time() - start
    
    log_result("GSE", 0.0, elapsed, "closed-form, no training")


def time_esol_model():
    """ESOL Model: Linear regression on 4 features."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from sklearn.linear_model import LinearRegression
    
    def compute_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n_heavy = mol.GetNumHeavyAtoms()
        return [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol),
            sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / n_heavy if n_heavy > 0 else 0
        ]
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = np.array([f for f in [compute_features(s) for s in train_df[SMILES_COL]] if f])
    y_train = train_df[TARGET_COL].values[:len(X_train)]
    X_test = np.array([f for f in [compute_features(s) for s in test_df[SMILES_COL]] if f])
    
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    _ = model.predict(X_test)
    inf_time = time.time() - start
    
    log_result("ESOL Model", train_time, inf_time, "linear regression")


def time_generic_baselines():
    """Train generic baselines."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import xgboost as xgb
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from featurizer import MoleculeFeaturizer
    
    print("  Generating molecular features...")
    featurizer = MoleculeFeaturizer()
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = featurizer.transform(train_df[SMILES_COL].tolist()).values
    y_train = train_df[TARGET_COL].values
    X_test = featurizer.transform(test_df[SMILES_COL].tolist()).values
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=SEED)
    
    # Decision Tree
    print("  Training Decision Tree...")
    start = time.time()
    dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=SEED)
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    _ = dt.predict(X_test)
    log_result("Decision Tree", train_time, time.time() - start)
    
    # Random Forest (100 trees for speed)
    print("  Training Random Forest (100 trees for speed)...")
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    _ = rf.predict(X_test)
    log_result("Random Forest", train_time, time.time() - start, "100 trees")
    
    # LightGBM (using 1000 iters for speed)
    print("  Training LightGBM (1000 iters for speed)...")
    start = time.time()
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, max_depth=8, random_state=SEED, verbose=-1)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    train_time = time.time() - start
    start = time.time()
    _ = lgbm.predict(X_test)
    log_result("LightGBM", train_time, time.time() - start)
    
    # XGBoost (using 1000 iters for speed)
    print("  Training XGBoost (1000 iters for speed)...")
    start = time.time()
    xgbm = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=8, random_state=SEED, verbosity=0)
    xgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    start = time.time()
    _ = xgbm.predict(X_test)
    log_result("XGBoost", train_time, time.time() - start)
    
    # ANN
    print("  Training ANN...")
    start = time.time()
    ann = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True, random_state=SEED)
    ann.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    _ = ann.predict(X_test)
    log_result("ANN", train_time, time.time() - start)


def time_ours():
    """Train Ours: CatBoost with custom features."""
    from catboost import CatBoostRegressor
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from featurizer import MoleculeFeaturizer
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print("  Generating molecular features...")
    featurizer = MoleculeFeaturizer()
    X_train = featurizer.transform(train_df[SMILES_COL].tolist())
    y_train = train_df[TARGET_COL].values
    X_test = featurizer.transform(test_df[SMILES_COL].tolist())
    
    start = time.time()
    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        verbose=1000,
        random_state=SEED,
        allow_writing_files=False,
        thread_count=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    _ = model.predict(X_test)
    inf_time = time.time() - start
    
    log_result("Ours", train_time, inf_time, "CatBoost 10000 iters")


def time_tayyebi():
    """Tayyebi: Mordred descriptors + RF."""
    from rdkit import Chem
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import VarianceThreshold
    
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        print("  ✗ Tayyebi: mordred not installed")
        return
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print("  Calculating Mordred descriptors (this takes a few minutes)...")
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Train features
    train_mols = [Chem.MolFromSmiles(s) for s in train_df[SMILES_COL]]
    train_valid = [(m, i) for i, m in enumerate(train_mols) if m is not None]
    train_valid_mols = [m for m, _ in train_valid]
    train_valid_idx = [i for _, i in train_valid]
    X_train_raw = calc.pandas(train_valid_mols).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = train_df[TARGET_COL].values[train_valid_idx]
    
    # Test features
    test_mols = [Chem.MolFromSmiles(s) for s in test_df[SMILES_COL]]
    test_valid = [(m, i) for i, m in enumerate(test_mols) if m is not None]
    test_valid_mols = [m for m, _ in test_valid]
    X_test_raw = calc.pandas(test_valid_mols).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Align columns
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)
    
    # Tayyebi filtering
    cols_to_drop = ['FilterItLogS', 'SLogP']
    X_train_clean = X_train_raw.drop(columns=cols_to_drop, errors='ignore')
    X_test_clean = X_test_raw.drop(columns=cols_to_drop, errors='ignore')
    
    vt = VarianceThreshold(threshold=0.1)
    vt.fit(X_train_clean)
    X_train_var = X_train_clean.loc[:, vt.get_support()]
    X_test_var = X_test_clean.loc[:, vt.get_support()]
    
    # Correlation filter
    corr_matrix = X_train_var.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]
    X_train_final = X_train_var.drop(columns=to_drop).values
    X_test_final = X_test_var.drop(columns=to_drop).values
    
    print(f"  Features after filtering: {X_train_final.shape[1]}")
    
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_final, y_train)
    train_time = time.time() - start
    
    start = time.time()
    _ = rf.predict(X_test_final)
    inf_time = time.time() - start
    
    log_result("Tayyebi", train_time, inf_time, "Mordred + RF 100 trees")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CPU-ONLY TIMING BENCHMARK (Regime-I)")
    print("=" * 70)
    
    print("\n[1/6] GSE...")
    time_gse()
    
    print("\n[2/6] ESOL Model...")
    time_esol_model()
    
    print("\n[3/6] Generic ML Baselines...")
    time_generic_baselines()
    
    print("\n[4/6] Tayyebi...")
    time_tayyebi()
    
    print("\n[5/6] Ours...")
    time_ours()
    
    # Print results
    print("\n" + "=" * 90)
    print("TIMING RESULTS (CPU-ONLY)")
    print("=" * 90)
    print(f"{'Method':20} | {'Train (s)':10} | {'Inference (s)':12} | {'ms/sample':10} | Notes")
    print("-" * 90)
    
    results_df = pd.DataFrame(RESULTS)
    results_df = results_df.sort_values("Train Time (s)")
    
    for _, row in results_df.iterrows():
        inf_s = f"{row['Inference Time (s)']:.4f}" if row['Inference Time (s)'] else "N/A"
        ms = f"{row['ms/sample']:.4f}" if row['ms/sample'] else "N/A"
        notes = row['Notes'] if row['Notes'] else ""
        print(f"{row['Method']:20} | {row['Train Time (s)']:10.2f} | {inf_s:>12} | {ms:>10} | {notes}")
    
    results_df.to_csv("timing_results_cpu.csv", index=False)
    print(f"\nSaved to timing_results_cpu.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()
