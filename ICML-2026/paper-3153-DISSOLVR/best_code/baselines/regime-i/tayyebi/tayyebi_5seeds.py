# tayyebi_5seeds.py
"""
Tayyebi et al. Baseline - 5 Seed Evaluation

Two methods:
1. Mordred descriptors + RandomForest (variance + correlation filtering)
2. Morgan-2048 fingerprints + RandomForest

Evaluates on AqSolDB, ESOL, SC2 datasets with 5 seeds.
Reports RMSE ± std and R² for each.
"""

import os
import warnings
import numpy as np
import pandas as pd
import time
from rdkit import Chem
from rdkit import RDLogger

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit.Chem import AllChem

# Try to import mordred
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    print("WARNING: mordred not installed. Mordred baseline will be skipped.")
    MORDRED_AVAILABLE = False


# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]

DATASETS = {
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}

SMILES_COL = "SMILES"
TARGET_COL = "LogS"


# ================= MORDRED FEATURES =================
def calc_mordred_features(smiles_list, name="Data"):
    """Calculate Mordred descriptors"""
    if not MORDRED_AVAILABLE:
        return None
        
    print(f"  [{name}] Calculating Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [m for m in mols if m is not None]
    
    if len(valid_mols) < len(mols):
        print(f"    Warning: {len(mols) - len(valid_mols)} invalid molecules dropped.")
    
    df_desc = calc.pandas(valid_mols)
    df_desc = df_desc.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df_desc, valid_indices


def apply_mordred_filtering(X_train, X_test):
    """Apply Tayyebi's filtering: drop specific cols, variance < 0.1, correlation > 0.8"""
    # Drop specific columns
    cols_to_drop = ['FilterItLogS', 'SLogP']
    X_train_clean = X_train.drop(columns=cols_to_drop, errors='ignore')
    X_test_clean = X_test.drop(columns=cols_to_drop, errors='ignore')
    
    # Variance threshold = 0.1
    vt = VarianceThreshold(threshold=0.1)
    vt.fit(X_train_clean)
    mask_var = vt.get_support()
    X_train_var = X_train_clean.loc[:, mask_var]
    X_test_var = X_test_clean.loc[:, mask_var]
    
    # Correlation filter > 0.8
    corr_matrix = X_train_var.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    
    X_train_final = X_train_var.drop(columns=to_drop)
    X_test_final = X_test_var.drop(columns=to_drop)
    
    return X_train_final, X_test_final


# ================= MORGAN FEATURES =================
def calc_morgan_2048(smiles_list, name="Data"):
    """Calculate Morgan-2048 fingerprints"""
    print(f"  [{name}] Generating Morgan-2048 fingerprints...")
    feats = []
    valid_indices = []
    
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            feats.append(list(fp))
            valid_indices.append(i)
    
    return pd.DataFrame(feats, index=valid_indices), valid_indices


# ================= EVALUATION =================
def evaluate_mordred(train_df, test_df, seed):
    """Evaluate Mordred + RF baseline for one seed"""
    if not MORDRED_AVAILABLE:
        return None, None
    
    # Generate features
    X_train_raw, train_valid = calc_mordred_features(train_df[SMILES_COL], "Train")
    X_test_raw, test_valid = calc_mordred_features(test_df[SMILES_COL], "Test")
    
    y_train = train_df.iloc[train_valid][TARGET_COL].values
    y_test = test_df.iloc[test_valid][TARGET_COL].values
    
    # Align test columns to train
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)
    
    # Apply filtering
    X_train_final, X_test_final = apply_mordred_filtering(X_train_raw, X_test_raw)
    
    # Train RF
    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_train_final, y_train)
    
    # Evaluate
    preds = model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return rmse, r2


def evaluate_morgan(train_df, test_df, seed):
    """Evaluate Morgan-2048 + RF baseline for one seed"""
    # Generate features
    X_train, train_valid = calc_morgan_2048(train_df[SMILES_COL], "Train")
    X_test, test_valid = calc_morgan_2048(test_df[SMILES_COL], "Test")
    
    y_train = train_df.iloc[train_valid][TARGET_COL].values
    y_test = test_df.iloc[test_valid][TARGET_COL].values
    
    # Train RF
    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return rmse, r2


def evaluate_dataset(name, train_path, test_path):
    """Evaluate both methods on a dataset across 5 seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # ===== PRECOMPUTE FEATURES ONCE =====
    print("\n  Precomputing features (once for all seeds)...")
    
    # Morgan features
    print("  Computing Morgan-2048 features...")
    X_train_morgan, train_valid_morgan = calc_morgan_2048(train_df[SMILES_COL], "Train")
    X_test_morgan, test_valid_morgan = calc_morgan_2048(test_df[SMILES_COL], "Test")
    y_train_morgan = train_df.iloc[train_valid_morgan][TARGET_COL].values
    y_test_morgan = test_df.iloc[test_valid_morgan][TARGET_COL].values
    
    # Mordred features (if available)
    X_train_mordred_final = None
    X_test_mordred_final = None
    y_train_mordred = None
    y_test_mordred = None
    
    if MORDRED_AVAILABLE:
        print("  Computing Mordred features...")
        X_train_mordred_raw, train_valid_mordred = calc_mordred_features(train_df[SMILES_COL], "Train")
        X_test_mordred_raw, test_valid_mordred = calc_mordred_features(test_df[SMILES_COL], "Test")
        
        y_train_mordred = train_df.iloc[train_valid_mordred][TARGET_COL].values
        y_test_mordred = test_df.iloc[test_valid_mordred][TARGET_COL].values
        
        # Align columns and apply filtering
        X_test_mordred_raw = X_test_mordred_raw.reindex(columns=X_train_mordred_raw.columns, fill_value=0)
        X_train_mordred_final, X_test_mordred_final = apply_mordred_filtering(X_train_mordred_raw, X_test_mordred_raw)
        
        print(f"    Mordred features after filtering: {X_train_mordred_final.shape[1]}")
    
    print("  Feature computation complete!\n")
    
    # ===== TRAIN MODELS PER SEED =====
    mordred_results = []
    morgan_results = []
    seed_times = []
    
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        seed_start = time.time()
        
        # Morgan RF
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_train_morgan, y_train_morgan)
        preds = model.predict(X_test_morgan)
        morgan_rmse = np.sqrt(mean_squared_error(y_test_morgan, preds))
        morgan_r2 = r2_score(y_test_morgan, preds)
        morgan_results.append((morgan_rmse, morgan_r2))
        print(f"    Morgan-2048: RMSE={morgan_rmse:.4f}, R²={morgan_r2:.4f}")
        
        # Mordred RF
        if MORDRED_AVAILABLE and X_train_mordred_final is not None:
            model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(X_train_mordred_final, y_train_mordred)
            preds = model.predict(X_test_mordred_final)
            mordred_rmse = np.sqrt(mean_squared_error(y_test_mordred, preds))
            mordred_r2 = r2_score(y_test_mordred, preds)
            mordred_results.append((mordred_rmse, mordred_r2))
            print(f"    Mordred:     RMSE={mordred_rmse:.4f}, R²={mordred_r2:.4f}")
        
        seed_elapsed = time.time() - seed_start
        seed_times.append(seed_elapsed)
        print(f"    Time: {seed_elapsed:.1f}s")
    
    return {
        "name": name,
        "morgan": {
            "rmse_mean": np.mean([r[0] for r in morgan_results]),
            "rmse_std": np.std([r[0] for r in morgan_results]),
            "r2_mean": np.mean([r[1] for r in morgan_results]),
            "r2_std": np.std([r[1] for r in morgan_results]),
        },
        "mordred": {
            "rmse_mean": np.mean([r[0] for r in mordred_results]) if mordred_results else None,
            "rmse_std": np.std([r[0] for r in mordred_results]) if mordred_results else None,
            "r2_mean": np.mean([r[1] for r in mordred_results]) if mordred_results else None,
            "r2_std": np.std([r[1] for r in mordred_results]) if mordred_results else None,
        },
        "avg_time": np.mean(seed_times),
        "total_time": np.sum(seed_times)
    }



def main():
    print("=" * 60)
    print("Tayyebi et al. Baseline - 5 Seed Evaluation")
    print("Methods: Mordred+RF and Morgan-2048+RF")
    print("=" * 60)
    
    all_results = []
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        all_results.append(result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - Tayyebi et al. Baseline")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        
        morgan = r['morgan']
        print(f"  Morgan-2048 + RF:")
        print(f"    RMSE: {morgan['rmse_mean']:.4f} ± {morgan['rmse_std']:.4f}")
        print(f"    R²:   {morgan['r2_mean']:.4f} ± {morgan['r2_std']:.4f}")
        
        mordred = r['mordred']
        if mordred['rmse_mean'] is not None:
            print(f"  Mordred + RF:")
            print(f"    RMSE: {mordred['rmse_mean']:.4f} ± {mordred['rmse_std']:.4f}")
            print(f"    R²:   {mordred['r2_mean']:.4f} ± {mordred['r2_std']:.4f}")
        
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s | Total: {r['total_time']:.1f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
