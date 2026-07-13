# aqsolpred_5seeds.py
"""
AqSolPred Baseline - 5 Seed Evaluation
Consensus model: Neural Nets + XGBoost + RandomForest

Adapted to work with simple SMILES + LogS datasets.
Uses same Mordred descriptors and LASSO selection as original.
CPU-based, can run locally on laptop.
Reports RMSE ± std and R² across 5 seeds with timing.
"""

import pandas as pd
import numpy as np
import time
import warnings
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# Silence warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]

DATASETS = {
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}

SMILES_COL = "SMILES"
TARGET_COL = "LogS"


def get_mordred_calculator():
    """Create the same Mordred calculator as original AqSolPred"""
    from mordred import Calculator
    from mordred import descriptors as desc
    
    calc = Calculator()
    
    calc.register(desc.AtomCount.AtomCount("X"))
    calc.register(desc.AtomCount.AtomCount("HeavyAtom"))
    calc.register(desc.Aromatic.AromaticAtomsCount())
    calc.register(desc.HydrogenBond.HBondAcceptor())
    calc.register(desc.HydrogenBond.HBondDonor())
    calc.register(desc.RotatableBond.RotatableBondsCount())
    calc.register(desc.BondCount.BondCount("any", False))
    calc.register(desc.Aromatic.AromaticBondsCount())
    calc.register(desc.BondCount.BondCount("heavy", False))
    calc.register(desc.BondCount.BondCount("single", False))
    calc.register(desc.BondCount.BondCount("double", False))
    calc.register(desc.BondCount.BondCount("triple", False))
    calc.register(desc.McGowanVolume.McGowanVolume())
    calc.register(desc.TopoPSA.TopoPSA(True))
    calc.register(desc.TopoPSA.TopoPSA(False))
    calc.register(desc.MoeType.LabuteASA())
    calc.register(desc.Polarizability.APol())
    calc.register(desc.Polarizability.BPol())
    calc.register(desc.AcidBase.AcidicGroupCount())
    calc.register(desc.AcidBase.BasicGroupCount())
    calc.register(desc.EccentricConnectivityIndex.EccentricConnectivityIndex())
    calc.register(desc.TopologicalCharge.TopologicalCharge("raw", 1))
    calc.register(desc.TopologicalCharge.TopologicalCharge("mean", 1))
    calc.register(desc.SLogP.SLogP())
    calc.register(desc.BertzCT.BertzCT())
    calc.register(desc.BalabanJ.BalabanJ())
    calc.register(desc.WienerIndex.WienerIndex(True))
    calc.register(desc.ZagrebIndex.ZagrebIndex(1, 1))
    calc.register(desc.ABCIndex.ABCIndex())
    calc.register(desc.RingCount.RingCount(None, False, False, None, None))
    calc.register(desc.RingCount.RingCount(None, False, False, None, True))
    calc.register(desc.RingCount.RingCount(None, False, False, True, None))
    calc.register(desc.RingCount.RingCount(None, False, False, True, True))
    calc.register(desc.RingCount.RingCount(None, False, False, False, None))
    calc.register(desc.RingCount.RingCount(None, False, True, None, None))
    calc.register(desc.EState)  # Register entire EState module
    
    return calc


def calculate_descriptors(df, calc, name="Data"):
    """Calculate Mordred descriptors for dataframe"""
    print(f"      Calculating Mordred descriptors for {name}...")
    
    descriptors = []
    valid_indices = []
    targets = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"      {name}"):
        smiles = row[SMILES_COL]
        target = row[TARGET_COL]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            mol = Chem.AddHs(mol)
            
            # Skip problematic molecules (same filters as original)
            if "." in smiles:  # Multi-fragment
                continue
            if "+" in smiles or "-" in smiles:  # Charged
                continue
            
            result = calc(mol)
            descriptors.append(result._values)
            valid_indices.append(idx)
            targets.append(target)
            
        except Exception:
            continue
    
    # Create descriptor names
    desc_names = [str(d) for d in calc.descriptors]
    
    # Create DataFrame
    desc_df = pd.DataFrame(descriptors, columns=desc_names)
    desc_df[TARGET_COL] = targets
    
    # Convert to numeric, fill NaN with 0
    desc_df = desc_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Drop MIN/MAX columns (same as original)
    drop_cols = [c for c in desc_df.columns if 'MIN' in c or 'MAX' in c]
    desc_df = desc_df.drop(columns=drop_cols, errors='ignore')
    
    print(f"      Valid samples: {len(desc_df)}/{len(df)}")
    
    return desc_df


def select_features_lasso(train_data, test_data):
    """LASSO feature selection (same as original)"""
    X_train = train_data.drop(columns=[TARGET_COL])
    y_train = train_data[TARGET_COL]
    X_test = test_data.drop(columns=[TARGET_COL])
    
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=1)
    lasso.fit(X_train, y_train)
    
    model = SelectFromModel(lasso, prefit=True)
    X_new = model.transform(X_train)
    
    # Get selected columns
    selected_features = pd.DataFrame(
        model.inverse_transform(X_new), 
        columns=X_train.columns
    )
    selected_cols = selected_features.columns[selected_features.var() != 0]
    
    print(f"      Selected {len(selected_cols)} features by LASSO")
    
    return train_data[selected_cols], test_data[selected_cols]


def train_and_evaluate(X_train, y_train, X_test, y_test, seed):
    """Train AqSolPred consensus model and return metrics"""
    
    # Train Neural Network (same params as original)
    mlp = MLPRegressor(
        activation='tanh', 
        hidden_layer_sizes=(500), 
        max_iter=500, 
        random_state=seed, 
        solver='adam'
    )
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    
    # Train XGBoost (same params as original)
    xgb = xgboost.XGBRegressor(n_estimators=1000, random_state=seed, verbosity=0)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    
    # Train Random Forest (same params as original)
    rf = RandomForestRegressor(n_estimators=1000, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    
    # Consensus prediction (average of 3 models)
    pred_consensus = (pred_mlp + pred_xgb + pred_rf) / 3
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred_consensus))
    r2 = r2_score(y_test, pred_consensus)
    
    # Print individual model RMSEs
    rmse_mlp = np.sqrt(mean_squared_error(y_test, pred_mlp))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    print(f"      NN: {rmse_mlp:.4f}, XGB: {rmse_xgb:.4f}, RF: {rmse_rf:.4f}, Consensus: {rmse:.4f}")
    
    return rmse, r2


def evaluate_dataset(name, train_path, test_path):
    """Evaluate on a dataset across 5 seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Calculate descriptors ONCE (shared across seeds)
    print("\n  Calculating descriptors (once for all seeds)...")
    calc = get_mordred_calculator()
    train_desc = calculate_descriptors(train_df, calc, "Train")
    test_desc = calculate_descriptors(test_df, calc, "Test")
    
    # Align columns
    common_cols = list(set(train_desc.columns) & set(test_desc.columns))
    train_desc = train_desc[common_cols]
    test_desc = test_desc[common_cols]
    
    # LASSO feature selection (once)
    print("\n  Applying LASSO feature selection...")
    X_train, X_test = select_features_lasso(train_desc, test_desc)
    y_train = train_desc[TARGET_COL].values
    y_test = test_desc[TARGET_COL].values
    
    print(f"\n  Training 5 seeds...")
    results = []
    seed_times = []
    
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        seed_start = time.time()
        
        rmse, r2 = train_and_evaluate(X_train, y_train, X_test, y_test, seed)
        results.append((rmse, r2))
        
        seed_elapsed = time.time() - seed_start
        seed_times.append(seed_elapsed)
        print(f"      Time: {seed_elapsed:.1f}s")
    
    return {
        "name": name,
        "rmse_mean": np.mean([r[0] for r in results]),
        "rmse_std": np.std([r[0] for r in results]),
        "r2_mean": np.mean([r[1] for r in results]),
        "r2_std": np.std([r[1] for r in results]),
        "avg_time": np.mean(seed_times),
        "total_time": np.sum(seed_times)
    }


def main():
    print("=" * 60)
    print("AqSolPred Baseline - 5 Seed Evaluation")
    print("Consensus: Neural Nets + XGBoost + RandomForest")
    print("=" * 60)
    
    all_results = []
    total_start = time.time()
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        all_results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - AqSolPred Baseline (Consensus)")
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
