# esol_model_baseline.py
"""
ESOL Model Baseline (Delaney, 2004)

Linear regression on 4 features:
- cLogP (MolLogP in RDKit)
- Molecular Weight
- Rotatable Bonds  
- Aromatic Proportion (aromatic atoms / total heavy atoms)

Deterministic - no seeds required.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def compute_esol_features(smiles):
    """Compute the 4 ESOL features from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # cLogP
    clogp = Descriptors.MolLogP(mol)
    
    # Molecular Weight
    mw = Descriptors.MolWt(mol)
    
    # Rotatable Bonds
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Aromatic Proportion (aromatic atoms / heavy atoms)
    n_heavy = mol.GetNumHeavyAtoms()
    n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_prop = n_aromatic / n_heavy if n_heavy > 0 else 0.0
    
    return [clogp, mw, rot_bonds, aromatic_prop]


def prepare_data(df):
    """Prepare features and targets from dataframe."""
    features = []
    targets = []
    valid_idx = []
    
    for i, row in df.iterrows():
        feats = compute_esol_features(row['SMILES'])
        if feats is not None:
            features.append(feats)
            targets.append(row['LogS'])
            valid_idx.append(i)
    
    return np.array(features), np.array(targets), valid_idx


def evaluate_dataset(name, train_path, test_path):
    """Train on train set, evaluate on test set."""
    print(f"\n{name}:")
    print("-" * 40)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare features
    print("  Computing features...")
    X_train, y_train, _ = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df)
    
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train linear regression (deterministic, closed-form)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    variance = np.var(y_test)
    r2 = 1 - (rmse**2 / variance) if variance > 0 else np.nan
    
    # Show learned coefficients
    feature_names = ['cLogP', 'MolWt', 'RotBonds', 'AromaticProp']
    print(f"  Coefficients: {dict(zip(feature_names, model.coef_.round(4)))}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return rmse, r2


def main():
    print("=" * 60)
    print("ESOL Model Baseline (Delaney, 2004)")
    print("Linear Regression: cLogP + MolWt + RotBonds + AromaticProp")
    print("=" * 60)
    
    datasets = {
        "AqSolDB": ("all_datasets/aqsoldb/train.csv", "all_datasets/aqsoldb/test.csv"),
        "ESOL": ("all_datasets/esol/train.csv", "all_datasets/esol/test.csv"),
        "SC2": ("all_datasets/sc2/train.csv", "all_datasets/sc2/test.csv"),
    }
    
    results = []
    for name, (train_path, test_path) in datasets.items():
        rmse, r2 = evaluate_dataset(name, train_path, test_path)
        results.append({"Dataset": name, "RMSE": rmse, "R²": r2})
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Test Set)")
    print("=" * 60)
    for r in results:
        print(f"{r['Dataset']:12s} | RMSE: {r['RMSE']:.4f} | R²: {r['R²']:.4f}")
    print("=" * 60)
    print("\nNote: Deterministic linear regression, no seeds required.")


if __name__ == "__main__":
    main()
