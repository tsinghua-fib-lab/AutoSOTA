# gse_baseline.py
"""
General Solubility Equation (GSE) Baseline
Yalkowsky & Valvani, 1980

log S = -log P - 0.01 * (Mpt - 25) + 0.5

where:
- log P: octanol-water partition coefficient (computed via RDKit MolLogP)
- Mpt: melting point in Celsius (estimated via Joback method)

No training required - closed-form equation applied to test sets.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# Joback melting point estimation (same as featurizer.py)
JOBACK_GROUPS = {
    "ch3": ("[CH3;X4;!R]", -5.10), "ch2_c": ("[CH2;X4;!R]", 11.27),
    "ch_c": ("[CH1;X4;!R]", 12.64), "c_c": ("[CH0;X4;!R]", 46.43),
    "ch2_r": ("[CH2;X4;R]", 8.25), "ch_r": ("[CH1;X4;R]", 20.15),
    "c_r": ("[CH0;X4;R]", 37.40), "c=c_c": ("[CX3;!R]=[CX3;!R]", 4.18),
    "c=c_r": ("[c,C;R]=[c,C;R]", 13.02), "F": ("[F]", 9.88),
    "Cl": ("[Cl]", 17.51), "Br": ("[Br]", 26.15), "I": ("[I]", 37.0),
    "oh_a": ("[OH;!#6a]", 20.0), "oh_p": ("[OH;a]", 44.45),
    "ether_c": ("[OD2;!R]([#6])[#6]", 22.42), "ether_r": ("[OD2;R]([#6])[#6]", 31.22),
    "co": ("[CX3]=[OX1]", 26.15), "ester": ("[CX3](=[OX1])[OX2H0]", 30.0),
    "nh2": ("[NH2]", 25.72), "nh_c": ("[NH1;!R]", 27.15), "nh_r": ("[NH1;R]", 30.12),
    "nitro": ("[NX3](=[OX1])=[OX1]", 45.0), "nitrile": ("[NX1]#[CX2]", 33.15)
}


def estimate_melting_point_celsius(mol):
    """
    Estimate melting point in Celsius using Joback method.
    Returns Kelvin - 273.15 to convert to Celsius.
    """
    if mol is None:
        return np.nan
    
    tm_sum = 122.5  # Joback constant
    for smarts, weight in JOBACK_GROUPS.values():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = len(mol.GetSubstructMatches(pattern))
            tm_sum += matches * weight
    
    # Joback gives Kelvin, convert to Celsius
    # Also apply minimum of 150K as in featurizer
    tm_kelvin = max(tm_sum, 150.0)
    return tm_kelvin - 273.15  # Convert to Celsius


def predict_gse(smiles):
    """
    Apply General Solubility Equation:
    log S = -log P - 0.01 * (Mpt - 25) + 0.5
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    
    logP = Descriptors.MolLogP(mol)
    Mpt_celsius = estimate_melting_point_celsius(mol)
    
    logS = -logP - 0.01 * (Mpt_celsius - 25) + 0.5
    return logS


def evaluate_dataset(name, test_path):
    """Evaluate GSE on a single dataset's test set."""
    df = pd.read_csv(test_path)
    
    # Apply GSE to all test samples
    preds = [predict_gse(s) for s in tqdm(df['SMILES'], desc=f"GSE on {name}", leave=False)]
    preds = np.array(preds)
    
    # Handle any NaN predictions
    valid_mask = ~np.isnan(preds)
    n_valid = valid_mask.sum()
    
    if n_valid == 0:
        return np.nan, np.nan, 0
    
    targets = df['LogS'].values[valid_mask]
    preds_valid = preds[valid_mask]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds_valid))
    variance = np.var(targets)
    r2 = 1 - (rmse**2 / variance) if variance > 0 else np.nan
    
    return rmse, r2, n_valid


def main():
    print("=" * 60)
    print("General Solubility Equation (GSE) Baseline")
    print("log S = -logP - 0.01*(Mpt - 25) + 0.5")
    print("=" * 60)
    
    # All datasets
    datasets = {
        "AqSolDB": "all_datasets/aqsoldb/test.csv",
        "ESOL": "all_datasets/esol/test.csv",
        "SC2": "all_datasets/sc2/test.csv",
    }
    
    results = []
    for name, path in datasets.items():
        rmse, r2, n_valid = evaluate_dataset(name, path)
        results.append({
            "Dataset": name,
            "RMSE": rmse,
            "R²": r2,
            "N": n_valid
        })
        print(f"{name}: RMSE = {rmse:.4f}, R² = {r2:.4f}, N = {n_valid}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"{r['Dataset']:12s} | RMSE: {r['RMSE']:.4f} | R²: {r['R²']:.4f}")
    
    print("=" * 60)
    print("\nNote: GSE is a closed-form equation, no training required.")
    print("Results reported on test sets only for fair comparison.")


if __name__ == "__main__":
    main()
