import pandas as pd
import numpy as np
from pathlib import Path
from math import log10
from thermo.chemical import Chemical
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from prepare_data import get_descs 

# --- CONFIG ---
RAW_DB = "../data/BigSolDB.csv"
USER_TRAIN = "../data/train.csv" # Your cleaned Top 19 Solvents file
USER_TEST = "../data/test.csv"   # Your cleaned "The Rest" file
OUTPUT_DIR = Path("data/baseline_fair")

# --- UTILS ---
enumerator = rdMolStandardize.TautomerEnumerator()

def canonicalize(smiles):
    try:
        if pd.isna(smiles) or str(smiles).strip() in ["-", ""]: return None
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if not mol: return None
        return Chem.MolToSmiles(enumerator.Canonicalize(mol), isomericSmiles=False)
    except: return None

def get_logs_molarity(row):
    """Exact MIT/Krasnov conversion formula using original float T,K"""
    try:
        m = Chemical(row['Solvent_Name_Raw'], T=row['T,K'])
        if m.MW is None or m.rho is None or m.rho <= 0:
            return np.nan
        # mol/L = mole_fraction / (MW_g_mol / density_g_L)
        return log10(row['Solubility'] / (m.MW / m.rho))
    except: return np.nan

def build_fair_baseline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Extracting unique chemical environments (Solute-Solvent Pairs) from your split...")
    df_user_train = pd.read_csv(USER_TRAIN)
    df_user_test = pd.read_csv(USER_TEST)
    
    # Create sets of unique tuples (Solute, Solvent)
    user_train_pairs = set(zip(df_user_train['Solute'], df_user_train['Solvent']))
    user_test_pairs = set(zip(df_user_test['Solute'], df_user_test['Solvent']))
    
    print(f"   Found {len(user_train_pairs)} unique pairs in your Train set.")
    print(f"   Found {len(user_test_pairs)} unique pairs in your Test set.")

    print("2. Loading Raw BigSolDB and applying basic Krasnov filters...")
    raw = pd.read_csv(RAW_DB)
    # Remove dots and polymers
    raw = raw[~raw["SMILES"].str.contains(".", regex=False)]
    raw = raw[~raw["Solvent"].isin(("PEG-400", "PEG-300", "PEG-200"))]

    print("3. Canonicalizing Raw SMILES (Strict Tautomer Standard)...")
    # We must canonicalize to match your user-split strings exactly
    raw['solute_canon'] = raw['SMILES'].apply(canonicalize)
    raw['solvent_canon'] = raw['SMILES_Solvent'].apply(canonicalize)
    raw = raw.dropna(subset=['solute_canon', 'solvent_canon'])

    print("4. Mapping Raw Rows to your pairs (Restoring all Temperatures & Sources)...")
    raw = raw.rename(columns={'Solvent': 'Solvent_Name_Raw'})
    
    # Function to check if a raw row belongs to your pairs
    def is_in_train(row): return (row['solute_canon'], row['solvent_canon']) in user_train_pairs
    def is_in_test(row): return (row['solute_canon'], row['solvent_canon']) in user_test_pairs

    # Filter raw DB to pull ALL available data for YOUR pairs
    baseline_train = raw[raw.apply(is_in_train, axis=1)].copy()
    baseline_test = raw[raw.apply(is_in_test, axis=1)].copy()

    print("5. Calculating LogS (Molarity) using raw float T,K and Thermo library...")
    baseline_train['logS'] = baseline_train.apply(get_logs_molarity, axis=1)
    baseline_test['logS'] = baseline_test.apply(get_logs_molarity, axis=1)
    
    baseline_train = baseline_train.dropna(subset=['logS'])
    baseline_test = baseline_test.dropna(subset=['logS'])

    # Format for MIT training script
    final_cols = ['solute_canon', 'solvent_canon', 'T,K', 'logS', 'Source']
    baseline_train = baseline_train[final_cols].rename(columns={
        'solute_canon': 'solute_smiles', 
        'solvent_canon': 'solvent_smiles', 
        'T,K': 'temperature', 
        'Source': 'source'
    })
    baseline_test = baseline_test[final_cols].rename(columns={
        'solute_canon': 'solute_smiles', 
        'solvent_canon': 'solvent_smiles', 
        'T,K': 'temperature', 
        'Source': 'source'
    })

    print(f"Result: MIT Train set has {len(baseline_train)} raw points for your {len(user_train_pairs)} pairs.")
    print(f"Result: MIT Test set has {len(baseline_test)} raw points for your {len(user_test_pairs)} pairs.")

    print("6. Generating FastProp Descriptors (Final Step)...")
    train_fastprop = get_descs(baseline_train)
    test_fastprop = get_descs(baseline_test)

    train_fastprop.to_csv(OUTPUT_DIR / "baseline_train.csv", index=False)
    test_fastprop.to_csv(OUTPUT_DIR / "baseline_test.csv", index=False)
    print(f"\nSUCCESS: Baseline data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    build_fair_baseline()
