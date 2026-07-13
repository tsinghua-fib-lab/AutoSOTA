import pandas as pd
import numpy as np
from pathlib import Path
from math import log10
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from thermo.chemical import Chemical  # Required for the fallback calculation

# --- CONFIG ---
RAW_DB = "data/BigSolDB 2.0 Dataset.csv" 
USER_TRAIN = "data/train.csv"
USER_TEST = "data/test.csv"
OUTPUT_DIR = Path("data/baseline_bigsol2")

# --- UTILS ---
enumerator = rdMolStandardize.TautomerEnumerator()

def canonicalize(smiles):
    """Standardize SMILES to match your split files"""
    try:
        if pd.isna(smiles) or str(smiles).strip() in ["-", ""]: return None
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if not mol: return None
        return Chem.MolToSmiles(enumerator.Canonicalize(mol), isomericSmiles=False)
    except: return None

def get_hybrid_logs(row):
    """
    HYBRID LOGIC:
    1. If 'LogS(mol/L)' exists, use it.
    2. If not, calculate from 'Solubility(mole_fraction)' using Thermo.
    """
    # 1. Try using the pre-provided value
    val = row.get('LogS_provided')
    if pd.notna(val):
        return float(val)

    # 2. Fallback: Calculate from mole fraction
    # We need: Solvent Name, Temperature, and Mole Fraction
    try:
        mole_frac = row.get('mole_fraction')
        if pd.isna(mole_frac):
            return np.nan
            
        solvent_name = row.get('Solvent_Name')
        temp = row.get('temperature')
        
        # Use Thermo library to find density/MW
        # Note: We use the exact same formula logic as your previous baseline
        m = Chemical(solvent_name, T=temp)
        
        if m.MW is None or m.rho is None or m.rho <= 0:
            return np.nan
            
        # Conversion: log10( mole_fraction / (MW / rho) )
        # MW is g/mol, rho is usually kg/m3 or g/ml in Thermo depending on version.
        # Assuming consistency with your previous script's success:
        return log10(mole_frac / (m.MW / m.rho))
        
    except Exception as e:
        # If thermo fails (unknown solvent name), we lose this point
        return np.nan

def get_descs(src_df, solute_col='solute_smiles', solvent_col='solvent_smiles'):
    """Vectorized descriptor calculation"""
    unique_smiles = np.unique(np.hstack((
        src_df[solvent_col].dropna().unique(), 
        src_df[solute_col].dropna().unique()
    )))
    
    print(f"   Calculating descriptors for {len(unique_smiles)} unique molecules...")
    mols = [Chem.MolFromSmiles(smi) for smi in unique_smiles]
    
    # FIX: Use positional arguments exactly as they appeared in your working prepare_data.py
    # (False corresponds to enable_cache)
    descs = get_descriptors(False, ALL_2D, mols).to_numpy(dtype=np.float32)
    
    desc_df = pd.DataFrame(descs, columns=ALL_2D)
    desc_df['smiles_key'] = unique_smiles
    
    print("   Mapping descriptors...")
    # Merge Solute
    merged_df = src_df.merge(
        desc_df.rename(columns={d: "solute_" + d for d in ALL_2D}),
        left_on=solute_col, right_on='smiles_key', how='left'
    ).drop(columns=['smiles_key'])
    
    # Merge Solvent
    merged_df = merged_df.merge(
        desc_df.rename(columns={d: "solvent_" + d for d in ALL_2D}),
        left_on=solvent_col, right_on='smiles_key', how='left'
    ).drop(columns=['smiles_key'])
    
    return merged_df

def build_hybrid_baseline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading User Split Definitions...")
    df_user_train = pd.read_csv(USER_TRAIN)
    df_user_test = pd.read_csv(USER_TEST)
    
    user_train_pairs = set(zip(df_user_train['Solute'], df_user_train['Solvent']))
    user_test_pairs = set(zip(df_user_test['Solute'], df_user_test['Solvent']))

    print("2. Loading BigSolDB 2.0...")
    raw = pd.read_csv(RAW_DB)
    
    # Rename for clarity
    raw = raw.rename(columns={
        'SMILES_Solute': 'solute_smiles',
        'SMILES_Solvent': 'solvent_smiles',
        'Temperature_K': 'temperature',
        'LogS(mol/L)': 'LogS_provided',          # The incomplete column
        'Solubility(mole_fraction)': 'mole_fraction', # The backup source
        'Solvent': 'Solvent_Name',
        'Source': 'source'
    })

    print("3. Calculating Hybrid LogS (Prioritize Provided -> Fallback to Thermo)...")
    # Apply the hybrid function
    raw['logS'] = raw.apply(get_hybrid_logs, axis=1)
    
    # Drop rows where we couldn't get a LogS either way
    before_len = len(raw)
    raw = raw.dropna(subset=['logS', 'solute_smiles', 'solvent_smiles', 'temperature'])
    print(f"   Rows retained: {len(raw)} / {before_len}")

    print("4. Canonicalizing SMILES...")
    # Remove dots
    raw = raw[~raw["solute_smiles"].str.contains(".", regex=False)]
    
    raw['solute_canon'] = raw['solute_smiles'].apply(canonicalize)
    raw['solvent_canon'] = raw['solvent_smiles'].apply(canonicalize)
    raw = raw.dropna(subset=['solute_canon', 'solvent_canon'])

    print("5. Filtering to match your splits...")
    def is_in_train(row): return (row['solute_canon'], row['solvent_canon']) in user_train_pairs
    def is_in_test(row): return (row['solute_canon'], row['solvent_canon']) in user_test_pairs

    baseline_train = raw[raw.apply(is_in_train, axis=1)].copy()
    baseline_test = raw[raw.apply(is_in_test, axis=1)].copy()
    
    # Clean up columns
    keep_cols = ['solute_canon', 'solvent_canon', 'temperature', 'logS', 'source']
    baseline_train = baseline_train[keep_cols].rename(columns={'solute_canon': 'solute_smiles', 'solvent_canon': 'solvent_smiles'})
    baseline_test = baseline_test[keep_cols].rename(columns={'solute_canon': 'solute_smiles', 'solvent_canon': 'solvent_smiles'})

    print(f"   Training Set Size: {len(baseline_train)}")
    print(f"   Testing Set Size: {len(baseline_test)}")

    print("6. Generating Descriptors...")
    train_ready = get_descs(baseline_train)
    test_ready = get_descs(baseline_test)

    train_ready.to_csv(OUTPUT_DIR / "baseline_train.csv", index=False)
    test_ready.to_csv(OUTPUT_DIR / "baseline_test.csv", index=False)
    
    print(f"\nSUCCESS: Hybrid baseline saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    build_hybrid_baseline()
