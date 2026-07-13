# FASTSOLV Model Recreation Guide
# ==================================
# This guide helps you recreate the FASTSOLV model from Attia et al. 2025
# for your own research baseline

"""
SETUP INSTRUCTIONS:
==================
1. Install dependencies:
   conda create -n fastsolv python=3.11
   conda activate fastsolv
   pip install torch pytorch-lightning fastprop astartes rdkit pandas numpy scikit-learn

2. Download data:
   - BigSolDB from: https://zenodo.org/records/6984601
   - SolProp from: https://zenodo.org/records/5970538
   - Leeds data from: https://github.com/BNNLab/Solubility_data

3. Directory structure:
   project/
   ├── data/
   │   ├── BigSolDB.csv
   │   ├── SolProp_v1.2/
   │   └── leeds_*.csv files
   ├── prepare_data.py
   ├── train_original.py
   ├── train_random_split.py
   └── train_solvent_split.py
"""

# ============================================================================
# FILE 1: prepare_data.py - Data Preparation Script
# ============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from math import log10
from thermo.chemical import Chemical

DROP_WATER = False  # Set to False to include aqueous solubility

def get_descs(src_df: pd.DataFrame, solute_col='solute_smiles', solvent_col='solvent_smiles'):
    """Calculate molecular descriptors for solutes and solvents (Vectorized)"""
    # 1. Get unique SMILES and calculate descriptors once
    unique_smiles = np.unique(np.hstack((
        src_df[solvent_col].unique(), 
        src_df[solute_col].unique()
    )))
    
    print(f"Calculating descriptors for {len(unique_smiles)} unique molecules...")
    descs = get_descriptors(
        False, 
        ALL_2D, 
        [Chem.MolFromSmiles(smi) for smi in unique_smiles]
    ).to_numpy(dtype=np.float32)
    
    # 2. Create a lookup DataFrame
    desc_df = pd.DataFrame(descs, columns=ALL_2D)
    desc_df['smiles_key'] = unique_smiles
    
    # 3. Merge descriptors for Solutes
    print("Mapping solute descriptors...")
    # Rename columns for solute merge
    solute_desc_cols = {d: "solute_" + d for d in ALL_2D}
    solute_lookup = desc_df.rename(columns=solute_desc_cols)
    
    merged_df = src_df.merge(
        solute_lookup, 
        left_on=solute_col, 
        right_on='smiles_key', 
        how='left'
    ).drop(columns=['smiles_key'])
    
    # 4. Merge descriptors for Solvents
    print("Mapping solvent descriptors...")
    # Rename columns for solvent merge
    solvent_desc_cols = {d: "solvent_" + d for d in ALL_2D}
    solvent_lookup = desc_df.rename(columns=solvent_desc_cols)
    
    merged_df = merged_df.merge(
        solvent_lookup, 
        left_on=solvent_col, 
        right_on='smiles_key', 
        how='left'
    ).drop(columns=['smiles_key'])
    
    # Ensure columns are in the expected order if necessary, or just return
    return merged_df


def prepare_bigsoldb(input_path="data/BigSolDB.csv", output_dir="data/processed"):
    """Prepare BigSolDB dataset"""
    print("Processing BigSolDB...")
    df = pd.read_csv(input_path)
    print(f"Original size: {len(df)}")
    
    # Remove PEG solvents
    df = df[~df["Solvent"].isin(("PEG-400", "PEG-300", "PEG-200"))]
    
    if DROP_WATER:
        df = df[~df["Solvent"].isin(("water",))]
    print(f"After filtering: {len(df)}")
    
    # Convert mole fraction to molarity
    def fraction_to_molarity(row):
        name = row["Solvent"]
        # Handle special names
        name_map = {
            "THF": "tetrahydrofuran",
            "n-heptane": "heptane",
            "DMS": "methylthiomethane",
            "2-ethyl-n-hexanol": "2-Ethyl hexanol",
            "3,6-dioxa-1-decanol": "butoxyethoxyethanol",
            "DEF": "diethylformamide"
        }
        name = name_map.get(name, name)
        
        try:
            m = Chemical(name, T=row["T,K"])
            return log10(row["Solubility"] / (m.MW / m.rho))
        except:
            print(f"Could not estimate {name}")
            return pd.NA
    
    df["logS"] = df[["T,K", "Solvent", "Solubility"]].apply(
        fraction_to_molarity, axis=1
    )
    df = df.dropna()
    print(f"After conversion: {len(df)}")
    
    # Rename columns
    df = df.rename(columns={
        "T,K": "temperature",
        "SMILES": "solute_smiles",
        "SMILES_Solvent": "solvent_smiles",
        "Source": "source"
    })
    
    # Remove multi-fragment molecules
    df = df[~df["solute_smiles"].str.contains(".", regex=False)]
    print(f"Final size: {len(df)}")
    
    # Calculate descriptors
    fastprop_data = get_descs(df)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_nonaq" if DROP_WATER else ""
    fastprop_data.to_csv(f"{output_dir}/bigsoldb_fastprop{suffix}.csv", index=False)
    
    print(f"Saved to {output_dir}/bigsoldb_fastprop{suffix}.csv")
    return fastprop_data


def prepare_solprop(input_dir="data/SolProp_v1.2/Data", output_dir="data/processed"):
    """Prepare SolProp test dataset"""
    print("Processing SolProp...")
    
    room_T = pd.read_csv(f"{input_dir}/CombiSolu-Exp.csv")
    high_T = pd.read_csv(f"{input_dir}/CombiSolu-Exp-HighT.csv")
    df = pd.concat([room_T, high_T])
    
    df = df[df["experimental_logS [mol/L]"].notna()].reset_index(drop=True)
    df = df.rename(columns={"experimental_logS [mol/L]": "logS"})
    
    if DROP_WATER:
        df = df[~df["solvent_smiles"].eq("O")].reset_index(drop=True)
    
    fastprop_data = get_descs(df)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_nonaq" if DROP_WATER else ""
    fastprop_data.to_csv(f"{output_dir}/solprop_fastprop{suffix}.csv", index=False)
    
    print(f"Saved to {output_dir}/solprop_fastprop{suffix}.csv")
    return fastprop_data


def prepare_leeds(input_files, output_dir="data/processed"):
    """Prepare Leeds test datasets"""
    print("Processing Leeds datasets...")
    
    solvent_map = {
        "acetone": "CC(=O)C",
        "benzene": "C1=CC=CC=C1",
        "ethanol": "OCC"
    }
    
    for file_path, solvent_name in input_files:
        print(f"Processing {solvent_name}...")
        df = pd.read_csv(file_path).dropna()
        df = df[df["T"].notna()].reset_index(drop=True)
        
        df["T"] = df["T"] + 273.15  # Convert to Kelvin
        df = df.rename(columns={
            "LogS": "logS",
            "T": "temperature",
            "SMILES": "solute_smiles"
        })
        df.insert(0, "solvent_smiles", solvent_map[solvent_name])
        
        fastprop_data = get_descs(df)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fastprop_data.to_csv(
            f"{output_dir}/leeds_{solvent_name}_fastprop.csv", 
            index=False
        )
        print(f"Saved to {output_dir}/leeds_{solvent_name}_fastprop.csv")


if __name__ == "__main__":
    # Prepare all datasets
    prepare_bigsoldb()
    prepare_solprop()
    
    leeds_files = [
        ("data/acetone_solubility_data.csv", "acetone"),
        ("data/benzene_solubility_data.csv", "benzene"),
        ("data/ethanol_solubility_data.csv", "ethanol")
    ]
    prepare_leeds(leeds_files)
    
    print("\nData preparation complete!")
