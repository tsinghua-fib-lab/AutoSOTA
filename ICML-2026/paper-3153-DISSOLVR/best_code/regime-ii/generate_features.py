# generate_features.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from featurizer import MoleculeFeaturizer
from council import CouncilExtractor
import warnings
from rdkit import RDLogger

# config
DATA_DIR = "data"
STORE_DIR = "feature_store"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# filenames
RAW_SOLUTE_FILE = os.path.join(STORE_DIR, "solute_raw.parquet")
RAW_SOLVENT_FILE = os.path.join(STORE_DIR, "solvent_raw.parquet")
SOLUTE_COUNCIL_FILE = os.path.join(STORE_DIR, "solute_council.parquet")
SOLVENT_COUNCIL_FILE = os.path.join(STORE_DIR, "solvent_council.parquet")
EXTRACTOR_FILE = os.path.join(STORE_DIR, "council_extractor.joblib")

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def build_full_feature_pipeline():
    print("Beginning feature generation based on train/test split...")
    ensure_dir(STORE_DIR)
    warnings.filterwarnings("ignore")
    RDLogger.DisableLog("rdApp.*")
    
    # 1. Load the existing train/test data
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    unique_solutes = df_full['Solute'].unique()
    unique_solvents = df_full['Solvent'].unique()
    print(f"Total unique molecules to process: {len(unique_solutes)} solutes, {len(unique_solvents)} solvents.")

    # 2. Generate RAW Features for ALL unique molecules (if not already present)
    featurizer = MoleculeFeaturizer()
    
    if not os.path.exists(RAW_SOLUTE_FILE):
        print("Generating ALL Solute Raw Features...")
        df_sol_raw = featurizer.transform(unique_solutes)
        df_sol_raw['SMILES_KEY'] = unique_solutes
        df_sol_raw.to_parquet(RAW_SOLUTE_FILE, index=False)
    else:
        print("Solute Raw Features already exist.")
        df_sol_raw = pd.read_parquet(RAW_SOLUTE_FILE)

    if not os.path.exists(RAW_SOLVENT_FILE):
        print("Generating ALL Solvent Raw Features...")
        df_solv_raw = featurizer.transform(unique_solvents)
        df_solv_raw['SMILES_KEY'] = unique_solvents
        df_solv_raw.to_parquet(RAW_SOLVENT_FILE, index=False)
    else:
        print("Solvent Raw Features already exist.")
        df_solv_raw = pd.read_parquet(RAW_SOLVENT_FILE)

    # 3. Determine the Training Distribution to FIT the Scaler
    print()
    print("Determining Training Distribution to FIT the Council Extractor...")
    
    # fitting the scaler using the solvents present in train only
    solvent_counts = df_train['Solvent'].value_counts()
    TOP_N_SOLVENTS_FOR_FIT = 128
    train_solvents_list = solvent_counts.head(TOP_N_SOLVENTS_FOR_FIT).index.tolist()
    
    train_df_rows = df_train[df_train['Solvent'].isin(train_solvents_list)]
    train_solutes = train_df_rows['Solute'].unique()
    train_solvents = train_df_rows['Solvent'].unique()
    
    # Filter raw features to get only molecules present in train for scaling
    train_sol_feats = df_sol_raw[df_sol_raw['SMILES_KEY'].isin(train_solutes)]
    train_solv_feats = df_solv_raw[df_solv_raw['SMILES_KEY'].isin(train_solvents)]
    
    training_data_for_scaling = pd.concat([
        train_sol_feats.drop(columns=['SMILES_KEY']), 
        train_solv_feats.drop(columns=['SMILES_KEY'])
    ], axis=0)
    
    # 4. Fit the Extractor on the training distribution
    extractor = CouncilExtractor()
    extractor.fit(training_data_for_scaling)
    joblib.dump(extractor, EXTRACTOR_FILE)
    print(f"Council Extractor fitted on distribution defined by top {len(train_solvents)} solvents.")
    
    # 5. transform all molecules using fitted features
    print("Transforming ALL molecules into Council Features...")

    # Transform all solutes (Train + Test molecules)
    df_sol_council = extractor.transform(df_sol_raw.drop(columns=['SMILES_KEY']))
    df_sol_council['SMILES_KEY'] = df_sol_raw['SMILES_KEY']
    
    # Transform all solvents (Train + Test molecules)
    df_solv_council = extractor.transform(df_solv_raw.drop(columns=['SMILES_KEY']))
    df_solv_council['SMILES_KEY'] = df_solv_raw['SMILES_KEY']
    
    # Save the final Council Stores
    df_sol_council.to_parquet(SOLUTE_COUNCIL_FILE, index=False)
    df_solv_council.to_parquet(SOLVENT_COUNCIL_FILE, index=False)
    
    print("Generated all required feature stores for the entire dataset, scaled using only the training distribution.")

if __name__ == "__main__":
    build_full_feature_pipeline()
