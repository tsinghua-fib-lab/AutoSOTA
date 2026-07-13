# train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import joblib
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor

from train_transformer import InteractionTransformer, DEVICE

# config
DATA_DIR, STORE_DIR, MODEL_DIR = "data", "feature_store", "model"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
OOF_EMBED_FILE = "train_embeddings.csv"
TRANSFORMER_PATH = "transformer.pth"
SEED = 42

warnings.filterwarnings("ignore")

def load_hyper_features(df, embed_df):
    # signed-Modulus Interaction terms for the 24 Council Members
    # order: [Raw Physics, Transformer Embeddings, 24 Interactions, Tm, T_red, T, T_inv]
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    
    # 1. Base Descriptors & Transformer Embeddings
    X_raw = np.hstack([sol_raw.loc[df['Solute']].values, solv_raw.loc[df['Solvent']].values])
    X_embed = embed_df[[c for c in embed_df.columns if c.startswith("Learned_")]].values
    
    # 2. Thermodynamic Hallmark Data
    Tm = sol_raw.loc[df['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_raw = df['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df['Temperature'].values).reshape(-1, 1).astype(np.float32)
    T_red = (T_raw / Tm).astype(np.float32)
    
    # 3. signed modulus interactions
    # Reshape (Samples, 24 Members, 32 Dims) -> Calculate Magnitude and Direction
    X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
    X_modulus = np.linalg.norm(X_reshaped, axis=2) 
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_modulus) * T_inv
    
    # Note: X_embed (768 latent) removed - keeping only X_interact (24 interaction terms)(if you ever want to add it bac, [X_raw, X_embed, X_interact, Tm, T_red, T_raw, T_inv])
    return np.hstack([X_raw, X_interact, Tm, T_red, T_raw, T_inv])

def generate_test_hyper_features(df_test):
    # Load feature stores
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    sol_c = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index('SMILES_KEY')
    solv_c = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index('SMILES_KEY')
    
    # Load Transformer
    model = InteractionTransformer().to(DEVICE)
    model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
    model.eval()
    
    # Batch Processing
    X_sol_all = sol_c.loc[df_test['Solute']].values.astype(np.float32)
    X_solv_all = solv_c.loc[df_test['Solvent']].values.astype(np.float32)
    
    batch_size = 512
    embed_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_sol_all), batch_size):
            b_sol = torch.tensor(X_sol_all[i : i+batch_size]).to(DEVICE)
            b_solv = torch.tensor(X_solv_all[i : i+batch_size]).to(DEVICE)
            _, feats, _ = model(b_sol, b_solv)
            embed_list.append(feats.cpu().numpy())
            
    X_embed = np.vstack(embed_list)
    
    # Thermodynamic Engineering
    T = df_test['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df_test['Temperature'].values).reshape(-1, 1).astype(np.float32)
    Tm = sol_raw.loc[df_test['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)
    
    X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
    X_interact = (np.sign(X_reshaped.mean(axis=2)) * np.linalg.norm(X_reshaped, axis=2)) * T_inv
    
    X_raw = np.hstack([sol_raw.loc[df_test['Solute']].values, solv_raw.loc[df_test['Solvent']].values])
    
    # Note: X_embed (768 latent) removed - keeping only X_interact (24 interaction terms)
    return np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])

def run_model_training():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Training model")
    
    # 1. Load and Engineer Features
    df_train = pd.read_csv(TRAIN_FILE)
    df_oof = pd.read_csv(OOF_EMBED_FILE)
    X_full = load_hyper_features(df_train, df_oof)
    
    # 2. Variance Pruning
    print(f"Initial features: {X_full.shape[1]}")
    selector = VarianceThreshold(threshold=0.0001)
    X_pruned = selector.fit_transform(X_full)
    joblib.dump(selector, os.path.join(MODEL_DIR, "selector.joblib"))
    print(f"Pruned features:  {X_pruned.shape[1]}")
    
    # 3. Physics Lock (Re-enabled)
    mono = [0] * X_pruned.shape[1]
    mono[-3], mono[-2], mono[-1] = 1, 1, -1  # T_red, T, 1/T
    print("Physical consistency: Monotonicity constraint ENABLED")

    # 4. Training
    X_tr, X_val, y_tr, y_val = train_test_split(X_pruned, df_train['LogS'].values, test_size=0.05, random_state=SEED)
    
    print("Training model...")
    model = CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=5,
        monotone_constraints=mono, 
        early_stopping_rounds=100, 
        random_seed=SEED, verbose=200, thread_count=-1
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

    # 5. Evaluation
    print("\nEvaluating on Rare Solvents (OOD)...")
    df_test = pd.read_csv(TEST_FILE)
    X_test = selector.transform(generate_test_hyper_features(df_test))
    preds = model.predict(X_test)
    
    print()
    print("Test Performance on LEEDS:")
    print()
    print(f"RMSE: {np.sqrt(mean_squared_error(df_test['LogS'], preds)):.4f}")
    print(f"R2:   {r2_score(df_test['LogS'], preds):.4f}")
    print()

if __name__ == "__main__":
    run_model_training()
