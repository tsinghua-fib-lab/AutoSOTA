import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import joblib
import torch
import time  # <--- Imported time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from train_transformer import InteractionTransformer, DEVICE

# config
DATA_DIR, STORE_DIR, MODEL_DIR = "data", "feature_store", "model"
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
TRANSFORMER_PATH = "transformer.pth"

def run_evaluation():
    print("Evaluation testing")
    
    # 1. Load Data
    df_test = pd.read_csv(TEST_FILE)
    y_true = df_test['LogS'].values
    
    # 2. Load Models & Stencils
    print("Loading model artifacts...")
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    selector = joblib.load(os.path.join(MODEL_DIR, "selector.joblib"))
    transformer = InteractionTransformer().to(DEVICE)
    transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
    transformer.eval()

    # 3. Load Feature Stores
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    sol_c = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index('SMILES_KEY')
    solv_c = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index('SMILES_KEY')

    # start benchmark timer
    start_time = time.time()

    # 4. Generate Transformer Embeddings
    print(f"Generating learned interactions for {len(df_test)} molecules...")
    X_sol_vals = sol_c.loc[df_test['Solute']].values.astype(np.float32)
    X_solv_vals = solv_c.loc[df_test['Solvent']].values.astype(np.float32)
    
    embed_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_sol_vals), batch_size):
            b_sol = torch.tensor(X_sol_vals[i : i+batch_size]).to(DEVICE)
            b_solv = torch.tensor(X_solv_vals[i : i+batch_size]).to(DEVICE)
            _, feats, _ = transformer(b_sol, b_solv)
            embed_list.append(feats.cpu().numpy())
    X_embed = np.vstack(embed_list)

    # 5. Engineering Final Feature Matrix
    print("Applying Thermodynamic Engineering...")
    T = df_test['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df_test['Temperature'].values).reshape(-1, 1).astype(np.float32)
    Tm = sol_raw.loc[df_test['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)
    
    # Signed-Modulus Interaction Engine (24 channels)
    X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
    X_mod = np.linalg.norm(X_reshaped, axis=2)
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_mod) * T_inv
    
    # Base Raw Descriptors
    X_raw = np.hstack([sol_raw.loc[df_test['Solute']].values, solv_raw.loc[df_test['Solvent']].values])

    # Final Stack: [Raw, Interact, Tm, T_red, T, T_inv] - X_embed removed
    X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])

    # 6. Predict
    print("Performing final predictions...")
    X_pruned = selector.transform(X_full)
    preds = model.predict(X_pruned)

    # stop benchmark timer
    end_time = time.time()
    total_time = end_time - start_time
    throughput = len(df_test) / total_time

    # 7. Final Metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print()
    print("Test Results")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("Inference timing results:")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Throughput: {throughput:.0f} pairs/sec")

if __name__ == "__main__":
    run_evaluation()
