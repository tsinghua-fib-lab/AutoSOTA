import os
import warnings
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
from featurizer import MoleculeFeaturizer

# config
SEEDS = [42, 101, 123, 456, 789]  # 5 seeds for variance estimation
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
MODEL_DIR = "./model" 
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.joblib")

# silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def main():
    # 1. Load data
    print(f"Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2. Featurization
    print("Featurizing molecules...")
    featurizer = MoleculeFeaturizer()
    
    # Transform raw SMILES into features
    X_train = featurizer.transform(train_df['SMILES'])
    y_train = train_df['LogS']
    
    X_test = featurizer.transform(test_df['SMILES'])
    y_test = test_df['LogS']
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # 3. Multi-seed Training
    print(f"\n{'='*60}")
    print(f"TRAINING WITH {len(SEEDS)} SEEDS")
    print(f"{'='*60}")
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    best_model = None
    best_rmse = float('inf')
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=5,
            verbose=1000,
            random_state=seed,
            allow_writing_files=False,
            thread_count=-1
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # 4. Report Results
    print(f"\n{'='*60}")
    print("FINAL RESULTS (5 seeds)")
    print(f"{'='*60}")
    print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    # 5. Save best model
    print(f"\nSaving best model (RMSE={best_rmse:.4f}) to {MODEL_SAVE_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    
    print("Done.")

if __name__ == "__main__":
    main()
