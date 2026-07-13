# eval_test.py
import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from featurizer import MoleculeFeaturizer
import warnings

# config
TEST_PATH = "./data/test.csv"
MODEL_PATH = "./model/model.joblib"

# silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def run_evaluation():
    # 1. Load Data
    print(f"Loading test data from {TEST_PATH}...")
    df_test = pd.read_csv(TEST_PATH)
    
    # Extract raw inputs and targets
    smiles_data = df_test['SMILES'].values
    y_true = df_test['LogS'].values

    # 2. Load Model & Tools
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    featurizer = MoleculeFeaturizer()

    print(f"Running inference on {len(df_test)} molecules...")
    

    # A. Featurization (SMILES -> Descriptors)
    X_test = featurizer.transform(smiles_data)

    # B. Prediction (Descriptors -> LogS)
    # start benchmark timer here
    start_time = time.time()
    preds = model.predict(X_test)

    # stop benchmark timer
    end_time = time.time()
    total_time = end_time - start_time
    throughput = len(df_test) / total_time

    # 3. Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    # 4. Print Report
    print()
    print("Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print()
    print("Inference Speed:")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Throughput: {throughput:.0f} mol/s")

if __name__ == "__main__":
    run_evaluation()
