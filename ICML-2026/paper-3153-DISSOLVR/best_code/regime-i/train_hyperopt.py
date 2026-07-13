# train_hyperopt.py
"""
Hyperparameter optimization script for training CatBoost models on different datasets.
Uses Optuna for hyperparameter search with cross-validation.

Usage:
    python train_hyperopt.py --dataset aqsoldb --n_trials 50
    python train_hyperopt.py --dataset all --n_trials 30
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
from featurizer import MoleculeFeaturizer

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Configuration
DATASETS = {
    "aqsoldb": {
        "train": "./all_datasets/aqsoldb/train.csv",
        "test": "./all_datasets/aqsoldb/test.csv",
    },
    "esol": {
        "train": "./all_datasets/esol/train.csv",
        "test": "./all_datasets/esol/test.csv",
    },
    "sc2": {
        "train": "./all_datasets/sc2/train.csv",
        "test": "./all_datasets/sc2/test.csv",
    },
}

MODEL_DIR = "./model"
SEED = 101


def load_and_featurize(train_path: str, test_path: str):
    """Load datasets and compute features."""
    print(f"Loading train data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    featurizer = MoleculeFeaturizer()
    
    print(f"Featurizing {len(train_df)} training molecules...")
    X_train = featurizer.transform(train_df['SMILES'])
    y_train = train_df['LogS'].values
    
    print(f"Featurizing {len(test_df)} test molecules...")
    X_test = featurizer.transform(test_df['SMILES'])
    y_test = test_df['LogS'].values
    
    return X_train, y_train, X_test, y_test


def create_objective(X_train, y_train, n_folds=5):
    """Create an Optuna objective function for hyperparameter optimization."""
    
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 5000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": SEED,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        
        model = CatBoostRegressor(**params)
        
        # Use cross-validation to evaluate
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)
    
    return objective


def train_dataset(dataset_name: str, n_trials: int, n_folds: int = 5):
    """Train a model on a specific dataset with hyperparameter optimization."""
    print(f"\n{'='*60}")
    print(f"Training on dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    paths = DATASETS[dataset_name]
    
    # Load and featurize data
    X_train, y_train, X_test, y_test = load_and_featurize(paths["train"], paths["test"])
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create Optuna study
    print(f"\nStarting hyperparameter optimization ({n_trials} trials, {n_folds}-fold CV)...")
    study = optuna.create_study(direction="minimize", study_name=f"{dataset_name}_study")
    
    objective = create_objective(X_train, y_train, n_folds)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_params["random_state"] = SEED
    best_params["verbose"] = 200
    best_params["allow_writing_files"] = False
    best_params["thread_count"] = -1
    
    print(f"\nBest parameters found:")
    for k, v in best_params.items():
        if k not in ["random_state", "verbose", "allow_writing_files", "thread_count"]:
            print(f"  {k}: {v}")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    
    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    preds = final_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n{'='*40}")
    print(f"TEST SET RESULTS ({dataset_name})")
    print(f"{'='*40}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Save outputs
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save best parameters
    params_to_save = {k: v for k, v in best_params.items() 
                      if k not in ["verbose", "allow_writing_files", "thread_count"]}
    params_to_save["best_cv_rmse"] = study.best_value
    params_to_save["test_rmse"] = rmse
    params_to_save["test_mae"] = mae
    params_to_save["test_r2"] = r2
    
    params_path = os.path.join(MODEL_DIR, f"{dataset_name}_params.json")
    with open(params_path, "w") as f:
        json.dump(params_to_save, f, indent=2)
    print(f"Parameters saved to: {params_path}")
    
    # Save optimization history
    history_df = study.trials_dataframe()
    history_path = os.path.join(MODEL_DIR, f"{dataset_name}_optuna_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Optimization history saved to: {history_path}")
    
    return {
        "dataset": dataset_name,
        "best_params": best_params,
        "best_cv_rmse": study.best_value,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CatBoost on solubility datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset to train on: aqsoldb, esol, sc2, or 'all' for all datasets")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of Optuna trials (default: 50)")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of cross-validation folds (default: 5)")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        datasets_to_train = list(DATASETS.keys())
    else:
        datasets_to_train = [args.dataset]
    
    results = []
    for dataset_name in datasets_to_train:
        result = train_dataset(dataset_name, args.n_trials, args.n_folds)
        results.append(result)
    
    # Print summary if multiple datasets
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL DATASETS")
        print(f"{'='*60}")
        print(f"{'Dataset':<12} {'CV RMSE':<10} {'Test RMSE':<10} {'Test MAE':<10} {'Test R²':<10}")
        print("-" * 52)
        for r in results:
            print(f"{r['dataset']:<12} {r['best_cv_rmse']:<10.4f} {r['test_rmse']:<10.4f} {r['test_mae']:<10.4f} {r['test_r2']:<10.4f}")
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(MODEL_DIR, "hyperopt_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
