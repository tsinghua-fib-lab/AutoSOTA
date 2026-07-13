#!/usr/bin/env python3
"""
Hyperparameter Optimization for CatBoost using Optuna
30 trials, no cross-validation (uses train/val split)

Usage:
    python hyperparam_sweep.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Config
DATA_DIR = "data"
STORE_DIR = "feature_store"
MODEL_DIR = "model"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

N_TRIALS = 10
VAL_SIZE = 0.15  # 15% for validation
RANDOM_STATE = 42

# Import feature generation from train.py
from train import load_hyper_features, generate_test_hyper_features, OOF_EMBED_FILE


def objective(trial, X, y, monotone_constraints):
    """Optuna objective function with 5-fold CV."""
    from sklearn.model_selection import KFold
    
    # Define hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 2500, 6000, step=500),
        'depth': trial.suggest_int('depth', 7, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    
    # Fixed params
    params['random_seed'] = RANDOM_STATE
    params['verbose'] = 0  # Quiet during CV
    params['thread_count'] = -1
    params['monotone_constraints'] = monotone_constraints
    params['loss_function'] = 'RMSE'
    
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    print(f"\n>>> Trial {trial.number}: depth={params['depth']}, iter={params['iterations']}, lr={params['learning_rate']:.4f}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        cv_scores.append(rmse)
        print(f"  Fold {fold+1}/5: RMSE={rmse:.4f}")
    
    mean_rmse = np.mean(cv_scores)
    print(f"  Mean CV RMSE: {mean_rmse:.4f}")
    
    return mean_rmse


def run_sweep():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION - CATBOOST")
    print(f"Trials: {N_TRIALS}, Validation Split: {VAL_SIZE*100:.0f}%")
    print("="*60)
    
    # 1. Load data
    print("\n[1] Loading train CSV...")
    df_train = pd.read_csv(TRAIN_FILE)
    print(f"  Loaded {len(df_train)} rows")
    
    print("  Loading OOF embeddings...")
    df_oof = pd.read_csv(OOF_EMBED_FILE)
    print(f"  Loaded OOF: {len(df_oof)} rows")
    
    print("  Generating hyper features (this may take a while)...")
    X_full = load_hyper_features(df_train, df_oof)
    y_full = df_train['LogS'].values
    print(f"  DONE! Feature dim: {X_full.shape}")
    
    print(f"  Train samples: {len(df_train)}")
    print(f"  Feature dim: {X_full.shape[1]}")
    
    print("\n[2] Applying feature selection...")
    print("  Loading selector...")
    selector = joblib.load(os.path.join(MODEL_DIR, "selector.joblib"))
    print("  Transforming features...")
    X_selected = selector.transform(X_full)
    print(f"  Selected features: {X_selected.shape[1]}")
    
    # 3. Setup monotone constraints for temperature features (last 4 columns)
    n_features = X_selected.shape[1]
    monotone_constraints = [0] * (n_features - 4) + [1, 1, 1, -1]  # T, T_m, T_red up; 1/T down
    
    # 4. Run Optuna with 5-fold CV
    print("\n[3] Running Optuna optimization with 5-fold CV...")
    print("  (Each trial runs 5 folds, ~10 min per trial)")
    print("="*60)
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: objective(trial, X_selected, y_full, monotone_constraints),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # 6. Results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"\nBest Validation RMSE: {best_rmse:.4f}")
    print("\nBest Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # 7. Train final model with best params on full train data
    print("\n[5] Training final model with best params...")
    best_params['random_seed'] = RANDOM_STATE
    best_params['verbose'] = 500
    best_params['thread_count'] = -1
    best_params['monotone_constraints'] = monotone_constraints
    best_params['loss_function'] = 'RMSE'
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_selected, y_full)
    
    # 8. Evaluate on test set
    print("\n[6] Evaluating on test set...")
    df_test = pd.read_csv(TEST_FILE)
    X_test = generate_test_hyper_features(df_test)
    X_test = selector.transform(X_test)
    
    preds_test = final_model.predict(X_test)
    y_test = df_test['LogS'].values
    
    test_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    test_r2 = r2_score(y_test, preds_test)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²:   {test_r2:.4f}")
    
    # 9. Save best model and params
    joblib.dump(final_model, os.path.join(MODEL_DIR, "model_optimized.joblib"))
    joblib.dump(best_params, os.path.join(MODEL_DIR, "best_params.joblib"))
    
    print(f"\nSaved: {MODEL_DIR}/model_optimized.joblib")
    print(f"Saved: {MODEL_DIR}/best_params.joblib")
    
    # 10. Save optimization history
    history_df = study.trials_dataframe()
    history_df.to_csv(os.path.join(MODEL_DIR, "optuna_history.csv"), index=False)
    print(f"Saved: {MODEL_DIR}/optuna_history.csv")
    
    print("\nDone!")


if __name__ == "__main__":
    run_sweep()
