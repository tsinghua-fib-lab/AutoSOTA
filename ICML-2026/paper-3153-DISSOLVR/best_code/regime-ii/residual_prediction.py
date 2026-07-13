#!/usr/bin/env python3
"""
Residual Error (Offset) Prediction Experiment

Hypothesis: Model predictions are offset by a systematic amount per solute-solvent pair.
If we can predict this offset, we can correct predictions and improve accuracy.

Workflow:
1. Compute mean offset per pair on train data
2. Train XGBoost to predict offset from molecular features (no temperature)
3. Evaluate on test data
4. Apply correction and measure improvement
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Config
DATA_DIR = "data"
STORE_DIR = "feature_store"
MODEL_DIR = "model"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.joblib")

from apelblat_analysis import generate_features_for_prediction


def load_pair_features(df):
    """
    Load molecular features for solute-solvent pairs (NO temperature).
    Uses raw features from feature_store.
    """
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    
    # Get unique pairs
    pairs = df[['Solute', 'Solvent']].drop_duplicates().reset_index(drop=True)
    
    # Get features for each pair
    X_sol = sol_raw.loc[pairs['Solute']].values
    X_solv = solv_raw.loc[pairs['Solvent']].values
    
    X_pair = np.hstack([X_sol, X_solv])
    
    return pairs, X_pair


def compute_pair_offsets(df, model, selector):
    """
    Compute mean signed offset for each solute-solvent pair.
    offset = prediction - actual (positive = model predicts too high)
    """
    print("  Generating predictions...")
    X = generate_features_for_prediction(df)
    X = selector.transform(X)
    preds = model.predict(X)
    
    # Compute errors
    df = df.copy()
    df['pred'] = preds
    df['error'] = df['pred'] - df['LogS']  # signed error
    
    # Group by pair and compute mean offset
    pair_stats = df.groupby(['Solute', 'Solvent']).agg({
        'error': ['mean', 'std', 'count'],
        'LogS': 'mean',
        'pred': 'mean'
    }).reset_index()
    
    pair_stats.columns = ['Solute', 'Solvent', 'offset_mean', 'offset_std', 
                          'n_samples', 'actual_mean', 'pred_mean']
    
    return pair_stats


def run_experiment():
    print("="*60)
    print("RESIDUAL ERROR PREDICTION EXPERIMENT")
    print("="*60)
    
    # 1. Load model
    print("\n[1] Loading model and data...")
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    
    print(f"  Train samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    
    # 2. Compute offsets on train data
    print("\n[2] Computing offsets on train data...")
    train_offsets = compute_pair_offsets(df_train, model, selector)
    print(f"  Train pairs: {len(train_offsets)}")
    print(f"  Mean offset: {train_offsets['offset_mean'].mean():.4f}")
    print(f"  Offset std:  {train_offsets['offset_mean'].std():.4f}")
    
    # 3. Load pair features for training XGBoost
    print("\n[3] Loading pair features (no temperature)...")
    train_pairs, X_train_pairs = load_pair_features(train_offsets)
    y_train_offset = train_offsets['offset_mean'].values
    
    print(f"  Pair feature dim: {X_train_pairs.shape}")
    
    # 4. Train CatBoost to predict offset
    print("\n[4] Training CatBoost offset predictor...")
    from catboost import CatBoostRegressor
    
    cb_offset = CatBoostRegressor(
        iterations=3000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        thread_count=-1
    )
    cb_offset.fit(X_train_pairs, y_train_offset)
    
    # Train performance
    train_offset_pred = cb_offset.predict(X_train_pairs)
    train_offset_rmse = np.sqrt(mean_squared_error(y_train_offset, train_offset_pred))
    train_mean_baseline = np.sqrt(mean_squared_error(y_train_offset, 
                                   np.full_like(y_train_offset, y_train_offset.mean())))
    
    print(f"  XGBoost offset RMSE (train): {train_offset_rmse:.4f}")
    print(f"  Mean baseline RMSE (train): {train_mean_baseline:.4f}")
    print(f"  Improvement: {(train_mean_baseline - train_offset_rmse) / train_mean_baseline * 100:.1f}%")
    
    # 5. Compute offsets on test data
    print("\n[5] Computing offsets on test data...")
    test_offsets = compute_pair_offsets(df_test, model, selector)
    print(f"  Test pairs: {len(test_offsets)}")
    
    # 6. Predict offsets on test pairs
    print("\n[6] Predicting offsets on test pairs...")
    test_pairs, X_test_pairs = load_pair_features(test_offsets)
    y_test_offset = test_offsets['offset_mean'].values
    
    # Check which test pairs were seen in training
    train_pair_set = set(zip(train_offsets['Solute'], train_offsets['Solvent']))
    test_pair_set = set(zip(test_offsets['Solute'], test_offsets['Solvent']))
    overlap = len(train_pair_set & test_pair_set)
    print(f"  Test pairs seen in train: {overlap} / {len(test_pair_set)}")
    
    test_offset_pred = cb_offset.predict(X_test_pairs)
    test_offset_rmse = np.sqrt(mean_squared_error(y_test_offset, test_offset_pred))
    test_mean_baseline = np.sqrt(mean_squared_error(y_test_offset, 
                                  np.full_like(y_test_offset, y_train_offset.mean())))
    
    print(f"  XGBoost offset RMSE (test): {test_offset_rmse:.4f}")
    print(f"  Mean baseline RMSE (test): {test_mean_baseline:.4f}")
    
    # 7. Apply correction to test predictions
    print("\n[7] Applying correction to test predictions...")
    
    # Get full test predictions
    X_test_full = generate_features_for_prediction(df_test)
    X_test_full = selector.transform(X_test_full)
    preds_original = model.predict(X_test_full)
    
    # Create offset lookup
    offset_lookup = dict(zip(zip(test_offsets['Solute'], test_offsets['Solvent']), 
                             test_offset_pred))
    
    # Apply correction per sample
    corrections = df_test.apply(
        lambda row: offset_lookup.get((row['Solute'], row['Solvent']), 0), axis=1
    ).values
    
    preds_corrected = preds_original - corrections
    
    # Compare
    y_test = df_test['LogS'].values
    rmse_original = np.sqrt(mean_squared_error(y_test, preds_original))
    rmse_corrected = np.sqrt(mean_squared_error(y_test, preds_corrected))
    r2_original = r2_score(y_test, preds_original)
    r2_corrected = r2_score(y_test, preds_corrected)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n{'Metric':<25} {'Original':<15} {'Corrected':<15} {'Δ':<10}")
    print("-"*65)
    print(f"{'RMSE':<25} {rmse_original:<15.4f} {rmse_corrected:<15.4f} {rmse_corrected - rmse_original:+.4f}")
    print(f"{'R²':<25} {r2_original:<15.4f} {r2_corrected:<15.4f} {r2_corrected - r2_original:+.4f}")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if test_offset_rmse < test_mean_baseline:
        improvement = (test_mean_baseline - test_offset_rmse) / test_mean_baseline * 100
        print(f"✅ XGBoost BEATS mean baseline by {improvement:.1f}%")
        print("   → Offset is SYSTEMATIC and PREDICTABLE")
        if rmse_corrected < rmse_original:
            print(f"   → Correction IMPROVED predictions by {rmse_original - rmse_corrected:.4f} RMSE")
        else:
            print(f"   → But correction didn't help; offset may vary within pairs")
    else:
        print("❌ XGBoost does NOT beat mean baseline")
        print("   → Offset is ALEATORIC (unpredictable)")
        print("   → Model already captures what's predictable from molecular structure")
    
    # Save XGBoost model
    joblib.dump(cb_offset, os.path.join(MODEL_DIR, "offset_predictor.joblib"))
    print(f"\nSaved offset predictor to {MODEL_DIR}/offset_predictor.joblib")
    
    print("\nDone!")


if __name__ == "__main__":
    run_experiment()
