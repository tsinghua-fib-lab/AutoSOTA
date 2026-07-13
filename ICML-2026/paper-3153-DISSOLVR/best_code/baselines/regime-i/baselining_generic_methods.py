#!/usr/bin/env python3
"""
Baselining Experiment for Regime-I

Trains multiple baseline models on aqueous solubility datasets:
- DecisionTree
- RandomForest
- LightGBM
- XGBoost
- ANN (MLPRegressor)

Each model is trained with 5 different seeds for variance estimation.
Results are aggregated and saved to benchmark_results/.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from tqdm.auto import tqdm
import joblib

from featurizer import MoleculeFeaturizer

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ["aqsoldb", "esol", "sc2"]
ALL_DATASETS_DIR = "all_datasets"
OUTPUT_DIR = "benchmark_results"
SEEDS = [42, 101, 123, 456, 789]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(dataset_name):
    """Load train and test data for a specific dataset."""
    train_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "train.csv")
    test_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "test.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    return df_train, df_test


def generate_features(df_train, df_test):
    """Generate molecular features for train and test sets."""
    print("  Generating molecular features...")
    featurizer = MoleculeFeaturizer()
    
    # Transform SMILES to features
    X_train = featurizer.transform(df_train['SMILES'].tolist())
    X_test = featurizer.transform(df_test['SMILES'].tolist())
    
    # Get targets
    y_train = df_train['LogS'].values
    y_test = df_test['LogS'].values
    
    return X_train.values, X_test.values, y_train, y_test


# ============================================================================
# BASELINE MODELS
# ============================================================================

def get_baseline_models(seed):
    """Return dictionary of baseline models.
    
    Args:
        seed: Random seed for reproducibility
    """
    return {
        'DecisionTree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=seed
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=10000,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=10000,
            learning_rate=0.02,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            random_state=seed,
            verbose=-1,
            early_stopping_rounds=100
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=10000,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            early_stopping_rounds=100,
            random_state=seed,
            verbosity=0
        ),
        'ANN': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed
        )
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_baselines(dataset_name, X_train, X_test, y_train, y_test):
    """Train all baseline models on all seeds."""
    print(f"\n{'='*60}")
    print(f"TRAINING BASELINES: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Feature shape: {X_train.shape}")
    print(f"  Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    # Initialize results storage
    results = {model_name: {'rmse': [], 'r2': []} 
               for model_name in get_baseline_models(42).keys()}
    
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        
        # Create 5% validation split for early stopping
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.05, random_state=seed
        )
        
        models = get_baseline_models(seed)
        
        for model_name, model in models.items():
            print(f"    Training {model_name}...", end=' ')
            
            # Train with validation set for models that support it
            if model_name == 'LightGBM':
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)]
                )
            elif model_name == 'XGBoost':
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )
            else:
                # DecisionTree, RandomForest, ANN train on full training set
                model.fit(X_train, y_train)
            
            # Predict on test set
            preds = model.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            results[model_name]['rmse'].append(rmse)
            results[model_name]['r2'].append(r2)
            
            print(f"RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # Save model
            output_dir = os.path.join(OUTPUT_DIR, dataset_name, "baselines", model_name)
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f"model_seed{seed}.joblib")
            joblib.dump(model, model_path)
    
    return results


# ============================================================================
# RESULTS REPORTING
# ============================================================================

def save_results(all_results):
    """Save aggregated results to JSON and CSV."""
    # Create summary DataFrame
    summary_data = []
    
    for dataset_name, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            rmse_mean = np.mean(metrics['rmse'])
            rmse_std = np.std(metrics['rmse'])
            r2_mean = np.mean(metrics['r2'])
            r2_std = np.std(metrics['r2'])
            
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'RMSE_mean': rmse_mean,
                'RMSE_std': rmse_std,
                'R2_mean': r2_mean,
                'R2_std': r2_std,
                'Seeds': len(metrics['rmse'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full results to: {json_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (Mean ± Std)")
    print("="*80)
    
    for dataset_name in DATASETS:
        print(f"\n{dataset_name}:")
        print("-"*80)
        dataset_df = summary_df[summary_df['Dataset'] == dataset_name].sort_values('RMSE_mean')
        for _, row in dataset_df.iterrows():
            print(f"  {row['Model']:<15} RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}  "
                  f"R²: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("REGIME-I BASELINE BENCHMARK")
    print("="*80)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Baseline models: 5 (DT, RF, LightGBM, XGBoost, ANN)")
    print(f"Seeds per model: {len(SEEDS)}")
    print(f"Total models to train: {len(DATASETS) * 5 * len(SEEDS)} "
          f"({len(DATASETS)} datasets × 5 models × {len(SEEDS)} seeds)")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = {}
    
    for dataset_name in DATASETS:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*80}")
        
        # Load data
        print(f"\nLoading {dataset_name} data...")
        df_train, df_test = load_dataset(dataset_name)
        print(f"  Train: {len(df_train)} samples")
        print(f"  Test: {len(df_test)} samples")
        
        # Generate features
        X_train, X_test, y_train, y_test = generate_features(df_train, df_test)
        
        # Train baselines
        results = train_baselines(dataset_name, X_train, X_test, y_train, y_test)
        all_results[dataset_name] = results
    
    # Save and display results
    save_results(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"All models saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
