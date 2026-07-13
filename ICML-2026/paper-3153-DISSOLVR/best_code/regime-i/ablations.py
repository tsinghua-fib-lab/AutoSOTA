# ablations_400_features.py
# Category-based ablation study on the FULL 400 features from featurizer
# NO features removed - uses everything the featurizer produces

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from featurizer import MoleculeFeaturizer

# Config
SEED = 123
SEEDS = [42, 101, 123, 456, 789]  # For baseline variance estimation
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def get_column_categories_400(columns):
    """
    Classify the 400 featurizer columns into 4 categories.
    This uses ALL features - nothing is removed.
    
    Categories:
    - COMPOSITIONAL: Atom counts (num_*), functional groups (fr_*), count-based descriptors
    - TOPOLOGICAL: AUTOCORR2D, MOSE motifs, graph connectivity indices
    - ENERGETIC: pred_Tm, Abraham parameters
    - PHYSICOCHEMICAL: Electronic, surface area, partial charge descriptors
    """
    categories = {
        'COMPOSITIONAL': [],
        'TOPOLOGICAL': [],
        'ENERGETIC': [],
        'PHYSICOCHEMICAL': [],
    }
    
    # Count-based features that are COMPOSITIONAL
    # Note: num_*, total_atoms, HeavyAtomMolWt, ExactMolWt excluded in featurizer.py
    count_features = {
        'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
        'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
        'RingCount', 'HeavyAtomCount', 'MolWt', 'NumBridgeheadAtoms', 
        'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'NumAmideBonds'
    }
    
    # Topological features (Chi*, FpDensityMorgan*, Kappa3 excluded in featurizer.py)
    topological_features = {
        'BalabanJ', 'BertzCT', 'Kappa1', 'Kappa2', 'HallKierAlpha', 'Phi'
    }
    
    # Physicochemical/Electronic descriptors (MolMR excluded in featurizer.py)
    # LabuteASA kept for regime-ii council compatibility
    physicochemical_features = {
        'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex',
        'MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge',
        'qed', 'MolLogP', 'TPSA', 'FractionCSP3', 'LabuteASA'
    }
    # VSA descriptors (SMR_VSA, SlogP_VSA, VSA_EState, BCUT2D excluded in featurizer.py)
    physicochemical_prefixes = ['PEOE_VSA', 'EState_VSA']
    
    for i, col in enumerate(columns):
        # COMPOSITIONAL: functional groups, ring/atom counts (num_* commented out)
        if (col.startswith('fr_') or col in count_features):
            categories['COMPOSITIONAL'].append(i)
        
        # TOPOLOGICAL: MOSE motifs, graph indices (AUTOCORR2D disabled in featurizer)
        elif col.startswith('mose_') or col in topological_features:
            categories['TOPOLOGICAL'].append(i)
        
        # ENERGETIC: pred_Tm and Abraham descriptors
        elif col == 'pred_Tm' or col.startswith('abraham_'):
            categories['ENERGETIC'].append(i)
        
        # PHYSICOCHEMICAL: electronic and surface area descriptors
        elif (col in physicochemical_features or 
              any(col.startswith(p) for p in physicochemical_prefixes)):
            categories['PHYSICOCHEMICAL'].append(i)
        
        # Catch any remaining as PHYSICOCHEMICAL (most RDKit descriptors)
        else:
            categories['PHYSICOCHEMICAL'].append(i)
    
    return categories


def train_and_evaluate(X_train, X_test, y_train, y_test, kept_indices, experiment_name, verbose=True):
    """Train model on subset of features and evaluate."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {len(kept_indices)} / {X_train.shape[1]}")
    
    X_tr = X_train.iloc[:, kept_indices]
    X_te = X_test.iloc[:, kept_indices]
    
    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        verbose=200 if verbose else 0,
        random_state=SEED,
        allow_writing_files=False,
        thread_count=-1
    )
    model.fit(X_tr, y_train)
    
    preds = model.predict(X_te)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Test R²: {r2:.6f}, RMSE: {rmse:.6f}")
    
    return {'Experiment': experiment_name, 'R2': r2, 'RMSE': rmse, 'N_Features': len(kept_indices)}


def run_baseline_multiseed(X_train, X_test, y_train, y_test, all_indices):
    """Run FULL MODEL with multiple seeds to estimate variance."""
    print(f"\n{'='*60}")
    print(f"BASELINE VARIANCE ESTIMATION (5 seeds)")
    print(f"{'='*60}")
    
    r2_scores = []
    rmse_scores = []
    
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        
        X_tr = X_train.iloc[:, all_indices]
        X_te = X_test.iloc[:, all_indices]
        
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=5,
            verbose=0,
            random_state=seed,
            allow_writing_files=False,
            thread_count=-1
        )
        model.fit(X_tr, y_train)
        
        preds = model.predict(X_te)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        print(f"R²={r2:.4f}, RMSE={rmse:.4f}")
    
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    
    print(f"\n  BASELINE: R² = {r2_mean:.4f} ± {r2_std:.4f}, RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
    
    return r2_mean, r2_std, rmse_mean, rmse_std


def run_ablation_study():
    """Run category-based ablation study on ALL 400 features."""
    print("=" * 60)
    print("ABLATION STUDY ON FULL 400 FEATURES")
    print("Categories: COMPOSITIONAL, TOPOLOGICAL, ENERGETIC, PHYSICOCHEMICAL")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Featurize
    print("Featurizing molecules...")
    featurizer = MoleculeFeaturizer()
    X_train = featurizer.transform(train_df['SMILES'])
    X_test = featurizer.transform(test_df['SMILES'])
    y_train = train_df['LogS']
    y_test = test_df['LogS']
    
    columns = list(X_train.columns)
    n_features = len(columns)
    all_indices = list(range(n_features))
    
    print(f"\nTotal features: {n_features}")
    
    # Get categories
    categories = get_column_categories_400(columns)
    
    print("\n--- Category Breakdown ---")
    for cat, indices in categories.items():
        print(f"  {cat}: {len(indices)} features")
        # Show first few features
        sample_feats = [columns[i] for i in indices[:5]]
        print(f"    Examples: {sample_feats}")
    
    # Save feature breakdown to CSV
    feature_breakdown = []
    for cat, indices in categories.items():
        for idx in indices:
            feature_breakdown.append({'Feature': columns[idx], 'Category': cat})
    pd.DataFrame(feature_breakdown).to_csv("feature_categories_400.csv", index=False)
    print("\nFeature breakdown saved to: feature_categories_400.csv")
    
    results = []
    
    # =========================================================================
    # 1. BASELINE VARIANCE ESTIMATION (5 seeds)
    # =========================================================================
    baseline_r2_mean, baseline_r2_std, baseline_rmse_mean, baseline_rmse_std = run_baseline_multiseed(
        X_train, X_test, y_train, y_test, all_indices
    )
    
    # =========================================================================
    # 2. FULL MODEL (single seed for comparison with ablations)
    # =========================================================================
    results.append(train_and_evaluate(
        X_train, X_test, y_train, y_test,
        all_indices, "FULL MODEL (400 features)"
    ))
    baseline_r2 = results[0]['R2']
    baseline_rmse = results[0]['RMSE']
    
    # =========================================================================
    # 2. REMOVE: Leave-One-Category-Out
    # =========================================================================
    print("\n" + "=" * 60)
    print("LEAVE-ONE-CATEGORY-OUT EXPERIMENTS")
    print("=" * 60)
    
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        remove_indices = set(categories[category])
        kept = [i for i in all_indices if i not in remove_indices]
        results.append(train_and_evaluate(
            X_train, X_test, y_train, y_test,
            kept, f"REMOVE: {category}"
        ))
    
    # =========================================================================
    # 3. ONLY: Use Only One Category
    # =========================================================================
    print("\n" + "=" * 60)
    print("SINGLE-CATEGORY-ONLY EXPERIMENTS")
    print("=" * 60)
    
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        kept = categories[category]
        if len(kept) > 0:
            results.append(train_and_evaluate(
                X_train, X_test, y_train, y_test,
                kept, f"ONLY: {category}"
            ))
    
    # =========================================================================
    # 4. Summary
    # =========================================================================
    df_results = pd.DataFrame(results)
    df_results['Delta_R2'] = df_results['R2'] - baseline_r2
    df_results['Delta_RMSE'] = df_results['RMSE'] - baseline_rmse
    
    df_results.to_csv("ablation_400_features_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(df_results.to_string(index=False))
    
    print(f"\n--- Baseline Variance (5 seeds) ---")
    print(f"R² = {baseline_r2_mean:.4f} ± {baseline_r2_std:.4f}")
    print(f"RMSE = {baseline_rmse_mean:.4f} ± {baseline_rmse_std:.4f}")
    print(f"\n(Compare ΔR² and ΔRMSE against ±{baseline_r2_std:.4f} and ±{baseline_rmse_std:.4f} respectively)")
    
    print(f"\n--- Key Insights ---")
    print(f"Baseline (seed={SEED}): R² = {baseline_r2:.6f}, RMSE = {baseline_rmse:.6f}")
    print("\nImpact when removing each category:")
    
    for category in ['COMPOSITIONAL', 'TOPOLOGICAL', 'ENERGETIC', 'PHYSICOCHEMICAL']:
        row = df_results[df_results['Experiment'] == f"REMOVE: {category}"].iloc[0]
        n_removed = len(categories[category])
        sig_r2 = "*" if abs(row['Delta_R2']) > 2*baseline_r2_std else ""
        sig_rmse = "*" if abs(row['Delta_RMSE']) > 2*baseline_rmse_std else ""
        print(f"  {category} ({n_removed} features): ΔR² = {row['Delta_R2']:+.6f}{sig_r2}, ΔRMSE = {row['Delta_RMSE']:+.6f}{sig_rmse}")
    
    print(f"\n* = exceeds 2σ baseline variance (likely significant)")
    print(f"\nResults saved to: ablation_400_features_results.csv")
    print("=" * 60)
    
    return df_results


if __name__ == "__main__":
    run_ablation_study()
