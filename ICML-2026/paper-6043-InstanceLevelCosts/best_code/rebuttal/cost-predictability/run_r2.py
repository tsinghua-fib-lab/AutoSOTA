"""
2D: R² cost predictability analysis (ICML rebuttal).

For each dataset, train Ridge and HistGBM regressors to predict |Δ| from features.
Hypothesis: synthetic high R², real datasets low — explaining why cost-sensitive
training doesn't help on real data.

Usage:
    python rebuttal/cost-predictability/run_r2.py
    python rebuttal/cost-predictability/run_r2.py --datasets jigsaw synthetic
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

RESULTS_DIR = Path("rebuttal/cost-predictability/results")
SEED = 42


def load_features_and_costs(dataset_name):
    """Load features and Δ for a dataset using the same splits as experiments."""
    from data import load_dataset

    dataset = load_dataset(dataset_name, val_size=0.1, test_size=0.1, seed=SEED)

    # Get features in numeric form
    X_train = dataset.train.X
    X_test = dataset.test.X
    # Return both signed and abs delta for analysis
    y_train = dataset.train.abs_delta
    y_test = dataset.test.abs_delta
    delta_signed_train = dataset.train.delta
    delta_signed_test = dataset.test.delta

    feature_type = dataset.feature_type

    if feature_type == 'text':
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
    elif feature_type == 'image':
        # For image datasets, try to load cached ResNet50 embeddings
        cache_path = PROJECT_ROOT / 'cache' / 'embeddings' / f'{dataset_name}_resnet50_mean_full.npy'
        if cache_path.exists():
            embeddings = np.load(cache_path)
            train_idx = dataset.train.indices
            test_idx = dataset.test.indices
            X_train = embeddings[train_idx]
            X_test = embeddings[test_idx]
        else:
            print(f"  WARNING: No cached embeddings for {dataset_name}, skipping")
            return None, None, None, None, None, None, feature_type
    elif feature_type == 'tabular':
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values.astype(np.float32)
            X_test = X_test.values.astype(np.float32)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

    return X_train, X_test, y_train, y_test, delta_signed_train, delta_signed_test, feature_type


def run_r2_analysis(dataset_name):
    """Run R² analysis for one dataset."""
    print(f"\n=== {dataset_name} ===")

    X_train, X_test, y_train, y_test, delta_signed_train, delta_signed_test, feature_type = load_features_and_costs(dataset_name)
    if X_train is None:
        return None

    print(f"  Feature type: {feature_type}")
    print(f"  Train: {X_train.shape if hasattr(X_train, 'shape') else 'sparse'}, Test: {X_test.shape if hasattr(X_test, 'shape') else 'sparse'}")
    print(f"  |Δ| range: [{y_train.min():.3f}, {y_train.max():.3f}], mean={y_train.mean():.3f}")

    from scipy import sparse
    is_sparse = sparse.issparse(X_train)

    results = {'dataset': dataset_name, 'feature_type': feature_type,
               'n_train': X_train.shape[0], 'n_test': X_test.shape[0],
               'delta_mean': float(y_train.mean()), 'delta_std': float(y_train.std())}

    # Ridge on signed delta (tests if delta is linearly predictable)
    print("  Training Ridge on signed Δ...")
    ridge_signed = Ridge(alpha=1.0)
    ridge_signed.fit(X_train, delta_signed_train)
    results['ridge_signed_r2'] = float(r2_score(delta_signed_test, ridge_signed.predict(X_test)))
    print(f"    signed Δ R² = {results['ridge_signed_r2']:.4f}")

    # Ridge on |delta|
    print("  Training Ridge on |Δ|...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results['ridge_r2'] = float(r2_score(y_test, y_pred_ridge))
    results['ridge_mae'] = float(mean_absolute_error(y_test, y_pred_ridge))
    print(f"    |Δ| R² = {results['ridge_r2']:.4f}, MAE = {results['ridge_mae']:.4f}")

    # HistGBM on |delta| — skip for sparse text (OOM on .toarray())
    if is_sparse:
        print("  Skipping HistGBM (sparse text features, would OOM)")
        results['hgbm_r2'] = float('nan')
        results['hgbm_mae'] = float('nan')
    else:
        print("  Training HistGBM on |Δ|...")
        hgbm = HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=SEED)
        hgbm.fit(X_train, y_train)
        y_pred_hgbm = hgbm.predict(X_test)
        results['hgbm_r2'] = float(r2_score(y_test, y_pred_hgbm))
        results['hgbm_mae'] = float(mean_absolute_error(y_test, y_pred_hgbm))
    print(f"    R² = {results['hgbm_r2']:.4f}, MAE = {results['hgbm_mae']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='R² cost predictability analysis')
    parser.add_argument('--datasets', nargs='+',
                        default=['jigsaw', 'nhanes', 'synthetic', 'turkey', 'inaturalist'])
    args = parser.parse_args()

    import os
    os.chdir(PROJECT_ROOT)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset in args.datasets:
        try:
            result = run_r2_analysis(dataset)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "r2_results.csv", index=False)

    # Print summary table
    print("\n" + "=" * 70)
    print("R² COST PREDICTABILITY SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Signed R²':>10} {'|Δ| Ridge':>10} {'|Δ| HGB':>10} {'Ridge MAE':>10} {'HGB MAE':>10}")
    print("-" * 65)
    for _, row in df.iterrows():
        signed = row.get('ridge_signed_r2', float('nan'))
        print(f"{row['dataset']:<15} {signed:>10.4f} {row['ridge_r2']:>10.4f} {row['hgbm_r2']:>10.4f} "
              f"{row['ridge_mae']:>10.4f} {row['hgbm_mae']:>10.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['ridge_r2'], width, label='Ridge', color='steelblue')
    ax.bar(x + width/2, df['hgbm_r2'], width, label='HistGBM', color='coral')
    ax.set_ylabel('Test R²')
    ax.set_title('Cost Predictability: Can |Δ| be predicted from features?')
    ax.set_xticks(x)
    ax.set_xticklabels(df['dataset'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.1, max(df[['ridge_r2', 'hgbm_r2']].max().max() + 0.1, 0.5))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_predictability.pdf", bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "cost_predictability.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
