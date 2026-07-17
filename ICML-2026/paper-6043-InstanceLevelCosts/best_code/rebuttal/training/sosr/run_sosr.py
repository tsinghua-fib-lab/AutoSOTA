"""
3A: SOSR-style regression of example-dependent costs (ICML rebuttal).

Train a secondary model to predict |Δ| from input features, then use
predicted costs as sample weights when training the main classifier.

Only runs on Jigsaw (TF-IDF) and Synthetic (logreg) — strongest contrast
for the predictability story.

Usage:
    python rebuttal/training/sosr/run_sosr.py
    python rebuttal/training/sosr/run_sosr.py --datasets jigsaw
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.chdir(PROJECT_ROOT)

from data import load_dataset
from models import get_model
from core.seed import set_seed

SEEDS = [42, 123, 456, 0, 1, 7, 13, 99, 2024, 314]

DATASET_MODEL_MAP = {
    'jigsaw': 'tfidf',
    'synthetic': 'logreg',
}


def run_sosr_experiment(dataset_name, model_name, seed, output_dir):
    """Run SOSR: predict costs, then use as weights."""
    out_path = output_dir / dataset_name / f"{model_name}_classification_sosr_s{seed}.csv"
    if out_path.exists():
        print(f"  SKIP: {out_path} exists")
        return

    set_seed(seed)
    dataset = load_dataset(dataset_name, val_size=0.1, test_size=0.1, seed=seed)

    # Step 1: Get features for cost prediction
    X_train = dataset.train.X
    X_test = dataset.test.X
    abs_delta_train = dataset.train.abs_delta

    # For text, vectorize
    if dataset.feature_type == 'text':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=10000, stop_words='english')
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        X_val_vec = vec.transform(dataset.val.X) if dataset.val else None
    else:
        X_train_vec = np.asarray(X_train)
        X_test_vec = np.asarray(X_test)
        X_val_vec = np.asarray(dataset.val.X) if dataset.val else None

    # Step 2: Train cost predictor (Ridge on features -> |Δ|)
    cost_predictor = Ridge(alpha=1.0)
    cost_predictor.fit(X_train_vec, abs_delta_train)

    # Predict costs for training examples
    predicted_costs = cost_predictor.predict(X_train_vec)
    predicted_costs = np.clip(predicted_costs, 1e-6, None)  # no negative weights

    # Step 3: Train classifier with predicted costs as sample weights
    model_kwargs = {}
    if model_name == 'roberta':
        model_kwargs['cache_name'] = dataset_name

    model = get_model(model_name, task='classification', **model_kwargs)
    model.fit(dataset.train.X, dataset.train.y, sample_weight=predicted_costs)

    # Step 4: Get predictions
    rows = []
    for split_name, split_data in [('val', dataset.val), ('test', dataset.test)]:
        if split_data is None:
            continue
        proba = model.predict_proba(split_data.X)
        preds = model.predict(split_data.X)

        for i in range(len(split_data.y)):
            idx = split_data.indices[i] if split_data.indices is not None else i
            rows.append({
                'instance_id': int(idx),
                'split': split_name,
                'y_true': int(split_data.y[i]),
                'y_pred': int(preds[i]),
                'y_proba_0': float(proba[i, 0]),
                'y_proba_1': float(proba[i, 1]),
                'delta_signed': float(split_data.delta[i]),
                'abs_delta': float(split_data.abs_delta[i]),
                'y_star': int(split_data.y[i]),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description='SOSR cost regression for rebuttal')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MODEL_MAP.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output-dir', type=str, default='rebuttal/predictions')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for dataset_name in args.datasets:
        model_name = DATASET_MODEL_MAP[dataset_name]
        print(f"\n=== {dataset_name} / {model_name} (SOSR) ===")
        for seed in args.seeds:
            print(f"  seed={seed}")
            try:
                run_sosr_experiment(dataset_name, model_name, seed, output_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
