"""
Zadrozny et al. (2003) "Costing" — cost-proportionate rejection sampling
with ensemble aggregation.

Algorithm from Section 2.3.4:
1. For i = 1 to t:
   (a) S' = rejection sample from S: accept example with prob c_i / Z
   (b) h_i = train classifier on S'
2. Predict by majority vote / averaged probabilities

Usage:
    python rebuttal/training/zadrozny/run_costing.py
    python rebuttal/training/zadrozny/run_costing.py --datasets nhanes --seeds 42
    python rebuttal/training/zadrozny/run_costing.py --datasets jigsaw --t 10
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.chdir(PROJECT_ROOT)

from data import load_dataset
from models import get_model
from core.seed import set_seed

SEEDS = [42, 123, 456, 0, 1, 7, 13, 99, 2024, 314]
T = 10

DATASET_MODEL_MAP = {
    'jigsaw': 'tfidf',
    'nhanes': 'histgbm',
    'synthetic': 'logreg',
}


def run_costing(dataset_name, model_name, seed, output_dir, t=T):
    out_path = output_dir / dataset_name / f"{model_name}_classification_costing_s{seed}.csv"
    if out_path.exists():
        print(f"  SKIP: {out_path} exists")
        return

    set_seed(seed)
    dataset = load_dataset(dataset_name, val_size=0.1, test_size=0.1, seed=seed)

    model_kwargs = {}
    if model_name == 'histgbm' and 'feature_names' in dataset.metadata:
        feature_names = dataset.metadata['feature_names']
        model_kwargs['num_features'] = [f for f in ['RIDAGEYR', 'BMXBMI'] if f in feature_names]
        model_kwargs['cat_features'] = [f for f in ['RIAGENDR', 'RIDRETH3'] if f in feature_names]

    X_train = dataset.train.X
    y_train = dataset.train.y
    costs = dataset.train.abs_delta
    Z = costs.max()

    rng = np.random.RandomState(seed)
    ensemble_models = []

    for i in range(t):
        accept = rng.random(len(y_train)) < (costs / Z)
        if accept.sum() < 10:
            accept[:10] = True

        if isinstance(X_train, list):
            X_sub = [X_train[j] for j in range(len(X_train)) if accept[j]]
        elif isinstance(X_train, pd.DataFrame):
            X_sub = X_train.iloc[accept]
        else:
            X_sub = X_train[accept]
        y_sub = y_train[accept]

        model = get_model(model_name, task='classification', **model_kwargs)
        model.fit(X_sub, y_sub)
        ensemble_models.append(model)
        print(f"    Member {i+1}/{t}: {accept.sum()}/{len(y_train)} accepted")

    rows = []
    for split_name, split_data in [('val', dataset.val), ('test', dataset.test)]:
        if split_data is None:
            continue
        proba_sum = None
        for m in ensemble_models:
            p = m.predict_proba(split_data.X)
            proba_sum = p if proba_sum is None else proba_sum + p
        avg_proba = proba_sum / len(ensemble_models)
        preds = (avg_proba[:, 1] >= 0.5).astype(int)

        for i in range(len(split_data.y)):
            idx = split_data.indices[i] if split_data.indices is not None else i
            rows.append({
                'instance_id': int(idx), 'split': split_name,
                'y_true': int(split_data.y[i]), 'y_pred': int(preds[i]),
                'y_proba_0': float(avg_proba[i, 0]), 'y_proba_1': float(avg_proba[i, 1]),
                'delta_signed': float(split_data.delta[i]),
                'abs_delta': float(split_data.abs_delta[i]),
                'y_star': int(split_data.y[i]),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Zadrozny costing for rebuttal')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MODEL_MAP.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--t', type=int, default=T, help='Number of ensemble members')
    parser.add_argument('--output-dir', type=str, default='rebuttal/predictions')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for dataset_name in args.datasets:
        model_name = DATASET_MODEL_MAP[dataset_name]
        print(f"\n=== {dataset_name} / {model_name} (Costing, t={args.t}) ===")
        for seed in args.seeds:
            print(f"  seed={seed}")
            try:
                run_costing(dataset_name, model_name, seed, output_dir, t=args.t)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
