"""
Focal loss training for ICML rebuttal.

Two-stage approach for sklearn-based models:
1. Train unweighted model, get predicted probabilities
2. Compute focal weights: w_i = (1 - p_t)^gamma where p_t is the predicted prob of the true class
3. Retrain with focal weights as sample_weight

Saves per-instance prediction CSVs to rebuttal/predictions/{dataset}/
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.chdir(PROJECT_ROOT)

from data import load_dataset
from models import get_model
from core.seed import set_seed


SEEDS = [42, 123, 456, 0, 1, 7, 13, 99, 2024, 314]

DATASET_MODEL_MAP = {
    'jigsaw': 'tfidf',
    'nhanes': 'histgbm',
    'synthetic': 'logreg',
    'turkey': 'resnet50',
    'inaturalist': 'resnet50',
}


def focal_weights(y_true, y_proba, gamma):
    """Compute focal loss weights: (1 - p_t)^gamma where p_t is prob of true class."""
    p_t = np.where(y_true == 1, y_proba[:, 1], y_proba[:, 0])
    weights = (1.0 - p_t) ** gamma
    # Clip to avoid zero weights
    weights = np.clip(weights, 1e-6, None)
    return weights


def run_focal_experiment(dataset_name, model_name, gamma, seed, output_dir):
    """Run a single focal loss experiment."""
    out_path = output_dir / dataset_name / f"{model_name}_classification_focal{gamma}_s{seed}.csv"
    if out_path.exists():
        print(f"  SKIP: {out_path} exists")
        return

    set_seed(seed)

    # Load dataset with same splits as original experiments
    dataset = load_dataset(dataset_name, val_size=0.1, test_size=0.1, seed=seed)

    # Model kwargs
    model_kwargs = {}
    if model_name == 'roberta':
        model_kwargs['cache_name'] = dataset_name
    if model_name == 'histgbm' and 'feature_names' in dataset.metadata:
        feature_names = dataset.metadata['feature_names']
        num_features = ['RIDAGEYR', 'BMXBMI']
        cat_features = ['RIAGENDR', 'RIDRETH3']
        model_kwargs['num_features'] = [f for f in num_features if f in feature_names]
        model_kwargs['cat_features'] = [f for f in cat_features if f in feature_names]

    # Stage 1: Train unweighted model
    model_stage1 = get_model(model_name, task='classification', **model_kwargs)
    model_stage1.fit(dataset.train.X, dataset.train.y)

    # Get training probabilities
    train_proba = model_stage1.predict_proba(dataset.train.X)

    # Compute focal weights
    fw = focal_weights(dataset.train.y, train_proba, gamma)

    # Stage 2: Retrain with focal weights
    model = get_model(model_name, task='classification', **model_kwargs)
    model.fit(dataset.train.X, dataset.train.y, sample_weight=fw)

    # Get predictions on all splits
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
    parser = argparse.ArgumentParser(description='Focal loss training for rebuttal')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MODEL_MAP.keys()))
    parser.add_argument('--gammas', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output-dir', type=str, default='rebuttal/predictions')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for dataset_name in args.datasets:
        model_name = DATASET_MODEL_MAP[dataset_name]
        print(f"\n=== {dataset_name} / {model_name} ===")

        for gamma in args.gammas:
            for seed in args.seeds:
                print(f"  gamma={gamma}, seed={seed}")
                try:
                    run_focal_experiment(dataset_name, model_name, gamma, seed, output_dir)
                except Exception as e:
                    print(f"  ERROR: {e}")


if __name__ == '__main__':
    main()
