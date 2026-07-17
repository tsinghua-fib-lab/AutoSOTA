"""
Training loss as cost proxy (ICML rebuttal).

Trains a regularized model, computes per-instance training loss,
and uses that loss as cost weights for a second classifier.

Unlike the existing reg_loss weighting in src/ (which uses a single
heavily-regularized model's loss as weights), this computes loss from
a normally-trained model — points with higher cross-entropy loss
are harder to classify and potentially more ambiguous.

Usage:
    python rebuttal/training/training_loss/run_training_loss.py
    python rebuttal/training/training_loss/run_training_loss.py --datasets jigsaw
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

DATASET_MODEL_MAP = {
    'jigsaw': 'tfidf',
    'nhanes': 'histgbm',
    'synthetic': 'logreg',
    'turkey': 'resnet50',
    'inaturalist': 'resnet50',
}


def cross_entropy_loss(y_true, y_proba):
    """Per-instance binary cross-entropy loss."""
    p = np.clip(y_proba, 1e-15, 1 - 1e-15)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def run_training_loss_experiment(dataset_name, model_name, seed, output_dir):
    """
    Two-stage approach:
    1. Train unweighted model, get training set probabilities
    2. Compute per-instance CE loss as cost proxy
    3. Retrain with loss-derived weights
    """
    out_path = output_dir / dataset_name / f"{model_name}_classification_trainloss_s{seed}.csv"
    if out_path.exists():
        print(f"  SKIP: {out_path} exists")
        return

    set_seed(seed)
    dataset = load_dataset(dataset_name, val_size=0.1, test_size=0.1, seed=seed)

    model_kwargs = {}
    if model_name == 'roberta':
        model_kwargs['cache_name'] = dataset_name
    if model_name == 'histgbm' and 'feature_names' in dataset.metadata:
        feature_names = dataset.metadata['feature_names']
        model_kwargs['num_features'] = [f for f in ['RIDAGEYR', 'BMXBMI'] if f in feature_names]
        model_kwargs['cat_features'] = [f for f in ['RIAGENDR', 'RIDRETH3'] if f in feature_names]

    # Stage 1: Train unweighted model
    model_s1 = get_model(model_name, task='classification', **model_kwargs)
    model_s1.fit(dataset.train.X, dataset.train.y)

    # Get training probabilities
    train_proba = model_s1.predict_proba(dataset.train.X)[:, 1]

    # Compute per-instance loss as cost proxy
    loss_weights = cross_entropy_loss(dataset.train.y, train_proba)
    # Normalize to mean 1 to keep scale similar to abs_delta
    loss_weights = loss_weights / (loss_weights.mean() + 1e-10)
    loss_weights = np.clip(loss_weights, 1e-6, None)

    # Stage 2: Retrain with loss-derived weights
    model = get_model(model_name, task='classification', **model_kwargs)
    model.fit(dataset.train.X, dataset.train.y, sample_weight=loss_weights)

    # Get predictions
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
    parser = argparse.ArgumentParser(description='Training loss as cost proxy')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MODEL_MAP.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output-dir', type=str, default='rebuttal/predictions')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for dataset_name in args.datasets:
        model_name = DATASET_MODEL_MAP[dataset_name]
        print(f"\n=== {dataset_name} / {model_name} (training loss) ===")
        for seed in args.seeds:
            print(f"  seed={seed}")
            try:
                run_training_loss_experiment(dataset_name, model_name, seed, output_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
