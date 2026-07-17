"""
Zadrozny et al. cost-proportionate reweighting with calibration for ICML rebuttal.

Like existing 'absdelta' weighting, but adds probability calibration via
isotonic regression on the validation set (CalibratedClassifierCV).

Saves per-instance prediction CSVs to rebuttal/predictions/{dataset}/
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV

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


def run_zadrozny_experiment(dataset_name, model_name, seed, output_dir):
    """Run a single Zadrozny cost-proportionate reweighting experiment."""
    out_path = output_dir / dataset_name / f"{model_name}_classification_zadrozny_s{seed}.csv"
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

    # Train with abs_delta as sample weights (cost-proportionate)
    sample_weight = np.abs(dataset.train.delta).astype(np.float32)
    sample_weight = np.clip(sample_weight, 1e-6, None)

    model = get_model(model_name, task='classification', **model_kwargs)
    model.fit(dataset.train.X, dataset.train.y, sample_weight=sample_weight)

    # Calibrate on validation set using isotonic regression
    # We need to get the underlying sklearn estimator for CalibratedClassifierCV
    # For models that wrap sklearn, extract the internal model
    # Then calibrate using val predictions

    # Get val features in the format the model uses internally
    # Instead of wrapping with CalibratedClassifierCV (which needs raw sklearn estimator),
    # we do manual isotonic calibration on the probability outputs
    from sklearn.isotonic import IsotonicRegression

    val_proba_uncalib = model.predict_proba(dataset.val.X)[:, 1]
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso_reg.fit(val_proba_uncalib, dataset.val.y)

    # Get calibrated predictions on all splits
    rows = []
    for split_name, split_data in [('val', dataset.val), ('test', dataset.test)]:
        if split_data is None:
            continue
        proba_uncalib = model.predict_proba(split_data.X)
        # Calibrate the positive class probability
        p1_calibrated = iso_reg.transform(proba_uncalib[:, 1])
        p0_calibrated = 1.0 - p1_calibrated
        preds = (p1_calibrated >= 0.5).astype(int)

        for i in range(len(split_data.y)):
            idx = split_data.indices[i] if split_data.indices is not None else i
            rows.append({
                'instance_id': int(idx),
                'split': split_name,
                'y_true': int(split_data.y[i]),
                'y_pred': int(preds[i]),
                'y_proba_0': float(p0_calibrated[i]),
                'y_proba_1': float(p1_calibrated[i]),
                'delta_signed': float(split_data.delta[i]),
                'abs_delta': float(split_data.abs_delta[i]),
                'y_star': int(split_data.y[i]),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Zadrozny cost-proportionate reweighting for rebuttal')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MODEL_MAP.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output-dir', type=str, default='rebuttal/predictions')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for dataset_name in args.datasets:
        model_name = DATASET_MODEL_MAP[dataset_name]
        print(f"\n=== {dataset_name} / {model_name} ===")

        for seed in args.seeds:
            print(f"  seed={seed}")
            try:
                run_zadrozny_experiment(dataset_name, model_name, seed, output_dir)
            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == '__main__':
    main()
