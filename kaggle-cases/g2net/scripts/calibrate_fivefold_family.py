"""Fold-specific Platt calibration using each fold's calibration half only.

Different checkpoint families have noticeable cross-fold score offsets. Fit on
the same seed-713 per-fold calibration halves used for fold-0 ensembles, never
on their evaluation halves. Each test fold is transformed before averaging.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from compare_predictions import ROOT, digest, load_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, nargs=5, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    raw = np.full(len(train), np.nan, np.float32)
    calibrated = raw.copy()
    test_outputs, plain_outputs, parameters, heldout_parts = [], [], [], []
    for fold, source in enumerate(args.sources):
        values, predicted, info = load_predictions(source, train, test)
        indices = np.flatnonzero(train.fold_48.values == fold)
        if not np.array_equal(np.isfinite(values), train.fold_48.values == fold):
            raise ValueError('Each input must cover its exact held-out fold')
        if info.get('fold', fold) != fold or info.get('fold_seed', 48) != 48:
            raise ValueError('Incorrect source fold metadata')
        y = train.target.values[indices]
        calibration, evaluation = train_test_split(np.arange(len(indices)), test_size=.5,
                                                   stratify=y, random_state=713)
        features = logit(np.clip(values[indices], 1e-6, 1-1e-6))[:, None]
        model = LogisticRegression(C=10, solver='lbfgs', max_iter=300).fit(features[calibration], y[calibration])
        slope, offset = float(model.coef_[0, 0]), float(model.intercept_[0])
        if not slope > 0:
            raise ValueError('Calibration must preserve within-fold ordering')
        output = expit(features[:, 0]*slope+offset)
        raw[indices] = values[indices]; calibrated[indices] = output
        test_outputs.append(expit(logit(np.clip(predicted, 1e-6, 1-1e-6))*slope+offset))
        plain_outputs.append(predicted)
        heldout_parts.append(indices[evaluation])
        parameters.append(dict(fold=fold, slope=slope, offset=offset, calibration_samples=len(calibration),
                               evaluation_samples=len(evaluation), raw_fold_auc=float(roc_auc_score(y, values[indices])),
                               source=str(source)))
    evaluation = np.concatenate(heldout_parts)
    predicted = np.mean(test_outputs, axis=0)
    if not np.isfinite(calibrated).all() or not np.isfinite(predicted).all():
        raise ValueError('Invalid calibrated probabilities')
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'oof.npy', calibrated)
    np.save(args.output / 'test_predictions.npy', predicted)
    np.save(args.output / 'evaluation_indices.npy', evaluation)
    pd.DataFrame({'id': test.id, 'target': predicted}).to_csv(args.output / 'submission.csv', index=False)
    evidence = dict(family='fold_calibrated_fivefold', fixture=False, data_scope='full official training data',
                    competition_validation_auc=float(roc_auc_score(train.target.values, calibrated)),
                    validation_samples=len(train), test_samples=len(test), train_id_sha256=digest(train.id),
                    test_id_sha256=digest(test.id), unpredicted_oof_rows=0, parameters=parameters,
                    raw_full_oof_auc=float(roc_auc_score(train.target.values, raw)),
                    raw_evaluation_auc=float(roc_auc_score(train.target.values[evaluation], raw[evaluation])),
                    calibrated_evaluation_auc=float(roc_auc_score(train.target.values[evaluation], calibrated[evaluation])),
                    scope='Calibration labels used only from per-fold seed-713 calibration halves; evaluation halves excluded',
                    sources=[str(s) for s in args.sources], full_competition_score_reproduction=False,
                    kaggle_public_auc=None, kaggle_private_auc=None)
    (args.output / 'metrics.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    main()
