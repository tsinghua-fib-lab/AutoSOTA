"""Fit nonnegative logit weights on calibration rows and evaluate separately."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from compare_predictions import ROOT, digest, load_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--per-fold-halves', action='store_true')
    args = parser.parse_args()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    sources = [args.baseline]+args.candidates
    loaded = [load_predictions(path, train, test) for path in sources]
    common = np.logical_and.reduce([np.isfinite(value[0]) for value in loaded])
    indices = np.flatnonzero(common)
    if len(indices) < 10000:
        raise ValueError('Insufficient genuinely held-out common observations')
    y = train.target.values[indices]
    values = np.stack([logit(np.clip(value[0][indices], 1e-6, 1-1e-6)) for value in loaded], 1)
    calibration, evaluation = train_test_split(np.arange(len(indices)), test_size=0.5,
                                               stratify=y, random_state=713)
    if args.per_fold_halves:
        calibration_parts, evaluation_parts = [], []
        for fold in range(5):
            positions = np.flatnonzero(train.fold_48.values[indices] == fold)
            if len(positions) < 10000:
                raise ValueError('Full-fivefold ensemble requires all five held-out folds')
            left, right = train_test_split(positions, test_size=.5, stratify=y[positions], random_state=713)
            calibration_parts.append(left); evaluation_parts.append(right)
        calibration, evaluation = np.concatenate(calibration_parts), np.concatenate(evaluation_parts)
    # Coordinate search over a small convex family. Hidden test scores never
    # enter this fitting process; all weight updates use calibration rows only.
    weights = np.zeros(len(sources))
    weights[0] = 1
    calibration_auc = roc_auc_score(y[calibration], values[calibration] @ weights)
    for _ in range(4):
        changed = False
        for col in range(1, len(sources)):
            chosen = (calibration_auc, 0., weights)
            for fraction in np.linspace(0, 1, 21)[1:]:
                proposed = weights*(1-fraction)
                proposed[col] += fraction
                score = roc_auc_score(y[calibration], values[calibration] @ proposed)
                if score > chosen[0]+1e-7:
                    chosen = (score, float(fraction), proposed)
            if chosen[1] > 0:
                calibration_auc, _, weights = chosen
                changed = True
        if not changed:
            break
    combined = values @ weights
    evaluation_auc = roc_auc_score(y[evaluation], combined[evaluation])
    baseline_eval_auc = roc_auc_score(y[evaluation], values[evaluation, 0])
    rng = np.random.default_rng(72026)
    labels = y[evaluation]
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    deltas = []
    for _ in range(args.bootstrap):
        sample = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        rows = evaluation[sample]
        deltas.append(roc_auc_score(y[rows], combined[rows])-roc_auc_score(y[rows], values[rows, 0]))
    interval = np.quantile(deltas, [0.025, 0.975]).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(train), np.nan, dtype=np.float32)
    oof[indices] = expit(combined)
    output = expit(np.stack([logit(np.clip(value[1], 1e-6, 1-1e-6)) for value in loaded], 1) @ weights)
    np.save(args.output / 'oof.npy', oof)
    np.save(args.output / 'test_predictions.npy', output)
    pd.DataFrame({'id': test.id, 'target': output}).to_csv(args.output / 'submission.csv', index=False)
    evidence = dict(family='nonnegative_logit_ensemble', fixture=False,
                    data_scope='full official training data', competition_validation_auc=float(roc_auc_score(y, combined)),
                    validation_samples=len(indices), test_samples=len(test), train_id_sha256=digest(train.id),
                    test_id_sha256=digest(test.id), sources=[str(s) for s in sources], weights=weights.tolist(),
                    calibration_samples=len(calibration), evaluation_samples=len(evaluation),
                    per_fold_halves=args.per_fold_halves,
                    independent_evaluation_auc=float(evaluation_auc), baseline_evaluation_auc=float(baseline_eval_auc),
                    evaluation_auc_gain=float(evaluation_auc-baseline_eval_auc), evaluation_gain_95pct_interval=interval,
                    baseline_full_validation_auc=float(roc_auc_score(y, values[:, 0])),
                    source_validation_auc=[float(roc_auc_score(y, values[:, col])) for col in range(len(sources))],
                    unpredicted_oof_rows=int(np.isnan(oof).sum()),
                    selection_policy='Adaptive model comparison on fixed validation data; weight fit uses calibration half only',
                    full_competition_score_reproduction=False, kaggle_public_auc=None, kaggle_private_auc=None)
    (args.output / 'metrics.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    main()
