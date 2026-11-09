"""Compare genuine held-out predictions and estimate paired AUC uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def digest(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def load_predictions(directory, train, test):
    metadata = json.loads((directory / 'metrics.json').read_text())
    if metadata.get('fixture'):
        raise ValueError('Generated fixtures cannot enter official comparisons')
    if metadata.get('train_id_sha256') != digest(train.id) or metadata.get('test_id_sha256') != digest(test.id):
        raise ValueError('Prediction IDs do not match the official data order')
    oof = np.load(directory / 'oof.npy')
    preds = np.load(directory / 'test_predictions.npy')
    if oof.shape != (len(train),) or preds.shape != (len(test),):
        raise ValueError('Invalid prediction shape')
    if not np.isfinite(preds).all() or np.any((preds < 0) | (preds > 1)):
        raise ValueError('Invalid test probabilities')
    valid = np.isfinite(oof)
    if valid.sum() != metadata['validation_samples'] or np.any((oof[valid] < 0) | (oof[valid] > 1)):
        raise ValueError('Invalid held-out probabilities or coverage')
    return oof, preds, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    first, first_test, first_info = load_predictions(args.baseline, train, test)
    second, second_test, second_info = load_predictions(args.candidate, train, test)
    indices = np.flatnonzero(np.isfinite(first) & np.isfinite(second))
    if len(indices) < 10000:
        raise ValueError('Insufficient common held-out observations for comparison')
    labels = train.target.values[indices]
    a, b = first[indices], second[indices]
    rng = np.random.default_rng(2026)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    deltas = []
    for _ in range(args.bootstrap):
        sample = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        deltas.append(roc_auc_score(labels[sample], b[sample])-roc_auc_score(labels[sample], a[sample]))
    calibration, evaluation = train_test_split(np.arange(len(indices)), test_size=0.5,
                                              stratify=labels, random_state=713)
    trials = []
    logits = np.stack([logit(np.clip(a, 1e-6, 1-1e-6)), logit(np.clip(b, 1e-6, 1-1e-6))], 1)
    for weight in np.linspace(0, 1, 21):
        fused = logits[:, 0]*(1-weight)+logits[:, 1]*weight
        trials.append((float(roc_auc_score(labels[calibration], fused[calibration])), float(weight)))
    _, weight = max(trials)
    fused = logits[:, 0]*(1-weight)+logits[:, 1]*weight
    eval_labels = labels[evaluation]
    eval_positive, eval_negative = np.flatnonzero(eval_labels == 1), np.flatnonzero(eval_labels == 0)
    fusion_deltas = []
    for _ in range(args.bootstrap):
        sample = np.r_[rng.choice(eval_positive, len(eval_positive), replace=True),
                       rng.choice(eval_negative, len(eval_negative), replace=True)]
        rows = evaluation[sample]
        fusion_deltas.append(roc_auc_score(labels[rows], fused[rows])-roc_auc_score(labels[rows], a[rows]))
    report = {'baseline': str(args.baseline), 'candidate': str(args.candidate), 'common_heldout_samples': len(indices),
              'same_heldout_coverage': bool(np.array_equal(np.isfinite(first), np.isfinite(second))),
              'baseline_auc': float(roc_auc_score(labels, a)), 'candidate_auc': float(roc_auc_score(labels, b)),
              'candidate_minus_baseline_auc': float(roc_auc_score(labels, b)-roc_auc_score(labels, a)),
              'paired_bootstrap_95pct_interval': np.quantile(deltas, [0.025, 0.975]).tolist(),
              'candidate_logit_weight_selected_on_calibration_half': weight,
              'ensemble_auc_on_independent_evaluation_half': float(roc_auc_score(labels[evaluation], fused[evaluation])),
              'baseline_auc_on_evaluation_half': float(roc_auc_score(labels[evaluation], a[evaluation])),
              'candidate_auc_on_evaluation_half': float(roc_auc_score(labels[evaluation], b[evaluation])),
              'ensemble_minus_baseline_on_evaluation_half': float(roc_auc_score(eval_labels, fused[evaluation])
                                                                  -roc_auc_score(eval_labels, a[evaluation])),
              'ensemble_delta_paired_bootstrap_95pct_interval_on_evaluation_half':
                  np.quantile(fusion_deltas, [0.025, 0.975]).tolist(),
              'evaluation_samples': len(evaluation), 'bootstrap_iterations': args.bootstrap,
              'scope': 'Common held-out predictions; not a complete five-fold ensemble',
              'kaggle_public_auc': None, 'kaggle_private_auc': None}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2))
    # Both weights were chosen without consulting hidden test labels or scores.
    output = expit(logit(np.clip(first_test, 1e-6, 1-1e-6))*(1-weight)
                   +logit(np.clip(second_test, 1e-6, 1-1e-6))*weight)
    pd.DataFrame({'id': test.id, 'target': output}).to_csv(args.output / 'submission.csv', index=False)
    oof = np.full(len(train), np.nan, dtype=np.float32)
    oof[indices] = expit(fused)
    np.save(args.output / 'oof.npy', oof)
    np.save(args.output / 'test_predictions.npy', output)
    evidence = {'family': 'logit_ensemble', 'fixture': False, 'data_scope': 'full official training data',
                'competition_validation_auc': float(roc_auc_score(labels, fused)),
                'validation_samples': len(indices), 'test_samples': len(test),
                'train_id_sha256': digest(train.id), 'test_id_sha256': digest(test.id),
                'sources': [str(args.baseline), str(args.candidate)], 'candidate_weight': weight,
                'calibration_samples': len(calibration), 'evaluation_samples': len(evaluation),
                'independent_evaluation_auc': report['ensemble_auc_on_independent_evaluation_half'],
                'unpredicted_oof_rows': int(np.isnan(oof).sum()),
                'full_competition_score_reproduction': False, 'kaggle_public_auc': None, 'kaggle_private_auc': None}
    (args.output / 'metrics.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
