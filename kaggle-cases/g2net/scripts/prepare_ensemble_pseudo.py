"""Soft labels from the existing fold-0 ensemble, with explicit teacher scope."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from compare_predictions import ROOT, digest, load_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path, default=ROOT / 'experiments/optimized_1d_ensemble')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/pseudo_campaign12h_ensemble_fold0')
    parser.add_argument('--sources', type=Path, nargs='+')
    parser.add_argument('--weights', type=float, nargs='+')
    args = parser.parse_args()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    teacher = args.teacher
    if args.sources:
        if not args.weights or len(args.weights) != len(args.sources):
            raise ValueError('Supply one frozen weight per same-fold teacher')
        weights = np.asarray(args.weights, dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Invalid frozen teacher weights')
        weights = weights/weights.sum()
        loaded = [load_predictions(source, train, test) for source in args.sources]
        for values, _, metadata in loaded:
            if metadata.get('family') != 'team3' or metadata.get('fold') != 0 or metadata.get('fold_seed') != 48:
                raise ValueError('Use direct same-fold models; full-fivefold test teachers include student validation data')
            if not np.array_equal(np.isfinite(values), train.fold_48.values == 0):
                raise ValueError('Each direct teacher must exclude the student validation fold')
        oof = np.full(len(train), np.nan, dtype=np.float32)
        mask = train.fold_48.values == 0
        oof[mask] = expit(np.stack([logit(np.clip(v[0][mask],1e-6,1-1e-6)) for v in loaded],1) @ weights)
        probabilities = expit(np.stack([logit(np.clip(v[1],1e-6,1-1e-6)) for v in loaded],1) @ weights)
        info = dict(family='frozen_samefold_teacher_ensemble', fixture=False, fold=0, fold_seed=48,
                    sources=[str(s) for s in args.sources], weights=weights.tolist(),
                    validation_samples=int(mask.sum()), test_samples=len(test),
                    train_id_sha256=digest(train.id), test_id_sha256=digest(test.id),
                    competition_validation_auc=float(roc_auc_score(train.target.values[mask],oof[mask])),
                    test_teacher_weights_exclude_student_validation_fold=True)
        teacher = 'Frozen direct fold-0 teacher components'
    else:
        oof, probabilities, info = load_predictions(teacher, train, test)
    if not np.array_equal(np.isfinite(oof), train.fold_48.values == 0):
        raise ValueError('Teacher must have exact fold-0 validation coverage')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': test.id, 'preds_Fold_0': np.clip(probabilities, 1e-5, 1-1e-5)}).to_csv(
        output / 'test_Fold_0.csv', index=False)
    evidence = dict(fold_seed=48, excluded_fold=0, teacher=str(teacher),
                    test_id_sha256=digest(test.id), teacher_metrics=info,
                    labels='Soft probabilities clipped to distinguish them from hard observed targets',
                    adaptive_validation_selection=True)
    evidence['test_teacher_weights_exclude_student_validation_fold'] = bool(info.get('test_teacher_weights_exclude_student_validation_fold', False))
    (output / 'teacher_scope.json').write_text(json.dumps(evidence, indent=2))
    print('Prepared same-fold ensemble teacher soft labels', flush=True)


if __name__ == '__main__':
    main()
