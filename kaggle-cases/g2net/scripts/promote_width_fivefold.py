"""Replace two fixed fold-0 blend components with calibrated five-fold bagging.

The prior ensemble weights are frozen. Fold-0 OOF scores cannot measure test
bagging gains; the promotion gate instead checks successful reproduction on
the other four folds, against the older same-fold width controls.
"""
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
    parser.add_argument('--families', type=Path, nargs=2, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Promotion output already exists')
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    base_oof, base_test, baseline = load_predictions(args.baseline, train, test)
    if not np.array_equal(np.isfinite(base_oof), train.fold_48.values == 0):
        raise ValueError('Fixed reference blend must have exact fold-0 coverage')
    control, _, control_info = load_predictions(args.control, train, test)
    families = [load_predictions(path, train, test) for path in args.families]
    if any(not np.isfinite(value[0]).all() for value in families) or not np.isfinite(control).all():
        raise ValueError('All family inputs must cover five exact held-out folds')
    paths = [Path(p) for p in baseline['sources']]
    weights = np.asarray(baseline['weights'])
    columns = []
    for suffix in ['lr3e4', 'lr1e3']:
        wanted = ROOT / f'experiments/campaign10h_width_physical_{suffix}/predictions'
        columns.append(paths.index(wanted))
    relative = weights[columns]/weights[columns].sum()
    score = sum(w*logit(np.clip(value[0], 1e-6, 1-1e-6)) for w, value in zip(relative, families))
    evaluation_parts = []
    fold_checks = []
    for fold in range(1, 5):
        indices = np.flatnonzero(train.fold_48.values == fold)
        _, evaluate = train_test_split(indices, test_size=.5, stratify=train.target.values[indices], random_state=713)
        evaluation_parts.append(evaluate)
        candidate_auc = float(roc_auc_score(train.target.values[evaluate], score[evaluate]))
        control_auc = float(roc_auc_score(train.target.values[evaluate], control[evaluate]))
        fold_checks.append(dict(fold=fold, evaluation_auc=candidate_auc, control_auc=control_auc, gain=candidate_auc-control_auc))
    evaluation = np.concatenate(evaluation_parts)
    labels = train.target.values[evaluation]
    a, b = control[evaluation], score[evaluation]
    gain = float(roc_auc_score(labels, b)-roc_auc_score(labels, a))
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(72026)
    deltas = []
    for _ in range(150):
        sample = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        deltas.append(roc_auc_score(labels[sample], b[sample])-roc_auc_score(labels[sample], a[sample]))
    interval = np.quantile(deltas, [.025, .975]).tolist()
    eligible = gain > .0003 and interval[0] > 0 and all(item['gain'] > -.0002 for item in fold_checks)
    args.output.mkdir(parents=True)
    # Map each model's probability to the corresponding fold-0 score scale
    # before averaging, so frozen component weights retain their interpretation.
    for family_index, ((oof, _, info), family_path, col) in enumerate(zip(families, args.families, columns)):
        reference = info['parameters'][0]
        transformed_oof = expit((logit(np.clip(oof, 1e-6, 1-1e-6))-reference['offset'])/reference['slope'])
        test_parts = []
        for fold, parameter in enumerate(info['parameters']):
            raw_oof, predicted, _ = load_predictions(Path(parameter['source']), train, test)
            features = logit(np.clip(predicted, 1e-6, 1-1e-6))
            test_parts.append(expit((features*parameter['slope']+parameter['offset']-reference['offset'])/reference['slope']))
            indices = train.fold_48.values == fold
            if fold == 0:
                # Preserve the original fold-0 score exactly, including tails.
                transformed_oof[indices] = raw_oof[indices]
        output = args.output / f'normalized_family_{family_index}'
        output.mkdir()
        transformed_test = np.mean(test_parts, axis=0)
        np.save(output / 'oof.npy', transformed_oof.astype(np.float32))
        np.save(output / 'test_predictions.npy', transformed_test)
        metadata = dict(info, family='fivefold_normalized_to_fold0', sources=[str(family_path)],
                        competition_validation_auc=float(roc_auc_score(train.target, transformed_oof)),
                        reference_scale=reference, test_bagging_gain_unobservable_from_single_fold_oof=True)
        (output / 'metrics.json').write_text(json.dumps(metadata, indent=2))
        paths[col] = output
    loaded = [load_predictions(path, train, test) for path in paths]
    mask = np.logical_and.reduce([np.isfinite(v[0]) for v in loaded])
    if not np.array_equal(mask, train.fold_48.values == 0):
        raise ValueError('Promotion changed validation exclusion scope')
    oof = np.full(len(train), np.nan, np.float32)
    oof[mask] = expit(np.stack([logit(np.clip(v[0][mask], 1e-6, 1-1e-6)) for v in loaded], 1) @ weights)
    predicted = expit(np.stack([logit(np.clip(v[1], 1e-6, 1-1e-6)) for v in loaded], 1) @ weights)
    if not np.isfinite(predicted).all() or np.max(np.abs(oof[mask]-base_oof[mask])) > 1e-5:
        raise ValueError('Promotion must preserve the fixed fold-0 validation recipe')
    np.save(args.output / 'oof.npy', oof); np.save(args.output / 'test_predictions.npy', predicted)
    pd.DataFrame({'id':test.id, 'target':predicted}).to_csv(args.output / 'submission.csv', index=False)
    info = dict(baseline, family='fixed_weight_fivefold_promotion', sources=[str(p) for p in paths],
                weights=weights.tolist(), train_id_sha256=digest(train.id), test_id_sha256=digest(test.id),
                competition_validation_auc=float(roc_auc_score(train.target.values[mask], oof[mask])),
                evaluation_auc_gain=0., baseline_full_validation_auc=baseline['competition_validation_auc'],
                promotion_eligible=eligible, replication_evaluation_samples=len(evaluation), replication_auc_gain=gain,
                replication_gain_95pct_interval=interval, replication_fold_checks=fold_checks,
                control_family=str(args.control), promoted_components=[str(p) for p in args.families],
                scope='Frozen prior weights; fold-0 scores unchanged; replication gate on fold1-4 evaluation halves',
                limitation='Replication gain measures improvement over old width models; test bagging gain is not directly measurable by OOF',
                kaggle_public_auc=None, kaggle_private_auc=None)
    (args.output / 'metrics.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
