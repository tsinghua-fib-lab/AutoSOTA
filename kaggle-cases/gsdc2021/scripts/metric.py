"""GSDC 2021 metric and strict timestamp/key validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['phone', 'millisSinceGpsEpoch']
COORD = ['latDeg', 'lngDeg']

def read_positions(path):
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = d.columns.str.strip()
    if 'phone' not in d and {'collectionName', 'phoneName'} <= set(d):
        d['phone'] = d.collectionName + '_' + d.phoneName
    d['phone'] = d.phone.str.strip()
    d['millisSinceGpsEpoch'] = d.millisSinceGpsEpoch.astype('int64')
    if d.duplicated(KEY).any():
        raise ValueError(f'Duplicate phone/timestamp keys: {path}')
    if not np.isfinite(d[COORD].to_numpy()).all():
        raise ValueError(f'Nonfinite coordinates: {path}')
    if not d.latDeg.between(-90, 90).all() or not d.lngDeg.between(-180, 180).all():
        raise ValueError(f'Invalid coordinate range: {path}')
    return d

def distances(pred, true):
    a, b = np.radians(pred), np.radians(true)
    delta = b - a
    h = np.sin(delta[:, 0]/2)**2 + np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin(delta[:, 1]/2)**2
    return 2 * 6_367_000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def evaluate(pred, truth):
    if pred.duplicated(KEY).any() or truth.duplicated(KEY).any():
        raise ValueError('Duplicate keys')
    m = truth[KEY+COORD].merge(pred[KEY+COORD], on=KEY, suffixes=('_gt', ''), how='outer', indicator=True, validate='one_to_one')
    if not m['_merge'].eq('both').all():
        raise ValueError(f'Key coverage mismatch: {m._merge.value_counts().to_dict()}')
    m['error_m'] = distances(m[COORD].to_numpy(), m[['latDeg_gt', 'lngDeg_gt']].to_numpy())
    if not np.isfinite(m.error_m).all():
        raise ValueError('Nonfinite errors')
    q = m.groupby('phone').error_m.quantile([0.5, 0.95]).unstack()
    q.columns = ['p50_m', 'p95_m']
    q['score_m'] = q.mean(axis=1)
    q['rows'] = m.groupby('phone').size()
    return float(q.score_m.mean()), q.reset_index(), m

def validate_submission(pred, sample):
    if list(pred.columns) != KEY+COORD:
        raise ValueError('Submission columns must be '+str(KEY+COORD))
    if pred.duplicated(KEY).any():
        raise ValueError('Duplicate submission keys')
    if not pred[KEY].reset_index(drop=True).equals(sample[KEY].reset_index(drop=True)):
        raise ValueError('Submission keys/order differ from sample')
    if not np.isfinite(pred[COORD].to_numpy()).all():
        raise ValueError('Submission contains nonfinite coordinates')
    if not pred.latDeg.between(-90,90).all() or not pred.lngDeg.between(-180,180).all():
        raise ValueError('Submission coordinate range invalid')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('prediction'); ap.add_argument('truth'); ap.add_argument('--output', default='artifacts/metric')
    args = ap.parse_args()
    score, phones, errors = evaluate(read_positions(args.prediction), read_positions(args.truth))
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    phones.to_csv(str(p)+'_phones.csv', index=False)
    result = dict(score_m=score, rows=len(errors), phones=len(phones), prediction=args.prediction, truth=args.truth)
    Path(str(p)+'.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
