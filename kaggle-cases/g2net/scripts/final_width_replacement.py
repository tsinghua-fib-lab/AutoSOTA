"""Build a held-out width reference, then replace its frozen blend allocation.

Only the two width components of the best submitted blend are replaced.
The reference calibrators fit seed713 calibration halves, never evaluation rows.
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
from compare_predictions import ROOT, load_predictions, digest

BASE = ROOT / 'experiments/campaign14h_promoted_fivefold'
REF = ROOT / 'experiments/campaign_final_width_reference'


def logits(x):
    return logit(np.clip(x, 1e-6, 1-1e-6))


def interval(y, a, b, n=200):
    rng = np.random.default_rng(72026)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(n):
        rows = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        values.append(roc_auc_score(y[rows], b[rows])-roc_auc_score(y[rows], a[rows]))
    return np.quantile(values, [.025, .975]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--replacement', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id':str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id':str})
    base_oof, base_test, base = load_predictions(BASE, train, test)
    sources = [Path(p) for p in base['sources']]
    columns = [sources.index(BASE / ('normalized_family_%d' % k)) for k in range(2)]
    allocation = float(sum(base['weights'][k] for k in columns))
    relative = np.array([base['weights'][k] for k in columns])/allocation
    old = [load_predictions(sources[k], train, test) for k in columns]
    raw_oof = sum(w*logits(v[0]) for w,v in zip(relative,old))
    raw_test = sum(w*logits(v[1]) for w,v in zip(relative,old))
    if not np.isfinite(raw_oof).all():
        raise ValueError('Width reference requires full held-out fold coverage')
    if args.prepare:
        if REF.exists(): raise FileExistsError(REF)
        parameters=[]; calibrated=np.zeros(len(train)); evaluation=[]
        for fold in range(5):
            indices=np.flatnonzero(train.fold_48.values==fold)
            left,right=train_test_split(indices,test_size=.5,stratify=train.target.values[indices],random_state=713)
            model=LogisticRegression(C=10,max_iter=300).fit(raw_oof[left,None],train.target.values[left])
            a,b=float(model.coef_[0,0]),float(model.intercept_[0])
            if a <= 0: raise ValueError('Nonpositive reference slope')
            calibrated[indices]=expit(a*raw_oof[indices]+b)
            parameters.append(dict(fold=fold,slope=a,offset=b)); evaluation.append(right)
        output=expit(parameters[0]['slope']*raw_test+parameters[0]['offset'])
        REF.mkdir(parents=True)
        np.save(REF/'oof.npy',calibrated.astype(np.float32));np.save(REF/'test_predictions.npy',output)
        np.save(REF/'evaluation_indices.npy',np.concatenate(evaluation))
        info=dict(family='calibrated_current_width_reference',fixture=False,
                  train_id_sha256=digest(train.id),test_id_sha256=digest(test.id),
                  validation_samples=len(train),test_samples=len(test),parameters=parameters,
                  sources=[str(sources[k]) for k in columns],weights=relative.tolist(),
                  competition_validation_auc=float(roc_auc_score(train.target,calibrated)),
                  calibration_scope='Per-fold seed713 calibration halves only',
                  test_scale='Exact current width blend mapped by fold0 calibrator',allocation=allocation)
        (REF/'metrics.json').write_text(json.dumps(info,indent=2))
        return
    reference, reference_test, ref_info=load_predictions(REF,train,test)
    candidate, candidate_test, candidate_info=load_predictions(args.replacement,train,test)
    if not np.isfinite(candidate).all(): raise ValueError('Incomplete replacement family')
    scale=ref_info['parameters'][0]
    f0=train.fold_48.values==0
    output_oof=np.full(len(train),np.nan,np.float32)
    def replace_delta(base, new, old):
        # Difference in the calibrated score space preserves the identity
        # exactly, including clipped calibration tails and float32 roundoff.
        delta=allocation*(logits(new)-logits(old))/scale['slope']
        changed=expit(logit(np.clip(base.astype(np.float64),1e-12,1-1e-12))+delta)
        return np.where(delta==0,base,changed)
    output_oof[f0]=replace_delta(base_oof[f0],candidate[f0],reference[f0])
    output_test=replace_delta(base_test,candidate_test,reference_test)
    evaluation=np.load(REF/'evaluation_indices.npy'); y=train.target.values
    f0eval=evaluation[train.fold_48.values[evaluation]==0]
    f0gain=float(roc_auc_score(y[f0eval],output_oof[f0eval])-roc_auc_score(y[f0eval],base_oof[f0eval]))
    f0ci=interval(y[f0eval],base_oof[f0eval],output_oof[f0eval])
    checks=[]
    for fold in range(5):
        rows=evaluation[train.fold_48.values[evaluation]==fold]
        checks.append(dict(fold=fold,gain=float(roc_auc_score(y[rows],candidate[rows])-roc_auc_score(y[rows],reference[rows]))))
    gain=candidate_info['evaluation_auc_gain']; ci=candidate_info['evaluation_gain_95pct_interval']
    eligible=(gain>.0001 and ci[0]>0 and f0gain>=0 and all(c['gain']>-.00015 for c in checks))
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    np.save(args.output/'oof.npy',output_oof); np.save(args.output/'test_predictions.npy',output_test)
    pd.DataFrame({'id':test.id,'target':output_test}).to_csv(args.output/'submission.csv',index=False)
    info=dict(family='frozen_allocation_width_replacement',fixture=False,
              data_scope='full official training data',validation_samples=int(f0.sum()),test_samples=len(test),
              train_id_sha256=digest(train.id),test_id_sha256=digest(test.id),
              competition_validation_auc=float(roc_auc_score(y[f0],output_oof[f0])),
              independent_evaluation_auc=float(roc_auc_score(y[f0eval],output_oof[f0eval])),
              evaluation_auc_gain=f0gain,evaluation_gain_95pct_interval=f0ci,
              replacement_evaluation_gain=gain,replacement_gain_95pct_interval=ci,
              replication_fold_checks=checks,submission_eligible=eligible,
              sources=[str(BASE),str(args.replacement),str(REF)],allocation=allocation,
              selection_policy='Full-fivefold component fit on calibration halves; test scores never used; adaptive validation reuse',
              submission_gate='Full-fivefold gain>0.0001 and pairedCI>0; every fold gain>-0.00015; actual fold0 final-blend gain>=0',
              kaggle_public_auc=None,kaggle_private_auc=None)
    (args.output/'metrics.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))


if __name__=='__main__': main()
