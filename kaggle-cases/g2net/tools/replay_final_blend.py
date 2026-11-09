"""Recompute the frozen final blend from saved component predictions; no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'replayed_submission.csv')
    args = ap.parse_args()
    old = json.loads((ROOT/'DELIVERY.json').read_text())['runtime_source_root']
    meta=json.loads((ROOT/'experiments/campaign_final_final_ensemble/metrics.json').read_text())
    columns=[]
    # Preserve all original columns, including zero-weight columns, to reproduce
    # the original matrix reduction order as closely as the math library allows.
    for source in meta['sources']:
        p=Path(source)
        p=ROOT/p.relative_to(old) if p.is_absolute() else ROOT/p
        x=np.load(p/'test_predictions.npy',allow_pickle=False)
        assert x.shape==(226000,) and np.isfinite(x).all()
        columns.append(logit(np.clip(x,1e-6,1-1e-6)))
    prediction=expit(np.stack(columns,1)@np.asarray(meta['weights']))
    expected=np.load(ROOT/'best_submission/test_predictions.npy',allow_pickle=False)
    error=float(np.max(np.abs(prediction-expected)))
    assert np.allclose(prediction,expected,rtol=0,atol=1e-12), error
    ids=pd.read_csv(ROOT/'data/packed/test.csv',dtype={'id':str}).id
    assert len(ids)==226000 and ids.is_unique
    assert args.output.resolve()!=(ROOT/'best_submission/submission.csv').resolve()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':ids,'target':prediction}).to_csv(args.output,index=False)
    print(json.dumps({'samples':len(ids),'max_absolute_error':error,'output':str(args.output),
                      'scope':'Frozen final layer replay from saved component predictions; not a new GPU inference run.'},indent=2))


if __name__=='__main__':main()
