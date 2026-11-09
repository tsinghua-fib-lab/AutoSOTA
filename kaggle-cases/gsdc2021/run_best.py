"""Run autosota's selected solution after regenerating external inputs."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from trajectory_v2 import run
from bias_correction import prepare,apply
from metric import read_positions,validate_submission,KEY,COORD

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
    p=json.loads((ROOT/'configs/best_private.json').read_text())
    required=['artifacts/rtklib_merged_test.csv','artifacts/ground_truth_train.csv','data/competition/sample_submission.csv',p['fitted_bias_file']]
    missing=[s for s in required if not (ROOT/s).exists()]
    if missing:raise SystemExit('Prepare external inputs first: python prepare_best.py --from-raw. Missing: '+str(missing))
    baseline=read_positions(ROOT/required[0])
    for phone in baseline.phone.unique():
        for directory in [*p['parameters']['velocity_dirs'],'pseudorange']:
            f=ROOT/'artifacts'/directory/'test'/f'{phone}.csv'
            if not f.exists():raise FileNotFoundError(f)
    pred,diag=run('test',p['parameters'],source=p['source'],workers=args.workers)
    pred=prepare(pred);mapped={d['collection'] for d in diag if d['road_factors']>0};mask=pred.collectionName.isin(mapped)
    bp=json.loads((ROOT/p['fitted_bias_file']).read_text())
    corrected=pd.concat([pred[mask],apply(pred[~mask],np.asarray(bp['coefficients']),bp['shrink'])],ignore_index=True)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(corrected[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    out=ROOT/'outputs/autosota_best_private.csv';out.parent.mkdir(exist_ok=True);sub.to_csv(out,index=False)
    h=hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({'implementation':'autosota','output':str(out),'rows':len(sub),'sha256':h,'matches_historical_csv_hash':h==p['submission_sha256'],'note':'No new Kaggle score is computed or submitted.'},indent=2))

if __name__=='__main__':main()
