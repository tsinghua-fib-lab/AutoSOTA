"""Generate external observations and fit the selected calibration recipe."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))

def step(name,*args,python=None,env=None):
    subprocess.run([python or sys.executable,str(ROOT/'scripts'/name),*args],cwd=ROOT,env=env,check=True)

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--from-raw',action='store_true');g.add_argument('--calibrate-only',action='store_true');ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    p=json.loads((ROOT/'configs/best_private.json').read_text())
    if a.from_raw:
        step('unpack_data.py');step('prepare_truth.py')
        subprocess.run(['bash',str(ROOT/'scripts/build_rtklib.sh')],cwd=ROOT,check=True)
        step('run_rtklib.py','--workers',str(a.workers));step('merge_rtklib.py')
        for variant,lag in [('midpoint',1),('lag5',5)]:
            step('gnss_velocity_v2.py','--workers',str(a.workers),env={**os.environ,'GNSS_VARIANT':variant,'GNSS_LAG':str(lag)})
        step('cache_base_observations.py',python=str(ROOT/'.venvs/ielenik/bin/python'))
        step('pseudorange_factors.py','--workers',str(a.workers))
    from trajectory_v2 import run
    from bias_correction import prepare,fit
    from metric import read_positions
    recipe=p['bias_training_recipe']
    pred,diag=run('train',p['parameters'],source=recipe['prediction_source'],workers=a.workers)
    folder=ROOT/'artifacts/autosota_best';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'calibration_train.csv';pred.to_csv(path,index=False)
    d=prepare(read_positions(path),read_positions(ROOT/'artifacts/ground_truth_train.csv'))
    d=d[~d.collectionName.str.contains(recipe['exclude_region'])&~d.collectionName.str[:10].isin(recipe['exclude_dates'])]
    coefficients=fit(d,recipe['huber'],recipe['ridge'])
    output=ROOT/p['fitted_bias_file'];output.write_text(json.dumps({'coefficients':coefficients.tolist(),'shrink':recipe['shrink'],'huber':recipe['huber'],'ridge':recipe['ridge'],'generated_locally':True},indent=2)+chr(10))
    print(json.dumps({'implementation':'autosota','fitted_bias_file':str(output),'calibration_rows':len(d),'note':'Fitted values are created locally; none were shipped in the source ZIP.'}))

if __name__=='__main__':main()
