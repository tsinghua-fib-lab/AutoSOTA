"""Replay an advanced selected experiment, optionally preparing raw intermediates."""
from pathlib import Path
import argparse,json,os,subprocess,sys,hashlib
import numpy as np
import pandas as pd
from trajectory_v2 import run
from metric import read_positions,validate_submission,KEY,COORD,distances
from bias_correction import prepare,apply
from map_matching import match
ROOT=Path(__file__).resolve().parents[1]

def step(script,*args,python=None,env=None):
    subprocess.run([python or sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,check=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',default='pseudorange');ap.add_argument('--source',default='published',choices=['published','raw']);ap.add_argument('--bias',default='after',choices=['none','after']);ap.add_argument('--prepare',action='store_true');ap.add_argument('--workers',type=int,default=12);ap.add_argument('--bias-scope',default='city_label',choices=['city_label','unmapped']);args=ap.parse_args()
    folder=ROOT/'artifacts/experiments'/args.experiment
    selected=json.loads((folder/'best.json').read_text());p=selected['parameters']
    if args.prepare:
        step('unpack_data.py');step('prepare_truth.py')
        if args.source=='raw':
            subprocess.run(['bash',str(ROOT/'scripts/build_rtklib.sh')],cwd=ROOT,check=True)
            step('run_rtklib.py','--workers',str(args.workers));step('merge_rtklib.py')
        dirs=p.get('velocity_dirs',[p.get('velocity_dir','velocity')])
        for directory in dirs:
            if directory=='velocity':step('gnss_velocity.py','--workers',str(args.workers))
            else:
                variant=directory.removeprefix('velocity_');lag=int(variant.removeprefix('lag')) if variant.startswith('lag') else 1
                env={**os.environ,'GNSS_VARIANT':variant,'GNSS_LAG':str(lag)}
                step('gnss_velocity_v2.py','--workers',str(args.workers),env=env)
        if p.get('pseudorange_sigma',0)>0:
            step('cache_base_observations.py',python=str(ROOT/'.venvs/ielenik/bin/python'))
            step('pseudorange_factors.py','--workers',str(args.workers))
    source='merged' if args.source=='published' else 'recomputed_merged'
    pred,diag=run('test',p,source=source,workers=args.workers)
    if not p.get('joint_road_sigma',0):
        truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
        pred=match(pred,truth,training=False,mode='road',limit=50.,drift_sigma=.1,road_sigma=.5,anchor_sigma=20.)
    bp=None
    if args.bias=='after':
        if p.get('bias_model'):raise ValueError('Bias already applied inside graph; use --bias none')
        bp=json.loads((folder/'bias_parameters.json').read_text())
        pred=prepare(pred);city=pred.collectionName.str.contains('-SJC-')
        if args.bias_scope=='unmapped':
            mapped={x['collection'] for x in diag if x['road_factors']>0}
            city=pred.collectionName.isin(mapped)
        pred=pd.concat([pred.loc[city],apply(pred.loc[~city],np.asarray(bp['coefficients']),bp['shrink'])],ignore_index=True)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(pred[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    name=f'replay_v2_{args.experiment}_{args.source}_{args.bias}'+('_unmapped' if args.bias_scope=='unmapped' else '')
    output=ROOT/f'submissions/{name}.csv';sub.to_csv(output,index=False)
    result=dict(experiment=args.experiment,source=args.source,bias_scope=args.bias_scope,parameters=p,bias_parameters=bp,rows=len(sub),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),output=str(output.relative_to(ROOT)),diagnostics=diag)
    if args.source=='published' and args.bias_scope=='city_label':
        ref=ROOT/f"submissions/v2_{args.experiment}{'_bias' if args.bias=='after' else ''}.csv"
        if ref.exists():
            d=sub.merge(read_positions(ref),on=KEY,suffixes=('','_ref'),validate='one_to_one')
            delta=distances(d[COORD].to_numpy(),d[['latDeg_ref','lngDeg_ref']].to_numpy())
            result['reference']=str(ref.relative_to(ROOT));result['max_difference_m']=float(delta.max())
            if delta.max()>1e-3:raise ValueError('Replay differs from scored file by more than 1 mm')
    (ROOT/f'artifacts/{name}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
