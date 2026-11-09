"""One-command replay of the scored submissions; no network submission side effect."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import hashlib
import numpy as np
from metric import read_positions,validate_submission,distances,KEY,COORD
from optimize_trajectory import run
from map_matching import match

ROOT=Path(__file__).resolve().parents[1]

def step(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',choices=['published','raw'],default='published')
    ap.add_argument('--profile',choices=['private','public'],default='private')
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    step('unpack_data.py');step('prepare_truth.py')
    if a.source=='raw':
        subprocess.run(['bash',str(ROOT/'scripts/build_rtklib.sh')],cwd=ROOT,check=True)
        step('run_rtklib.py','--workers',str(a.workers))
        step('merge_rtklib.py')
    step('gnss_velocity.py','--workers',str(a.workers))
    parameter_file='best_parameters.json' if a.profile=='private' else 'refined_parameters.json'
    p=json.loads((ROOT/'artifacts'/parameter_file).read_text())
    road=json.loads((ROOT/'artifacts/best_road_parameters.json').read_text())
    rp={k:road[k] for k in ['mode','alpha','limit','drift_sigma','road_sigma','anchor_sigma']}
    source='merged' if a.source=='published' else 'recomputed_merged'
    name=f'replay_{a.source}_{a.profile}'
    pred,diag=run('test',p,source=source,workers=a.workers)
    intermediate=ROOT/f'artifacts/{name}_before_road.csv'
    pred.to_csv(intermediate,index=False)
    # The original private-grid runner reloaded its intermediate CSV, whereas
    # the extended public-grid runner applied the road stage directly.
    if a.profile=='private':pred=read_positions(intermediate)
    truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
    pred=match(pred,truth,training=False,**rp)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(pred[KEY+COORD],on=KEY,how='left',validate='one_to_one')
    validate_submission(sub,sample)
    output=ROOT/f'submissions/{name}.csv';sub.to_csv(output,index=False)
    manifest=dict(source=a.source,profile=a.profile,parameters=p,road_parameters=rp,rows=len(sub),phones=sub.phone.nunique(),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),output=str(output.relative_to(ROOT)),diagnostics=diag)
    ref='fgo_raw_reproduced.csv' if a.source=='raw' and a.profile=='private' else ('fgo_road_selected.csv' if a.profile=='private' else 'fgo_refined_selected.csv')
    reference=ROOT/'submissions'/ref
    if reference.exists():
        original=read_positions(reference)
        joined=sub.merge(original,on=KEY,suffixes=('','_ref'),validate='one_to_one')
        delta=distances(joined[COORD].to_numpy(),joined[['latDeg_ref','lngDeg_ref']].to_numpy())
        manifest['reference']=ref
        manifest['max_coordinate_difference_m']=float(np.max(delta))
        if a.source=='published' and np.max(delta)>1e-3:raise RuntimeError('Replay differs from scored submission by more than 1 mm')
    (ROOT/f'artifacts/{name}.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
