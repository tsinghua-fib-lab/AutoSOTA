"""Linux orchestration of Everett's released 2021 RTKLIB smartphone solution."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone'
BINARY=ROOT/'upstream/RTKLIB_smartphone/app/consapp/rnx2rtkp/gcc/rnx2rtkp'

def run_phone(item):
    split, folder=item
    collection,phone=folder.parent.name,folder.name
    out=ROOT/'artifacts/rtklib'/split/collection/phone
    out.mkdir(parents=True,exist_ok=True)
    raw=folder/f'{phone}_GnssLog.txt';obs=out/'rover.obs';pos=out/'solution.pos'
    aux=RELEASE/'Google'/split/collection
    base=sorted(aux.glob('slac*.*o'));nav=sorted(aux.glob('BRDM*.rnx'))
    if not base or not nav:return {'split':split,'phone':collection+'_'+phone,'status':'missing_aux'}
    env=os.environ.copy();env['PYTHONPATH']=str(RELEASE/'python');env['OPENBLAS_NUM_THREADS']='1'
    try:
        if not obs.exists():
            cmd=[sys.executable,str(RELEASE/'python/android_rinex/gnsslogger_to_rnx.py'),str(raw),'--output',str(obs),'--fix-bias','--slip-mask','0','--time-adj','1e-7']
            with (out/'convert.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True,timeout=900)
        if not pos.exists():
            cmd=[str(BINARY),'-x','0','-y','2','-k',str(RELEASE/'Google/ppk_slac.conf'),'-o',str(pos),str(obs)]+[str(f) for f in base+nav]
            with (out/'positioning.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=log,check=True,timeout=1800)
        x=np.loadtxt(pos,comments='%')
        if x.ndim!=2 or len(x)<2:raise ValueError('Insufficient output positions')
        return {'split':split,'phone':collection+'_'+phone,'status':'ok','epochs':len(x)}
    except Exception as e:
        return {'split':split,'phone':collection+'_'+phone,'status':'error','error':str(e)}

def collect(split):
    baseline=pd.read_csv(ROOT/f'data/competition/baseline_locations_{split}.csv')
    dfs=[]
    for phone,b in baseline.groupby('phone',sort=False):
        collection,name=phone.rsplit('_',1)
        f=ROOT/'artifacts/rtklib'/split/collection/name/'solution.pos'
        if not f.exists():raise FileNotFoundError(f)
        x=np.loadtxt(f,comments='%');t=np.rint(1000*(x[:,0]*604800+x[:,1])).astype('int64')
        order=np.argsort(t);x=x[order];t=t[order]
        for col,j in [('latDeg',2),('lngDeg',3),('heightAboveWgs84EllipsoidM',4),('stdn',7),('stde',8),('stdu',9)]:
            if col.startswith('std'):
                b[col]=np.interp(b.millisSinceGpsEpoch,t,x[:,j],left=1000,right=1000)
            else:b[col]=np.interp(b.millisSinceGpsEpoch,t,x[:,j])
        b['Q']=np.interp(b.millisSinceGpsEpoch,t,x[:,5]);b['ns']=np.interp(b.millisSinceGpsEpoch,t,x[:,6]);dfs.append(b)
    d=pd.concat(dfs,ignore_index=True)
    d.to_csv(ROOT/f'artifacts/rtklib_{split}.csv',index=False)
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test','both'],default='both');p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);args=p.parse_args()
    splits=['train','test'] if args.split=='both' else [args.split]
    items=[(s,f) for s in splits for f in sorted((ROOT/'data/competition'/s).glob('*/*')) if f.is_dir()]
    if args.limit:items=items[:args.limit]
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for future in as_completed([ex.submit(run_phone,i) for i in items]):
            r=future.result();results.append(r);print(json.dumps(r),flush=True)
    (ROOT/'artifacts/rtklib_run_status.json').write_text(json.dumps(results,indent=2))
    if any(r['status']!='ok' for r in results):sys.exit(1)
    if not args.limit:
        for split in splits:collect(split)
