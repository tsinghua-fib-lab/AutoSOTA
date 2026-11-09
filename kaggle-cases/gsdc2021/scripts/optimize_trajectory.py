"""Joint robust sparse trajectory optimization from PPK and GNSS velocities.

Each collection has a shared vehicle path. Linear interpolation aligns every
phone at its actual timestamps; TDCP links two times, Doppler observes a
central time derivative, and acceleration regularizes gaps. Training labels
are used exclusively by the external evaluation/parameter-selection code.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from concurrent.futures import ProcessPoolExecutor,as_completed
from gnss_velocity import enu_matrix,ecef
from metric import evaluate,read_positions,validate_submission,KEY,COORD

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone/Google'

def interpolation(t,n):
    u=np.clip(t,0,n-1);i=np.minimum(np.floor(u).astype(int),n-2);f=u-i
    rows=np.repeat(np.arange(len(u)),2)
    cols=np.column_stack([i,i+1]).ravel();data=np.column_stack([1-f,f]).ravel()
    return sparse.csr_matrix((data,(rows,cols)),shape=(len(u),n))

def solve_collection(task):
    split,collection,b,p=task
    t0=float(b.millisSinceGpsEpoch.min());t1=float(b.millisSinceGpsEpoch.max())
    n=int(np.ceil((t1-t0)/1000))+1
    lat=float(b.latDeg.mean());lon=float(b.lngDeg.mean());R=enu_matrix(lat,lon)
    origin=ecef(np.array([[lat,lon,0.]]))[0]
    # Convert observations to a collection-local tangent plane; ~10 km extents.
    xyz=ecef(np.column_stack([b.latDeg,b.lngDeg,np.zeros(len(b))]))
    obs=((xyz-origin)@R.T)[:,:2]
    t=(b.millisSinceGpsEpoch.to_numpy()-t0)/1000
    C=interpolation(t,n)
    ns=np.maximum(b.get('ns',pd.Series(np.full(len(b),10))).to_numpy(),1)
    std=np.sqrt(b.get('stde',pd.Series(np.zeros(len(b)))).to_numpy()**2+b.get('stdn',pd.Series(np.zeros(len(b)))).to_numpy()**2)
    sig=np.maximum(p['position_sigma'],std)
    # Unavailable/extrapolated PPK samples carry negligible information.
    weights=1/np.maximum(sig,0.2)
    mats=[C];vals=[obs];scales=[weights];types=[np.zeros(len(b),dtype=int)]
    tdcp_count=doppler_count=0
    for phone in b.phone.unique():
        f=ROOT/'artifacts/velocity'/split/f'{phone}.csv'
        if not f.exists():continue
        v=pd.read_csv(f)
        if p['tdcp_sigma']>0 and 'tdcp_dx' in v:
            good=v.tdcp_dx.notna()&v.tdcp_t0.ge(t0)&v.millisSinceGpsEpoch.le(t1)
            z=v.loc[good].copy()
            if len(z):
                v0=(z.tdcp_t0.to_numpy()-t0)/1000;v1=(z.millisSinceGpsEpoch.to_numpy()-t0)/1000
                mats.append(interpolation(v1,n)-interpolation(v0,n))
                vals.append((z[['tdcp_dx','tdcp_dy','tdcp_dz']].to_numpy()@R.T)[:,:2])
                sg=np.maximum(p['tdcp_sigma'],z.tdcp_error.to_numpy())
                scales.append(1/sg);types.append(np.ones(len(z),dtype=int));tdcp_count+=len(z)
        if p['doppler_sigma']>0 and 'dop_vx' in v:
            good=v.dop_vx.notna()&v.millisSinceGpsEpoch.ge(t0+500)&v.millisSinceGpsEpoch.le(t1-500)
            z=v.loc[good].copy()
            if len(z):
                tv=(z.millisSinceGpsEpoch.to_numpy()-t0)/1000
                mats.append(interpolation(tv+0.5,n)-interpolation(tv-0.5,n))
                vals.append((z[['dop_vx','dop_vy','dop_vz']].to_numpy()@R.T)[:,:2])
                sg=np.maximum(p['doppler_sigma'],z.dop_error.to_numpy())
                scales.append(1/sg);types.append(np.full(len(z),2,dtype=int));doppler_count+=len(z)
    D=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csr')
    mats.append(D);vals.append(np.zeros((n-2,2)));scales.append(np.full(n-2,1/p['accel_sigma']));types.append(np.full(n-2,3,dtype=int))
    A=sparse.vstack(mats,format='csr');Y=np.vstack(vals);w=np.concatenate(scales);kind=np.concatenate(types);rob=np.ones(len(w))
    for _ in range(6):
        W=w*np.sqrt(rob);WA=A.multiply(W[:,None]);normal=WA.T@WA+sparse.eye(n)*1e-10
        xy=spsolve(normal.tocsc(),WA.T@(Y*W[:,None]))
        residual=np.linalg.norm(A@xy-Y,axis=1)
        huber=np.select([kind==0,kind==1,kind==2],[p['position_huber'],0.4,1.0],default=np.inf)
        rob=np.minimum(1,huber/np.maximum(residual,1e-9))
    fit=C@xy
    # Exact inverse of the tangent-plane projection at fixed zero ellipsoid height.
    # Latitude/longitude update with the analytic ellipsoid Jacobian is sub-mm
    # for corrections of tens of meters, and retains the input earth curvature.
    correction=fit-obs
    latrad=np.radians(b.latDeg.to_numpy());a=6378137.;e2=6.6943799901413165e-3
    M=a*(1-e2)/(1-e2*np.sin(latrad)**2)**1.5
    N=a/np.sqrt(1-e2*np.sin(latrad)**2)
    b=b.copy();b['latDeg']+=np.degrees(correction[:,1]/M);b['lngDeg']+=np.degrees(correction[:,0]/(N*np.cos(latrad)))
    return b,{'collection':collection,'tdcp_factors':tdcp_count,'doppler_factors':doppler_count}

def run(split,p,source='merged',workers=8,collections=None):
    if source=='release':
        f=ROOT/f'artifacts/rtklib_author_{split}.csv'
    elif source=='merged':
        f=RELEASE/(f'baseline_locations_merged_{split}_12_31.csv' if split=='train' else 'baseline_locations_merged_test_12_30.csv')
    elif source=='recomputed':f=ROOT/f'artifacts/rtklib_{split}.csv'
    elif source=='recomputed_merged':f=ROOT/f'artifacts/rtklib_merged_{split}.csv'
    else:raise ValueError(source)
    b=read_positions(f)
    items=[(split,c,g.copy(),p) for c,g in b.groupby('collectionName') if collections is None or c in collections]
    dfs=[];diagnostics=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(solve_collection,x) for x in items]):
            d,diag=f.result();dfs.append(d);diagnostics.append(diag)
    d=pd.concat(dfs,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    return d,diagnostics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='train',choices=['train','test']);ap.add_argument('--source',choices=['release','merged','recomputed','recomputed_merged'],default='merged');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--collection',action='append');ap.add_argument('--position-sigma',type=float,default=2.);ap.add_argument('--tdcp-sigma',type=float,default=.15);ap.add_argument('--doppler-sigma',type=float,default=.3);ap.add_argument('--accel-sigma',type=float,default=1.5);ap.add_argument('--position-huber',type=float,default=5.);ap.add_argument('--name',default='fgo');args=ap.parse_args()
    p={k:getattr(args,k) for k in ['position_sigma','tdcp_sigma','doppler_sigma','accel_sigma','position_huber']}
    d,diag=run(args.split,p,args.source,args.workers,args.collection)
    d.to_csv(ROOT/f'artifacts/{args.name}_{args.split}.csv',index=False)
    result={'parameters':p,'source':args.source,'split':args.split,'diagnostics':diag}
    if args.split=='train':
        truth=read_positions(ROOT/'artifacts/ground_truth_train.csv');truth=truth[truth.phone.isin(d.phone.unique())]
        score,phones,_=evaluate(d,truth);result['train_score_m']=score;phones.to_csv(ROOT/f'artifacts/{args.name}_train_scores.csv',index=False)
    elif not args.collection:
        sample=read_positions(ROOT/'data/competition/sample_submission.csv')
        sub=sample[KEY].merge(d[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
        sub.to_csv(ROOT/f'submissions/{args.name}.csv',index=False)
    (ROOT/f'artifacts/{args.name}_{args.split}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
