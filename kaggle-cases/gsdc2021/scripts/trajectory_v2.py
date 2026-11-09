"""Joint robust sparse trajectory optimization from PPK and GNSS velocities.

Each collection has a shared vehicle path. Linear interpolation aligns every
phone at its actual timestamps; TDCP links two times, Doppler observes a
central time derivative, and acceleration regularizes gaps. Optional road
factors use date-excluded training labels; bias coefficients are fitted by
external date-grouped calibration. Hidden test labels are never available.
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
from road_factors import RoadFactors
from scipy.ndimage import gaussian_filter1d,minimum_filter1d

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
    if p.get('normalize_position_phones',False):weights/=np.sqrt(b.phone.nunique())
    mats=[C];vals=[obs];scales=[weights];types=[np.zeros(len(b),dtype=int)]
    tdcp_count=doppler_count=0
    for phone in b.phone.unique():
        if p.get('pseudorange_sigma',0)>0:
            code_path=ROOT/'artifacts/pseudorange'/split/f'{phone}.csv'
            if code_path.exists():
                code=pd.read_csv(code_path)
                code=code[code.millisSinceGpsEpoch.between(t0,t1)].copy()
                if len(code):
                    code_xyz=ecef(np.column_stack([code.latDeg,code.lngDeg,np.zeros(len(code))]))
                    code_xy=((code_xyz-origin)@R.T)[:,:2]
                    mats.append(interpolation((code.millisSinceGpsEpoch.to_numpy()-t0)/1000,n))
                    vals.append(code_xy)
                    code_sigma=np.maximum(p['pseudorange_sigma'],np.hypot(code.stde,code.stdn).to_numpy())
                    scales.append(1/code_sigma);types.append(np.full(len(code),4,dtype=int))
        if p.get('velocity_phones') and phone.rsplit('_',1)[1] not in p['velocity_phones']:continue
        parts=[]
        for velocity_dir in p.get('velocity_dirs',[p.get('velocity_dir','velocity')]):
            f=ROOT/'artifacts'/velocity_dir/split/f'{phone}.csv'
            if f.exists():parts.append(pd.read_csv(f))
        if not parts:continue
        v=pd.concat(parts,ignore_index=True)
        if p['tdcp_sigma']>0 and 'tdcp_dx' in v:
            good=v.tdcp_dx.notna()&v.tdcp_t0.ge(t0)&v.millisSinceGpsEpoch.le(t1)
            z=v.loc[good].copy()
            if len(z):
                v0=(z.tdcp_t0.to_numpy()-t0)/1000;v1=(z.millisSinceGpsEpoch.to_numpy()-t0)/1000
                mats.append(interpolation(v1,n)-interpolation(v0,n))
                vals.append((z[['tdcp_dx','tdcp_dy','tdcp_dz']].to_numpy()@R.T)[:,:2])
                sg=np.maximum(p['tdcp_sigma'],z.tdcp_error.to_numpy())
                if p.get('tdcp_covariance',False):sg=np.maximum(sg,z.tdcp_horizontal_sd.to_numpy())
                scales.append(1/sg);types.append(np.ones(len(z),dtype=int));tdcp_count+=len(z)
        if p['doppler_sigma']>0 and 'dop_vx' in v:
            good=v.dop_vx.notna()&v.millisSinceGpsEpoch.ge(t0+500)&v.millisSinceGpsEpoch.le(t1-500)&~v.duplicated('millisSinceGpsEpoch')
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
    road=None
    if p.get('joint_road_sigma',0)>0 and '-SJC-' in collection:
        road=RoadFactors(ROOT,collection,split,R,origin,p.get('excluded_dates',()))
    road_count=0
    stop_count=0
    for iteration in range(p.get('iterations',10)):
        if iteration==3 and p.get('stop_speed',0)>0:
            # Freeze a conservative sustained-stop detector before adding ZUPT.
            speed=np.linalg.norm(np.gradient(gaussian_filter1d(xy,1,axis=0),axis=0),axis=1)
            stopped=minimum_filter1d((speed<p['stop_speed']).astype(int),size=p.get('stop_window',5),mode='constant',cval=0)>0
            keep=stopped[:-1]&stopped[1:]
            stop_count=int(keep.sum())
            if stop_count:
                Z=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')[keep]
                A=sparse.vstack([A,Z],format='csr');Y=np.vstack([Y,np.zeros((stop_count,2))])
                w=np.r_[w,np.full(stop_count,1/p.get('stop_sigma',.05))]
                kind=np.r_[kind,np.full(stop_count,5)];rob=np.r_[rob,np.ones(stop_count)]
        if iteration==3 and p.get('bias_model'):
            models=json.loads((ROOT/p['bias_model']).read_text())
            coeff=np.asarray(models['by_date'].get(collection[:10],models['default']) if split=='train' else models['default'])
            velocity=np.gradient(gaussian_filter1d(xy,2,axis=0),axis=0)
            speed=np.linalg.norm(velocity,axis=1)
            forward=velocity/np.maximum(speed[:,None],.5)
            left=np.column_stack([-forward[:,1],forward[:,0]])
            shift=coeff[:2]+coeff[2]*forward+coeff[3]*left+coeff[4]*velocity
            shift*=np.minimum(1,3/np.maximum(np.linalg.norm(shift,axis=1),1e-8))[:,None]
            Y[:len(b)]=obs+(C@shift)*p.get('bias_shrink',1.)
        W=w*np.sqrt(rob);WA=A.multiply(W[:,None]);normal=WA.T@WA+sparse.eye(n)*1e-10
        rhs=WA.T@(Y*W[:,None])
        if road is not None and iteration>=3:
            rc,ry,road_count=road.factors(xy,p['joint_road_sigma'],p.get('joint_road_limit',50.),p.get('joint_road_huber',5.),p.get('joint_road_direction','axis'),p.get('heading_fill',False))
            coupled=sparse.block_diag([normal,normal],format='csc')+rc.T@rc
            answer=spsolve(coupled.tocsc(),rhs.T.ravel()+rc.T@ry)
            xy=answer.reshape(2,n).T
        else:
            xy=spsolve(normal.tocsc(),rhs)
        residual=np.linalg.norm(A@xy-Y,axis=1)
        huber=np.select([kind==0,kind==1,kind==2,kind==4,kind==5],[p['position_huber'],p.get('tdcp_huber',.4),p.get('doppler_huber',1.),p.get('pseudorange_huber',5.),p.get('stop_huber',.1)],default=np.inf)
        rob=np.minimum(1,huber/np.maximum(residual,1e-9))
        if p.get('tdcp_kernel')=='cauchy' and iteration>=4:
            threshold=p.get('tdcp_huber',.1)*max(1.,(10-iteration)/2)
            rob[kind==1]=1/(1+(residual[kind==1]/threshold)**2)
    fit=C@xy
    # Approximate local inverse using the ellipsoid latitude/longitude Jacobian.
    # Retaining each observation's coordinates avoids flattening the whole route.
    correction=fit-obs
    latrad=np.radians(b.latDeg.to_numpy());a=6378137.;e2=6.6943799901413165e-3
    M=a*(1-e2)/(1-e2*np.sin(latrad)**2)**1.5
    N=a/np.sqrt(1-e2*np.sin(latrad)**2)
    b=b.copy();b['latDeg']+=np.degrees(correction[:,1]/M);b['lngDeg']+=np.degrees(correction[:,0]/(N*np.cos(latrad)))
    return b,{'collection':collection,'tdcp_factors':tdcp_count,'doppler_factors':doppler_count,'road_factors':road_count,'stop_factors':stop_count}

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
