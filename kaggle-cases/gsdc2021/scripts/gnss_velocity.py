"""2021 raw/derived alignment and robust Doppler/TDCP displacement estimates.

TDCP uses the differential carrier range and satellite clock/orbit changes;
it fits a receiver clock term alongside vehicle displacement. No labels enter
the measurement model. Intermediate arrays are cached for experiments.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import io
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
POS=['xSatPosM','ySatPosM','zSatPosM']
VEL=['xSatVelMps','ySatVelMps','zSatVelMps']

def ecef(llh):
    lat,lon=np.radians(llh[:,0]),np.radians(llh[:,1]);h=llh[:,2]
    a=6378137.;e2=6.6943799901413165e-3
    n=a/np.sqrt(1-e2*np.sin(lat)**2)
    return np.column_stack([(n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)])

def enu_matrix(lat,lon):
    lat,lon=np.radians([lat,lon]);s,c=np.sin(lat),np.cos(lat);sl,cl=np.sin(lon),np.cos(lon)
    return np.array([[-sl,cl,0],[-s*cl,-s*sl,c],[c*cl,c*sl,s]])

def load_raw(path):
    names=None;lines=[]
    with path.open() as f:
        for line in f:
            if line.startswith('# Raw,'):names=line.strip()[2:].split(',')
            elif line.startswith('Raw,'):lines.append(line)
    if names is None or not lines:raise ValueError('Raw section missing')
    want=['TimeNanos','FullBiasNanos','Svid','ConstellationType','CarrierFrequencyHz','Cn0DbHz',
          'AccumulatedDeltaRangeMeters','AccumulatedDeltaRangeState','AccumulatedDeltaRangeUncertaintyMeters',
          'PseudorangeRateMetersPerSecond','PseudorangeRateUncertaintyMetersPerSecond','HardwareClockDiscontinuityCount']
    d=pd.read_csv(io.StringIO(''.join(lines)),names=names,usecols=want)
    # Integer arithmetic preserves nanosecond timestamp precision.
    d['millisSinceGpsEpoch']=(d.TimeNanos.astype('int64')-d.FullBiasNanos.astype('int64'))//1_000_000
    band5=d.CarrierFrequencyHz<1.3e9
    d['signalType']=np.select([d.ConstellationType.eq(1)&~band5,d.ConstellationType.eq(1)&band5,
                              d.ConstellationType.eq(6)&~band5,d.ConstellationType.eq(6)&band5,
                              d.ConstellationType.eq(3),d.ConstellationType.eq(5)],
                             ['GPS_L1','GPS_L5','GAL_E1','GAL_E5A','GLO_G1','BDS_B1I'],default='unknown')
    return d.rename(columns={'Svid':'svid','ConstellationType':'constellationType'})

def robust_fit(a,b,sigma,threshold,rng):
    good=np.isfinite(a).all(axis=1)&np.isfinite(b)&np.isfinite(sigma)&(sigma>0)
    a,b,sigma=a[good],b[good],sigma[good]
    if len(b)<5 or np.linalg.matrix_rank(a)<a.shape[1]:return None
    best=None;best_score=-1
    # Exact minimal samples plus an uncertainty-weighted initial fit.
    trials=[np.arange(len(b))]+[rng.choice(len(b),4,replace=False) for _ in range(24)]
    for ids in trials:
        try:
            w=1/np.maximum(sigma[ids],0.02)
            x=np.linalg.lstsq(a[ids]*w[:,None],b[ids]*w,rcond=None)[0]
            residual=np.abs(a@x-b)
            valid=residual<threshold
            score=int(valid.sum())
            if score>best_score:best_score=score;best=(x,valid)
        except np.linalg.LinAlgError:continue
    if best is None or best_score<5 or best_score<len(b)*0.6:return None
    x,valid=best
    for _ in range(3):
        r=np.abs(a@x-b)
        w=valid/np.maximum(sigma,0.02)*np.minimum(1,threshold/np.maximum(r,1e-6))
        x=np.linalg.lstsq(a*w[:,None],b*w,rcond=None)[0]
    rmse=float(np.sqrt(np.mean((a[valid]@x-b[valid])**2)))
    if np.linalg.norm(x[:3])>80:return None
    return x,rmse,int(valid.sum())

def estimate(item):
    split,folder=item
    collection,phone=folder.parent.name,folder.name
    out=ROOT/'artifacts/velocity'/split/f'{collection}_{phone}.csv';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():return {'phone':collection+'_'+phone,'status':'cached'}
    raw=load_raw(folder/f'{phone}_GnssLog.txt')
    der=pd.read_csv(folder/f'{phone}_derived.csv')
    ts=np.sort(raw.millisSinceGpsEpoch.unique())
    # Host's documented epoch correction: derived entries refer to the prior raw epoch.
    ix=np.searchsorted(ts,der.millisSinceGpsEpoch.to_numpy())-1
    der=der.loc[ix>=0].copy();der['millisSinceGpsEpoch']=ts[ix[ix>=0]]
    key=['millisSinceGpsEpoch','constellationType','svid','signalType']
    d=raw.drop_duplicates(key).merge(der.drop_duplicates(key),on=key,how='inner',validate='one_to_one')
    d=d[d.Cn0DbHz>20].sort_values(key)
    base=pd.read_csv(ROOT/f'data/competition/baseline_locations_{split}.csv')
    base=base[base.phone.eq(collection+'_'+phone)].sort_values('millisSinceGpsEpoch')
    tref=base.millisSinceGpsEpoch.to_numpy();llh=base[['latDeg','lngDeg','heightAboveWgs84EllipsoidM']].to_numpy()
    rec=ecef(llh)
    dts=d.millisSinceGpsEpoch.to_numpy();rp=np.column_stack([np.interp(dts,tref,rec[:,k]) for k in range(3)])
    sat=d[POS].to_numpy();dist=np.linalg.norm(sat-rp,axis=1);theta=7.2921151467e-5*dist/299792458.
    c,s=np.cos(theta),np.sin(theta);sat=np.column_stack([c*sat[:,0]+s*sat[:,1],-s*sat[:,0]+c*sat[:,1],sat[:,2]])
    los=(sat-rp)/np.linalg.norm(sat-rp,axis=1)[:,None]
    d[POS]=sat;d[['losx','losy','losz']]=los;d[['recx','recy','recz']]=rp
    sv=d[VEL].to_numpy();d['doppler_b']=d.PseudorangeRateMetersPerSecond.to_numpy()-np.sum(los*sv,axis=1)+d.satClkDriftMps.to_numpy()
    rows=[];rng=np.random.default_rng(2021)
    for t,g in d.groupby('millisSinceGpsEpoch',sort=True):
        a=np.column_stack([-g[['losx','losy','losz']].to_numpy(),np.ones(len(g))]);fit=robust_fit(a,g.doppler_b.to_numpy(),g.PseudorangeRateUncertaintyMetersPerSecond.to_numpy()+0.05,0.7,rng)
        r={'millisSinceGpsEpoch':int(t)}
        if fit is not None:
            x,err,n=fit;r.update(dict(zip(['dop_vx','dop_vy','dop_vz'],x[:3])));r.update(dop_error=err,dop_sats=n)
        rows.append(r)
    delta=[]
    for _,g in d.groupby(['constellationType','svid','signalType']):
        g=g.sort_values('millisSinceGpsEpoch').copy()
        t=g.millisSinceGpsEpoch.to_numpy();dt=np.diff(t)/1000
        if len(t)<2:continue
        state=g.AccumulatedDeltaRangeState.fillna(0).to_numpy(dtype='int64')
        valid=((state&1)>0)&((state&6)==0)
        ac=g.AccumulatedDeltaRangeMeters.to_numpy();p=g[POS].to_numpy();rx=g[['recx','recy','recz']].to_numpy()
        motion=np.linalg.norm(p[1:]-rx[:-1],axis=1)-np.linalg.norm(p[:-1]-rx[:-1],axis=1)
        clk=g.satClkBiasM.to_numpy()
        y=np.diff(ac)-motion+np.diff(clk)
        sg=g.AccumulatedDeltaRangeUncertaintyMeters.to_numpy()
        mask=valid[1:]&valid[:-1]&(dt>0.2)&(dt<2.1)&(np.abs(y)<200)&(np.diff(g.HardwareClockDiscontinuityCount)==0)
        ll=(g[['losx','losy','losz']].to_numpy()[1:]+g[['losx','losy','losz']].to_numpy()[:-1])/2
        for k in np.flatnonzero(mask):
            delta.append([t[k],t[k+1],*ll[k],y[k],np.sqrt(sg[k]**2+sg[k+1]**2)+0.01])
    if delta:
        dd=pd.DataFrame(delta,columns=['t0','t1','ux','uy','uz','b','sigma'])
        estimates={}
        for (t0,t1),g in dd.groupby(['t0','t1']):
            a=np.column_stack([-g[['ux','uy','uz']].to_numpy(),np.ones(len(g))]);fit=robust_fit(a,g.b.to_numpy(),g.sigma.to_numpy(),0.25,rng)
            if fit is not None:
                x,err,n=fit;v=x[:3]/((t1-t0)/1000)
                if np.linalg.norm(v)<80:
                    estimates[t1]={'tdcp_t0':t0,**dict(zip(['tdcp_dx','tdcp_dy','tdcp_dz'],x[:3])),'tdcp_error':err,'tdcp_sats':n}
        for r in rows:r.update(estimates.get(r['millisSinceGpsEpoch'],{}))
    result=pd.DataFrame(rows);result.to_csv(out,index=False)
    return {'phone':collection+'_'+phone,'epochs':len(result),'doppler':int(result.get('dop_vx',pd.Series(dtype=float)).notna().sum()),'tdcp':int(result.get('tdcp_dx',pd.Series(dtype=float)).notna().sum()),'status':'ok'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test','both'],default='both');p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);a=p.parse_args()
    splits=['train','test'] if a.split=='both' else [a.split]
    items=[(s,f) for s in splits for f in sorted((ROOT/'data/competition'/s).glob('*/*')) if f.is_dir()]
    if a.limit:items=items[:a.limit]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for future in as_completed([ex.submit(estimate,i) for i in items]):
            try:r=future.result()
            except Exception as e:r={'status':'error','error':str(e)}
            results.append(r);print(json.dumps(r),flush=True)
    (ROOT/'artifacts/velocity_status.json').write_text(json.dumps(results,indent=2))
    if any(r['status']=='error' for r in results):raise SystemExit(1)
