"""Independent base-corrected code positions for absolute GNSS factor fusion.

Uses only raw/derived observations, public SLAC RINEX, and the official baseline
as a linearization point. No ground-truth file is read by this module.
"""
from pathlib import Path
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import median_filter
from gnss_velocity import load_raw,ecef,enu_matrix,POS,VEL
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone/Google'
LIGHT=299792458.;OMEGA=7.2921151467e-5

def rotate(sat,rho):
    theta=OMEGA*rho/LIGHT;c=np.cos(theta);s=np.sin(theta)
    return np.column_stack([c*sat[:,0]+s*sat[:,1],-s*sat[:,0]+c*sat[:,1],sat[:,2]])

def solve(a,y,sigma,groups):
    unique,gi=np.unique(groups,return_inverse=True)
    A=np.column_stack([a,np.eye(len(unique))[gi]])
    if len(y)<A.shape[1]+2 or np.linalg.matrix_rank(A)<A.shape[1]:return None
    x=np.r_[np.zeros(3),[np.median(y[gi==i]) for i in range(len(unique))]]
    for _ in range(8):
        residual=y-A@x
        robust=np.minimum(1,5/np.maximum(np.abs(residual),1e-6))
        w=np.sqrt(robust)/sigma;aw=A*w[:,None]
        x=np.linalg.lstsq(aw,y*w,rcond=None)[0]
    if np.linalg.norm(x[:3])>200:return None
    information=aw.T@aw
    covariance=np.linalg.pinv(information)[:3,:3]
    residual=y-A@x
    scale=max(1.,float(np.sqrt(np.mean(np.sort((residual/sigma)**2)[:max(4,int(.8*len(y)))]))))
    covariance*=scale**2
    return x[:3],np.sqrt(np.maximum(np.diag(covariance),0.)),float(np.median(np.abs(residual)))

def process(item):
    split,folder=item;collection,phone=folder.parent.name,folder.name
    out=ROOT/'artifacts/pseudorange'/split/f'{collection}_{phone}.csv';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():return {'phone':collection+'_'+phone,'status':'cached'}
    paths=list((RELEASE/split/collection).glob('slac*.*o'))
    if not paths:raise FileNotFoundError(collection)
    base=np.load(ROOT/'artifacts/base_observations'/split/collection/f'{paths[0].name}.npz')
    # Same published station position as the original RTKLIB configuration.
    base_xyz=ecef(np.array([[37.4165169989997,-122.204262,63.6962000001222]]))[0]
    raw=load_raw(folder/f'{phone}_GnssLog.txt')
    der=pd.read_csv(folder/f'{phone}_derived.csv')
    ts=np.sort(raw.millisSinceGpsEpoch.unique());ix=np.searchsorted(ts,der.millisSinceGpsEpoch)-1
    der=der.loc[ix>=0].copy();der['millisSinceGpsEpoch']=ts[ix[ix>=0]]
    key=['millisSinceGpsEpoch','constellationType','svid','signalType']
    d=raw.drop_duplicates(key).merge(der.drop_duplicates(key),on=key,validate='one_to_one')
    d=d[d.Cn0DbHz.gt(20)&d.constellationType.isin([1,3,6])].copy()
    reference=pd.read_csv(ROOT/f'data/competition/baseline_locations_{split}.csv')
    reference=reference[reference.phone.eq(collection+'_'+phone)].sort_values('millisSinceGpsEpoch')
    ref_t=reference.millisSinceGpsEpoch.to_numpy()
    llh=reference[['latDeg','lngDeg','heightAboveWgs84EllipsoidM']].to_numpy();ref_xyz=ecef(llh)
    rt=d.millisSinceGpsEpoch.to_numpy();rx=np.column_stack([np.interp(rt,ref_t,ref_xyz[:,i]) for i in range(3)])
    sat=d[POS].to_numpy();rho=np.linalg.norm(sat-rx,axis=1);sat=rotate(sat,rho)
    rho=np.linalg.norm(sat-rx,axis=1);los=(sat-rx)/rho[:,None]
    d[['rx','ry','rz']]=rx;d[['ux','uy','uz']]=los;d['rho']=rho
    d['base_residual']=np.nan
    for (const,svid,signal),g in d.groupby(['constellationType','svid','signalType']):
        letter={1:'G',3:'R',6:'E'}[const];sv=f'{letter}{int(svid):02d}'
        if sv not in base['sv']:continue
        band='C5' if signal in ['GPS_L5','GAL_E5A'] else 'C1'
        if band not in base:continue
        j=list(base['sv']).index(sv);code=base[band][:,j]
        g=g.sort_values('millisSinceGpsEpoch').drop_duplicates('millisSinceGpsEpoch')
        if len(g)<3:continue
        epoch=float(g.millisSinceGpsEpoch.iloc[0]);t=(g.millisSinceGpsEpoch.to_numpy()-epoch)/1000
        bt=(base['time_ms']-epoch)/1000
        mask=np.isfinite(code)&(bt>=t.min()-30)&(bt<=t.max()+30)
        if mask.sum()<3:continue
        bt=bt[mask];code=code[mask]
        spline=CubicHermiteSpline(t,g[POS].to_numpy(),g[VEL].to_numpy(),extrapolate=True)
        sp=spline(bt);vel=spline(bt,1)
        rx_i=np.column_stack([np.interp(bt,t,g[k]) for k in ['rx','ry','rz']])
        rho_phone=np.linalg.norm(sp-rx_i,axis=1);rho_base=np.linalg.norm(sp-base_xyz,axis=1)
        # Base and rover observe different transmit epochs for the same receive time.
        sp+=vel*((rho_phone-rho_base)/LIGHT)[:,None]
        sp=rotate(sp,np.linalg.norm(sp-base_xyz,axis=1))
        geometric=np.linalg.norm(sp-base_xyz,axis=1)
        clock=np.interp(bt,t,g.satClkBiasM.to_numpy())
        residual=code+clock-geometric
        # Short robust smoothing suppresses reference-code noise while retaining drift.
        residual=median_filter(residual,size=3,mode='nearest')
        interpolated=np.interp(t,bt,residual,left=np.nan,right=np.nan)
        d.loc[g.index,'base_residual']=interpolated
    d=d[d.base_residual.notna()].copy()
    # Single-differenced atmospheric, orbit and satellite-clock errors.
    d['corrected']=d.rawPrM+d.satClkBiasM-d.base_residual
    rows=[]
    R=enu_matrix(float(reference.latDeg.mean()),float(reference.lngDeg.mean()))
    for t,g in d.groupby('millisSinceGpsEpoch'):
        A=-g[['ux','uy','uz']].to_numpy()@R.T
        y=(g.corrected-g.rho).to_numpy()
        sigma=np.maximum(g.rawPrUncM.to_numpy(),1.5)*np.maximum(.5,10**((35-g.Cn0DbHz.to_numpy())/40))
        fitted=solve(A,y,sigma,g.signalType.to_numpy())
        if fitted is None:continue
        shift,sd,residual=fitted
        latitude=np.interp(t,ref_t,llh[:,0]);longitude=np.interp(t,ref_t,llh[:,1]);height=np.interp(t,ref_t,llh[:,2])
        latitude+=np.degrees(shift[1]/6378137.);longitude+=np.degrees(shift[0]/(6378137.*np.cos(np.radians(latitude))))
        rows.append(dict(millisSinceGpsEpoch=int(t),latDeg=latitude,lngDeg=longitude,heightAboveWgs84EllipsoidM=height+shift[2],stde=sd[0],stdn=sd[1],stdu=sd[2],residual=residual,ns=len(g)))
    result=pd.DataFrame(rows)
    if not len(result):raise ValueError('No pseudorange solutions '+collection+'_'+phone)
    result.to_csv(out,index=False)
    return {'phone':collection+'_'+phone,'status':'ok','epochs':len(result),'median_sd':float(np.median(np.hypot(result.stde,result.stdn)))}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='both',choices=['train','test','both']);ap.add_argument('--workers',type=int,default=12);ap.add_argument('--limit',type=int);args=ap.parse_args()
    items=[(s,f) for s in (['train','test'] if args.split=='both' else [args.split]) for f in sorted((ROOT/'data/competition'/s).glob('*/*')) if f.is_dir()]
    if args.limit:items=items[:args.limit]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(process,item) for item in items]):
            try:r=fut.result()
            except Exception as exc:r={'status':'error','error':str(exc)}
            results.append(r);print(json.dumps(r),flush=True)
    (ROOT/'artifacts/pseudorange_status.json').write_text(json.dumps(results,indent=2))
    if any(r['status']=='error' for r in results):raise SystemExit(1)
