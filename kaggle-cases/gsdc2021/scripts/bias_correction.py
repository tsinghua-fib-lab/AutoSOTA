"""Date-OOF calibration of bounded physical antenna offset and timestamp bias."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from metric import read_positions,evaluate,validate_submission,KEY,COORD
ROOT=Path(__file__).resolve().parents[1]

def prepare(pred,truth=None):
    d=pred.copy()
    if 'collectionName' not in d:
        d[['collectionName','phoneName']]=d.phone.str.rsplit('_',n=1,expand=True)
    frames=[]
    for phone,g in d.groupby('phone',sort=False):
        g=g.sort_values('millisSinceGpsEpoch').copy();t=g.millisSinceGpsEpoch.to_numpy()/1000
        radius=6378137.;cos=np.cos(np.radians(g.latDeg.mean()))
        xy=np.column_stack([np.radians(g.lngDeg.to_numpy())*radius*cos,np.radians(g.latDeg.to_numpy())*radius])
        vel=np.gradient(gaussian_filter1d(xy,2,axis=0),t,axis=0)
        speed=np.linalg.norm(vel,axis=1);direction=vel/np.maximum(speed[:,None],.5)
        g['ve']=vel[:,0];g['vn']=vel[:,1];g['ue']=direction[:,0];g['un']=direction[:,1]
        frames.append(g)
    d=pd.concat(frames,ignore_index=True)
    if truth is not None:
        d=d.merge(truth[KEY+COORD],on=KEY,suffixes=('','_gt'),validate='one_to_one')
        d['target_e']=np.radians(d.lngDeg_gt-d.lngDeg)*6378137.*np.cos(np.radians(d.latDeg))
        d['target_n']=np.radians(d.latDeg_gt-d.latDeg)*6378137.
    return d

def design(d):
    n=len(d);A=np.zeros((n,2,5))
    A[:,0,0]=1;A[:,1,1]=1
    A[:,0,2]=d.ue;A[:,1,2]=d.un
    A[:,0,3]=-d.un;A[:,1,3]=d.ue
    A[:,0,4]=d.ve;A[:,1,4]=d.vn
    return A

def fit(d,huber=1.5,ridge=.1):
    # Each complete phone receives equal total weight, like the competition.
    w=1/d.groupby('phone').phone.transform('size').to_numpy()
    A=design(d);Y=d[['target_e','target_n']].to_numpy();b=np.zeros(5)
    prior=np.array([1.,1.,1.,1.,100.])*ridge
    for i in range(12):
        residual=np.linalg.norm(Y-A@b,axis=1)
        W=w*np.minimum(1,huber/np.maximum(residual,1e-6))
        lhs=np.einsum('ndi,ndj,n->ij',A,A,W)+np.diag(prior)
        rhs=np.einsum('ndi,nd,n->i',A,Y,W)
        b=np.linalg.solve(lhs,rhs)
    return b

def apply(d,b,shrink=1.):
    d=d.copy();shift=design(d)@b*shrink
    shift*=np.minimum(1,3/np.maximum(np.linalg.norm(shift,axis=1),1e-8))[:,None]
    d['latDeg']+=np.degrees(shift[:,1]/6378137.)
    d['lngDeg']+=np.degrees(shift[:,0]/(6378137.*np.cos(np.radians(d.latDeg))))
    return d

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',default='joint_road');args=ap.parse_args()
    folder=ROOT/'artifacts/experiments'/args.experiment
    best=json.loads((folder/'best.json').read_text())
    truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
    pred=read_positions(folder/f"train_{best['config']}.csv")
    dates=json.loads((ROOT/'artifacts/cv_partitions.json').read_text())['holdout_dates']
    all_data=prepare(pred,truth)
    data=all_data[~all_data.collectionName.str.contains('-SJC-')].copy()
    cal=data[~data.collectionName.str[:10].isin(dates)].copy()
    val=data[data.collectionName.str[:10].isin(dates)].copy()
    results=[];candidates={}
    for huber in [.5,1.5,3.]:
        for ridge in [.1,1.,10.]:
            folds=[]
            for date in sorted(cal.collectionName.str[:10].unique()):
                train=cal[cal.collectionName.str[:10]!=date];test=cal[cal.collectionName.str[:10]==date]
                b=fit(train,huber,ridge)
                for shrink in [.5,1.]:
                    fitted=apply(test,b,shrink);fitted['shrink']=shrink;folds.append(fitted)
            oof=pd.concat(folds,ignore_index=True)
            for shrink in [.5,1.]:
                d=oof[oof.shrink==shrink].copy()
                score,_,_=evaluate(d,truth[truth.phone.isin(d.phone.unique())])
                i=len(results);r=dict(config=i,huber=huber,ridge=ridge,shrink=shrink,calibration_oof_m=score)
                results.append(r);candidates[i]=d;print(json.dumps(r),flush=True)
    baseline=evaluate(cal,truth[truth.phone.isin(cal.phone.unique())])[0]
    best_bias=min(results,key=lambda r:r['calibration_oof_m'])
    if best_bias['calibration_oof_m']>=baseline:
        best_bias['shrink']=0.
    b=fit(cal,best_bias['huber'],best_bias['ridge'])
    corrected=apply(val,b,best_bias['shrink'])
    vs,_,_=evaluate(corrected,truth[truth.phone.isin(val.phone.unique())])
    best_bias.update(coefficients=b.tolist(),baseline_calibration_m=baseline,holdout_non_sjc_m=vs,source_experiment=args.experiment)
    (folder/'bias_parameters.json').write_text(json.dumps(best_bias,indent=2))
    pd.DataFrame(results).to_csv(folder/'bias_calibration.csv',index=False)
    test=prepare(read_positions(ROOT/f'submissions/v2_{args.experiment}.csv'))
    urban=test.collectionName.str.contains('-SJC-')
    fitted=pd.concat([test.loc[urban],apply(test.loc[~urban],b,best_bias['shrink'])],ignore_index=True)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(fitted[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    sub.to_csv(ROOT/f'submissions/v2_{args.experiment}_bias.csv',index=False)
    print(json.dumps({'best':best_bias}),flush=True)
