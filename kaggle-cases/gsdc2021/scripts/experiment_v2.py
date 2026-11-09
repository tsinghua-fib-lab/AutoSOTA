"""Resumable physics experiments; calibration selection before holdout scoring."""
from pathlib import Path
import json
import argparse
import time
import pandas as pd
from trajectory_v2 import run
from metric import read_positions,evaluate,validate_submission,KEY,COORD
from map_matching import match
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='joint_road');ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    folder=ROOT/'artifacts/experiments'/args.name;folder.mkdir(parents=True,exist_ok=True)
    truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
    dates=json.loads((ROOT/'artifacts/cv_partitions.json').read_text())['holdout_dates']
    holdout=truth.collectionName.str[:10].isin(dates)
    base=dict(position_sigma=16.,tdcp_sigma=.1,doppler_sigma=.5,accel_sigma=1.5,position_huber=5.,excluded_dates=dates,iterations=10)
    configs=[]
    if args.name=='joint_road':
        for ps in [8.,16.]:
            for rs in [.2,.5,1.,2.,5.]:configs.append({**base,'position_sigma':ps,'joint_road_sigma':rs})
    elif args.name=='reliability':
        for ps in [8.,16.]:
            for hub in [.05,.1,.2,.4]:configs.append({**base,'position_sigma':ps,'tdcp_huber':hub})
        for ps in [6.,10.]:configs.append({**base,'position_sigma':ps,'normalize_position_phones':True})
        for phones in [['SamsungS20Ultra','Mi8'],['Pixel4','SamsungS20Ultra','Mi8'],['SamsungS20Ultra','Pixel4XL','Mi8']]:
            configs.append({**base,'velocity_phones':phones})
    elif args.name=='combined':
        for hub in [.05,.1,.2]:
            for rs in [.5,1.,2.,5.]:configs.append({**base,'tdcp_huber':hub,'joint_road_sigma':rs})
        for hub in [.05,.1,.2]:
            configs.append({**base,'tdcp_huber':hub,'joint_road_sigma':2.,'tdcp_kernel':'cauchy','iterations':16})
    elif args.name=='directional':
        for fill in [False,True]:
            for rs in [.5,1.,2.,5.]:
                configs.append({**base,'tdcp_huber':.05,'joint_road_sigma':rs,'joint_road_direction':'signed','heading_fill':fill})
    elif args.name=='midpoint':
        for ps in [8.,16.]:
            for cov in [False,True]:
                for hub in [.05,.2]:
                    configs.append({**base,'position_sigma':ps,'tdcp_huber':hub,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dir':'velocity_midpoint','tdcp_covariance':cov})
    elif args.name=='lag5':
        for ps in [8.,16.]:
            for dirs in [['velocity_lag5'],['velocity_midpoint','velocity_lag5']]:
                for sigma in [.1,.2]:
                    for hub in [.05,.2]:
                        configs.append({**base,'position_sigma':ps,'tdcp_sigma':sigma,'tdcp_huber':hub,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':dirs})
    elif args.name=='multiscale':
        for ps in [8.,16.,24.]:
            for dirs in [['velocity_midpoint','velocity_lag10'],['velocity_midpoint','velocity_lag5','velocity_lag10']]:
                for sigma in [.1,.2]:
                    configs.append({**base,'position_sigma':ps,'tdcp_sigma':sigma,'tdcp_huber':.05,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':dirs})
    elif args.name=='pseudorange':
        for ps in [8.,16.]:
            for code_sigma in [2.,5.,10.,20.]:
                configs.append({**base,'position_sigma':ps,'tdcp_huber':.05,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':['velocity_midpoint','velocity_lag5'],'pseudorange_sigma':code_sigma})
    elif args.name=='code_strength':
        for ps in [16.,24.,32.]:
            for code_sigma in [5.,10.,20.]:
                for sigma in [.05,.1]:
                    configs.append({**base,'position_sigma':ps,'tdcp_sigma':sigma,'tdcp_huber':.05,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':['velocity_midpoint','velocity_lag5'],'pseudorange_sigma':code_sigma})
    elif args.name=='code_bias_prior':
        for ps in [16.,32.]:
            for shrink in [.5,1.]:
                for code_sigma in [5.,10.,20.]:
                    configs.append({**base,'position_sigma':ps,'tdcp_huber':.05,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':['velocity_midpoint','velocity_lag5'],'pseudorange_sigma':code_sigma,'bias_model':'artifacts/experiments/pseudorange/bias_by_date.json','bias_shrink':shrink})
    elif args.name=='convergence':
        for iterations in [10,20,30]:
            for huber in [.02,.05,.1]:
                configs.append({**base,'iterations':iterations,'tdcp_huber':huber,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':['velocity_midpoint','velocity_lag5'],'pseudorange_sigma':10.})
    elif args.name=='stops':
        for speed in [.2,.5,1.]:
            for sigma in [.02,.1]:
                for window in [5,11]:
                    configs.append({**base,'tdcp_huber':.05,'joint_road_sigma':2.,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':['velocity_midpoint','velocity_lag5'],'pseudorange_sigma':10.,'stop_speed':speed,'stop_sigma':sigma,'stop_window':window})
    elif args.name=='bias_prior':
        for ps in [8.,16.]:
            for shrink in [.5,1.]:
                for rs in [.5,2.]:
                    configs.append({**base,'position_sigma':ps,'tdcp_huber':.05,'joint_road_sigma':rs,'joint_road_direction':'signed','heading_fill':True,'velocity_dirs':['velocity_midpoint','velocity_lag5'],'bias_model':'artifacts/experiments/lag5/bias_by_date.json','bias_shrink':shrink})
    else:raise ValueError(args.name)
    results=[]
    for i,p in enumerate(configs):
        checkpoint=folder/f'config_{i}.json';output=folder/f'train_{i}.csv'
        if checkpoint.exists() and output.exists():r=json.loads(checkpoint.read_text())
        else:
            start=time.time();pred,diag=run('train',p,source='merged',workers=args.workers)
            if not p.get('joint_road_sigma',0):
                pred=match(pred,truth,training=True,excluded_dates=dates,mode='road',limit=50.,drift_sigma=.1,road_sigma=.5,anchor_sigma=20.)
            cal=pred[~pred.collectionName.str[:10].isin(dates)]
            score,phones,_=evaluate(cal,truth.loc[~holdout])
            regions={}
            for city in ['SJC','other']:
                mask=cal.collectionName.str.contains('-SJC-')==(city=='SJC')
                gt=truth[truth.phone.isin(cal.loc[mask,'phone'].unique())]
                regions[city]=evaluate(cal.loc[mask],gt)[0]
            pred.to_csv(output,index=False)
            r=dict(config=i,calibration_m=score,regions=regions,parameters=p,seconds=time.time()-start)
            checkpoint.write_text(json.dumps(r,indent=2))
        results.append(r);print(json.dumps(r),flush=True)
        (folder/'progress.json').write_text(json.dumps({'completed':len(results),'total':len(configs),'latest':r},indent=2))
    best=min(results,key=lambda r:r['calibration_m'])
    pred=read_positions(folder/f"train_{best['config']}.csv")
    val=pred[pred.collectionName.str[:10].isin(dates)]
    best['holdout_m'],phones,_=evaluate(val,truth.loc[holdout]);phones.to_csv(folder/'holdout_phones.csv',index=False)
    (folder/'best.json').write_text(json.dumps(best,indent=2))
    p=best['parameters'];pred,diag=run('test',p,source='merged',workers=args.workers)
    if not p.get('joint_road_sigma',0):pred=match(pred,truth,training=False,mode='road',limit=50.,drift_sigma=.1,road_sigma=.5,anchor_sigma=20.)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(pred[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    sub.to_csv(ROOT/f'submissions/v2_{args.name}.csv',index=False)
    print(json.dumps({'best':best,'status':'complete'}),flush=True)
