"""Extend the original calibration grid; never select using held-out scores."""
from pathlib import Path
import json
from metric import read_positions,evaluate,validate_submission,KEY,COORD
from optimize_trajectory import run
from map_matching import match
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
    partition=json.loads((ROOT/'artifacts/cv_partitions.json').read_text())
    dates=partition['holdout_dates'];holdout=truth.collectionName.str[:10].isin(dates)
    road=json.loads((ROOT/'artifacts/best_road_parameters.json').read_text())
    rp={k:road[k] for k in ['mode','alpha','limit','drift_sigma','road_sigma','anchor_sigma']}
    configs=[(8.,.1,.5),(8.,.05,.3),(16.,.05,.3),(32.,.05,.3),(16.,.1,.5),(32.,.1,.5),(64.,.1,.5),(16.,.2,.5),(32.,.2,.5)]
    results=[]
    for i,(ps,td,dp) in enumerate(configs):
        p=dict(position_sigma=ps,tdcp_sigma=td,doppler_sigma=dp,accel_sigma=1.5,position_huber=5.)
        pred,diag=run('train',p,source='merged')
        pred.to_csv(ROOT/f'artifacts/refine_{i}_unmapped_train.csv',index=False)
        fitted=match(pred,truth,training=True,excluded_dates=dates,**rp)
        cal=fitted[~fitted.collectionName.str[:10].isin(dates)]
        score,_,_=evaluate(cal,truth.loc[~holdout])
        r=dict(config=i,calibration_score_m=score,**p)
        results.append(r);print(json.dumps(r),flush=True)
        fitted.to_csv(ROOT/f'artifacts/refine_{i}_train.csv',index=False)
    import pandas as pd
    pd.DataFrame(results).to_csv(ROOT/'artifacts/refined_calibration.csv',index=False)
    best=min(results,key=lambda r:r['calibration_score_m'])
    fitted=read_positions(ROOT/f"artifacts/refine_{best['config']}_train.csv")
    vs,phones,_=evaluate(fitted[fitted.collectionName.str[:10].isin(dates)],truth.loc[holdout])
    best.update(holdout_score_m=vs,road_parameters=rp,selection='calibration dates only; same held-out dates as original grid')
    (ROOT/'artifacts/refined_parameters.json').write_text(json.dumps(best,indent=2))
    phones.to_csv(ROOT/'artifacts/refined_holdout_scores.csv',index=False)
    pred,diag=run('test',best,source='merged')
    fitted=match(pred,truth,training=False,**rp)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(fitted[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    sub.to_csv(ROOT/'submissions/fgo_refined_selected.csv',index=False)
    print(json.dumps(best),flush=True)
