"""Calibrate automatic road constraints without own-date or held-out labels."""
from pathlib import Path
import json
import pandas as pd
from metric import read_positions,evaluate,validate_submission,KEY,COORD
from map_matching import match
ROOT=Path(__file__).resolve().parents[1]
truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
params=json.loads((ROOT/'artifacts/best_parameters.json').read_text());partition=json.loads((ROOT/'artifacts/cv_partitions.json').read_text())
dates=partition['holdout_dates'];holdout=truth.collectionName.str[:10].isin(dates)
pred=read_positions(ROOT/f"artifacts/tune_{params['config']}_train.csv")
results=[]
configs=[{'mode':'snap','alpha':1.,'limit':limit} for limit in [20.,35.,50.]]+[
    {'mode':'road','alpha':1.,'limit':limit,'drift_sigma':drift,'road_sigma':.5,'anchor_sigma':20.}
    for limit,drift in [(20.,.02),(35.,.02),(50.,.02),(35.,.05),(50.,.05),(50.,.1)]]
for i,p in enumerate(configs):
    fitted=match(pred,truth,training=True,excluded_dates=dates,**p)
    cal=fitted[~fitted.collectionName.str[:10].isin(dates)]
    score,_,_=evaluate(cal,truth.loc[~holdout]);results.append({'config':i,'calibration_score_m':score,**p})
    fitted.to_csv(ROOT/f'artifacts/road_{i}_train.csv',index=False);print(json.dumps(results[-1]),flush=True)
best=min(results,key=lambda r:r['calibration_score_m']);p=configs[best['config']]
fitted=read_positions(ROOT/f"artifacts/road_{best['config']}_train.csv")
val=fitted[fitted.collectionName.str[:10].isin(dates)];vs,phones,_=evaluate(val,truth.loc[holdout]);best['holdout_score_m']=vs
best['hyperparameters_selected_on']='calibration dates only';(ROOT/'artifacts/best_road_parameters.json').write_text(json.dumps(best,indent=2));pd.DataFrame(results).to_csv(ROOT/'artifacts/road_calibration.csv',index=False);phones.to_csv(ROOT/'artifacts/road_holdout_scores.csv',index=False)
test=read_positions(ROOT/'artifacts/fgo_cv_selected_test.csv')
fitted=match(test,truth,training=False,**p);sample=read_positions(ROOT/'data/competition/sample_submission.csv')
sub=sample[KEY].merge(fitted[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample);sub.to_csv(ROOT/'submissions/fgo_road_selected.csv',index=False)
print(json.dumps(best),flush=True)
