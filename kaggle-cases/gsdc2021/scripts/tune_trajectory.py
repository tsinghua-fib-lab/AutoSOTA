"""Select physics parameters on calibration dates, then report reserved dates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from optimize_trajectory import run,ROOT
from metric import read_positions,evaluate
from map_matching import match

truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
# Reserve entire dates, including every phone and collection on each date.
holdout_dates=['2020-05-21','2020-06-11','2021-04-22','2021-04-15']
holdout=truth.collectionName.str[:10].isin(holdout_dates)
(ROOT/'artifacts/cv_partitions.json').write_text(json.dumps({'holdout_dates':holdout_dates,'holdout_collections':sorted(truth.loc[holdout,'collectionName'].unique()),'calibration_collections':sorted(truth.loc[~holdout,'collectionName'].unique()),'grouping':'date; all phones and drives on a date stay together'},indent=2))
configs=[]
for ps,ts,ds in [(1.,.05,.3),(2.,.05,.3),(4.,.05,.3),(2.,.1,.3),(4.,.1,.3),(4.,.2,.5),(8.,.1,.5),(4.,.05,0.)]:
    configs.append(dict(position_sigma=ps,tdcp_sigma=ts,doppler_sigma=ds,accel_sigma=1.5,position_huber=5.))
results=[]
for i,p in enumerate(configs):
    pred,diag=run('train',p,source='merged',workers=8)
    for alpha in [0.,.5,1.]:
        mapped=match(pred,truth,alpha=alpha,limit=12.,training=True,excluded_dates=holdout_dates) if alpha else pred
        calibration=mapped[~mapped.collectionName.str[:10].isin(holdout_dates)]
        score,_,_=evaluate(calibration,truth.loc[~holdout])
        results.append(dict(config=i,map_alpha=alpha,calibration_score_m=score,**p))
    pred.to_csv(ROOT/f'artifacts/tune_{i}_train.csv',index=False)
    print(json.dumps({'config':i,'calibration_results':results[-3:]}),flush=True)
pd.DataFrame(results).to_csv(ROOT/'artifacts/parameter_calibration.csv',index=False)
best=min(results,key=lambda r:r['calibration_score_m']);p=configs[best['config']]
pred=read_positions(ROOT/f"artifacts/tune_{best['config']}_train.csv")
mapped=match(pred,truth,alpha=best['map_alpha'],limit=12.,training=True,excluded_dates=holdout_dates) if best['map_alpha'] else pred
validation=mapped[mapped.collectionName.str[:10].isin(holdout_dates)]
validation_score,phones,_=evaluate(validation,truth.loc[holdout]);phones.to_csv(ROOT/'artifacts/holdout_scores.csv',index=False)
reference=read_positions(ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone/Google/baseline_locations_merged_train_12_31.csv')
reference=reference[reference.collectionName.str[:10].isin(holdout_dates)]
reference_score,_,_=evaluate(reference,truth.loc[holdout])
best.update(holdout_score_m=validation_score,reference_holdout_score_m=reference_score,map_limit_m=12.,source='merged',selection='calibration dates only; holdout dates excluded from road reference for both sets')
(ROOT/'artifacts/best_parameters.json').write_text(json.dumps(best,indent=2));mapped.to_csv(ROOT/'artifacts/fgo_selected_train.csv',index=False)
print(json.dumps(best),flush=True)
