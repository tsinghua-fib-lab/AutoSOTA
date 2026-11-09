"""Evaluate a frozen replay on reserved dates; never select parameters here."""
import argparse,json
import numpy as np
import pandas as pd
from trajectory_v2 import ROOT,run,RELEASE
from bias_correction import prepare,apply
from metric import read_positions,evaluate

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',default='pseudorange');ap.add_argument('--source',default='raw',choices=['raw','published']);a=ap.parse_args()
    folder=ROOT/'artifacts/experiments'/a.experiment
    selected=json.loads((folder/'best.json').read_text());p=selected['parameters']
    dates=json.loads((ROOT/'artifacts/cv_partitions.json').read_text())['holdout_dates']
    truth=read_positions(ROOT/'artifacts/ground_truth_train.csv');truth=truth[truth.collectionName.str[:10].isin(dates)]
    pred,diag=run('train',p,'recomputed_merged' if a.source=='raw' else 'merged',workers=12,collections=set(truth.collectionName))
    if not p.get('bias_model'):
        bp=json.loads((folder/'bias_parameters.json').read_text());pred=prepare(pred)
        mapped={x['collection'] for x in diag if x['road_factors']>0};urban=pred.collectionName.isin(mapped)
        pred=pd.concat([pred[urban],apply(pred[~urban],np.asarray(bp['coefficients']),bp['shrink'])],ignore_index=True)
    output=ROOT/'artifacts/validation';output.mkdir(exist_ok=True)
    frames={'selected':pred,'google':read_positions(ROOT/'data/competition/baseline_locations_train.csv'),'raw_ppk':read_positions(ROOT/'artifacts/rtklib_merged_train.csv'),'published_ppk':read_positions(RELEASE/'baseline_locations_merged_train_12_31.csv')}
    result={'experiment':a.experiment,'source':a.source,'dates':dates,'interpretation':'Repeatedly inspected development holdout, not a fresh blind test','scores':{}}
    for name,d in frames.items():
        d=d[d.phone.isin(truth.phone)];score,phones,errors=evaluate(d,truth)
        result['scores'][name]=score;phones.to_csv(output/f'{a.experiment}_{a.source}_{name}_phones.csv',index=False)
        if name=='selected':
            d.to_csv(output/f'{a.experiment}_{a.source}_holdout.csv',index=False)
            errors.to_csv(output/f'{a.experiment}_{a.source}_errors.csv',index=False)
    (output/f'{a.experiment}_{a.source}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
