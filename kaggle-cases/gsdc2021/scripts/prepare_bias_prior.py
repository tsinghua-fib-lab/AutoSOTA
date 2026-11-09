"""Freeze calibration-date-excluded coefficients for an in-graph bias prior."""
import argparse,json
from bias_correction import ROOT,prepare,fit
from metric import read_positions

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',default='pseudorange');a=ap.parse_args()
    folder=ROOT/'artifacts/experiments'/a.experiment
    best=json.loads((folder/'best.json').read_text())
    selected=json.loads((folder/'bias_parameters.json').read_text())
    dates=json.loads((ROOT/'artifacts/cv_partitions.json').read_text())['holdout_dates']
    d=prepare(read_positions(folder/f"train_{best['config']}.csv"),read_positions(ROOT/'artifacts/ground_truth_train.csv'))
    d=d[~d.collectionName.str.contains('-SJC-')&~d.collectionName.str[:10].isin(dates)]
    p={'default':selected['coefficients'],'by_date':{},'source_experiment':a.experiment,'excluded_dates':dates}
    for date in sorted(d.collectionName.str[:10].unique()):
        p['by_date'][date]=fit(d[d.collectionName.str[:10]!=date],selected['huber'],selected['ridge']).tolist()
    (folder/'bias_by_date.json').write_text(json.dumps(p,indent=2))
    print(json.dumps(p))
