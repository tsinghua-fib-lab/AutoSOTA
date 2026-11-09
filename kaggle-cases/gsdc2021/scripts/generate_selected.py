"""Generate a submission from the saved, calibration-selected parameters."""
from pathlib import Path
import argparse
import json
from optimize_trajectory import run
from map_matching import match
from metric import read_positions, validate_submission, KEY, COORD

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['merged','recomputed_merged'],default='merged');ap.add_argument('--name',default='fgo_selected_replay');args=ap.parse_args()
    p=json.loads((ROOT/'artifacts/best_parameters.json').read_text())
    road=json.loads((ROOT/'artifacts/best_road_parameters.json').read_text())
    road={k:road[k] for k in ['mode','alpha','limit','drift_sigma','road_sigma','anchor_sigma']}
    pred,diagnostics=run('test',p,source=args.source)
    pred.to_csv(ROOT/f'artifacts/{args.name}_test.csv',index=False)
    truth=read_positions(ROOT/'artifacts/ground_truth_train.csv')
    pred=match(pred,truth,training=False,**road)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(pred[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    path=ROOT/f'submissions/{args.name}.csv';sub.to_csv(path,index=False)
    (ROOT/f'artifacts/{args.name}.json').write_text(json.dumps({'source':args.source,'trajectory_parameters':p,'road_parameters':road,'diagnostics':diagnostics,'submission':str(path)},indent=2))
    print(str(path),flush=True)
