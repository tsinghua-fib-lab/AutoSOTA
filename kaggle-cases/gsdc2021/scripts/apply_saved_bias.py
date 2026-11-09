"""Apply an already calibrated bias model without re-fitting or reading labels."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
from bias_correction import prepare,apply
from metric import read_positions,validate_submission,KEY,COORD
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('parameters');ap.add_argument('output');a=ap.parse_args()
    p=json.loads(Path(a.parameters).read_text());test=prepare(read_positions(a.input))
    urban=test.collectionName.str.contains('-SJC-')
    fitted=pd.concat([test.loc[urban],apply(test.loc[~urban],np.asarray(p['coefficients']),p['shrink'])],ignore_index=True)
    sample=read_positions(ROOT/'data/competition/sample_submission.csv')
    sub=sample[KEY].merge(fitted[KEY+COORD],on=KEY,how='left',validate='one_to_one');validate_submission(sub,sample)
    sub.to_csv(a.output,index=False)
    Path(a.output+'.json').write_text(json.dumps({'input':a.input,'parameters':a.parameters,'parameter_values':p},indent=2))
