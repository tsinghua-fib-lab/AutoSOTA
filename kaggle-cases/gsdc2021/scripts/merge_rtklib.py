"""Execute the author's unchanged phone-fusion algorithm on raw replay outputs."""
from pathlib import Path
import argparse
import os
import json
import numpy as np
from metric import read_positions, evaluate, validate_submission, KEY, COORD, distances

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone'

def merge(split):
    work = ROOT/'artifacts/rtklib_merge'/split
    work.mkdir(parents=True, exist_ok=True)
    d = read_positions(ROOT/f'artifacts/rtklib_{split}.csv')
    # The original script reads by column position. North SD precedes east SD
    # in RTKLIB's output, despite reversed labels in the author's CSV header.
    cols = ['collectionName','phoneName','millisSinceGpsEpoch','latDeg','lngDeg',
            'heightAboveWgs84EllipsoidM','stdn','stde','stdu','Q','ns','phone']
    d['Q'] = np.rint(d.Q).astype(int)
    d['ns'] = np.rint(d.ns).astype(int)
    d[cols].to_csv(work/'phone_ppk.csv', index=False)
    original = (RELEASE/'python/mergePhones.py').read_text()
    original = original.replace("datapath = r'C:\\gps\\rtklib\\bins\\demo5_b34e_phone\\Google'", f'datapath = {str(work)!r}')
    original = original.replace("DATA_SET = 'test'", f'DATA_SET = {split!r}')
    original = original.replace("BASELINE_PPK_FILE = 'baseline_locations_test_PPK.csv'", "BASELINE_PPK_FILE = 'phone_ppk.csv'")
    original = original.replace("BASELINE_PPP_FILE = 'baseline_locations_' + DATA_SET + '.csv'", f'BASELINE_PPP_FILE = {str(ROOT/f"data/competition/baseline_locations_{split}.csv")!r}')
    (work/'original_merge_port.py').write_text(original)
    old = Path.cwd()
    try:
        exec(compile(original, str(work/'original_merge_port.py'), 'exec'), {'__name__':'__main__'})
    finally:
        os.chdir(old)
    fitted = read_positions(work/f'baseline_locations_merged_PPK_{split}.csv')
    fitted.to_csv(ROOT/f'artifacts/rtklib_merged_{split}.csv', index=False)
    ref_name = f'baseline_locations_merged_{split}_12_31.csv' if split=='train' else 'baseline_locations_merged_test_12_30.csv'
    ref = read_positions(RELEASE/'Google'/ref_name)
    comparison = ref[KEY+COORD].merge(fitted[KEY+COORD],on=KEY,suffixes=('_ref',''),validate='one_to_one')
    error = distances(comparison[COORD].to_numpy(),comparison[['latDeg_ref','lngDeg_ref']].to_numpy())
    result = {'split':split,'rows':len(fitted),'reference_difference_m':{'mean':float(error.mean()),'p50':float(np.median(error)),'p95':float(np.quantile(error,.95)),'max':float(error.max())}}
    if split=='train':
        score,phones,_ = evaluate(fitted,read_positions(ROOT/'artifacts/ground_truth_train.csv'))
        result['train_score_m'] = score
        phones.to_csv(ROOT/'artifacts/rtklib_recomputed_phones.csv',index=False)
    else:
        sample = read_positions(ROOT/'data/competition/sample_submission.csv')
        sub = sample[KEY].merge(fitted[KEY+COORD],on=KEY,how='left',validate='one_to_one')
        validate_submission(sub,sample)
        sub.to_csv(ROOT/'submissions/rtklib_raw_reproduced.csv',index=False)
    (ROOT/f'artifacts/rtklib_merged_{split}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','test','both'],default='both');args=ap.parse_args()
    for split in ['train','test'] if args.split=='both' else [args.split]:merge(split)
