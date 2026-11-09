"""Apply a frozen small residual calibration to a weighted binary-mask baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from residual_calibration import ROOT,make_features,project


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'artifacts/residual-calibration/frozen_config.json');parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());baseline=pl.read_parquet(args.baseline)
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);columns=sample.columns[1:];weights=sample.select(columns).to_numpy()[0]
    ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
    if baseline.columns!=sample.columns or not np.array_equal(ids,baseline['sample_id'].to_numpy()):raise ValueError('Schema/IDs differ')
    if not np.isin(weights,[0,1]).all():raise ValueError('Non-binary masks require explicit deweighting')
    p=baseline.select(columns).to_numpy()
    names=[f'state_t_{i}' for i in range(60)]+[f'{name}_{i}' for name in ['state_q0001','state_q0002','state_q0003'] for i in range(48,60)]+['state_ps','pbuf_LHFLX','pbuf_SHFLX','pbuf_SOLIN']
    raw=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=names).select(names).to_numpy()
    rh=np.load(ROOT/'artifacts/test-inference/rh-raw.npy',mmap_mode='r')
    features=make_features(raw,names,rh,p);z=np.column_stack([np.ones(len(p)),np.clip((features-np.asarray(config['feature_mean']))/np.asarray(config['feature_std']),-5,5)])
    recipe=config['recipe']
    if recipe['strength']:
        fit=config['fits'][str(recipe['regularization'])];scale=np.asarray(config['target_std']);pn=(p-np.asarray(config['prediction_mean']))/scale
        correction=(np.clip(z@np.asarray(fit['coefficient']),-.05,.05)+pn*np.asarray(fit['slopes'])[None])*scale
        p=p+recipe['strength']*correction
    humidity=np.load(ROOT/'artifacts/test-inference/humidity.npy',mmap_mode='r');p=project(p,humidity,weights)
    if not np.isfinite(p).all():raise FloatingPointError('Non-finite residual candidate')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial')
    pl.DataFrame({'sample_id':ids,**{c:p[:,i] for i,c in enumerate(columns)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output)
    h=hashlib.sha256()
    with args.output.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'rows':len(ids),'targets':368,'config':str(args.config.resolve()),'baseline':str(args.baseline.resolve()),'recipe':recipe,'sha256':h.hexdigest(),'weights_applied_once':True,'test_labels_used':False},indent=2))
    print('Residual submission ready:',args.output,flush=True)


if __name__=='__main__':main()
