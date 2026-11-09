"""Apply frozen state-dependent weights/biases to cached test predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from conditional_ensemble import ROOT,GROUPS,feature,interpolate_weights,project


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'artifacts/conditional-ensemble/frozen_config.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    models=json.loads((ROOT/'configs/champion_models.json').read_text())
    if config['models']!=[m['id'] for m in models]:raise ValueError('Model order differs')
    cache=ROOT/'artifacts/test-inference';ids=np.load(cache/'ids.npy');values=feature(cache)
    if len(values)!=len(ids):raise ValueError('State feature rows differ')
    humidity=np.load(cache/'humidity.npy',mmap_mode='r')
    arrays=[np.load(cache/(m['id']+'.npy'),mmap_mode='r') for m in models]
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1)
    columns=sample.columns[1:];weights=sample.select(columns).to_numpy()[0]
    sample_ids=pl.read_csv(ROOT/'data/competition/sample_submission.csv',columns=['sample_id'])['sample_id'].to_numpy()
    if not np.array_equal(ids,sample_ids) or len(columns)!=368:raise ValueError('Submission schema/IDs differ')
    baseline=np.zeros((len(ids),368),np.float64)
    for name,start,end in GROUPS:
        for weight,prediction in zip(config['base_group_weights'][name],arrays):baseline[:,start:end]+=weight*prediction[:,start:end]
    baseline=project(baseline,humidity,weights);recipe=config['recipe'];centers=np.asarray(config['centers'])
    if recipe['kind']=='covariance':
        gated=np.zeros_like(baseline);reg=str(recipe['regularization'])
        for name,start,end in GROUPS:
            row_weights=interpolate_weights(values,centers,np.asarray(config['conditional_weights'][reg][name]))
            if not np.allclose(row_weights.sum(axis=1),1,rtol=0,atol=1e-9):raise ValueError('Weights do not sum to one')
            for i,prediction in enumerate(arrays):gated[:,start:end]+=row_weights[:,i,None]*prediction[:,start:end]
        result=(1-recipe['strength'])*baseline+recipe['strength']*gated
    elif recipe['kind']=='mean_bias':result=baseline+recipe['strength']*interpolate_weights(values,centers,np.asarray(config['bias']))
    else:result=baseline
    result=project(result,humidity,weights)
    if not np.isfinite(result).all():raise FloatingPointError('Non-finite candidate')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial')
    pl.DataFrame({'sample_id':ids,**{c:result[:,i] for i,c in enumerate(columns)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output)
    digest=hashlib.sha256()
    with args.output.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'rows':len(ids),'targets':368,'config':str(args.config.resolve()),'recipe':recipe,'sha256':digest.hexdigest(),'weights_applied_once':True,'test_labels_used':False},indent=2))
    print('Conditional submission ready:',args.output,flush=True)


if __name__=='__main__':main()
