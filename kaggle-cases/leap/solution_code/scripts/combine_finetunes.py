"""Mix qualified fine-tunes into the frozen conditional ensemble and calibration."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from conditional_ensemble import ROOT,GROUPS,feature,interpolate_weights,project
from residual_calibration import apply_calibration


def conditional_weights(kind,config):
    cache=ROOT/('data/finetune-holdout' if kind=='validation' else 'artifacts/test-inference')
    values=feature(cache);recipe=config['recipe'];centers=np.asarray(config['centers']);result={}
    for name,_,_ in GROUPS:
        base=np.asarray(config['base_group_weights'][name]);weights=np.broadcast_to(base,(len(values),len(base)))
        if recipe['kind']=='covariance':
            gated=interpolate_weights(values,centers,np.asarray(config['conditional_weights'][str(recipe['regularization'])][name]));weights=(1-recipe['strength'])*weights+recipe['strength']*gated
        result[name]=weights
    return result


def raw_ensemble(kind,config,coefficients):
    path=ROOT/'artifacts/overnight'/f'conditional-{kind}-raw.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    n=len(next(iter(coefficients.values())));result=np.zeros((n,368),np.float64)
    for i,model in enumerate(config['models']):
        file=ROOT/'artifacts/champion-validation'/model/'predictions.npy' if kind=='validation' else ROOT/'artifacts/test-inference'/(model+'.npy')
        p=np.load(file,mmap_mode='r')
        for name,start,end in GROUPS:result[:,start:end]+=coefficients[name][:,i,None]*p[:,start:end]
    if config['recipe']['kind']=='mean_bias':
        cache=ROOT/('data/finetune-holdout' if kind=='validation' else 'artifacts/test-inference');result+=config['recipe']['strength']*interpolate_weights(feature(cache),np.asarray(config['centers']),np.asarray(config['bias']))
    np.save(path,result);return np.load(path,mmap_mode='r')


def replacement(kind,parent,child,fraction,coefficients,conditional):
    index=conditional['models'].index(parent)
    old_file=ROOT/'artifacts/champion-validation'/parent/'predictions.npy' if kind=='validation' else ROOT/'artifacts/test-inference'/(parent+'.npy')
    old=np.load(old_file,mmap_mode='r');new=np.load(ROOT/'artifacts/extra-inference'/child/kind/'predictions.npy',mmap_mode='r');delta=np.zeros_like(new)
    for name,start,end in GROUPS:delta[:,start:end]=fraction*coefficients[name][:,index,None]*(new[:,start:end]-old[:,start:end])
    return delta


def input_context(kind):
    if kind=='validation':
        cache=ROOT/'data/finetune-holdout';raw=np.load(cache/'raw_features.npy',mmap_mode='r')
        with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
    else:
        cache=ROOT/'artifacts/test-inference';columns=[f'state_t_{i}' for i in range(60)]+[f'{name}_{i}' for name in ['state_q0001','state_q0002','state_q0003'] for i in range(48,60)]+['state_ps','pbuf_LHFLX','pbuf_SHFLX','pbuf_SOLIN'];raw=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=columns).select(columns).to_numpy()
    return raw,columns,np.load(cache/'rh-raw.npy',mmap_mode='r'),np.load(cache/'humidity.npy',mmap_mode='r')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--recipe',type=Path);parser.add_argument('--children',nargs='+');parser.add_argument('--build',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    conditional=json.loads((ROOT/'artifacts/conditional-ensemble/frozen_config.json').read_text());ridge=json.loads((ROOT/'artifacts/residual-calibration/frozen_config.json').read_text());sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);columns=sample.columns[1:];weights=sample.select(columns).to_numpy()[0]
    with threadpool_limits(limits=8):
        if not args.build:
            coeff=conditional_weights('validation',conditional);raw_base=raw_ensemble('validation',conditional,coeff);raw,names,rh,humidity=input_context('validation');y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r')
            def finish(p):return project(apply_calibration(project(p,humidity,weights),raw,names,rh,ridge),humidity,weights)
            baseline=finish(raw_base);known=np.load(ROOT/'artifacts/residual-calibration/predictions.npy',mmap_mode='r');rmse=float(np.sqrt(np.mean(((baseline-known)/np.maximum(np.asarray(ridge['target_std']),1e-20))**2)))
            if rmse>1e-10:raise ValueError(f'Frozen baseline mismatch: {rmse}')
            grouped={}
            for child in args.children or []:
                folder=ROOT/'artifacts/finetune'/child
                if not all((ROOT/'artifacts/extra-inference'/child/k/'completed.json').exists() for k in ['validation','test']):raise ValueError('Incomplete candidate inference: '+child)
                cfg=json.loads((folder/'config.json').read_text());grouped.setdefault(cfg['model_id'],[]).append(child)
            score=lambda p,part:float(r2_score(y[part]*weights,p[part]*weights));selection=slice(75000,100000);audit=slice(100000,200000);best_score=score(baseline,selection);current=raw_base.copy();choices=[];scores={'baseline':best_score};selected=baseline
            for parent,children in grouped.items():
                winner=None;winner_delta=None
                for child in children:
                    delta=replacement('validation',parent,child,1,coeff,conditional)
                    for fraction in [.25,.5,1]:
                        prediction=finish(current+fraction*delta);value=score(prediction,selection);scores[child+f'-{fraction}']=value
                        if value>best_score:best_score=value;winner={'parent':parent,'child':child,'fraction':fraction};winner_delta=fraction*delta;selected=prediction
                if winner is not None:choices.append(winner);current+=winner_delta
                print('Combined parent',parent,'choice',winner,flush=True)
            config={'replacements':choices,'selection_scores':scores,'baseline_selection':scores['baseline'],'selected_selection':best_score,'baseline_audit':score(baseline,audit),'selected_audit':score(selected,audit),'baseline_reproduction_normalized_rmse':rmse,'conditional_config':str(ROOT/'artifacts/conditional-ensemble/frozen_config.json'),'ridge_config':str(ROOT/'artifacts/residual-calibration/frozen_config.json'),'selection_rows':[75000,100000],'audit_rows':[100000,200000],'test_labels_used':False}
            config['audit_gain']=config['selected_audit']-config['baseline_audit'];(args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));np.save(args.output/'predictions.npy',selected);np.save(args.output/'raw_predictions.npy',project(current,humidity,weights));print('Combined audit',config['audit_gain'],flush=True)
        else:
            config=json.loads(args.recipe.read_text())
            if config['audit_gain']<=1e-6:raise ValueError('Combined candidate did not pass audit')
            coeff=conditional_weights('test',conditional);result=raw_ensemble('test',conditional,coeff).copy()
            for member in config['replacements']:result+=replacement('test',member['parent'],member['child'],member['fraction'],coeff,conditional)
            raw,names,rh,humidity=input_context('test');result=project(apply_calibration(project(result,humidity,weights),raw,names,rh,ridge),humidity,weights);ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
            if not np.isfinite(result).all():raise FloatingPointError('Non-finite combined predictions')
            if not np.array_equal(ids,pl.read_csv(ROOT/'data/competition/sample_submission.csv',columns=['sample_id'])['sample_id'].to_numpy()):raise ValueError('ID order differs')
            output=ROOT/'artifacts/submissions'/(args.output.name+'.parquet');output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+'.partial');pl.DataFrame({'sample_id':ids,**{name:result[:,i] for i,name in enumerate(columns)}}).write_parquet(temporary,compression='zstd');temporary.replace(output)
            digest=hashlib.sha256()
            with output.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            output.with_name(output.name+'.manifest.json').write_text(json.dumps({'config':str(args.recipe.resolve()),'rows':len(ids),'targets':368,'sha256':digest.hexdigest(),'replacements':config['replacements'],'weights_applied_once':True,'test_labels_used':False},indent=2));print('Combined submission',output,flush=True)


if __name__=='__main__':main()
