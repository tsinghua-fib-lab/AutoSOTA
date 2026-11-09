"""Regularized binned monotone mean calibration of each ensemble output."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from conditional_ensemble import ROOT,project
from physics_projection import albedo_features,apply_recipe


def correction(p,fit,scale):
    delta=np.zeros_like(p)
    for column,curve in fit.items():
        j=int(column);z=p[:,j]/scale[j];calibrated=np.interp(z,curve['x'],curve['y']);delta[:,j]=np.clip(calibrated-z,-.05,.05)*scale[j]
    return delta


def build(args):
    config=json.loads(args.recipe.read_text())
    if config['recipe'] is None or config['audit_gain']<=1e-6 or config['selected_selection']<=config['baseline_selection']:raise ValueError('Output calibration did not improve both checks')
    if abs(config['mixed_precision_audit_difference'])>1e-8:raise ValueError('Mixed precision changed audit')
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);columns=sample.columns[1:];mask=sample.select(columns).to_numpy()[0];baseline=pl.read_parquet(args.baseline);ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
    if baseline.columns!=sample.columns or not np.array_equal(ids,baseline['sample_id'].to_numpy()) or not np.isin(mask,[0,1]).all():raise ValueError('Schema, mask, or IDs differ')
    prediction=baseline.select(columns).to_numpy();recipe=config['recipe'];prediction+=recipe['strength']*correction(prediction,config['fits'][recipe['fit']],np.asarray(config['target_std']));humidity=np.load(ROOT/'artifacts/test-inference/humidity.npy',mmap_mode='r');solar=json.loads(Path(config['solar_config']).read_text());prediction=apply_recipe(project(prediction,humidity,mask),albedo_features('test'),solar)
    if not np.isfinite(prediction).all():raise FloatingPointError('Non-finite recalibrated output')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial');pl.DataFrame({'sample_id':ids,**{c:prediction[:,i].astype(np.float32) if not 132<=i<148 else prediction[:,i] for i,c in enumerate(columns)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output);h=hashlib.sha256()
    with args.output.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'config':str(args.recipe.resolve()),'baseline':str(args.baseline.resolve()),'rows':len(ids),'targets':368,'sha256':h.hexdigest(),'weights_applied_once':True,'test_labels_used':False},indent=2));print('Output calibration submission ready',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build',action='store_true');parser.add_argument('--recipe',type=Path);parser.add_argument('--baseline',type=Path);args=parser.parse_args()
    if args.build:build(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    p=np.load(args.predictions,mmap_mode='r');y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r');humidity=np.load(ROOT/'data/finetune-holdout/humidity.npy',mmap_mode='r')
    with (ROOT/'data/greysnow/leap-msm-ds/std_y.p').open('rb') as f:sd=np.asarray(pickle.load(f)).reshape(-1)
    scale=np.where(sd>0,sd,1);sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);mask=sample.select(sample.columns[1:]).to_numpy()[0];active=(mask!=0)&(sd>0);active[132:148]=False
    solar=json.loads((ROOT/'artifacts/physics-projection-conditional/frozen_config.json').read_text());albedo=albedo_features('validation');selection=slice(75000,100000);audit=slice(100000,200000)
    def finish(v):return apply_recipe(project(v,humidity,mask),albedo,solar)
    score=lambda v,part:float(r2_score(y[part]*mask,v[part]*mask));baseline_selection=score(p,selection);best=baseline_selection;winner=None;best_prediction=np.asarray(p).copy();scores={'baseline':baseline_selection};fits={}
    with threadpool_limits(limits=8):
        for bins in [8,16,32]:
            curves={str(a):{} for a in [0,5000,25000]}
            for j in np.flatnonzero(active):
                x=p[:75000,j]/scale[j];target=y[:75000,j]/scale[j];thresholds=np.unique(np.quantile(x,np.linspace(0,1,bins+1)[1:-1]));assigned=np.digitize(x,thresholds,right=True);count=np.bincount(assigned,minlength=len(thresholds)+1);valid=count>=200
                centers=np.bincount(assigned,weights=x,minlength=len(count))[valid]/count[valid];means=np.bincount(assigned,weights=target,minlength=len(count))[valid]/count[valid];counts=count[valid]
                if len(centers)<2:continue
                for anchor in [0,5000,25000]:
                    regression=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(centers,(counts*means+anchor*centers)/(counts+anchor),sample_weight=counts+anchor)
                    curves[str(anchor)][str(j)]={'x':regression.X_thresholds_.tolist(),'y':regression.y_thresholds_.tolist()}
            for anchor,curve in curves.items():
                key=f'bins{bins}-anchor{anchor}';fits[key]=curve;delta=correction(p,curve,scale)
                for strength in [.25,.5,1]:
                    prediction=finish(p+strength*delta);name=key+f'-alpha{strength}';scores[name]=score(prediction,selection)
                    if scores[name]>best:best=scores[name];winner={'fit':key,'strength':strength};best_prediction=prediction
                print('Fitted output mean calibration',key,'best',winner,flush=True)
        compact=best_prediction.astype(np.float32).astype(np.float64);compact[:,132:148]=best_prediction[:,132:148]
        config={'baseline_predictions':str(args.predictions.resolve()),'target_std':scale.tolist(),'fits':fits,'recipe':winner,'selection_scores':scores,'baseline_selection':baseline_selection,'selected_selection':best,'baseline_audit':score(p,audit),'selected_audit':score(best_prediction,audit),'calibration_rows':[0,75000],'selection_rows':[75000,100000],'audit_rows':[100000,200000],'normalized_correction_cap':.05,'solar_config':str(ROOT/'artifacts/physics-projection-conditional/frozen_config.json'),'mixed_precision_audit_difference':score(compact,audit)-score(best_prediction,audit),'test_labels_used':False,'audit_note':'Repeated training-era diagnostic; comparator is strongest already scored complete pipeline'};config['audit_gain']=config['selected_audit']-config['baseline_audit'];(args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));np.save(args.output/'predictions.npy',best_prediction);print('Output calibration result',winner,'audit gain',config['audit_gain'],flush=True)


if __name__=='__main__':main()
