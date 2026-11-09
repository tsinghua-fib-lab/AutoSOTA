"""Blend warm-start children within a parent's fixed conditional weight budget."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from combine_finetunes import conditional_weights,raw_ensemble,replacement
from conditional_ensemble import ROOT,GROUPS,project
from residual_calibration import apply_calibration
from extend_submission import apply_extensions
from physics_projection import albedo_features


def change(kind,parent,old_child,new_child,amount,weights,conditional):
    old=np.load(ROOT/'artifacts/extra-inference'/old_child/kind/'predictions.npy',mmap_mode='r');new=np.load(ROOT/'artifacts/extra-inference'/new_child/kind/'predictions.npy',mmap_mode='r');index=conditional['models'].index(parent);delta=np.zeros_like(new)
    for name,start,end in GROUPS:delta[:,start:end]=amount*weights[name][:,index,None]*(new[:,start:end]-old[:,start:end])
    return delta


def context(kind):
    cache=ROOT/('data/finetune-holdout' if kind=='validation' else 'artifacts/test-inference')
    if kind=='validation':
        raw=np.load(cache/'raw_features.npy',mmap_mode='r')
        with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
    else:
        columns=[f'state_t_{i}' for i in range(60)]+[f'{name}_{i}' for name in ['state_q0001','state_q0002','state_q0003'] for i in range(48,60)]+[f'{name}_{i}' for name in ['state_u','state_v'] for i in range(30,60)]+['state_ps','pbuf_LHFLX','pbuf_SHFLX','pbuf_SOLIN']+['cam_in_'+n for n in ['ASDIR','ALDIR','ASDIF','ALDIF']];raw=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=columns).select(columns).to_numpy()
    return raw,columns,np.load(cache/'rh-raw.npy',mmap_mode='r'),np.load(cache/'humidity.npy',mmap_mode='r'),albedo_features(kind)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--combo',type=Path,required=True);parser.add_argument('--extensions',type=Path,required=True);parser.add_argument('--children',nargs='+');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--recipe',type=Path);parser.add_argument('--build',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    combo=json.loads(args.combo.read_text());extension=json.loads(args.extensions.read_text());conditional=json.loads(Path(combo['conditional_config']).read_text());ridge=json.loads(Path(combo['ridge_config']).read_text());kind='test' if args.build else 'validation';coefficient=conditional_weights(kind,conditional);current=raw_ensemble(kind,conditional,coefficient).copy()
    for m in combo['replacements']:current+=replacement(kind,m['parent'],m['child'],m['fraction'],coefficient,conditional)
    raw,columns,rh,humidity,albedo=context(kind);sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);names=sample.columns[1:];weights=sample.select(names).to_numpy()[0]
    def finish(p,part):
        r=raw[part];h=humidity[part];rrh=rh[part];a=albedo[part]
        p=project(apply_calibration(project(p,h,weights),r,columns,rrh,ridge),h,weights)
        return apply_extensions(p,r,columns,rrh,h,weights,extension,kind,albedo_override=a)
    with threadpool_limits(limits=8):
        if not args.build:
            y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r');selection=slice(75000,100000);audit=slice(100000,200000);baseline_selection=finish(current[selection],selection);best_score=float(r2_score(y[selection]*weights,baseline_selection*weights));base_score=best_score;scores={'baseline':best_score};choices=[]
            for member in combo['replacements']:
                winner=None;winner_delta=None
                for child in args.children or []:
                    if child==member['child']:continue
                    cfg=json.loads((ROOT/'artifacts/finetune'/child/'config.json').read_text())
                    if cfg['model_id']!=member['parent']:continue
                    if not all((ROOT/'artifacts/extra-inference'/child/k/'completed.json').exists() for k in ['validation','test']):continue
                    delta=change('validation',member['parent'],member['child'],child,member['fraction'],coefficient,conditional)
                    for fraction in [.25,.5,1]:
                        candidate=finish(current[selection]+fraction*delta[selection],selection);score=float(r2_score(y[selection]*weights,candidate*weights));key=child+f'-mix{fraction}';scores[key]=score
                        if score>best_score:best_score=score;winner={'parent':member['parent'],'old_child':member['child'],'new_child':child,'parent_fraction':member['fraction'],'blend_fraction':fraction};winner_delta=fraction*delta
                if winner is not None:choices.append(winner);current+=winner_delta
                print('Parent-budget blend',member['parent'],winner,flush=True)
            prediction=finish(current,slice(None));baseline_audit=float(r2_score(y[audit]*weights,finish((raw_ensemble(kind,conditional,coefficient)+sum((replacement(kind,m['parent'],m['child'],m['fraction'],coefficient,conditional) for m in combo['replacements']),np.zeros_like(current)))[audit],audit)*weights));selected_audit=float(r2_score(y[audit]*weights,prediction[audit]*weights));compact=prediction.astype(np.float32).astype(np.float64);compact[:,132:148]=prediction[:,132:148]
            config={'combo':str(args.combo.resolve()),'extensions':str(args.extensions.resolve()),'blends':choices,'selection_scores':scores,'baseline_selection':base_score,'selected_selection':best_score,'selection_gain':best_score-base_score,'baseline_audit':baseline_audit,'selected_audit':selected_audit,'audit_gain':selected_audit-baseline_audit,'mixed_precision_audit_difference':float(r2_score(y[audit]*weights,compact[audit]*weights))-selected_audit,'test_labels_used':False,'budget_note':'Each parent keeps its original conditional weight and selected replacement fraction; child weights remain nonnegative and sum to that budget'}
            (args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));np.save(args.output/'predictions.npy',prediction);print('Parent-budget audit gain',config['audit_gain'],flush=True)
        else:
            config=json.loads(args.recipe.read_text())
            if config['audit_gain']<=1e-6 or config['selection_gain']<=0:raise ValueError('Parent blend failed checks')
            for m in config['blends']:current+=change('test',m['parent'],m['old_child'],m['new_child'],m['parent_fraction']*m['blend_fraction'],coefficient,conditional)
            prediction=finish(current,slice(None));ids=np.load(ROOT/'artifacts/test-inference/ids.npy');compact=abs(config['mixed_precision_audit_difference'])<=1e-8
            if not np.isfinite(prediction).all():raise FloatingPointError('Non-finite mixed child prediction')
            if not np.isin(weights,[0,1]).all() or not np.array_equal(ids,pl.read_csv(ROOT/'data/competition/sample_submission.csv',columns=['sample_id'])['sample_id'].to_numpy()):raise ValueError('Mask or IDs differ')
            output=ROOT/'artifacts/submissions'/(args.output.name+'.parquet');temporary=output.with_name(output.name+'.partial');pl.DataFrame({'sample_id':ids,**{c:prediction[:,i].astype(np.float32) if compact and not 132<=i<148 else prediction[:,i] for i,c in enumerate(names)}}).write_parquet(temporary,compression='zstd');temporary.replace(output);digest=hashlib.sha256()
            with output.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            output.with_name(output.name+'.manifest.json').write_text(json.dumps({'config':str(args.recipe.resolve()),'rows':len(ids),'targets':368,'sha256':digest.hexdigest(),'blends':config['blends'],'test_labels_used':False,'weights_applied_once':True},indent=2));print('Parent-budget submission ready',output,flush=True)


if __name__=='__main__':main()
