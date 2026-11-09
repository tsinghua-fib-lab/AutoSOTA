"""Bounded two-dimensional physical-regime weights for the current model pool."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from combine_finetunes import conditional_weights, input_context, raw_ensemble, replacement
from conditional_ensemble import ROOT, GROUPS, project
from extend_submission import apply_extensions
from residual_calibration import apply_calibration
from mix_finetune_children import context


def regime_features(kind):
    cache=ROOT/('data/finetune-holdout' if kind=='validation' else 'artifacts/test-inference')
    rh=np.load(cache/'rh-raw.npy',mmap_mode='r')
    names=['state_t_59','pbuf_SOLIN']+[f'{q}_{i}' for q in ['state_q0002','state_q0003'] for i in range(30,60)]
    if kind=='validation':
        with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
        all_raw=np.load(cache/'raw_features.npy',mmap_mode='r');raw=all_raw[:,[columns.index(n) for n in names]]
    else:raw=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=names).select(names).to_numpy()
    return {'low_rh':np.mean(rh[:,48:60],axis=1),'surface_temperature':raw[:,0],'solar_input':raw[:,1],'mid_rh':np.mean(rh[:,30:48],axis=1),'cloud':np.log1p(np.maximum(raw[:,2:],0).mean(axis=1)*1e6)}


def bilinear(x,y,cx,cy,grid):
    ix=np.clip(np.searchsorted(cx,x,side='right')-1,0,len(cx)-2);iy=np.clip(np.searchsorted(cy,y,side='right')-1,0,len(cy)-2)
    tx=np.clip((x-cx[ix])/(cx[ix+1]-cx[ix]),0,1)[:,None];ty=np.clip((y-cy[iy])/(cy[iy+1]-cy[iy]),0,1)[:,None]
    return (1-tx)*(1-ty)*grid[ix,iy]+tx*(1-ty)*grid[ix+1,iy]+(1-tx)*ty*grid[ix,iy+1]+tx*ty*grid[ix+1,iy+1]


def build(args):
    config=json.loads(args.recipe.read_text())
    if config['recipe'] is None or config['audit_gain']<=1e-6 or config['selected_selection']<=config['baseline_selection']:raise ValueError('Physical regime did not pass both checks')
    combo=json.loads(Path(config['combo_config']).read_text());conditional=json.loads(Path(combo['conditional_config']).read_text());ridge=json.loads(Path(combo['ridge_config']).read_text());extensions=json.loads(Path(config['extension_config']).read_text())
    coeff=conditional_weights('test',conditional);base=raw_ensemble('test',conditional,coeff).copy()
    for r in combo['replacements']:base+=replacement('test',r['parent'],r['child'],r['fraction'],coeff,conditional)
    features=regime_features('test');spec=config['specs'][config['recipe']['spec_index']];gated=np.zeros_like(base)
    for name,start,end in GROUPS:
        row_weights=bilinear(features['low_rh'],features[spec['feature']],np.asarray(spec['x_centers']),np.asarray(spec['y_centers']),np.asarray(spec['weights'][name]))
        for i,model in enumerate(config['models']):
            path=ROOT/'artifacts/extra-inference'/config['full_parent_replacements'][model]/'test/predictions.npy' if model in config['full_parent_replacements'] else ROOT/'artifacts/test-inference'/(model+'.npy')
            a=np.load(path,mmap_mode='r');gated[:,start:end]+=row_weights[:,i,None]*a[:,start:end]
    strength=config['recipe']['strength'];prediction=(1-strength)*base+strength*gated;raw,columns,rh,humidity,albedo=context('test');sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);names=sample.columns[1:];mask=sample.select(names).to_numpy()[0]
    with threadpool_limits(limits=8):
        prediction=project(apply_calibration(project(prediction,humidity,mask),raw,columns,rh,ridge),humidity,mask)
        prediction=apply_extensions(prediction,raw,columns,rh,humidity,mask,extensions,'test',albedo_override=albedo)
    ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
    if not np.isfinite(prediction).all() or not np.isin(mask,[0,1]).all():raise ValueError('Invalid values or mask')
    if not np.array_equal(ids,pl.read_csv(ROOT/'data/competition/sample_submission.csv',columns=['sample_id'])['sample_id'].to_numpy()):raise ValueError('ID order differs')
    validation=np.load(args.recipe.parent/'predictions.npy',mmap_mode='r');labels=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r');compact=validation[100000:].astype(np.float32).astype(np.float64);compact[:,132:148]=validation[100000:,132:148];difference=float(r2_score(labels[100000:]*mask,compact*mask))-config['selected_audit']
    if abs(difference)>1e-8:raise ValueError('Mixed precision changed audit')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial');pl.DataFrame({'sample_id':ids,**{c:prediction[:,i].astype(np.float32) if not 132<=i<148 else prediction[:,i] for i,c in enumerate(names)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output)
    digest=hashlib.sha256()
    with args.output.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):digest.update(block)
    args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'config':str(args.recipe.resolve()),'rows':len(ids),'targets':368,'sha256':digest.hexdigest(),'mixed_precision_audit_difference':difference,'weights_applied_once':True,'test_labels_used':False},indent=2));print('Physical-regime submission ready',args.output,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--combo',type=Path,default=ROOT/'artifacts/combined/thirteen-completed/frozen_config.json');p.add_argument('--extensions',type=Path,default=ROOT/'artifacts/combined/thirteen-completed-rbf-momentum/frozen_config.json');p.add_argument('--build',action='store_true');p.add_argument('--recipe',type=Path);args=p.parse_args()
    if args.build:build(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    combo=json.loads(args.combo.read_text());conditional=json.loads(Path(combo['conditional_config']).read_text());ridge=json.loads(Path(combo['ridge_config']).read_text());extensions=json.loads(args.extensions.read_text())
    if any(r['fraction']!=1 for r in combo['replacements']):raise ValueError('This experiment requires full parent replacement')
    children={r['parent']:r['child'] for r in combo['replacements']};models=conditional['models'];arrays=[]
    for model in models:
        path=ROOT/'artifacts/extra-inference'/children[model]/'validation/predictions.npy' if model in children else ROOT/'artifacts/champion-validation'/model/'predictions.npy'
        arrays.append(np.load(path,mmap_mode='r'))
    labels=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r');features=regime_features('validation');raw,columns,rh,humidity=input_context('validation')
    # The extension's wind features require its fuller input ordering.
    with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
    raw=np.load(ROOT/'data/finetune-holdout/raw_features.npy',mmap_mode='r');sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);mask=sample.select(sample.columns[1:]).to_numpy()[0]
    coeff=conditional_weights('validation',conditional);current=raw_ensemble('validation',conditional,coeff).copy()
    for r in combo['replacements']:current+=replacement('validation',r['parent'],r['child'],1,coeff,conditional)
    finish=lambda v:apply_extensions(project(apply_calibration(project(v,humidity,mask),raw,columns,rh,ridge),humidity,mask),raw,columns,rh,humidity,mask,extensions,'validation')
    baseline=finish(current);known=np.load(ROOT/'artifacts/combined/thirteen-completed-rbf-momentum/predictions.npy',mmap_mode='r')
    if not np.array_equal(baseline,known):raise ValueError('Current baseline reconstruction differs')
    with (ROOT/'data/greysnow/leap-msm-ds/std_y.p').open('rb') as f:source_std=np.asarray(pickle.load(f)).reshape(-1)
    sd=np.maximum(labels[:75000].std(axis=0),.05*source_std);active=(mask!=0)&(sd>0);active[132:148]=False;lr=np.array([m.startswith('lr_') for m in models],float)
    specs=[]
    for second in ['surface_temperature','solar_input','cloud','mid_rh']:
        x,y=features['low_rh'],features[second];tx=np.unique(np.quantile(x[:75000],[1/3,2/3]));ty=np.unique(np.quantile(y[:75000],[1/3,2/3]))
        if len(tx)!=2 or len(ty)!=2:continue
        bx,by=np.digitize(x[:75000],tx,right=True),np.digitize(y[:75000],ty,right=True)
        if any(not np.any(bx==i) or not np.any(by==i) for i in range(3)):continue
        cx=np.array([x[:75000][bx==i].mean() for i in range(3)]);cy=np.array([y[:75000][by==i].mean() for i in range(3)])
        for reg in [0,.1]:specs.append({'feature':second,'regularization':reg,'x_centers':cx.tolist(),'y_centers':cy.tolist(),'x_thresholds':tx.tolist(),'y_thresholds':ty.tolist(),'weights':{},'counts':[[int(np.count_nonzero((bx==i)&(by==j))) for j in range(3)] for i in range(3)]})
    with threadpool_limits(limits=8):
        for name,start,end in GROUPS:
            cols=np.arange(start,end)[active[start:end]];residuals=np.stack([(a[:75000,cols]-labels[:75000,cols])/sd[cols] for a in arrays],axis=-1);r=residuals.reshape(-1,len(models));global_gram=r.T@r/len(r);prior=np.array(conditional['base_group_weights'][name])
            for spec in specs:
                bx=np.digitize(features['low_rh'][:75000],spec['x_thresholds'],right=True);by=np.digitize(features[spec['feature']][:75000],spec['y_thresholds'],right=True);grid=[]
                for i in range(3):
                    row=[]
                    for j in range(3):
                        sel=(bx==i)&(by==j);r=residuals[sel].reshape(-1,len(models));local=r.T@r/max(len(r),1);shrink=2000/(np.count_nonzero(sel)+2000);gram=(1-shrink)*local+shrink*global_gram;penalty=spec['regularization']*max(float(np.trace(gram)/len(models)),1e-12)
                        fit=minimize(lambda w:float(w@gram@w+penalty*np.sum((w-prior)**2)),prior,jac=lambda w:2*gram@w+2*penalty*(w-prior),method='SLSQP',bounds=[(.85*w,1.15*w) for w in prior],constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(models))},{'type':'eq','fun':lambda w:w@lr-.18,'jac':lambda w:lr}],options={'ftol':1e-12,'maxiter':500})
                        if not fit.success:raise RuntimeError(fit.message)
                        row.append(fit.x.tolist())
                    grid.append(row)
                spec['weights'][name]=grid
            print('Fitted two-dimensional physical regimes',name,flush=True)
        selection=slice(75000,100000);audit=slice(100000,200000);score=lambda a,part:float(r2_score(labels[part]*mask,a[part]*mask));best=score(baseline,selection);winner=None;prediction=baseline;scores={'baseline':best}
        for k,spec in enumerate(specs):
            gated=np.zeros_like(current)
            for name,start,end in GROUPS:
                row_weights=bilinear(features['low_rh'],features[spec['feature']],np.asarray(spec['x_centers']),np.asarray(spec['y_centers']),np.asarray(spec['weights'][name]))
                if np.max(np.abs(row_weights.sum(1)-1))>1e-10 or np.max(np.abs(row_weights@lr-.18))>1e-10:raise ValueError('Parent budget constraint violated')
                for i,a in enumerate(arrays):gated[:,start:end]+=row_weights[:,i,None]*a[:,start:end]
            for strength in [.25,.5,1]:
                value=finish((1-strength)*current+strength*gated);key=f'{spec["feature"]}-reg{spec["regularization"]}-alpha{strength}';scores[key]=score(value,selection)
                if scores[key]>best:best=scores[key];winner={'spec_index':k,'strength':strength};prediction=value
            print('Scored physical regime',spec['feature'],spec['regularization'],flush=True)
        report={'models':models,'full_parent_replacements':children,'combo_config':str(args.combo.resolve()),'extension_config':str(args.extensions.resolve()),'specs':specs,'recipe':winner,'selection_scores':scores,'baseline_selection':scores['baseline'],'selected_selection':best,'baseline_audit':score(baseline,audit),'selected_audit':score(prediction,audit),'calibration_rows':[0,75000],'selection_rows':[75000,100000],'audit_rows':[100000,200000],'audit_note':'Repeated diagnostic holdout; strongest previously scored recipe is the comparator','conditional_relative_radius':.15,'local_global_shrinkage_pseudocount':2000,'test_labels_used':False};report['audit_gain']=report['selected_audit']-report['baseline_audit'];np.save(args.output/'predictions.npy',prediction);(args.output/'frozen_config.json').write_text(json.dumps(report,indent=2));print('Physical-regime result',winner,'audit_gain',report['audit_gain'],flush=True)


if __name__=='__main__':main()
