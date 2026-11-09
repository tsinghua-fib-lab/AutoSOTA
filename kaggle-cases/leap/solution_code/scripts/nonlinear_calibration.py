"""Small, strongly regularized nonlinear corrections of ensemble mean bias.

Only the first 75k original training-era holdout labels fit coefficients.
The next 25k choose a recipe; the last 100k provide a diagnostic audit.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from residual_calibration import ROOT,make_features,project,FEATURE_NAMES

EXTRA_NAMES=['lower_u','lower_v','mid_u','mid_v','vertical_wind_shear',
             'solar_albedo_ASDIR','solar_albedo_ALDIR','solar_albedo_ASDIF','solar_albedo_ALDIF']


def features(raw,columns,rh,p):
    base=make_features(raw,columns,rh,p)
    def average(name,start,end):return np.asarray(raw[:,[columns.index(f'{name}_{i}') for i in range(start,end)]],np.float64).mean(axis=1)
    lu=average('state_u',48,60);lv=average('state_v',48,60)
    mu=average('state_u',30,48);mv=average('state_v',30,48)
    albedo=np.asarray(raw[:,[columns.index('cam_in_'+n) for n in ['ASDIR','ALDIR','ASDIF','ALDIF']]],np.float64)
    return np.column_stack([base,lu,lv,mu,mv,np.hypot(lu-mu,lv-mv),albedo])


def basis(z,recipe):
    if recipe['kind']=='quadratic':
        i,j=np.triu_indices(z.shape[1]);return np.column_stack([z,z[:,i]*z[:,j]])
    centers=np.asarray(recipe['centers']);distance=np.maximum((z*z).sum(axis=1)[:,None]+(centers*centers).sum(axis=1)[None]-2*z@centers.T,0)
    return np.column_stack([z,np.exp(-distance/(2*recipe['bandwidth']**2))])


def correction(features_,config,fit):
    z=np.clip((features_-np.asarray(config['feature_mean']))/np.asarray(config['feature_std']),-3,3)
    b=basis(z,fit['basis']);b=np.clip((b-np.asarray(fit['basis_mean']))/np.asarray(fit['basis_std']),-5,5)
    design=np.column_stack([np.ones(len(z)),b])
    return np.clip(design@np.asarray(fit['coefficient']),-.05,.05)*np.asarray(config['target_std'])


def fit_main(args):
    p=np.load(args.predictions,mmap_mode='r');y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r')
    cache=ROOT/'data/finetune-holdout';raw=np.load(cache/'raw_features.npy',mmap_mode='r');rh=np.load(cache/'rh-raw.npy',mmap_mode='r');humidity=np.load(cache/'humidity.npy',mmap_mode='r')
    with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as handle:columns=list(pickle.load(handle))
    with (ROOT/'data/greysnow/leap-msm-ds/std_y.p').open('rb') as handle:sd=np.asarray(pickle.load(handle)).reshape(-1)
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);weights=sample.select(sample.columns[1:]).to_numpy()[0]
    f=features(raw,columns,rh,p);mean=f[:75000].mean(axis=0);scale=np.maximum(f[:75000].std(axis=0),1e-10);z=np.clip((f-mean)/scale,-3,3)
    target_scale=np.where(sd>0,sd,1);residual=(y[:75000]-p[:75000])/target_scale
    inactive=(weights==0)|(sd==0);inactive[132:148]=True;residual[:,inactive]=0
    config={'feature_names':FEATURE_NAMES+EXTRA_NAMES,'feature_mean':mean.tolist(),'feature_std':scale.tolist(),'target_std':target_scale.tolist(),'fits':{},'feature_clip':[-3,3],'basis_clip':[-5,5],'normalized_correction_cap':.05,'baseline_predictions':str(args.predictions.resolve()),'calibration_rows':[0,75000],'selection_rows':[75000,100000],'audit_rows':[100000,200000],'test_labels_used':False}
    selection=slice(75000,100000);audit=slice(100000,200000)
    score=lambda candidate,part:float(r2_score(y[part]*weights,candidate[part]*weights))
    selection_scores={'baseline':score(p,selection)};best_key='baseline';best_prediction=np.asarray(p).copy();best_score=selection_scores['baseline']
    with threadpool_limits(limits=8):
        kmeans=MiniBatchKMeans(n_clusters=64,random_state=43,batch_size=2048,n_init=3,max_iter=50).fit(z[:75000]);centers=kmeans.cluster_centers_
        radii=np.linalg.norm(z[:75000:10]-centers[kmeans.predict(z[:75000:10])],axis=1);radius=max(float(np.median(radii)),.5)
        recipes=[{'kind':'quadratic'}]+[{'kind':'rbf','centers':centers.tolist(),'bandwidth':radius*factor} for factor in [1,2]]
        for index,recipe in enumerate(recipes):
            b=basis(z,recipe);bm=b[:75000].mean(axis=0);bs=np.maximum(b[:75000].std(axis=0),.05)
            design=np.column_stack([np.ones(len(z)),np.clip((b-bm)/bs,-5,5)]);gram=design[:75000].T@design[:75000]/75000;rhs=design[:75000].T@residual/75000
            for regularization in [1,10,100]:
                penalty=np.eye(design.shape[1])*regularization;penalty[0,0]=0;coef=np.linalg.solve(gram+penalty,rhs);coef[:,inactive]=0
                key=f"{recipe['kind']}{index}-ridge{regularization}";fit={'basis':recipe,'basis_mean':bm.tolist(),'basis_std':bs.tolist(),'regularization':regularization,'coefficient':coef.tolist()};config['fits'][key]=fit
                delta=np.clip(design@coef,-.05,.05)*target_scale
                for strength in [.25,.5,1]:
                    name=key+f'-alpha{strength}';candidate=project(p+strength*delta,humidity,weights);current=score(candidate,selection);selection_scores[name]=current
                    if current>best_score:best_key=name;best_score=current;best_prediction=candidate.copy();config['recipe']={'fit':key,'strength':strength}
                print('Fitted nonlinear basis',key,'best',best_key,best_score,flush=True)
    if best_key=='baseline':config['recipe']={'fit':None,'strength':0}
    config.update({'selected':best_key,'selection_scores':selection_scores})
    baseline_audit=score(p,audit);selected_audit=score(best_prediction,audit);report={**config,'baseline_audit':baseline_audit,'selected_audit':selected_audit,'audit_gain':selected_audit-baseline_audit,'audit_note':'Repeated diagnostic holdout; no competition test labels'}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));(args.output/'metrics.json').write_text(json.dumps(report,indent=2));np.save(args.output/'predictions.npy',best_prediction)
    print(json.dumps({k:report[k] for k in ['selected','baseline_audit','selected_audit','audit_gain']}),flush=True)


def build_main(args):
    config=json.loads(args.config.read_text());baseline=pl.read_parquet(args.baseline);sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);columns=sample.columns[1:];weights=sample.select(columns).to_numpy()[0];ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
    if baseline.columns!=sample.columns or not np.array_equal(ids,baseline['sample_id'].to_numpy()) or not np.isin(weights,[0,1]).all():raise ValueError('Schema, IDs or weights differ')
    names=[f'state_t_{i}' for i in range(60)]+[f'{name}_{i}' for name in ['state_q0001','state_q0002','state_q0003'] for i in range(48,60)]+['state_ps','pbuf_LHFLX','pbuf_SHFLX','pbuf_SOLIN']+[f'{name}_{i}' for name in ['state_u','state_v'] for i in range(30,60)]+['cam_in_'+name for name in ['ASDIR','ALDIR','ASDIF','ALDIF']]
    raw=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=names).select(names).to_numpy();rh=np.load(ROOT/'artifacts/test-inference/rh-raw.npy',mmap_mode='r');p=baseline.select(columns).to_numpy();f=features(raw,names,rh,p);recipe=config['recipe']
    with threadpool_limits(limits=8):
        if recipe['strength']:
            fit=config['fits'][recipe['fit']]
            for start in range(0,len(p),20000):p[start:start+20000]+=recipe['strength']*correction(f[start:start+20000],config,fit)
    humidity=np.load(ROOT/'artifacts/test-inference/humidity.npy',mmap_mode='r');p=project(p,humidity,weights)
    if not np.isfinite(p).all():raise FloatingPointError('Non-finite candidate')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial');pl.DataFrame({'sample_id':ids,**{c:p[:,i] for i,c in enumerate(columns)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output)
    digest=hashlib.sha256()
    with args.output.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'rows':len(p),'targets':368,'config':str(args.config.resolve()),'baseline':str(args.baseline.resolve()),'recipe':recipe,'sha256':digest.hexdigest(),'weights_applied_once':True,'test_labels_used':False},indent=2));print('Nonlinear submission ready',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    fit=sub.add_parser('fit');fit.add_argument('--predictions',type=Path,required=True);fit.add_argument('--output',type=Path,required=True)
    build=sub.add_parser('build');build.add_argument('--config',type=Path,required=True);build.add_argument('--baseline',type=Path,required=True);build.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();fit_main(args) if args.action=='fit' else build_main(args)


if __name__=='__main__':main()
