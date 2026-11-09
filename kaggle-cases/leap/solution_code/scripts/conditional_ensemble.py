"""Fit bounded conditional covariance weights and state-dependent mean corrections."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
GROUPS=[('temperature',0,60),('vapor',60,120),('liquid',120,180),('ice',180,240),('wind_u',240,300),('wind_v',300,360),('scalars',360,368)]


def feature(cache):
    rh=np.load(cache/'rh-raw.npy',mmap_mode='r')
    return np.asarray(rh[:,48:60]).mean(axis=1)


def interpolate_weights(values,centers,weights):
    return np.stack([np.interp(values,centers,weights[:,i]) for i in range(weights.shape[1])],axis=1)


def project(p,humidity,weights):
    p=p.copy();p[:,60:240]=np.maximum(p[:,60:240],-humidity/1200);p[:,weights==0]=0
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/conditional-ensemble')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    models=json.loads((ROOT/'configs/champion_models.json').read_text())
    prior=json.loads((ROOT/'artifacts/constrained-ensemble/frozen_config.json').read_text())
    if prior['models']!=[m['id'] for m in models]:raise ValueError('Model order differs')
    arrays=[np.load(ROOT/'artifacts/champion-validation'/m['id']/'predictions.npy',mmap_mode='r') for m in models]
    labels=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r')
    humidity=np.load(ROOT/'data/finetune-holdout/humidity.npy',mmap_mode='r')
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);weights=sample.select(sample.columns[1:]).to_numpy()[0]
    values=feature(ROOT/'data/finetune-holdout');thresholds=np.unique(np.quantile(values[:75000],[.25,.5,.75]));bins=np.digitize(values,thresholds)
    centers=np.array([values[:75000][bins[:75000]==i].mean() for i in range(len(thresholds)+1)])
    with (ROOT/'data/greysnow/leap-msm-ds/std_y.p').open('rb') as f:source_std=np.asarray(pickle.load(f)).reshape(-1)
    sd=np.maximum(labels[:75000].std(axis=0),.05*source_std);active=(weights!=0)&(sd>0);active[132:148]=False
    lr=np.asarray([m['id'].startswith('lr_') for m in models],np.float64)
    baseline=np.zeros(labels.shape,np.float64);conditional={str(reg):{} for reg in [0,.01,.1,1]};correlations={}
    for name,start,end in GROUPS:
        original=np.asarray(prior['group_weights'][name]);columns=np.arange(start,end)[active[start:end]]
        for weight,p in zip(original,arrays):baseline[:,start:end]+=weight*p[:,start:end]
        residuals=np.stack([(p[:75000,columns]-labels[:75000,columns])/sd[columns] for p in arrays],axis=-1)
        with threadpool_limits(limits=8):global_gram=residuals.reshape(-1,len(models)).T@residuals.reshape(-1,len(models))/np.prod(residuals.shape[:2])
        mean=residuals.mean(axis=(0,1));covariance=global_gram-np.outer(mean,mean);denom=np.sqrt(np.maximum(np.diag(covariance),1e-12))
        correlation=covariance/(denom[:,None]*denom[None]);correlations[name]={'mean_off_diagonal':float(correlation[np.triu_indices(len(models),1)].mean()),'matrix':correlation.tolist()}
        fits={str(reg):[] for reg in [0,.01,.1,1]}
        for bin_index in range(len(centers)):
            r=residuals[bins[:75000]==bin_index].reshape(-1,len(models))
            with threadpool_limits(limits=8):gram=r.T@r/len(r)
            # Include nonzero residual means in the second moment used by MSE.
            scale=max(float(np.trace(gram)/len(models)),1e-12)
            for regularization in [0,.01,.1,1]:
                penalty=regularization*scale
                objective=lambda w:float(w@gram@w+penalty*np.sum((w-original)**2))
                gradient=lambda w:2*gram@w+2*penalty*(w-original)
                fit=minimize(objective,original,jac=gradient,method='SLSQP',bounds=[(w*.85,w*1.15) for w in original],constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(models))},{'type':'eq','fun':lambda w:w@lr-.18,'jac':lambda w:lr}],options={'ftol':1e-12,'maxiter':500})
                if not fit.success:raise RuntimeError(fit.message)
                fits[str(regularization)].append(fit.x.tolist())
        for reg in fits:conditional[reg][name]=fits[reg]
        print('Conditional covariance fitted',name,'mean_error_correlation',correlations[name]['mean_off_diagonal'],flush=True)
    baseline=project(baseline,humidity,weights)
    selection=slice(75000,100000);audit=slice(100000,200000)
    def score(prediction,part):return float(r2_score(labels[part]*weights,prediction[part]*weights))
    selection_scores={'baseline':score(baseline,selection)};candidate_files={'baseline':None};recipes={'baseline':{'kind':'baseline'}}
    for regularization in [0,.01,.1,1]:
        gated=np.zeros_like(baseline)
        for name,start,end in GROUPS:
            row_weights=interpolate_weights(values,centers,np.asarray(conditional[str(regularization)][name]))
            for i,p in enumerate(arrays):gated[:,start:end]+=row_weights[:,i,None]*p[:,start:end]
        for strength in [.25,.5,1]:
            key=f'cov-reg{regularization}-alpha{strength}'
            raw=(1-strength)*baseline+strength*gated
            candidate=project(raw,humidity,weights);selection_scores[key]=score(candidate,selection)
            path=args.output/(key+'.npy');np.save(path,candidate);candidate_files[key]=str(path);recipes[key]={'kind':'covariance','regularization':regularization,'strength':strength}
    # Mean corrections can address shared biases that convex averaging cannot.
    residual=labels[:75000]-baseline[:75000]
    bias=np.stack([residual[bins[:75000]==i].mean(axis=0) for i in range(len(centers))])
    bias=np.clip(bias,-.05*source_std,.05*source_std);bias[:,weights==0]=0;bias[:,132:148]=0
    row_bias=interpolate_weights(values,centers,bias)
    for strength in [.05,.1,.25,.5,1]:
        key=f'bias-alpha{strength}';candidate=project(baseline+strength*row_bias,humidity,weights)
        selection_scores[key]=score(candidate,selection);path=args.output/(key+'.npy');np.save(path,candidate);candidate_files[key]=str(path);recipes[key]={'kind':'mean_bias','strength':strength}
    selected=max(selection_scores,key=selection_scores.get)
    frozen={'models':prior['models'],'groups':[list(g) for g in GROUPS],'base_group_weights':prior['group_weights'],'humidity_feature':'mean RH at levels 48 through 59, derived only from available inputs','thresholds':thresholds.tolist(),'centers':centers.tolist(),'conditional_weights':conditional,'bias':bias.tolist(),'selected':selected,'recipe':recipes[selected],'selection_scores':selection_scores,'calibration_rows':[0,75000],'selection_rows':[75000,100000],'audit_rows':[100000,200000],'conditional_relative_radius':.15,'test_labels_used':False}
    (args.output/'frozen_config.json').write_text(json.dumps(frozen,indent=2))
    chosen=baseline if selected=='baseline' else np.load(candidate_files[selected],mmap_mode='r')
    audit_baseline=score(baseline,audit);audit_selected=score(chosen,audit)
    report={**frozen,'baseline_audit':audit_baseline,'selected_audit':audit_selected,'audit_gain':audit_selected-audit_baseline,'error_correlations':correlations,'audit_note':'Repeated training-era diagnostic holdout, not an unseen competition test'}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    print('Frozen conditional result',json.dumps({'selected':selected,'baseline_audit':audit_baseline,'selected_audit':audit_selected,'audit_gain':report['audit_gain']}),flush=True)


if __name__=='__main__':main()
