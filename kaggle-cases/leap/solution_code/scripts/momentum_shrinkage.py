"""Audit soft pressure-weighted wind-integral shrinkage without asserting closure."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from residual_calibration import ROOT


def pressure_mass(ps):
    with (ROOT/'data/greysnow/leap-gdata-ds/gdata.p').open('rb') as f:grid=pickle.load(f)
    return (100000*np.diff(np.asarray(grid[1]))[None]+np.asarray(ps)[:,None]*np.diff(np.asarray(grid[3]))[None])/9.80616


def apply_shrinkage(p,ps,config):
    result=np.asarray(p,np.float64).copy();mass=pressure_mass(ps)
    for name,start in [('u',240),('v',300)]:
        fit=config['wind'][name];recipe=fit['recipe']
        if not recipe['strength']:continue
        scale=np.asarray(fit['scale']);prediction=result[:,start:start+60]/scale;a=mass*scale
        direction=a@np.asarray(fit['matrices'][recipe['matrix']]).T;den=(a*direction).sum(axis=1)
        flux=(a*prediction).sum(axis=1)-fit['target_integral_mean'];delta=-np.divide(flux,den,out=np.zeros(len(p)),where=den>1e-30)[:,None]*direction
        result[:,start:start+60]=(prediction+recipe['strength']*delta)*scale
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    p=np.load(args.predictions,mmap_mode='r');y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r')
    with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
    raw=np.load(ROOT/'data/finetune-holdout/raw_features.npy',mmap_mode='r');mass=pressure_mass(raw[:,columns.index('state_ps')]);sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);weights=sample.select(sample.columns[1:]).to_numpy()[0]
    selection=slice(75000,100000);audit=slice(100000,200000);result=np.asarray(p).copy();config={'wind':{},'calibration_rows':[0,75000],'selection_rows':[75000,100000],'audit_rows':[100000,200000],'test_labels_used':False,'interpretation':'Soft learned shrinkage toward the calibration mean, not exact momentum conservation'}
    with threadpool_limits(limits=8):
        for name,start in [('u',240),('v',300)]:
            active=weights[start:start+60]!=0;scale=np.maximum(y[:75000,start:start+60].std(axis=0),1e-15);prediction=p[:,start:start+60]/scale;labels=y[:,start:start+60]/scale
            a=mass*scale;err=prediction[:75000]-labels[:75000];cov=err.T@err/len(err);cov[~active]=0;cov[:,~active]=0
            target_cov=np.diag(active.astype(float));matrices={'target_variance':target_cov,'error_covariance':cov+.01*target_cov*np.trace(cov)/max(active.sum(),1),'error_variance':np.diag(np.diag(cov))}
            target_integral=(mass[:75000]*y[:75000,start:start+60]).sum(axis=1);offset=float(target_integral.mean())
            base_sel=float(r2_score(labels[selection,active],prediction[selection,active]));base_audit=float(r2_score(labels[audit,active],prediction[audit,active]));best_score=base_sel;recipe={'matrix':None,'strength':0};best_prediction=prediction.copy();scores={};deltas={}
            for key,covariance in matrices.items():
                direction=a@covariance.T;den=(a*direction).sum(axis=1);flux=(a*prediction).sum(axis=1)-offset;delta=-np.divide(flux,den,out=np.zeros(len(p)),where=den>1e-30)[:,None]*direction;deltas[key]=delta
                for strength in [.125,.25,.5,.75,1]:
                    candidate=prediction+strength*delta;current=float(r2_score(labels[selection,active],candidate[selection,active]));scores[key+f'-{strength}']=current
                    if current>best_score:best_score=current;recipe={'matrix':key,'strength':strength};best_prediction=candidate.copy()
            changed_audit=float(r2_score(labels[audit,active],best_prediction[audit,active]));result[:,start:start+60]=best_prediction*scale;result[:,start:start+60][:,~active]=0
            config['wind'][name]={'scale':scale.tolist(),'matrices':{k:v.tolist() for k,v in matrices.items()},'target_integral_mean':offset,'target_integral_rms':float(np.sqrt(np.mean(target_integral**2))),'prediction_integral_rms':float(np.sqrt(np.mean((mass[:75000]*p[:75000,start:start+60]).sum(axis=1)**2))),'recipe':recipe,'selection_scores':scores,'baseline_selection':base_sel,'selected_selection':best_score,'baseline_audit':base_audit,'selected_audit':changed_audit,'full_macro_audit_gain':(changed_audit-base_audit)*active.sum()/368}
            print(name,json.dumps({k:config['wind'][name][k] for k in ['recipe','target_integral_rms','prediction_integral_rms','full_macro_audit_gain']}),flush=True)
    baseline_audit=float(r2_score(y[audit]*weights,p[audit]*weights));selected_audit=float(r2_score(y[audit]*weights,result[audit]*weights));config.update({'baseline_audit':baseline_audit,'selected_audit':selected_audit,'audit_gain':selected_audit-baseline_audit})
    (args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));np.save(args.output/'predictions.npy',result);print('Momentum shrinkage audit gain',config['audit_gain'],flush=True)


if __name__=='__main__':main()
