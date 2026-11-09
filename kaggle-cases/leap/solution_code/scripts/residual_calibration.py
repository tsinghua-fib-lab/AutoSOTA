"""Conservative ridge calibration of shared state-dependent prediction biases."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
FEATURE_NAMES=['lower_RH','upper_RH','surface_T','lower_T','vertical_T_difference','log_liquid','log_ice','log_vapor','surface_pressure','latent_flux','sensible_flux','solar_input','daylight','daylight_lower_RH','log_predicted_precipitation','predicted_snow_fraction']


def make_features(raw,columns,rh,p):
    def col(name):return np.asarray(raw[:,columns.index(name)],np.float64)
    t=np.stack([col(f'state_t_{i}') for i in range(60)],axis=1)
    def average(name):return np.stack([col(f'{name}_{i}') for i in range(48,60)],axis=1).mean(axis=1)
    lower=np.asarray(rh[:,48:60]).mean(axis=1);upper=np.asarray(rh[:,30:48]).mean(axis=1);solar=col('pbuf_SOLIN');day=(solar>0).astype(float)
    liquid=np.log1p(np.maximum(average('state_q0002'),0)/1e-6);ice=np.log1p(np.maximum(average('state_q0003'),0)/1e-6);vapor=np.log1p(np.maximum(average('state_q0001'),0)/1e-3)
    precipitation=np.log1p(np.maximum(p[:,363],0)/1e-8);snow_fraction=np.clip(np.maximum(p[:,362],0)/np.maximum(p[:,363],1e-8),0,1)
    return np.column_stack([lower,upper,t[:,59],t[:,48:60].mean(axis=1),t[:,48]-t[:,59],liquid,ice,vapor,col('state_ps'),col('pbuf_LHFLX'),col('pbuf_SHFLX'),solar,day,day*lower,precipitation,snow_fraction])


def project(p,humidity,weights):
    p=p.copy();p[:,60:240]=np.maximum(p[:,60:240],-humidity/1200);p[:,weights==0]=0
    return p


def apply_calibration(p,raw,columns,rh,config,batch_size=20000):
    """Apply exactly the frozen fit; also usable after changing ensemble members."""
    result=np.asarray(p,np.float64).copy();recipe=config['recipe']
    if not recipe['strength']:return result
    fit=config['fits'][str(recipe['regularization'])];scale=np.asarray(config['target_std'])
    for start in range(0,len(p),batch_size):
        stop=min(start+batch_size,len(p));prediction=p[start:stop]
        f=make_features(raw[start:stop],columns,rh[start:stop],prediction)
        z=np.column_stack([np.ones(len(f)),np.clip((f-np.asarray(config['feature_mean']))/np.asarray(config['feature_std']),-5,5)])
        pn=(prediction-np.asarray(config['prediction_mean']))/scale
        delta=(np.clip(z@np.asarray(fit['coefficient']),-.05,.05)+pn*np.asarray(fit['slopes'])[None])*scale
        result[start:stop]=prediction+recipe['strength']*delta
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    p=np.load(args.predictions,mmap_mode='r');y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r');cache=ROOT/'data/finetune-holdout'
    raw=np.load(cache/'raw_features.npy',mmap_mode='r');rh=np.load(cache/'rh-raw.npy',mmap_mode='r');humidity=np.load(cache/'humidity.npy',mmap_mode='r')
    with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
    with (ROOT/'data/greysnow/leap-msm-ds/std_y.p').open('rb') as f:std=np.asarray(pickle.load(f)).reshape(-1)
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);weights=sample.select(sample.columns[1:]).to_numpy()[0]
    features=make_features(raw,columns,rh,p);mean=features[:75000].mean(axis=0);scale=np.maximum(features[:75000].std(axis=0),1e-10)
    z=np.column_stack([np.ones(len(p)),np.clip((features-mean)/scale,-5,5)])
    target_scale=np.where(std>0,std,1);pmean=p[:75000].mean(axis=0);pn=(p-pmean)/target_scale;residual=(y[:75000]-p[:75000])/target_scale
    inactive=(weights==0)|(std==0);inactive[132:148]=True
    selection=slice(75000,100000);audit=slice(100000,200000)
    score=lambda prediction,part:float(r2_score(y[part]*weights,prediction[part]*weights))
    base_score=score(p,selection);selection_scores={'baseline':base_score};fits={};best_key='baseline';best_selection=base_score;best_prediction=np.asarray(p).copy()
    with threadpool_limits(limits=8):
        zz=z[:75000].T@z[:75000]/75000;cross=z[:75000].T@pn[:75000]/75000;rhs=z[:75000].T@residual/75000
        pp=(pn[:75000]**2).mean(axis=0)+.1;pe=(pn[:75000]*residual).mean(axis=0)
        for regularization in [1,10,100]:
            penalty=np.eye(z.shape[1])*regularization;penalty[0,0]=0
            inverse=np.linalg.inv(zz+penalty);ib=inverse@rhs;ic=inverse@cross
            slopes=np.clip((pe-(cross*ib).sum(axis=0))/np.maximum(pp-(cross*ic).sum(axis=0),1e-10),-.02,.02)
            slopes[inactive]=0;coef=inverse@(rhs-cross*slopes[None]);coef[:,inactive]=0
            correction=(np.clip(z@coef,-.05,.05)+pn*slopes[None])*target_scale
            fits[str(regularization)]={'coefficient':coef.tolist(),'slopes':slopes.tolist()}
            for strength in [.05,.1,.25,.5,1]:
                name=f'ridge{regularization}-alpha{strength}';candidate=project(p+strength*correction,humidity,weights);current=score(candidate,selection);selection_scores[name]=current
                if current>best_selection:best_key=name;best_selection=current;best_prediction=candidate.copy()
            print('Ridge calibrated',regularization,'best_selection',best_selection,flush=True)
    if best_key=='baseline':recipe={'regularization':None,'strength':0}
    else:
        regularization,best_strength=best_key.split('-alpha');recipe={'regularization':int(regularization.removeprefix('ridge')),'strength':float(best_strength)}
    config={'selected':best_key,'recipe':recipe,'fits':fits,'feature_names':FEATURE_NAMES,'feature_mean':mean.tolist(),'feature_std':scale.tolist(),'prediction_mean':pmean.tolist(),'target_std':target_scale.tolist(),'feature_clip':[-5,5],'state_bias_clip':[-.05,.05],'slope_bounds':[-.02,.02],'selection_scores':selection_scores,'calibration_rows':[0,75000],'selection_rows':[75000,100000],'audit_rows':[100000,200000],'baseline_predictions':str(args.predictions.resolve()),'test_labels_used':False}
    (args.output/'frozen_config.json').write_text(json.dumps(config,indent=2))
    baseline_audit=score(p,audit);candidate_audit=score(best_prediction,audit)
    report={**config,'baseline_audit':baseline_audit,'selected_audit':candidate_audit,'audit_gain':candidate_audit-baseline_audit,'audit_note':'Repeated training-era diagnostic holdout; calibration is not backbone training'}
    np.save(args.output/'predictions.npy',best_prediction);(args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    print('Residual result',json.dumps({'selected':best_key,'baseline_audit':baseline_audit,'selected_audit':candidate_audit,'audit_gain':report['audit_gain']}),flush=True)


if __name__=='__main__':main()
