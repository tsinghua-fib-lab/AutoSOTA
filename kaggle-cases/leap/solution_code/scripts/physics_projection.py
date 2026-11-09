"""Joint solar-flux closure and snow/total-precipitation consistency."""
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
SOLAR=[360,364,365,366,367]
ALBEDO_NAMES=['cam_in_ASDIR','cam_in_ALDIR','cam_in_ASDIF','cam_in_ALDIF']


def albedo_features(kind):
    path=ROOT/'data/physical-features'/f'albedo-{kind}.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
    if kind=='validation':
        raw=np.load(ROOT/'data/finetune-holdout/raw_features.npy',mmap_mode='r')
        values=np.asarray(raw[:,[columns.index(c) for c in ALBEDO_NAMES]],np.float64)
    else:values=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=ALBEDO_NAMES).select(ALBEDO_NAMES).to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('Invalid albedos')
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,values)
    return np.load(path,mmap_mode='r')


def solar_coefficients(albedo):return np.column_stack([np.ones(len(albedo)),-(1-albedo)])


def positive_diagonal_projection(p,a,variances):
    # KKT solution y_i = max(p_i-lambda*variance_i*a_i,0).
    # Its scalar constraint is monotone. These brackets avoid divisions by
    # tiny absorbed fractions when a surface albedo is close to one.
    variances=np.asarray(variances)
    if not np.all(a[:,0]==1) or (a[:,1:]>0).any() or (variances<=0).any():raise ValueError('Unexpected solar constraint')
    absorbed=(-a[:,1:]*np.maximum(p[:,1:],0)).sum(axis=1)
    initial=np.maximum(p[:,0],0)-absorbed
    lower=np.where(initial<0,(p[:,0]-absorbed)/variances[0],0)
    upper=np.where(initial>0,np.maximum(p[:,0],0)/variances[0],0)
    direction=a*variances[None]
    for _ in range(60):
        mid=(lower+upper)/2
        result=np.maximum(p-mid[:,None]*direction,0)
        residual=(a*result).sum(axis=1)
        lower=np.where(residual>0,mid,lower);upper=np.where(residual>0,upper,mid)
    result=np.maximum(p-((lower+upper)/2)[:,None]*direction,0)
    if np.max(np.abs((a*result).sum(axis=1)))>1e-7:raise ValueError('Solar closure projection failed')
    return result


def solar_projection(p,albedo,covariance,target_variance):
    a=solar_coefficients(albedo);solar=p[:,SOLAR]
    direction=a@np.asarray(covariance).T;denominator=(a*direction).sum(axis=1)
    residual=(a*solar).sum(axis=1)
    multiplier=np.divide(residual,denominator,out=np.zeros(len(p)),where=denominator>1e-30)
    projected=solar-multiplier[:,None]*direction
    # Finish in the positive cone while retaining the exact flux relationship.
    projected=positive_diagonal_projection(projected,a,np.asarray(target_variance))
    result=p.copy();result[:,SOLAR]=projected
    return result


def precipitation_projection(p,variances):
    result=p.copy();snow=result[:,362];total=result[:,363]
    snow_var,total_var=np.maximum(np.asarray(variances),1e-40)
    violating=snow>total
    shared=np.maximum((snow/snow_var+total/total_var)/(1/snow_var+1/total_var),0)
    snow[violating]=shared[violating];total[violating]=shared[violating]
    snow[:]=np.maximum(snow,0);total[:]=np.maximum(total,0)
    if (snow>total+1e-20).any():raise ValueError('Precipitation ordering failed')
    return result


def apply_recipe(p,albedo,config):
    recipe=config['recipe'];result=np.asarray(p,np.float64)
    if recipe.get('solar'):
        matrix=config['solar_matrices'][recipe['solar']]
        result=solar_projection(result,albedo,matrix,config['solar_target_variance'])
    if recipe.get('precipitation'):result=precipitation_projection(result,config['precipitation_target_variance'])
    return result
