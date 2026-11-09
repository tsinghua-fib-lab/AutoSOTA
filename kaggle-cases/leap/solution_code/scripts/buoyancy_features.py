"""Plume buoyancy from equations 19-22 of Beucler et al. (2024).

The archived bmse_calc uses actual q in a variable called h_satenv, whereas
equation 20 uses saturation humidity. This feature follows the paper equation.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from scipy.integrate import cumulative_trapezoid
from rh_features import ROOT,liquid_pressure,ice_pressure

STATISTICS=ROOT/'data/physical-features/buoyancy-statistics-lr2m.json'


def calculate(t,q,ps,hyam,hybm):
    p=100000*hyam[None]+ps[:,None]*hybm[None]
    omega=np.clip((t-253.16)/20,0,1);es=omega*liquid_pressure(t)+(1-omega)*ice_pressure(t)
    invalid=(~np.isfinite(es))|(es<=0);es=np.where(invalid,liquid_pressure(np.clip(t,193.16,350)),es)
    qs=287*es/(461*p);cp=1004.64;lv=2.501e6;gravity=9.80616
    integral=cumulative_trapezoid(t*(287+(461-287)*q)/(gravity*p),x=p,axis=1,initial=0)
    height=integral[:,[-1]]-integral
    plume=cp*t[:,[-1]]+lv*q[:,[-1]]
    saturated_environment=cp*t+lv*qs+gravity*height
    kappa=1+lv*lv*qs/(461*cp*t*t)
    b=gravity*(plume-saturated_environment)/(kappa*cp*t)
    if not np.isfinite(b).all() or (p<=0).any() or (qs<=0).any():raise FloatingPointError('Invalid buoyancy feature')
    return np.clip(b,-30,30),int(invalid.sum())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--fit',action='store_true');parser.add_argument('--test-csv',type=Path);args=parser.parse_args();manifest=json.loads((args.cache/'manifest.json').read_text()) if (args.cache/'manifest.json').exists() else {}
    if args.fit and (manifest.get('validation_only') or manifest.get('resolution')!='lr'):raise ValueError('Fit input statistics on LR training inputs only')
    grid_file='data_hr.p' if manifest.get('resolution')=='hr' else 'gdata.p'
    with (ROOT/'data/greysnow/leap-gdata-ds'/grid_file).open('rb') as f:grid=pickle.load(f)
    hyam=np.asarray(grid[2]);hybm=np.asarray(grid[4]);names=[f'state_t_{i}' for i in range(60)]+[f'state_q0001_{i}' for i in range(60)]+['state_ps']
    if args.test_csv:raw=pl.read_csv(args.test_csv,columns=names).select(names).to_numpy();indices=list(range(121))
    else:
        raw=np.load(args.cache/'raw_features.npy',mmap_mode='r')
        with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
        indices=[columns.index(n) for n in names]
    values=np.lib.format.open_memmap(args.cache/'buoyancy-raw.npy',mode='w+',dtype=np.float32,shape=(len(raw),60));sums=np.zeros(60);squares=np.zeros(60);fallback=0
    for start in range(0,len(raw),16384):
        x=np.asarray(raw[start:start+16384][:,indices],np.float64);b,count=calculate(x[:,:60],x[:,60:120],x[:,120],hyam,hybm);fallback+=count;values[start:start+len(x)]=b;sums+=b.sum(axis=0);squares+=(b*b).sum(axis=0)
    values.flush()
    if args.fit:
        mean=sums/len(raw);std=np.maximum(np.sqrt(np.maximum(squares/len(raw)-mean*mean,0)),.05);STATISTICS.write_text(json.dumps({'mean':mean.tolist(),'std':std.tolist(),'rows':len(raw),'fit_cache':str(args.cache.resolve()),'equations':'Beucler et al. 2024 SI equations 19-22','hsat_uses':'saturation specific humidity','height_reference':'lowest model midpoint','feature_clip':[-30,30],'labels_used':False,'test_inputs_used_to_fit':False},indent=2))
    stats=json.loads(STATISTICS.read_text());out=np.lib.format.open_memmap(args.cache/'buoyancy.npy',mode='w+',dtype=np.float32,shape=values.shape)
    for start in range(0,len(raw),16384):out[start:start+16384]=(values[start:start+16384]-np.asarray(stats['mean']))/np.asarray(stats['std'])
    out.flush();(args.cache/'buoyancy.manifest.json').write_text(json.dumps({'rows':len(raw),'statistics':str(STATISTICS),'statistics_sha256':hashlib.sha256(STATISTICS.read_bytes()).hexdigest(),'grid':grid_file,'saturation_polynomial_fallback_cells':fallback,'input_features_only':True},indent=2));print('Buoyancy features ready:',args.cache,len(raw),flush=True)


if __name__=='__main__':main()
