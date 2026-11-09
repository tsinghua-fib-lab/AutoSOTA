"""Derive RH from available column inputs using the host thermodynamic formulas.

Saturation pressure formulas follow climsim_utils/data_utils.py in the Apache-2.0
host reference repository. No sample location, time, or target enters features.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
STATISTICS=ROOT/"data/physical-features/rh-statistics-lr2m.json"


def liquid_pressure(t):
    coefficients=np.array([-.976195544e-15,-.952447341e-13,.640689451e-10,.206739458e-7,.302950461e-5,.264847430e-3,.142986287e-1,.443987641,6.11239921])
    return 100*np.polyval(coefficients,np.maximum(-80,t-273.16))


def ice_pressure(t):
    coefficients=np.array([.252751365e-14,.146898966e-11,.385852041e-9,.602588177e-7,.615021634e-5,.420895665e-3,.188439774e-1,.503160820,6.11147274])
    cold=np.maximum(-100,t-273.16)
    return np.where(t>273.15,liquid_pressure(t),np.where(t>185,100*np.polyval(coefficients,t-273.16),100*(.00763685+cold*(.000151069+cold*7.48215e-7))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--fit",action="store_true")
    parser.add_argument("--test-csv",type=Path)
    args=parser.parse_args()
    cache_manifest=json.loads((args.cache/'manifest.json').read_text()) if (args.cache/'manifest.json').exists() else {}
    grid_file='data_hr.p' if cache_manifest.get('resolution')=='hr' else 'gdata.p'
    with (ROOT/"data/greysnow/leap-gdata-ds"/grid_file).open('rb') as f:grid=pickle.load(f)
    hyam=np.asarray(grid[2]).reshape(60);hybm=np.asarray(grid[4]).reshape(60)
    with (ROOT/"data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p").open('rb') as f:columns=list(pickle.load(f))
    names=[f'state_t_{i}' for i in range(60)]+[f'state_q0001_{i}' for i in range(60)]+['state_ps']
    if args.test_csv:
        frame=pl.read_csv(args.test_csv,columns=names).select(names)
        values=frame.to_numpy();indices=np.arange(121)
    else:
        values=np.load(args.cache/'raw_features.npy',mmap_mode='r');indices=[columns.index(n) for n in names]
    if args.fit:
        manifest=json.loads((args.cache/'manifest.json').read_text())
        if manifest.get('validation_only') or manifest.get('resolution','lr')!='lr':raise ValueError('Fit RH normalization on LR training features only')
    raw_path=args.cache/'rh-raw.npy';raw=np.lib.format.open_memmap(raw_path,mode='w+',dtype=np.float32,shape=(len(values),60))
    sums=np.zeros(60);squares=np.zeros(60);fallback_count=0
    for start in range(0,len(values),16384):
        features=np.asarray(values[start:start+16384][:,indices],dtype=np.float64)
        t=features[:,:60];q=features[:,60:120];p=100000*hyam[None]+features[:,120,None]*hybm[None]
        omega=np.clip((t-253.16)/20,0,1);es=omega*liquid_pressure(t)+(1-omega)*ice_pressure(t)
        invalid=(~np.isfinite(es))|(es<=0)
        # Rare simulated temperatures exceed the fitted polynomial's range.
        # Keep the original model inputs; stabilize only this added feature.
        fallback_count+=int(invalid.sum())
        es=np.where(invalid,liquid_pressure(np.clip(t,193.16,350)),es)
        qs=287*es/(461*p)
        if not np.isfinite(qs).all() or (qs<=0).any():raise ValueError('Invalid saturation humidity')
        rh=np.clip(q/qs,0,2)
        raw[start:start+len(features)]=rh;sums+=rh.sum(axis=0);squares+=(rh**2).sum(axis=0)
    raw.flush()
    if args.fit:
        mean=sums/len(values);std=np.maximum(np.sqrt(np.maximum(squares/len(values)-mean**2,0)),.05)
        STATISTICS.parent.mkdir(parents=True,exist_ok=True)
        STATISTICS.write_text(json.dumps({'mean':mean.tolist(),'std':std.tolist(),'fit_cache':str(args.cache.resolve()),'rows':len(values),'clip':[0,2],'formula_source':'leap-stc/climsim-kaggle-edition/climsim_utils/data_utils.py','labels_used':False,'location_or_time_used':False},indent=2))
    statistics=json.loads(STATISTICS.read_text());mean=np.array(statistics['mean']);std=np.array(statistics['std'])
    temporary=args.cache/'rh.npy.partial';out=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=raw.shape)
    for start in range(0,len(raw),16384):out[start:start+16384]=(raw[start:start+16384]-mean)/std
    out.flush();del out;temporary.replace(args.cache/'rh.npy')
    (args.cache/'rh.manifest.json').write_text(json.dumps({'rows':len(raw),'statistics':str(STATISTICS),'statistics_sha256':hashlib.sha256(STATISTICS.read_bytes()).hexdigest(),'grid_coefficients':grid_file,'polynomial_fallback_cells':fallback_count,'input_features_only':True},indent=2))
    print(f'RH representation ready: {args.cache}, rows={len(raw)}',flush=True)


if __name__=='__main__':main()
