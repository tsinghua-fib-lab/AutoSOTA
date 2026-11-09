"""Apply frozen RBF, soft momentum and optional exact solar recipes in order."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from nonlinear_calibration import features,correction,ROOT
from momentum_shrinkage import apply_shrinkage
from physics_projection import albedo_features,apply_recipe
from residual_calibration import project


def apply_extensions(p,raw,columns,rh,humidity,weights,config,kind,albedo_override=None):
    result=p.copy()
    if config.get('nonlinear_config'):
        fit=json.loads(Path(config['nonlinear_config']).read_text());recipe=fit['recipe'];f=features(raw,columns,rh,result)
        if recipe['strength']:
            for start in range(0,len(p),20000):result[start:start+20000]+=recipe['strength']*correction(f[start:start+20000],fit,fit['fits'][recipe['fit']])
        result=project(result,humidity,weights)
    if config.get('momentum_config'):
        fit=json.loads(Path(config['momentum_config']).read_text());result=apply_shrinkage(result,raw[:,columns.index('state_ps')],fit)
    if config.get('solar_config'):
        fit=json.loads(Path(config['solar_config']).read_text());albedo=albedo_features(kind) if albedo_override is None else albedo_override;result=apply_recipe(result,albedo,fit)
    return project(result,humidity,weights)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation-base',type=Path);parser.add_argument('--test-base',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--recipe',type=Path);parser.add_argument('--nonlinear',action='store_true');parser.add_argument('--momentum',action='store_true');parser.add_argument('--solar',action='store_true');parser.add_argument('--compact',action='store_true');args=parser.parse_args()
    sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);out_columns=sample.columns[1:];weights=sample.select(out_columns).to_numpy()[0]
    with threadpool_limits(limits=8):
        if args.validation_base:
            args.output.mkdir(parents=True,exist_ok=True);cache=ROOT/'data/finetune-holdout';p=np.load(args.validation_base,mmap_mode='r');raw=np.load(cache/'raw_features.npy',mmap_mode='r');rh=np.load(cache/'rh-raw.npy',mmap_mode='r');humidity=np.load(cache/'humidity.npy',mmap_mode='r')
            with (ROOT/'data/greysnow/leap-sample-submission-stds-ds/FEAT_COLS.p').open('rb') as f:columns=list(pickle.load(f))
            config={'nonlinear_config':str(ROOT/'artifacts/nonlinear-calibration/frozen_config.json') if args.nonlinear else None,'momentum_config':str(ROOT/'artifacts/momentum-shrinkage/frozen_config.json') if args.momentum else None,'solar_config':str(ROOT/'artifacts/physics-projection-conditional/frozen_config.json') if args.solar else None,'baseline':str(args.validation_base.resolve()),'test_labels_used':False,'recipes_refit':False}
            prediction=apply_extensions(p,raw,columns,rh,humidity,weights,config,'validation');y=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r')
            score=lambda v,part:float(r2_score(y[part]*weights,v[part]*weights));sel=slice(75000,100000);audit=slice(100000,200000)
            config.update({'baseline_selection':score(p,sel),'selected_selection':score(prediction,sel),'baseline_audit':score(p,audit),'selected_audit':score(prediction,audit),'audit_note':'Repeated training-era diagnostic holdout; extensions originally fitted on prior baseline'})
            config['selection_gain']=config['selected_selection']-config['baseline_selection'];config['audit_gain']=config['selected_audit']-config['baseline_audit'];q=prediction.astype(np.float32).astype(np.float64);config['float32_audit_difference']=score(q,audit)-config['selected_audit']
            q[:,132:148]=prediction[:,132:148];config['mixed_precision_audit_difference']=score(q,audit)-config['selected_audit']
            (args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));np.save(args.output/'predictions.npy',prediction);print(json.dumps(config,indent=2),flush=True)
        else:
            config=json.loads(args.recipe.read_text())
            if config['selection_gain']<=0 or config['audit_gain']<=1e-6:raise ValueError('Extended candidate did not improve both checks')
            baseline=pl.read_parquet(args.test_base);ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
            if baseline.columns!=sample.columns or not np.array_equal(ids,baseline['sample_id'].to_numpy()):raise ValueError('Schema or IDs differ')
            if not np.isin(weights,[0,1]).all():raise ValueError('Non-binary mask would require explicit deweighting')
            columns=[f'state_t_{i}' for i in range(60)]+[f'{name}_{i}' for name in ['state_q0001','state_q0002','state_q0003'] for i in range(48,60)]+['state_ps','pbuf_LHFLX','pbuf_SHFLX','pbuf_SOLIN']+[f'{name}_{i}' for name in ['state_u','state_v'] for i in range(30,60)]+['cam_in_'+name for name in ['ASDIR','ALDIR','ASDIF','ALDIF']]
            raw=pl.read_csv(ROOT/'data/greysnow/leap-test-sample/test.csv',columns=columns).select(columns).to_numpy();cache=ROOT/'artifacts/test-inference';rh=np.load(cache/'rh-raw.npy',mmap_mode='r');humidity=np.load(cache/'humidity.npy',mmap_mode='r');p=baseline.select(out_columns).to_numpy();prediction=apply_extensions(p,raw,columns,rh,humidity,weights,config,'test')
            if args.compact:
                if abs(config['mixed_precision_audit_difference'])>1e-8:raise ValueError('Compact precision changed audit beyond threshold')
            if not np.isfinite(prediction).all():raise FloatingPointError('Non-finite extended candidate')
            args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial');pl.DataFrame({'sample_id':ids,**{c:prediction[:,i].astype(np.float32) if args.compact and not 132<=i<148 else prediction[:,i] for i,c in enumerate(out_columns)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output);digest=hashlib.sha256()
            with args.output.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'config':str(args.recipe.resolve()),'baseline':str(args.test_base.resolve()),'rows':len(ids),'targets':368,'sha256':digest.hexdigest(),'physical_dtype':'float32 except deterministic q0002 levels 12-27 kept float64' if args.compact else 'float64','mixed_precision_audit_difference':config['mixed_precision_audit_difference'],'weights_applied_once':True,'test_labels_used':False},indent=2));print('Extended candidate ready',args.output,flush=True)


if __name__=='__main__':main()
