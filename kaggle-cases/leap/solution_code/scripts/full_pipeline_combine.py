"""Choose parent replacements after all frozen calibration and physical steps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from combine_finetunes import conditional_weights,raw_ensemble,replacement
from conditional_ensemble import ROOT,project
from extend_submission import apply_extensions
from mix_finetune_children import context
from residual_calibration import apply_calibration


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--children',nargs='+');p.add_argument('--initial-combo',type=Path,default=ROOT/'artifacts/combined/thirteen-completed/frozen_config.json');p.add_argument('--extensions',type=Path,default=ROOT/'artifacts/combined/thirteen-completed-rbf-momentum/frozen_config.json');p.add_argument('--compare-validation',type=Path,required=True);p.add_argument('--recipe',type=Path);p.add_argument('--build',action='store_true');args=p.parse_args()
    config=json.loads(args.recipe.read_text()) if args.build else None
    combo=json.loads(Path(config['initial_combo_config'] if config else args.initial_combo).read_text());extension_path=Path(config['extension_config'] if config else args.extensions);extension=json.loads(extension_path.read_text());conditional=json.loads(Path(combo['conditional_config']).read_text());ridge=json.loads(Path(combo['ridge_config']).read_text());kind='test' if args.build else 'validation';coeff=conditional_weights(kind,conditional);base=raw_ensemble(kind,conditional,coeff);current=base.copy()
    members={r['parent']:r for r in (config['replacements'] if args.build else combo['replacements'])}
    for r in members.values():current+=replacement(kind,r['parent'],r['child'],r['fraction'],coeff,conditional)
    raw,columns,rh,humidity,albedo=context(kind);sample=pl.read_csv(ROOT/'data/competition/sample_submission.csv',n_rows=1);names=sample.columns[1:];mask=sample.select(names).to_numpy()[0]
    def finish(v,part):
        r,h,rr,a=raw[part],humidity[part],rh[part],albedo[part];v=project(apply_calibration(project(v,h,mask),r,columns,rr,ridge),h,mask)
        return apply_extensions(v,r,columns,rr,h,mask,extension,kind,albedo_override=a)
    with threadpool_limits(limits=8):
        if not args.build:
            args.output.mkdir(parents=True,exist_ok=True);labels=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r');selection=slice(75000,100000);audit=slice(100000,200000);compare=np.load(args.compare_validation,mmap_mode='r');score=lambda v,part:float(r2_score(labels[part]*mask,v*mask));initial=finish(current[selection],selection);best=score(initial,selection);scores={'initial':best};baseline_score=score(compare[selection],selection)
            grouped={}
            for child in args.children or []:
                folder=ROOT/'artifacts/finetune'/child;cfg=json.loads((folder/'config.json').read_text())
                if not all((ROOT/'artifacts/extra-inference'/child/k/'completed.json').exists() for k in ['validation','test']):raise ValueError('Incomplete prediction caches')
                grouped.setdefault(cfg['model_id'],[]).append(child)
            for parent in conditional['models']:
                if parent not in grouped:continue
                old=members.get(parent);old_delta=replacement('validation',parent,old['child'],old['fraction'],coeff,conditional) if old else np.zeros_like(current);winner=None;winner_delta=None
                for child in sorted(grouped[parent]):
                    delta=replacement('validation',parent,child,1,coeff,conditional)
                    for fraction in [.25,.5,1]:
                        shift=fraction*delta-old_delta;value=score(finish(current[selection]+shift[selection],selection),selection);scores[child+f'-{fraction}']=value
                        if value>best+1e-14:best=value;winner={'parent':parent,'child':child,'fraction':fraction};winner_delta=shift
                if winner is not None:members[parent]=winner;current+=winner_delta
                print('Full-pipeline parent',parent,'selected',members.get(parent),flush=True)
            prediction=finish(current,slice(None));selected_audit=score(prediction[audit],audit);base_audit=score(compare[audit],audit);compact=prediction[audit].astype(np.float32).astype(np.float64);compact[:,132:148]=prediction[audit,132:148]
            config={'replacements':list(members.values()),'initial_combo_config':str(args.initial_combo.resolve()),'conditional_config':combo['conditional_config'],'ridge_config':combo['ridge_config'],'extension_config':str(extension_path.resolve()),'compare_validation':str(args.compare_validation.resolve()),'selection_scores':scores,'baseline_selection':baseline_score,'selected_selection':best,'selection_gain':best-baseline_score,'baseline_audit':base_audit,'selected_audit':selected_audit,'audit_gain':selected_audit-base_audit,'mixed_precision_audit_difference':score(compact,audit)-selected_audit,'selection_rows':[75000,100000],'audit_rows':[100000,200000],'audit_note':'Repeated training-era diagnostic; full frozen pipeline is inside selection; comparator is current best scored recipe','test_labels_used':False};np.save(args.output/'predictions.npy',prediction);(args.output/'frozen_config.json').write_text(json.dumps(config,indent=2));print('Full-pipeline gains',config['selection_gain'],config['audit_gain'],flush=True)
        else:
            if config['selection_gain']<=0 or config['audit_gain']<=1e-6:raise ValueError('Candidate did not pass checks')
            if abs(config['mixed_precision_audit_difference'])>1e-8:raise ValueError('Mixed precision changed audit')
            prediction=finish(current,slice(None));ids=np.load(ROOT/'artifacts/test-inference/ids.npy')
            if not np.isfinite(prediction).all() or not np.isin(mask,[0,1]).all():raise ValueError('Invalid values or sample mask')
            if not np.array_equal(ids,pl.read_csv(ROOT/'data/competition/sample_submission.csv',columns=['sample_id'])['sample_id'].to_numpy()):raise ValueError('ID order differs')
            args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(args.output.name+'.partial');pl.DataFrame({'sample_id':ids,**{c:prediction[:,i].astype(np.float32) if not 132<=i<148 else prediction[:,i] for i,c in enumerate(names)}}).write_parquet(temporary,compression='zstd');temporary.replace(args.output);h=hashlib.sha256()
            with args.output.open('rb') as f:
                for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
            args.output.with_name(args.output.name+'.manifest.json').write_text(json.dumps({'config':str(args.recipe.resolve()),'rows':len(ids),'targets':368,'sha256':h.hexdigest(),'weights_applied_once':True,'test_labels_used':False},indent=2));print('Full-pipeline submission ready',args.output,flush=True)


if __name__=='__main__':main()
