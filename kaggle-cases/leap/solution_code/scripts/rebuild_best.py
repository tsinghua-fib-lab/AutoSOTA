"""Rebuild the strongest scored submission from frozen recipes and verify it."""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--current',type=Path,default=ROOT/'configs/sota-current.json');args=p.parse_args();current=json.loads(args.current.read_text());original=Path(current['file']);ref=current['best_submission']['ref'];manifest=json.loads(original.with_name(original.name+'.manifest.json').read_text());recipe_path=Path(manifest['config']);config=json.loads(recipe_path.read_text());output=ROOT/'artifacts/submissions'/('rebuilt-submission-'+ref+'.parquet')
    if sha(original)!=manifest['sha256']:raise ValueError('Scored file checksum differs')
    if 'specs' in config:
        run(['scripts/regime_ensemble.py','--build','--recipe',recipe_path,'--output',output])
    elif 'initial_combo_config' in config:
        run(['scripts/full_pipeline_combine.py','--build','--recipe',recipe_path,'--compare-validation',config['compare_validation'],'--output',output])
    elif 'fits' in config and 'normalized_correction_cap' in config and config.get('recipe',{}).get('fit','').startswith('bins'):
        run(['scripts/output_recalibration.py','--build','--recipe',recipe_path,'--baseline',manifest['baseline'],'--output',output])
    elif 'nonlinear_config' in config:
        base_manifest=json.loads(Path(manifest['baseline']+'.manifest.json').read_text());parent=ROOT/'artifacts/combined'/('rebuild-parent-'+ref)
        run(['scripts/combine_finetunes.py','--build','--recipe',base_manifest['config'],'--output',parent]);base=ROOT/'artifacts/submissions'/(parent.name+'.parquet')
        run(['scripts/extend_submission.py','--test-base',base,'--recipe',recipe_path,'--output',output,'--compact'])
    else:raise ValueError('Unsupported best recipe; no artifact overwritten')
    a,b=pq.ParquetFile(original),pq.ParquetFile(output)
    if a.schema_arrow!=b.schema_arrow:raise ValueError('Schema differs')
    mismatches=0;rows=0;maximum=0.
    for x,y in zip(a.iter_batches(batch_size=10000),b.iter_batches(batch_size=10000)):
        if x.num_rows!=y.num_rows:raise ValueError('Batch size differs')
        for j in range(x.num_columns):
            u,v=x.column(j).to_numpy(zero_copy_only=False),y.column(j).to_numpy(zero_copy_only=False);mismatches+=int(np.count_nonzero(u!=v))
            if j:maximum=max(maximum,float(np.max(np.abs(u-v))))
        rows+=x.num_rows
    if rows!=a.metadata.num_rows or rows!=b.metadata.num_rows or mismatches:raise ValueError(f'Rebuilt values differ: {mismatches}')
    report={'submission_ref':ref,'original_file':str(original.relative_to(ROOT)),'rebuilt_file':str(output.relative_to(ROOT)),'original_sha256':sha(original),'rebuilt_sha256':sha(output),'rows':rows,'targets':a.metadata.num_columns-1,'schema_equal':True,'values_and_ids_equal':True,'mismatched_values':mismatches,'maximum_absolute_difference':maximum,'reconstruction':'checkpoint prediction caches plus frozen JSON recipe chain','full_checkpoint_reinference_performed':False,'test_labels_used':False,'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()};folder=ROOT/'release'/('submission-'+ref);folder.mkdir(parents=True,exist_ok=True);(folder/'cached-reconstruction-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
