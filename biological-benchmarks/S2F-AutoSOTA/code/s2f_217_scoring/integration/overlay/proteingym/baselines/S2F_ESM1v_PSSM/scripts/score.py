#!/usr/bin/env python3
"""Portable, label-independent entry point for the frozen 217-assay method."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    'S2F_SCORING_ESM': 'ESM-1v-ens5', 'S2F_ESM_WEIGHT': '0.20',
    'S2F_WT_ESM_WEIGHT': '0.30', 'S2F_PSSM_WEIGHT': '0.25',
    'S2F_PSSM_ADAPTIVE': '1', 'S2F_MULTI_WINDOW': '1',
    'S2F_SITE_AGG_ALPHA': '0',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--DMS-data-folder', type=Path, required=True)
    p.add_argument('--DMS-reference-file', type=Path, default=ROOT / 'reference/DMS_substitutions.csv')
    p.add_argument('--structure-folder', type=Path, required=True)
    p.add_argument('--msa-folder', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--esm-weights', type=Path, required=True)
    p.add_argument('--output-scores-folder', type=Path, required=True)
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument('--all', action='store_true', help='Score every assay, including Binding')
    select.add_argument('--assay-id', action='append')
    select.add_argument('--DMS-index', type=int, help='Zero-based row index in the reference CSV')
    p.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value for this process')
    p.add_argument('--verify-weights', action='store_true', help='Hash all ESM files, in addition to the checkpoint')
    p.add_argument('--prepare-only', action='store_true', help='Validate inputs and stage config; do not run inference')
    a = p.parse_args()
    for name in ('DMS_data_folder', 'DMS_reference_file', 'structure_folder', 'msa_folder',
                 'checkpoint', 'esm_weights', 'output_scores_folder'):
        setattr(a, name, getattr(a, name).resolve())
    meta = read_rows(a.DMS_reference_file)
    if len({r['DMS_id'] for r in meta}) != len(meta):
        raise ValueError('Duplicate reference DMS IDs')
    if a.DMS_index is not None:
        if not 0 <= a.DMS_index < len(meta):
            raise ValueError('DMS index out of range')
        meta = [meta[a.DMS_index]]
    elif a.assay_id:
        requested = set(a.assay_id)
        meta = [r for r in meta if r['DMS_id'] in requested]
        if {r['DMS_id'] for r in meta} != requested:
            raise ValueError('Unknown assay ID')
    if not meta:
        raise ValueError('No assays selected')
    weights = json.loads((ROOT / 'reference/weights.json').read_text())
    for w in weights:
        path = a.checkpoint if w['filename'] == 'model_best.pth' else a.esm_weights / w['filename']
        if not path.is_file() or path.stat().st_size != w['bytes']:
            raise ValueError('Missing or wrong-size weight: %s' % path)
        if (w['filename'] == 'model_best.pth' or a.verify_weights) and sha256(path) != w['sha256']:
            raise ValueError('Weight SHA256 mismatch: %s' % path)

    run = a.output_scores_folder / '_runs' / uuid.uuid4().hex
    runtime = run / 'runtime'
    data = runtime / 'dataset' / 'assays'
    data.mkdir(parents=True)
    for name in ('s3f', 'script', 'util.py'):
        (runtime / name).symlink_to(ROOT / 'vendor/s3f' / name)
    (runtime / 'dataset/DMS_msa_files').symlink_to(a.msa_folder, target_is_directory=True)
    # The archived evaluator calls abspath(__file__), so the runtime script symlink
    # keeps its relative MSA/cache path local to this isolated run.
    originals = {}
    for row in meta:
        filename = row['DMS_filename']
        if Path(filename).name != filename:
            raise ValueError('DMS filenames must be basenames')
        for pdb in row['pdb_file'].split('|'):
            if not (a.structure_folder / pdb).is_file():
                raise ValueError('Missing structure: %s' % pdb)
        if row.get('MSA_filename') and not (a.msa_folder / row['MSA_filename']).is_file():
            raise ValueError('Missing alignment: %s' % row['MSA_filename'])
        original = read_rows(a.DMS_data_folder / filename)
        mutants = [r['mutant'] for r in original]
        if not mutants or len(mutants) != len(set(mutants)):
            raise ValueError('Empty or duplicate input mutants: %s' % filename)
        for mutant in mutants:
            for m in mutant.split(':'):
                pos = int(m[1:-1])
                if not 1 <= pos <= len(row['target_seq']) or row['target_seq'][pos - 1] != m[0]:
                    raise ValueError('Mutation/target mismatch: %s %s' % (filename, m))
                if m[-1] not in 'ACDEFGHIKLMNPQRSTVWY':
                    raise ValueError('Nonstandard mutant residue: %s' % m)
        originals[row['DMS_id']] = mutants
        # Discard experimental labels; the legacy evaluator requires a numeric
        # target only for diagnostic metrics. Its scores do not use these zeros.
        write_rows(data / filename, [{'mutant': m, 'DMS_score': 0} for m in mutants], ['mutant', 'DMS_score'])
    write_rows(data / 'DMS_substitutions.csv', meta, list(meta[0]))
    cfg = json.loads((ROOT / 'reference/model_config.json').read_text())
    cfg.update(output_dir=str(run / 'raw'), structure_path=str(a.structure_folder),
               model_checkpoint=str(a.checkpoint), id_list=[r['DMS_id'] for r in meta])
    cfg['summary']['path'] = str(data)
    cfg['task']['model']['sequence_model']['path'] = str(a.esm_weights)
    # JSON is valid YAML and avoids adding a dependency to this launcher.
    config = run / 'config.yaml'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    manifest = {'assays': cfg['id_list'], 'settings': SETTINGS, 'seed': 1024,
                'experimental_labels_passed_to_inference': False, 'checkpoint_sha256': weights[0]['sha256'],
                'status': 'prepared'}
    report = run / 'run.json'
    report.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Run directory: %s' % run, flush=True)
    if a.prepare_only:
        return
    env = {k: v for k, v in os.environ.items() if not k.startswith('S2F_')}
    env.update(SETTINGS)
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, SLURM_JOB_ID=run.name, PYTHONHASHSEED='1024')
    env.setdefault('MPLCONFIGDIR', str(run / 'matplotlib'))
    cmd = [sys.executable, str(runtime / 'script/evaluate.py'), '-c', str(config), '-s', '1024']
    with open(run / 'inference.log', 'w') as log:
        result = subprocess.run(cmd, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Inference failed; inspect %s' % (run / 'inference.log'))
    for row in meta:
        files = list((run / 'raw').rglob(row['DMS_filename']))
        if len(files) != 1:
            raise RuntimeError('Expected one prediction file for %s' % row['DMS_id'])
        predicted = read_rows(files[0])
        by_mutant = {r['mutant']: r for r in predicted}
        expected = originals[row['DMS_id']]
        if len(predicted) != len(by_mutant) or set(by_mutant) != set(expected):
            raise RuntimeError('Prediction mutant set mismatch')
        ordered = [{'mutant': m, 'model_score': by_mutant[m]['model_score']} for m in expected]
        if not all(math.isfinite(float(r['model_score'])) for r in ordered):
            raise RuntimeError('Nonfinite predictions')
        dest = a.output_scores_folder / (row['DMS_id'] + '.csv')
        tmp = dest.with_suffix('.csv.tmp')
        write_rows(tmp, ordered, ['mutant', 'model_score'])
        tmp.replace(dest)
    manifest.update(status='complete', total_mutants=sum(map(len, originals.values())))
    report.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
