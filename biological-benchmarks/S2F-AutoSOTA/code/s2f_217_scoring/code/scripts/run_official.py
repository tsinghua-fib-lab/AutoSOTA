#!/usr/bin/env python3
"""Run the archived, unmodified ProteinGym merge and performance scripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'S2F_ESM1v_PSSM'  # provisional stable identifier, not a publication title


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.scores, a.data, a.output = a.scores.resolve(), a.data.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    ref = ROOT / 'reference/DMS_substitutions_official.csv'
    meta = pd.read_csv(ref)
    assert len(meta) == 217 and meta.DMS_id.is_unique
    expected = {x + '.csv' for x in meta.DMS_id}
    assert {x.name for x in a.scores.glob('*.csv')} == expected, 'Score file set mismatch'
    total = 0
    hashes = {}
    for row in meta.itertuples(index=False):
        truth = pd.read_csv(a.data / row.DMS_filename)
        path = a.scores / (row.DMS_id + '.csv')
        scores = pd.read_csv(path)
        assert len(truth) == row.DMS_total_number_mutants, row.DMS_id
        assert truth.mutant.is_unique and scores.mutant.is_unique, row.DMS_id
        assert list(truth.mutant) == list(scores.mutant), row.DMS_id
        assert np.isfinite(scores.model_score).all(), row.DMS_id
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        total += len(scores)
    (a.output / 'prediction_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    model_root = a.output / 'model_scores'
    model_root.mkdir(exist_ok=True)
    link = model_root / MODEL
    if link.exists():
        assert link.resolve() == a.scores
    else:
        link.symlink_to(a.scores, target_is_directory=True)
    vendor = ROOT / 'vendor/proteingym/proteingym'
    config = ROOT / 'reference/proteingym_config.json'
    merged = model_root / 'merged_scores'
    commands = [
        [sys.executable, str(vendor / 'merge.py'), '--DMS_assays_location', str(a.data),
         '--model_scores_location', str(model_root), '--DMS_reference_file', str(ref),
         '--config_file', str(config)],
        [sys.executable, '-c',
         'import sys,runpy,numpy as np; np.random.seed(1024); path=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(path,run_name="__main__")',
         str(vendor / 'performance_DMS_benchmarks.py'), '--input_scoring_files_folder', str(merged),
         '--output_performance_file_folder', str(a.output / 'performance'),
         '--DMS_reference_file_path', str(ref), '--DMS_data_folder', str(a.data),
         '--config_file', str(config), '--performance_by_depth'],
    ]
    (a.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for i, cmd in enumerate(commands):
        print('Running official %s' % ('merge' if i == 0 else 'performance'), flush=True)
        with open(a.output / ('merge.log' if i == 0 else 'performance.log'), 'w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        if i == 0:
            for row in meta.itertuples(index=False):
                d = pd.read_csv(merged / (row.DMS_id + '.csv'))
                assert len(d) == row.DMS_total_number_mutants and np.isfinite(d[MODEL]).all()
    result = {'assays': len(meta), 'mutants': total, 'model_id': MODEL, 'metrics': {}}
    for metric in ['Spearman', 'AUC', 'MCC', 'NDCG', 'Top_recall']:
        folder = a.output / 'performance' / metric
        d = pd.read_csv(folder / ('DMS_substitutions_%s_DMS_level.csv' % metric))
        assert set(d['DMS ID']) == set(meta.DMS_id)
        if metric == 'Spearman':
            assert np.isfinite(d[MODEL]).all()
        aggregate = d.groupby(['UniProt ID', 'Selection Type'])[MODEL].mean().groupby('Selection Type').mean()
        summary = pd.read_csv(folder / ('Summary_performance_DMS_substitutions_%s.csv' % metric))
        assert abs(summary['Average_' + metric].iloc[0] - aggregate.mean()) <= 0.00050001
        result['metrics'][metric] = {'official_display': float(summary['Average_' + metric].iloc[0]),
                                    'aggregate_from_official_DMS_table': float(aggregate.mean()),
                                    'undefined_assays': int(d[MODEL].isna().sum())}
    result['note'] = ('Only the submitted model was scored in this run. The generated rank 1 is local to this one-model table; '
                      'the bootstrap SE is a difference from the top included model (itself), hence zero. Neither is a global ranking or uncertainty estimate.')
    (a.output / 'verified_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
