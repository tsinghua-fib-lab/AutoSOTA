"""Rebuild autosota submission 57180922 from its frozen cached prediction recipe."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_workspace import stage, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-source', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--link-mode', choices=['auto', 'hardlink', 'copy'], default='auto')
    parser.add_argument('--verify-assets', action='store_true')
    args = parser.parse_args()
    root = stage(args.assets_source, args.workspace, 'cached', args.link_mode, args.verify_assets)
    subprocess.run([sys.executable, str(root / 'scripts/rebuild_best.py')], cwd=root, check=True)
    proof_path = root / 'release/submission-57180922/cached-reconstruction-check.json'
    proof = json.loads(proof_path.read_text())
    if not proof['values_and_ids_equal'] or proof['original_sha256'] != proof['rebuilt_sha256']:
        raise RuntimeError('Reconstruction verification failed')
    generated = root / proof['rebuilt_file']
    output = (args.output or root / 'autosota-best-57180922.parquet').resolve()
    if output == (root / proof['original_file']).resolve() or output == (args.assets_source / proof['original_file']).resolve():
        raise ValueError('Output must not overwrite the scored reference file')
    if output.exists():
        raise FileExistsError('Output already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated), output)
    report = {'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'submission_ref': '57180922', 'public_score': '0.79431', 'private_score': '0.79165',
              'output': str(output), 'output_sha256': sha256(output),
              'rows': proof['rows'], 'targets': proof['targets'],
              'values_and_ids_equal': proof['values_and_ids_equal'],
              'schema_equal': proof['schema_equal'], 'mismatched_values': proof['mismatched_values'],
              'scored_reference_sha256': proof['original_sha256'],
              'reproduction': 'frozen learned recipe applied to existing checkpoint prediction caches',
              'checkpoint_reinference_performed': False, 'submitted_to_kaggle': False}
    (root / 'AUTOSOTA_REPRODUCTION_RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
