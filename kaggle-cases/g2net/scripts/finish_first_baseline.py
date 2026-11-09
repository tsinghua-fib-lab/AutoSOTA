"""Compare the paired full-data pilots and submit one checked baseline automatically."""
import json
import os
import time

import run_phase1 as runner


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/baseline_submission_status.json'
    runner.status['stage'] = 'waiting_for_full_data_predictions'
    runner.save()
    tags = ['real_team3_v2_scratch', 'real_team3_v2_synthetic_finetune']
    while not all((root / f'experiments/{tag}/predictions/metrics.json').exists() for tag in tags):
        initial = json.loads((root / 'reports/phase1_status.json').read_text())
        for tag in tags:
            for name in [tag, tag+'_predict']:
                if initial['jobs'].get(name, {}).get('state') == 'failed':
                    raise RuntimeError(f'Required baseline artifact failed: {name}')
        time.sleep(15)
    env = os.environ.copy()
    runner.status['stage'] = 'comparing_and_submitting'
    runner.save()
    python = str(root / 'envs/tools/bin/python')
    runner.run('compare_normalized_synthetic_to_scratch', [python, 'scripts/compare_predictions.py',
                '--baseline', f'experiments/{tags[0]}/predictions',
                '--candidate', f'experiments/{tags[1]}/predictions',
                '--output', 'experiments/comparison_normalized_synthetic'], env)
    evaluated = [(json.loads((root / f'experiments/{tag}/predictions/metrics.json').read_text())['competition_validation_auc'], tag)
                 for tag in tags]
    auc, tag = max(evaluated)
    runner.status.update(selected_model=tag, selected_validation_auc=auc,
                         selection_scope='Same fold 0, seed 48, 112000 held-out official observations')
    runner.save()
    runner.run('submit_first_full_data_baseline', [python, 'scripts/submit_checked.py',
                '--predictions', f'experiments/{tag}/predictions',
                '--message', f'C33 first full-data baseline {tag} fold0 sign-and-LIGO-swap TTA'], env)
    project_path = root / 'reports/project_status.json'
    project = json.loads(project_path.read_text())
    records = sorted((root / 'reports/submissions').glob('*.json'))
    project['kaggle_submissions'] = sum(json.loads(path.read_text()).get('submission') is not None for path in records)
    project['baseline_selection'] = {'model': tag, 'validation_auc': auc}
    project_path.write_text(json.dumps(project, indent=2))
    runner.status.update(stage='finished', completed=True)
    runner.save()


if __name__ == '__main__':
    main()
