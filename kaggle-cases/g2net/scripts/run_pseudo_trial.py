"""Use same-fold teacher predictions for a controlled soft-pseudo-label trial."""
import json
import os
import time

import numpy as np
import pandas as pd

import run_phase1 as runner


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/pseudo_trial_status.json'
    runner.status.update(stage='waiting_for_same_fold_teacher',
                         validation_policy='Teacher and student both exclude fold 0, seed 48')
    runner.save()
    selection_path = root / 'reports/baseline_submission_status.json'
    while True:
        selection = json.loads(selection_path.read_text())
        teacher = selection.get('selected_model')
        if teacher:
            break
        time.sleep(15)
    predictions = root / f'experiments/{teacher}/predictions'
    evidence = json.loads((predictions / 'metrics.json').read_text())
    if evidence.get('fixture') or evidence['fold'] != 0 or evidence['fold_seed'] != 48:
        raise ValueError('Teacher must correspond to the exact student validation split')
    test = pd.read_csv(root / 'data/packed/test.csv', dtype={'id': str})
    values = np.load(predictions / 'test_predictions.npy')
    tag = 'real_team3_v2_pseudo_low_lr'
    labels = root / f'experiments/{tag}/pseudo'
    labels.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': test.id, 'preds_Fold_0': values}).to_csv(labels / 'test_Fold_0.csv', index=False)
    runner.status.update(stage='training_and_prediction', teacher=teacher,
                         pseudo_samples=len(values), selection_metric=evidence['competition_validation_auc'])
    runner.save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    try:
        runner.run(tag, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '3',
                        '--lr', '0.001', '--epochs', '6', '--workers', '8',
                        '--checkpoint-dir', f'experiments/{teacher}/fold0', '--pseudo-label-dir', str(labels),
                        '--experiment-name', tag], env)
        runner.run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', 'V2',
                    '--checkpoint', f'experiments/{tag}/fold0/Fold_0_best_model.pth',
                    '--output', f'experiments/{tag}/predictions', '--fold', '0', '--gpu', '3', '--tta', '--mc-folds', '64'], env)
        runner.run('compare_pseudo_to_teacher', [str(root / 'envs/tools/bin/python'), 'scripts/compare_predictions.py',
                    '--baseline', str(predictions), '--candidate', f'experiments/{tag}/predictions',
                    '--output', 'experiments/comparison_pseudo'], env)
        runner.status.update(stage='finished', completed=True)
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
