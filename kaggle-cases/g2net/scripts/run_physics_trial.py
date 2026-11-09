"""Queue a KDL-generator experiment after the original V2 baseline releases GPU 1."""
import json
import os
import time

import run_phase1 as runner


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/physics_trial_status.json'
    runner.status.update(stage='waiting_for_gpu1_baseline',
                         architecture='Original third-place V2, not the unavailable KDL neural network',
                         prior_scope='External published mass predictor with unknown historical training scope')
    runner.save()
    while True:
        first_round = json.loads((root / 'reports/phase1_status.json').read_text())
        training = first_round['jobs'].get('real_team3_v2_scratch', {})
        prediction = first_round['jobs'].get('real_team3_v2_scratch_predict', {})
        if prediction.get('state') in ['completed', 'failed'] or training.get('state') == 'failed':
            break
        time.sleep(15)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    pre = 'real_team3_kdl_physical_pretrain24'
    fine = 'real_team3_kdl_physical_finetune'
    runner.status['stage'] = 'training_and_prediction'
    runner.save()
    try:
        runner.run(pre, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '1',
                         '--epochs', '24', '--lr', '0.003', '--workers', '8', '--no-augmentation',
                         '--synthetic-bank', str(root / 'data/synthetic_kdl_50k_fold0'),
                         '--experiment-name', pre], env)
        runner.run(fine, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '1',
                          '--epochs', '2', '--lr', '0.0001', '--workers', '8', '--no-augmentation',
                          '--checkpoint-dir', f'experiments/{pre}/fold0', '--experiment-name', fine], env)
        runner.run(fine+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', 'V2',
                    '--checkpoint', f'experiments/{fine}/fold0/Fold_0_best_model.pth',
                    '--output', f'experiments/{fine}/predictions', '--fold', '0', '--gpu', '1', '--tta'], env)
        runner.run('compare_kdl_physical_to_scratch', [str(root / 'envs/tools/bin/python'), 'scripts/compare_predictions.py',
                    '--baseline', 'experiments/real_team3_v2_scratch/predictions',
                    '--candidate', f'experiments/{fine}/predictions',
                    '--output', 'experiments/comparison_kdl_physical'], env)
        runner.status.update(stage='finished', completed=True)
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
