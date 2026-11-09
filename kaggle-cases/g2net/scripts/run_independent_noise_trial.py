"""Controlled physical-signal trial using independent generated noise."""
import json
import os
import time

import run_phase1 as runner


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/independent_noise_trial_status.json'
    runner.status.update(stage='waiting_for_gpu3_pseudo_trial',
                         validation_policy='Fold 0 seed 48; physical mass prior excludes this fold',
                         comparison='Same original V2, physical 50k bank, epochs/lr; noise source changed')
    runner.save()
    while True:
        previous = json.loads((root / 'reports/pseudo_trial_status.json').read_text())
        if previous['stage'] in ['finished', 'failed']:
            break
        time.sleep(15)
    audit = json.loads((root / 'reports/kdl_noise_bank_audit.json').read_text())
    if not audit['passed']:
        raise ValueError('Independent noise bank has not passed its distribution audit')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    pre, fine = 'real_team3_kdl_independent_noise_pretrain24', 'real_team3_kdl_independent_noise_finetune'
    runner.status['stage'] = 'training_and_prediction'
    runner.save()
    try:
        runner.run(pre, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '3',
                    '--epochs', '24', '--lr', '0.003', '--workers', '8', '--no-augmentation',
                    '--synthetic-bank', str(root / 'data/synthetic_kdl_50k_fold0'),
                    '--noise-bank', str(root / 'data/synthetic_kdl_noise_250k'), '--experiment-name', pre], env)
        runner.run(fine, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '3',
                    '--epochs', '2', '--lr', '0.0001', '--workers', '8', '--no-augmentation',
                    '--checkpoint-dir', f'experiments/{pre}/fold0', '--experiment-name', fine], env)
        runner.run(fine+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', 'V2',
                    '--checkpoint', f'experiments/{fine}/fold0/Fold_0_best_model.pth',
                    '--output', f'experiments/{fine}/predictions', '--fold', '0', '--gpu', '3', '--tta'], env)
        runner.run('compare_independent_noise_to_current_ensemble', [str(root / 'envs/tools/bin/python'),
                    'scripts/compare_predictions.py', '--baseline', 'experiments/comparison_normalized_synthetic',
                    '--candidate', f'experiments/{fine}/predictions',
                    '--output', 'experiments/comparison_independent_noise'], env)
        runner.status.update(stage='finished', completed=True)
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
