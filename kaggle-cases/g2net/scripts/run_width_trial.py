"""Compare the original wider stochastic-depth model on the same held-out fold."""
import os

import run_phase1 as runner


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/width_trial_status.json'
    runner.status.update(stage='training_and_prediction',
                         architecture='Original third-place V-V2SD, 32 channels, survival 0.8',
                         validation_policy='Fold 0, seed 48, training-fold-only noise spectrum',
                         override='Original FP32 precision retained; workers set to 8',
                         superseded_attempt='Mixed-precision run aborted because training loss was NaN; evidence retained')
    runner.save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    common = [python, 'scripts/run_team3.py', '--config', 'V-V2SD', '--fold', '0', '--gpu', '0',
              '--workers', '8']
    scratch = 'real_team3_width32_sd_fp32_scratch'
    pre = 'real_team3_width32_sd_fp32_synthetic_pretrain'
    fine = 'real_team3_width32_sd_fp32_synthetic_finetune'

    def predict(tag):
        runner.run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3',
                    '--config', 'V-V2SD', '--checkpoint', f'experiments/{tag}/fold0/Fold_0_best_model.pth',
                    '--output', f'experiments/{tag}/predictions', '--fold', '0', '--gpu', '0', '--tta'], env)

    def compare(name, baseline, candidate):
        runner.run(name, [str(root / 'envs/tools/bin/python'), 'scripts/compare_predictions.py',
                    '--baseline', baseline, '--candidate', candidate, '--output', f'experiments/{name}'], env)

    try:
        runner.run(scratch, common+['--epochs', '6', '--experiment-name', scratch], env)
        predict(scratch)
        compare('comparison_width32_to_v2', 'experiments/comparison_normalized_synthetic',
                f'experiments/{scratch}/predictions')
        runner.run(pre, common+['--epochs', '2', '--synthetic-bank', str(root / 'data/synthetic_team3_300k'),
                               '--experiment-name', pre], env)
        runner.run(fine, common+['--epochs', '6', '--checkpoint-dir', f'experiments/{pre}/fold0',
                                '--experiment-name', fine], env)
        predict(fine)
        compare('comparison_width32_initializations', f'experiments/{scratch}/predictions',
                f'experiments/{fine}/predictions')
        compare('comparison_four_v2_models', 'experiments/comparison_normalized_synthetic',
                'experiments/comparison_width32_initializations')
        runner.status.update(stage='finished', completed=True)
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
