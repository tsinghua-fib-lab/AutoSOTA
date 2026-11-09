"""Run an EventHorizon configuration and aligned predictions on its assigned GPU."""
import argparse
import os
from pathlib import Path

import run_phase1 as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    root = runner.ROOT
    if Path(args.tag).name != args.tag:
        raise ValueError('Invalid experiment tag')
    runner.STATE = root / f'reports/{args.tag}_status.json'
    runner.status.update(stage='training_and_prediction',
                         adaptation='Explicit current CUDA device fixes upstream single-device GPU selection')
    runner.save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    try:
        runner.run(args.tag, [python, 'scripts/run_eventhorizon.py', '--config', args.config,
                             '--fold', '0', '--gpu', str(args.gpu), '--workers', '8',
                             '--experiment-name', args.tag], env)
        runner.run(args.tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'eventhorizon',
                   '--config', args.config, '--checkpoint', f'results/{args.tag}/fold0.pt',
                   '--output', f'experiments/{args.tag}/predictions', '--fold', '0', '--gpu', str(args.gpu), '--tta'], env)
        runner.status.update(stage='finished', completed=True)
    except Exception as error:
        runner.status.update(stage='failed', error=str(error), completed=False)
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
