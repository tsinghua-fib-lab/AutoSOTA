"""Durable first-round real-data experiments, using one process per GPU.

Waits for the checked archive, packs data, and runs two upstream families plus
a paired synthetic-pretraining control. It records failures rather than silently
treating them as completed experiments. No Kaggle submission is made here.
"""
import concurrent.futures
import datetime
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'reports/phase1_status.json'
LOCK = threading.Lock()
status = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'stage': 'waiting_for_archive', 'jobs': {}, 'completed': False}


def save():
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(STATE)


def run(name, arguments, env):
    with LOCK:
        status['jobs'][name] = {'state': 'running', 'command': arguments,
                                'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'log': str(ROOT / f'logs/{name}.log')}
        save()
    started = time.monotonic()
    with (ROOT / f'logs/{name}.log').open('w') as handle:
        process = subprocess.Popen(arguments, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        with LOCK:
            status['jobs'][name]['pid'] = process.pid
            save()
        code = process.wait()
    with LOCK:
        status['jobs'][name].update(state='completed' if code == 0 else 'failed',
                                  exit_code=code, seconds=time.monotonic()-started)
        save()
    if code != 0:
        raise RuntimeError(f'{name} failed, see logs/{name}.log')


def main():
    os.chdir(ROOT)
    save()
    archive = ROOT / 'data/raw/g2net-gravitational-wave-detection.zip'
    partial = archive.with_suffix('.ranges.partial')
    while not (partial.exists() and archive.with_suffix('.ranges.json').exists()) and not (
            archive.exists() and archive.stat().st_size == 74086993965
            and not archive.with_name(archive.name+'.kaggle-partial').exists()):
        time.sleep(15)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(ROOT / '.cache/torch'), HF_HOME=str(ROOT / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(ROOT / '.cache/numba'), MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(ROOT / '.cache'))
    status['stage'] = 'packing_official_data'
    save()
    prepare = [str(ROOT / 'envs/tools/bin/python'), 'scripts/prepare_data.py', '--workers', '32']
    if partial.exists():
        prepare.extend(['--streaming', '--archive', str(partial)])
    run('prepare_official_data', prepare, env)
    metadata = json.loads((ROOT / 'data/packed/metadata.json').read_text())
    if metadata.get('fixture'):
        raise RuntimeError('Official phase refuses fixture data')
    project_path = ROOT / 'reports/project_status.json'
    project = json.loads(project_path.read_text())
    project.update(official_data_downloaded=True, full_score_reproduction='first-round real-data training running',
                   optimization_experiments_on_competition_data='paired synthetic-pretraining trial running')
    project_path.write_text(json.dumps(project, indent=2))
    status['stage'] = 'training_and_prediction'
    save()
    python = str(ROOT / 'envs/reproduce/bin/python')

    def event_job(config, gpu, tag):
        run(tag, [python, 'scripts/run_eventhorizon.py', '--config', config,
                  '--fold', '0', '--gpu', str(gpu), '--workers', '8', '--experiment-name', tag], env)
        run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'eventhorizon', '--config', config,
                            '--checkpoint', f'results/{tag}/fold0.pt', '--output', f'experiments/{tag}/predictions',
                            '--fold', '0', '--gpu', str(gpu), '--tta'], env)

    def scratch_job():
        tag = 'real_team3_v2_scratch'
        run(tag, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '1',
                  '--workers', '8', '--experiment-name', tag], env)
        run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', 'V2',
                            '--checkpoint', f'experiments/{tag}/fold0/Fold_0_best_model.pth',
                            '--output', f'experiments/{tag}/predictions', '--fold', '0', '--gpu', '1', '--tta'], env)

    def synthetic_job():
        bank = ROOT / 'data/synthetic_team3_300k'
        while not (bank / 'metadata.json').exists():
            time.sleep(15)
        pretrain = 'real_team3_v2_synthetic_pretrain'
        run(pretrain, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '3',
                       '--epochs', '2', '--synthetic-bank', str(bank), '--workers', '8',
                       '--experiment-name', pretrain], env)
        tag = 'real_team3_v2_synthetic_finetune'
        run(tag, [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', '0', '--gpu', '3',
                  '--checkpoint-dir', f'experiments/{pretrain}/fold0', '--workers', '8', '--experiment-name', tag], env)
        run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', 'V2',
                            '--checkpoint', f'experiments/{tag}/fold0/Fold_0_best_model.pth',
                            '--output', f'experiments/{tag}/predictions', '--fold', '0', '--gpu', '3', '--tta'], env)

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(event_job, 'Seq03aug3', 0, 'real_event_seq03'),
                   executor.submit(scratch_job),
                   executor.submit(event_job, 'Nspec22aug1', 2, 'real_event_wavegram'),
                   executor.submit(synthetic_job)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
    with LOCK:
        status.update(stage='finished' if not errors else 'finished_with_failures',
                      errors=errors, completed=not errors)
        save()
    if errors:
        raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
