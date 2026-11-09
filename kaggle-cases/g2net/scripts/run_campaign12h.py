"""Durable 12-hour, four-GPU exploration campaign with explicit time limits.

Existing five-fold jobs retain their GPUs until their last prediction exits.
The campaign owns only its own process groups; budget enforcement cannot stop
the previously deployed run. Every new model uses seed-48 fold 0. Final ensemble
weights are fit on calibration rows and checked on the separate evaluation half.
"""
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / 'envs/reproduce/bin/python')
TOOLS = str(ROOT / 'envs/tools/bin/python')
PHYSICS = str(ROOT / 'envs/physics/bin/python')
STATE = ROOT / 'reports/campaign12h_status.json'
PLAN = ROOT / 'reports/campaign12h_plan.json'
LOCK = threading.RLock()
BANK_READY = threading.Event()
STOP = threading.Event()
STATUS = {}
DEADLINE = None
TRAINING_END = None
CANDIDATES = []
BANK = ROOT / 'data/synthetic_kdl_300k_fold0_campaign12h'
NOISE = ROOT / 'data/synthetic_kdl_noise_1m_campaign12h'
ENV = os.environ.copy()
ENV.update(OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', PYTHONUNBUFFERED='1',
           TORCH_HOME=str(ROOT / '.cache/torch'), HF_HOME=str(ROOT / '.cache/huggingface'),
           NUMBA_CACHE_DIR=str(ROOT / '.cache/numba'), MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'),
           XDG_CACHE_HOME=str(ROOT / '.cache'))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def save():
    # Caller holds the shared lock.
    STATUS['last_updated_at'] = now()
    STATUS['remaining_seconds'] = max(0., DEADLINE-time.time())
    atomic_json(STATE, STATUS)


def request_stop(*unused):
    STOP.set()


class BudgetReached(Exception):
    pass


def run(name, command, gpu=None, budget_seconds=None, usable_checkpoint=None, physics=False):
    if STOP.is_set() or time.time() >= DEADLINE:
        raise BudgetReached('Campaign deadline or stop request')
    end = min(DEADLINE, time.time()+budget_seconds) if budget_seconds else DEADLINE
    log = ROOT / f'logs/{name}.log'
    entry = dict(state='running', command=command, gpu=gpu, started_at=now(), log=str(log),
                 job_deadline_utc=dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat())
    began = time.time()
    env = dict(ENV, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1') if physics else ENV
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        entry['pid'] = process.pid
        with LOCK:
            STATUS['jobs'][name] = entry; save()
        limited = False
        while process.poll() is None:
            if STOP.is_set() or time.time() >= end:
                limited = True
                # Only the child process group created above belongs to this job.
                os.killpg(process.pid, signal.SIGTERM)
                grace = min(90., max(0., DEADLINE-time.time()))
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            STOP.wait(min(15., max(.1, end-time.time())))
        code = process.wait()
    checkpoint_ok = bool(usable_checkpoint and usable_checkpoint.exists())
    accepted_limit = limited and checkpoint_ok and not STOP.is_set() and time.time() < DEADLINE
    with LOCK:
        entry.update(state='budget_limited_checkpoint_available' if accepted_limit else
                     ('completed' if code == 0 else 'failed'), exit_code=code,
                     seconds=time.time()-began, ended_at=now(), budget_limited=limited,
                     usable_checkpoint=str(usable_checkpoint) if checkpoint_ok else None)
        save()
    if code != 0 and not accepted_limit:
        if limited:
            raise BudgetReached(f'{name} ended at its time limit')
        raise RuntimeError(f'{name} failed; see {log}')


def wait_for_gpu(gpu):
    previous = ROOT / 'reports/five_fold_optimized_status.json'
    fold = gpu+1
    final_job = f'fivefold_optimized_f{fold}_teacher_mc64_predict'
    with LOCK:
        STATUS['gpu_workers'][str(gpu)] = dict(state='waiting_for_existing_fold', prior_fold=fold,
                                               release_marker=final_job)
        save()
    while not STOP.is_set() and time.time() < TRAINING_END-1800:
        state = json.loads(previous.read_text())
        finished = state.get('jobs', {}).get(final_job, {}).get('state') == 'completed'
        finished |= state.get('stage') in ['finished', 'failed']
        if finished:
            # Confirm the actual GPU is empty as well as the dependency status.
            active = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if active.returncode == 0 and not active.stdout.strip():
                with LOCK:
                    STATUS['gpu_workers'][str(gpu)].update(state='running_new_experiments', released_at=now())
                    STATUS['stage'] = 'exploring_with_released_gpus'; save()
                return True
        STOP.wait(30)
        with LOCK:
            save()
    return False


def checkpoint(tag):
    return ROOT / f'experiments/{tag}/fold0/Fold_0_best_model.pth'


def train(tag, config, gpu, epochs, extra=(), minutes=120):
    remaining = TRAINING_END-time.time()
    if remaining < 600:
        raise BudgetReached('Training window exhausted')
    end = min(TRAINING_END, time.time()+minutes*60)
    command = [PYTHON, 'scripts/run_team3.py', '--config', config, '--fold', '0', '--gpu', str(gpu),
               '--workers', '6', '--epochs', str(epochs), '--experiment-name', tag, '--fp32',
               '--stop-at', str(end)]+list(extra)
    # Source trainer stops at the next epoch boundary; allow 12 minutes to save
    # its current epoch. Hard process-group limit remains within the 12h window.
    run(tag, command, gpu=gpu, budget_seconds=min(end+720, DEADLINE)-time.time(),
        usable_checkpoint=checkpoint(tag))


def predict(tag, config, gpu, mc=0):
    target = ROOT / f'experiments/{tag}/predictions'
    run(tag+'_predict', [PYTHON, 'scripts/predict_upstream.py', '--family', 'team3', '--config', config,
                        '--checkpoint', str(checkpoint(tag)), '--output', str(target),
                        '--fold', '0', '--gpu', str(gpu), '--tta', '--mc-folds', str(mc)], gpu=gpu)
    add_candidate(tag, target)


def add_candidate(tag, target):
    evidence = json.loads((target / 'metrics.json').read_text())
    with LOCK:
        CANDIDATES.append(str(target))
        STATUS['candidates'][tag] = dict(predictions=str(target),
                                       validation_auc=evidence['competition_validation_auc'])
        save()
    # Comparisons are reports, not an input to hidden-score optimization.
    run(tag+'_compare', [TOOLS, 'scripts/compare_predictions.py', '--baseline',
                        str(ROOT / 'experiments/optimized_1d_ensemble'), '--candidate', str(target),
                        '--output', str(ROOT / f'experiments/{tag}/comparison')])
    comparison = json.loads((ROOT / f'experiments/{tag}/comparison/comparison.json').read_text())
    with LOCK:
        STATUS['candidates'][tag]['comparison'] = comparison; save()


def prepare_banks():
    try:
        if not (BANK / 'metadata.json').exists():
            run('campaign12h_generate_300k_signals', [PHYSICS, 'scripts/generate_kdl_bank.py', '--fold', '0',
                '--count', '300000', '--workers', '32', '--output', str(BANK)], physics=True)
        if not (NOISE / 'metadata.json').exists():
            run('campaign12h_generate_1m_noise', [PHYSICS, 'scripts/generate_kdl_noise_bank.py',
                '--count', '1000000', '--workers', '32', '--output', str(NOISE)], physics=True)
        run('campaign12h_prepare_signal_snr', [PHYSICS, 'scripts/prepare_physical_snr.py', '--bank', str(BANK)], physics=True)
        run('campaign12h_audit_expanded_noise', [TOOLS, 'scripts/audit_kdl_noise_bank.py', '--bank', str(NOISE),
            '--report', str(ROOT / 'reports/campaign12h_noise_audit.json')])
        bank = json.loads((BANK / 'metadata.json').read_text())
        noise = json.loads((NOISE / 'metadata.json').read_text())
        snr = json.loads((BANK / 'snr_audit.json').read_text())
        if bank['count'] != 300000 or bank['excluded_prior_fold'] != 0 or bank['fold_seed'] != 48:
            raise ValueError('Incorrect campaign signal bank')
        if noise['count'] != 1000000 or not snr['passed']:
            raise ValueError('Incorrect campaign noise/SNR bank')
        with LOCK:
            STATUS['banks_ready'] = True; STATUS['bank_snr_audit'] = snr; save()
    except Exception as error:
        with LOCK:
            STATUS['bank_error'] = str(error); save()
    finally:
        BANK_READY.set()


def require_banks():
    while not BANK_READY.wait(15):
        if STOP.is_set() or time.time() >= TRAINING_END:
            raise BudgetReached('Signal generation exceeded training budget')
    if not STATUS.get('banks_ready'):
        raise RuntimeError('Physical bank preparation failed; see bank_error')


def physical_pretrain(tag, gpu, epochs, minutes, snr=False, real_mix=0., initial=None):
    require_banks()
    initial = initial or ROOT / 'experiments/real_team3_kdl_independent_noise_pretrain24/fold0'
    extra = ['--lr', '.001', '--no-augmentation', '--synthetic-bank', str(BANK), '--noise-bank', str(NOISE),
             '--noise-real-probability', str(real_mix), '--checkpoint-dir', str(initial)]
    if snr:
        extra += ['--synthetic-snr-range', '5', '12']
    train(tag, 'V2', gpu, epochs, extra, minutes)


def physical_finetune(pre, tag, gpu, lr='.0001'):
    train(tag, 'V2', gpu, 2, ['--lr', lr, '--no-augmentation', '--checkpoint-dir', str(checkpoint(pre).parent)], 30)
    predict(tag, 'V2', gpu)


def route_long(gpu):
    pre = 'campaign12h_physical_long100'
    physical_pretrain(pre, gpu, 100, 250)
    for lr, suffix in [('.0001', 'lr1e4'), ('.0003', 'lr3e4')]:
        physical_finetune(pre, 'campaign12h_physical_long_'+suffix, gpu, lr)
    if TRAINING_END-time.time() > 5400:
        pre = 'campaign12h_physical_long_continue32'
        physical_pretrain(pre, gpu, 32, 110, initial=checkpoint('campaign12h_physical_long100').parent)
        physical_finetune(pre, 'campaign12h_physical_long_continue_fine', gpu)


def route_snr(gpu):
    for mode in ['natural_control', 'low_snr']:
        pre = 'campaign12h_'+mode+'_pre48'
        physical_pretrain(pre, gpu, 48, 135, snr=mode == 'low_snr')
        physical_finetune(pre, 'campaign12h_'+mode+'_fine', gpu)
    if TRAINING_END-time.time() > 3600:
        pre = 'campaign12h_low_snr_real_noise_mix_pre24'
        physical_pretrain(pre, gpu, 24, 85, snr=True, real_mix=.5)
        physical_finetune(pre, 'campaign12h_low_snr_real_noise_mix_fine', gpu)


def route_reconstructed(gpu):
    require_banks()
    for weight, name in [('.05', 'aux'), ('0', 'noaux')]:
        remaining = TRAINING_END-time.time()
        if remaining < 3600:
            raise BudgetReached('Insufficient time for another reconstructed model')
        # Divide available time evenly between the two controlled variants.
        cap = min(180*60, (remaining-2400)/2) if name == 'aux' else min(180*60, remaining-1200)
        pre = 'campaign12h_split_resnet_'+name+'_pre24'
        destination = ROOT / f'experiments/{pre}'
        end = time.time()+cap
        run(pre, [PYTHON, 'scripts/train_kdl_inspired.py', '--tag', pre, '--gpu', str(gpu), '--epochs', '24',
                  '--bank', str(BANK), '--noise-bank', str(NOISE), '--auxiliary-weight', weight,
                  '--stop-at', str(end)], gpu=gpu, budget_seconds=min(end+360, DEADLINE)-time.time(),
            usable_checkpoint=destination / 'best.pth')
        tag = 'campaign12h_split_resnet_'+name+'_fine'
        end = min(TRAINING_END+600, time.time()+1800)
        output = ROOT / f'experiments/{tag}'
        run(tag, [PYTHON, 'scripts/train_kdl_inspired.py', '--tag', tag, '--gpu', str(gpu), '--epochs', '1',
                  '--lr', '.0001', '--checkpoint', str(destination / 'best.pth'), '--stop-at', str(end)],
            gpu=gpu, budget_seconds=min(end+360, DEADLINE)-time.time(), usable_checkpoint=output / 'best.pth')
        target = output / 'predictions'
        run(tag+'_predict', [PYTHON, 'scripts/predict_kdl_inspired.py', '--checkpoint', str(output / 'best.pth'),
                            '--output', str(target), '--gpu', str(gpu)], gpu=gpu)
        add_candidate(tag, target)


def route_originals(gpu):
    tag = 'campaign12h_split_attention_scratch'
    train(tag, 'M-1DS32', gpu, 6, minutes=70); predict(tag, 'M-1DS32', gpu)
    pre, fine = 'campaign12h_split_attention_pre4', 'campaign12h_split_attention_fine'
    train(pre, 'M-1DS32_pretrain', gpu, 4, ['--synthetic-bank', str(ROOT / 'data/synthetic_team3_300k')], 70)
    train(fine, 'M-1DS32', gpu, 6, ['--checkpoint-dir', str(checkpoint(pre).parent)], 90)
    predict(fine, 'M-1DS32', gpu)
    rank = 'campaign12h_split_attention_rank'
    train(rank, 'M-1DS32_adjust', gpu, 2, ['--checkpoint-dir', str(checkpoint(fine).parent)], 40)
    predict(rank, 'M-1DS32_adjust', gpu)
    teacher = ROOT / 'experiments/real_team3_width32_sd_fp32_synthetic_finetune/fold0'
    tag = 'campaign12h_width_ensemble_pseudo_weight025'
    train(tag, 'V-V2SD', gpu, 3, ['--checkpoint-dir', str(teacher), '--lr', '.0003',
          '--pseudo-label-dir', str(ROOT / 'data/pseudo_campaign12h_ensemble_fold0'), '--pseudo-weight', '.25'], 75)
    predict(tag, 'V-V2SD', gpu, mc=64)
    tag = 'campaign12h_width_rank'
    train(tag, 'V-V2SD', gpu, 2, ['--checkpoint-dir', str(teacher), '--lr', '.0002', '--criterion', 'rank'], 40)
    predict(tag, 'V-V2SD', gpu, mc=64)
    if TRAINING_END-time.time() > 7200:
        pre, fine = 'campaign12h_width_ranger_pre4', 'campaign12h_width_ranger_fine'
        train(pre, 'M-1D_pretrain', gpu, 4, ['--synthetic-bank', str(ROOT / 'data/synthetic_team3_300k')], 55)
        train(fine, 'M-1D', gpu, 6, ['--checkpoint-dir', str(checkpoint(pre).parent)], 65)
        predict(fine, 'M-1D', gpu)
    if TRAINING_END-time.time() > 3600:
        tag = 'campaign12h_width_seed6148'
        train(tag, 'V-V2SD', gpu, 6, ['--training-seed', '6148'], 55)
        predict(tag, 'V-V2SD', gpu)


def gpu_worker(gpu, route):
    lease_path = ROOT / f'reports/gpu_leases/gpu{gpu}.lock'
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not wait_for_gpu(gpu):
                raise BudgetReached('Previous job did not release GPU within training window')
            route(gpu)
            result = dict(state='finished', ended_at=now())
        except BudgetReached as error:
            result = dict(state='budget_exhausted', error=str(error), ended_at=now())
        except Exception as error:
            result = dict(state='failed', error=str(error), ended_at=now())
        with LOCK:
            STATUS['gpu_workers'].setdefault(str(gpu), {}).update(result); save()


def finalize():
    with LOCK:
        STATUS['stage'] = 'fitting_and_evaluating_final_ensemble'; save()
    if not CANDIDATES or STOP.is_set() or DEADLINE-time.time() < 300:
        return
    output = ROOT / 'experiments/campaign12h_final_ensemble'
    run('campaign12h_fit_final_ensemble', [TOOLS, 'scripts/ensemble_predictions.py', '--baseline',
        str(ROOT / 'experiments/optimized_1d_ensemble'), '--candidates', *CANDIDATES, '--output', str(output)])
    info = json.loads((output / 'metrics.json').read_text())
    eligible = info['evaluation_auc_gain'] > .0001 and info['evaluation_gain_95pct_interval'][0] > 0
    with LOCK:
        STATUS.update(final_ensemble=str(output), final_metrics=info, submission_eligible=eligible); save()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    records = [json.loads(p.read_text()) for p in (ROOT / 'reports/submissions').glob('*.json')]
    if eligible and sum(r['submitted_at'].startswith(today) for r in records) < 5 and DEADLINE-time.time() > 600:
        run('campaign12h_submit_validated_ensemble', [TOOLS, 'scripts/submit_checked.py', '--predictions', str(output),
            '--message', 'C33 12h research: physical/SNR/multiscale/rank diverse models; held-out validation weights'])
        if DEADLINE-time.time() > 120:
            run('campaign12h_refresh_scores', [TOOLS, 'scripts/refresh_submissions.py', '--wait-seconds',
                str(min(600, int(DEADLINE-time.time()-60)))])
    else:
        with LOCK:
            STATUS['submission_skipped'] = 'Gain unsupported, daily submission budget reached, or time reserved for artifacts'; save()


def main():
    global DEADLINE, TRAINING_END
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=12)
    args = parser.parse_args()
    started = time.time(); DEADLINE = started+args.hours*3600; TRAINING_END = DEADLINE-90*60
    if STATE.exists() or PLAN.exists():
        raise FileExistsError('Campaign files already exist; refuse duplicate campaign')
    manager_lock = (ROOT / 'reports/campaign12h_manager.lock').open('a')
    fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    STATUS.update(started_at=now(), deadline_utc=dt.datetime.fromtimestamp(DEADLINE, dt.timezone.utc).isoformat(),
                  deadline_china=dt.datetime.fromtimestamp(DEADLINE, dt.timezone(dt.timedelta(hours=8))).isoformat(),
                  training_end_utc=dt.datetime.fromtimestamp(TRAINING_END, dt.timezone.utc).isoformat(),
                  planned_hours=args.hours, manager_pid=os.getpid(), stage='preparing_banks_and_waiting_for_existing_gpus',
                  jobs={}, gpu_workers={}, candidates={}, banks_ready=False, completed=False,
                  validation_policy='All exploratory models exclude fold_48 == 0; 112000 held-out rows, calibration/evaluation half seed 713',
                  model_selection_policy='Adaptive reuse of fixed validation data; no hidden-score weight fitting',
                  prior_scope='Published mass estimates/PSD are external historical artifacts with unknown original training scope',
                  deadline_policy='12h wall-clock budget includes earlier jobs; 90min reserved for inference and aggregation; only campaign children may be stopped',
                  max_campaign_submissions=1)
    save()
    plan = dict(started_at=STATUS['started_at'], deadline_utc=STATUS['deadline_utc'], deadline_china=STATUS['deadline_china'],
                duration_hours=args.hours, gpus=4, maximum_aggregate_gpu_hours=4*args.hours,
                existing_jobs_preserved=True, generation=dict(physical_signals=300000, synthetic_noise=1000000,
                signal_bank=str(BANK), noise_bank=str(NOISE), prior_excludes_fold=0, physics_snr_range=[5, 12]),
                routes={
                    '0': ['V2 physical continuation: up to 100 epochs on 6x signal and 4x noise banks',
                          'Real fine tuning at 1e-4 and 3e-4; optional further 32-epoch continuation'],
                    '1': ['48 epochs natural-SNR control', '48 epochs uniform network SNR 5-12',
                          'Optional 24 epochs low-SNR with 50% real training-negative noise'],
                    '2': ['New split multi-scale encoders + ResNet34, up to 24 epochs with mass/q/SNR auxiliary tasks',
                          'Same seed/model without auxiliary task, up to 24 epochs; each then real fine tuning'],
                    '3': ['Original V2S split attention/RangerLars/spectral dropout scratch',
                          'Legacy synthetic pretraining + real fine tuning + low-lr rank-loss adjustment',
                          'Width32 ensemble-teacher soft pseudo labels weighted 0.25',
                          'Width32 low-lr rank adjustment', 'Optional width32 RangerLars branch and extra seed']},
                precision='FP32 due to previously observed FP16 NaNs',
                final_selection='Nonnegative logit weights on calibration half; paired bootstrap on evaluation half',
                submission_gate='AUC gain > 0.0001 and paired 95% CI lower bound > 0; project limit 5/day, campaign max 1',
                exact_winner_model_reproduction=False, new_submission_guaranteed=False,
                expected_candidates='11 primary variants; optional jobs depend on remaining wall-clock budget',
                early_stop='SIGTERM manager requests stop only for campaign children; completed artifacts are retained')
    atomic_json(PLAN, plan)
    print(json.dumps(plan, indent=2), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(prepare_banks)]
        futures += [executor.submit(gpu_worker, gpu, route) for gpu, route in enumerate(
            [route_long, route_snr, route_reconstructed, route_originals])]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    try:
        finalize()
    except Exception as error:
        with LOCK:
            STATUS['finalization_error'] = str(error); save()
    with LOCK:
        failures = [v['error'] for v in STATUS['gpu_workers'].values() if v['state'] == 'failed']
        STATUS.update(stage='stopped' if STOP.is_set() else ('finished_with_failures' if failures or STATUS.get('finalization_error') else 'finished'),
                      completed=not STOP.is_set() and not failures and not STATUS.get('finalization_error'),
                      finished_at=now(), actual_hours=(time.time()-started)/3600, errors=failures)
        save()
    print(json.dumps(STATUS, indent=2), flush=True)


if __name__ == '__main__':
    main()
