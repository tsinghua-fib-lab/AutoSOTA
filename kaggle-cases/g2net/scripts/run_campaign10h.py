"""Ten-hour four-GPU campaign: deeper physics, fold replication and new controls.

Uses explicitly assigned CUDA devices and CPU checkpoint deserialization. Every
training chain reserves time for real-data fine tuning and aligned inference.
Data generation runs concurrently. Final submission is conditional on held-out
calibration/evaluation evidence and never uses hidden-score fitting.
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
PY = str(ROOT / 'envs/reproduce/bin/python')
TPY = str(ROOT / 'envs/tools/bin/python')
PPY = str(ROOT / 'envs/physics/bin/python')
STATE = ROOT / 'reports/campaign10h_status.json'
LOCK = threading.RLock()
STOP = threading.Event()
STATUS = {}
DEADLINE = 0.
BASELINE = ROOT / 'experiments/campaign12h_final_ensemble'
OLD_BANK = ROOT / 'data/synthetic_kdl_300k_fold0_campaign12h'
OLD_NOISE = ROOT / 'data/synthetic_kdl_noise_1m_campaign12h'
NOISE = ROOT / 'data/synthetic_kdl_noise_2m_campaign10h'
READY = {fold: threading.Event() for fold in range(5)}
NOISE_READY = threading.Event()
CANDIDATES = []
FAMILY_SOURCES = {}
ENV = os.environ.copy()
ENV.update(OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', PYTHONUNBUFFERED='1',
           TORCH_HOME=str(ROOT / '.cache/torch'), HF_HOME=str(ROOT / '.cache/huggingface'),
           NUMBA_CACHE_DIR=str(ROOT / '.cache/numba'), MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'),
           XDG_CACHE_HOME=str(ROOT / '.cache'))


class TimeBudget(Exception):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def save():
    STATUS['last_updated_at'] = now()
    STATUS['remaining_seconds'] = max(0., DEADLINE-time.time())
    atomic_json(STATE, STATUS)


def stop(*unused):
    STOP.set()


def bank(fold):
    return ROOT / f'data/synthetic_kdl_{"1m" if fold == 0 else "300k"}_fold{fold}_campaign10h'


def ck(tag, fold=0, selection='best'):
    return ROOT / f'experiments/{tag}/fold{fold}/Fold_{fold}_{selection}_model.pth'


def run(tag, command, gpu=None, end=None, checkpoint=None, physics=False):
    if STOP.is_set() or time.time() >= DEADLINE:
        raise TimeBudget('Campaign deadline or explicit stop request')
    end = min(end or DEADLINE, DEADLINE)
    log = ROOT / f'logs/{tag}.log'
    started = time.time()
    env = dict(ENV, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1') if physics else ENV
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        entry = dict(state='running', pid=process.pid, command=command, gpu=gpu, log=str(log), started_at=now(),
                     deadline_utc=dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat())
        with LOCK:
            STATUS['jobs'][tag] = entry; save()
        limited = False
        while process.poll() is None:
            if STOP.is_set() or time.time() >= end:
                limited = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=min(60., max(0., DEADLINE-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                break
            STOP.wait(min(15., max(.1, end-time.time())))
        code = process.wait()
    usable = bool(checkpoint and checkpoint.exists())
    accepted = limited and usable and not STOP.is_set() and time.time() < DEADLINE
    with LOCK:
        entry.update(state='budget_limited_checkpoint_available' if accepted else ('completed' if code == 0 else 'failed'),
                     ended_at=now(), seconds=time.time()-started, exit_code=code, budget_limited=limited,
                     checkpoint=str(checkpoint) if usable else None)
        save()
    if code and not accepted:
        if limited:
            raise TimeBudget(tag+' reached its process budget')
        raise RuntimeError(tag+' failed; see '+str(log))


def trial(name, work):
    """Keep unrelated hypotheses running when one independent trial fails."""
    try:
        work()
        return True
    except (TimeBudget, RuntimeError, ValueError, FileNotFoundError) as error:
        with LOCK:
            STATUS['trial_outcomes'][name] = dict(state='budget_exhausted' if isinstance(error, TimeBudget) else 'failed',
                                                 error=str(error), at=now()); save()
        return False


def train(tag, gpu, config, epochs, extra=(), minutes=90, fold=0, reserve=30, precision='bf16'):
    end = min(time.time()+minutes*60, DEADLINE-reserve*60)
    if end-time.time() < 300:
        raise TimeBudget('Not enough time to train and finish the dependent artifacts')
    command = [PY, 'scripts/run_team3.py', '--config', config, '--fold', str(fold), '--gpu', str(gpu),
               '--workers', '6', '--epochs', str(epochs), '--experiment-name', tag, '--'+precision,
               '--keep-last', '--stop-at', str(end)]+list(extra)
    run(tag, command, gpu, end=min(end+600, DEADLINE-10*60), checkpoint=ck(tag, fold))


def predict(tag, gpu, config, fold=0, mc=0, mode='sign_swap', checkpoint=None, suffix=''):
    target = ROOT / f'experiments/{tag}/predictions{suffix}'
    run(tag+'_predict'+suffix, [PY, 'scripts/predict_upstream.py', '--family', 'team3', '--config', config,
        '--checkpoint', str(checkpoint or ck(tag, fold)), '--output', str(target), '--fold', str(fold),
        '--gpu', str(gpu), '--tta-mode', mode, '--mc-folds', str(mc)], gpu=gpu)
    if fold == 0:
        add_candidate(tag+suffix, target)
    return target


def add_candidate(tag, directory):
    if time.time() >= DEADLINE or STOP.is_set():
        return
    info = json.loads((directory / 'metrics.json').read_text())
    with LOCK:
        if str(directory) not in CANDIDATES:
            CANDIDATES.append(str(directory))
        STATUS['candidates'][tag] = dict(directory=str(directory), validation_auc=info['competition_validation_auc']); save()
    run(tag+'_compare', [TPY, 'scripts/compare_predictions.py', '--baseline', str(BASELINE),
                         '--candidate', str(directory), '--output', str(ROOT / f'experiments/{tag}/comparison')])
    comparison = json.loads((ROOT / f'experiments/{tag}/comparison/comparison.json').read_text())
    with LOCK:
        STATUS['candidates'][tag]['comparison'] = comparison; save()


def require_data(fold):
    while not (READY[fold].is_set() and NOISE_READY.is_set()):
        if STOP.is_set() or DEADLINE-time.time() < 3600:
            raise TimeBudget('Data generation left insufficient training time')
        with LOCK:
            error = STATUS.get('data_errors', {}).get(str(fold)) or STATUS.get('data_errors', {}).get('noise')
        if error:
            raise RuntimeError(error)
        STOP.wait(15)


def prepare_data():
    def audit_snr(fold):
        try:
            run(f'campaign10h_snr_audit_f{fold}', [PPY, 'scripts/prepare_physical_snr.py', '--bank', str(bank(fold))], physics=True)
            info = json.loads((bank(fold) / 'metadata.json').read_text())
            audit = json.loads((bank(fold) / 'snr_audit.json').read_text())
            if info['excluded_prior_fold'] != fold or info['fold_seed'] != 48 or not audit['passed']:
                raise ValueError('Invalid per-fold signal prior or SNR audit')
            READY[fold].set()
            with LOCK:
                STATUS['data_ready'][str(fold)] = dict(count=info['count'], bank=str(bank(fold)), at=now()); save()
        except Exception as error:
            with LOCK:
                STATUS['data_errors'][str(fold)] = str(error); save()
    try:
        run('campaign10h_generate_2m_noise', [PPY, 'scripts/generate_kdl_noise_bank.py', '--count', '2000000',
             '--workers', '32', '--prefix-bank', str(OLD_NOISE), '--output', str(NOISE)], physics=True)
        run('campaign10h_audit_2m_noise', [TPY, 'scripts/audit_kdl_noise_bank.py', '--bank', str(NOISE),
             '--report', str(ROOT / 'reports/campaign10h_noise_audit.json')])
        NOISE_READY.set()
    except Exception as error:
        with LOCK:
            STATUS['data_errors']['noise'] = str(error); save()
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as audits:
        futures = []
        # Fold 1 first enables independent validation while the larger fold-0
        # bank grows. CPU generation continues while completed banks are audited.
        for fold in [1, 0, 2, 3, 4]:
            try:
                prefix = OLD_BANK if fold == 0 else ROOT / f'data/synthetic_kdl_50k_fold{fold}'
                count = '1000000' if fold == 0 else '300000'
                run(f'campaign10h_generate_signals_f{fold}', [PPY, 'scripts/generate_kdl_bank.py', '--fold', str(fold),
                    '--count', count, '--workers', '48', '--prefix-bank', str(prefix), '--output', str(bank(fold))], physics=True)
                futures.append(audits.submit(audit_snr, fold))
            except Exception as error:
                with LOCK:
                    STATUS['data_errors'][str(fold)] = str(error); save()
        for future in concurrent.futures.as_completed(futures):
            future.result()


def physical_pre(tag, gpu, config, initial, epochs, minutes, fold=0, selection='best', snr=None, large=True, reserve=50):
    if large:
        require_data(fold)
    signals = bank(fold) if large else OLD_BANK
    noise = NOISE if large else OLD_NOISE
    extra = ['--checkpoint-dir', str(initial), '--checkpoint-selection', selection,
             '--lr', '.001', '--no-augmentation', '--synthetic-bank', str(signals), '--noise-bank', str(noise)]
    if snr:
        extra += ['--synthetic-snr-range', str(snr[0]), str(snr[1])]
    train(tag, gpu, config, epochs, extra, minutes, fold, reserve)


def fine(pre, tag, gpu, config='V2', fold=0, selection='best', lr='.0003', epochs=2, mc=0):
    train(tag, gpu, config, epochs, ['--lr', lr, '--no-augmentation', '--checkpoint-dir', str(ck(pre, fold).parent),
          '--checkpoint-selection', selection], minutes=35, fold=fold, reserve=10)
    return predict(tag, gpu, config, fold=fold, mc=mc)


def route_width(gpu):
    trial('finish_long_continuation', lambda: fine('campaign12h_physical_long_continue32',
          'campaign10h_long_continue_fine', gpu))
    warm = 'campaign10h_width_physical_warm20'
    if not trial(warm, lambda: physical_pre(warm, gpu, 'V-V2SD',
        ROOT / 'experiments/real_team3_width32_sd_fp32_synthetic_pretrain/fold0', 20, 65, large=False)):
        return
    pre = 'campaign10h_width_physical_1m_pre100'
    if not trial(pre, lambda: physical_pre(pre, gpu, 'V-V2SD', ck(warm).parent, 100, 290, selection='last')):
        return
    for lr, suffix in [('.0003', 'lr3e4'), ('.001', 'lr1e3')]:
        trial('width_fine_'+suffix, lambda lr=lr, suffix=suffix: fine(pre, 'campaign10h_width_physical_'+suffix,
              gpu, config='V-V2SD', lr=lr, epochs=3, mc=64))
    if DEADLINE-time.time() > 100*60:
        low = 'campaign10h_width_low_snr_pre24'
        if trial(low, lambda: physical_pre(low, gpu, 'V-V2SD', ck(pre).parent, 24, 70, selection='last', snr=(5, 12), reserve=35)):
            trial('width_low_snr_fine', lambda: fine(low, 'campaign10h_width_low_snr_fine', gpu, 'V-V2SD',
                  selection='last', epochs=2, mc=64))


def train_kdl(tag, gpu, initial, epochs, minutes, large=True, snr=None, lr='.001', reserve=35):
    if large:
        require_data(0)
    end = min(time.time()+minutes*60, DEADLINE-reserve*60)
    if end-time.time() < 300:
        raise TimeBudget('Insufficient time for reconstructed model stage')
    command = [PY, 'scripts/train_kdl_inspired.py', '--tag', tag, '--gpu', str(gpu), '--epochs', str(epochs),
               '--checkpoint', str(initial), '--lr', lr, '--stop-at', str(end), '--precision', 'fp32',
               '--batch-size', '256']
    if snr != 'real':
        command += ['--bank', str(bank(0) if large else OLD_BANK), '--noise-bank', str(NOISE if large else OLD_NOISE),
                    '--auxiliary-weight', '.05']
        if snr:
            command += ['--snr-range', str(snr[0]), str(snr[1])]
    destination = ROOT / f'experiments/{tag}/best.pth'
    run(tag, command, gpu, end=min(end+300, DEADLINE-8*60), checkpoint=destination)


def predict_kdl(tag, gpu, mode='none', suffix=''):
    target = ROOT / f'experiments/{tag}/predictions{suffix}'
    run(tag+'_predict'+suffix, [PY, 'scripts/predict_kdl_inspired.py', '--checkpoint',
        str(ROOT / f'experiments/{tag}/best.pth'), '--output', str(target), '--gpu', str(gpu), '--tta-mode', mode], gpu)
    add_candidate(tag+suffix, target)


def route_kdl(gpu):
    warm = 'campaign10h_kdl_aux_warm12'
    if not trial(warm, lambda: train_kdl(warm, gpu, ROOT / 'experiments/campaign12h_split_resnet_aux_pre24/best.pth',
                                        12, 65, large=False)):
        return
    pre = 'campaign10h_kdl_aux_1m_pre100'
    if not trial(pre, lambda: train_kdl(pre, gpu, ROOT / f'experiments/{warm}/last.pth', 100, 365, reserve=70)):
        return
    for lr, epochs, suffix in [('.0001', 1, 'lr1e4_ep1'), ('.0003', 1, 'lr3e4_ep1'), ('.0001', 3, 'lr1e4_ep3')]:
        tag = 'campaign10h_kdl_aux_'+suffix
        def execute(tag=tag, lr=lr, epochs=epochs):
            train_kdl(tag, gpu, ROOT / f'experiments/{pre}/best.pth', epochs, 30, large=False, snr='real', lr=lr, reserve=15)
            predict_kdl(tag, gpu)
        trial(tag, execute)
    if DEADLINE-time.time() > 90*60:
        low = 'campaign10h_kdl_aux_low_snr_pre12'
        if trial(low, lambda: train_kdl(low, gpu, ROOT / f'experiments/{pre}/last.pth', 12, 55, snr=(5, 12), lr='.0003')):
            tag = 'campaign10h_kdl_aux_low_snr_fine'
            def execute():
                train_kdl(tag, gpu, ROOT / f'experiments/{low}/best.pth', 1, 15, large=False, snr='real', lr='.0001', reserve=15)
                predict_kdl(tag, gpu)
            trial(tag, execute)


def route_folds(gpu):
    trial('finish_mixed_noise', lambda: fine('campaign12h_low_snr_real_noise_mix_pre24',
          'campaign10h_low_snr_noise_mix_fine', gpu))
    FAMILY_SOURCES[0] = ROOT / 'experiments/campaign12h_low_snr_fine/predictions'
    for fold in range(1, 5):
        pre, tag = f'campaign10h_low_snr_f{fold}_pre43', f'campaign10h_low_snr_f{fold}_fine'
        initial = ROOT / f'experiments/fivefold_optimized_f{fold}_independent_noise_pretrain/fold{fold}'
        def execute(fold=fold, pre=pre, tag=tag, initial=initial):
            physical_pre(pre, gpu, 'V2', initial, 43, 125, fold=fold, snr=(5, 12), reserve=30)
            target = fine(pre, tag, gpu, fold=fold, lr='.0003', epochs=2)
            with LOCK:
                FAMILY_SOURCES[fold] = target
                STATUS['fold_replication'][str(fold)] = json.loads((target / 'metrics.json').read_text()); save()
        trial(tag, execute)
    if len(FAMILY_SOURCES) == 5:
        output = ROOT / 'experiments/campaign10h_low_snr_calibrated_fivefold'
        run('campaign10h_calibrate_new_low_snr_fivefold', [TPY, 'scripts/calibrate_fivefold_family.py',
             '--sources', *[str(FAMILY_SOURCES[f]) for f in range(5)], '--output', str(output)])
        add_candidate('campaign10h_low_snr_calibrated_fivefold', output)
    if DEADLINE-time.time() > 80*60:
        pre = 'campaign10h_v2_1m_low_snr_pre24'
        if trial(pre, lambda: physical_pre(pre, gpu, 'V2', ROOT / 'experiments/campaign12h_low_snr_pre48/fold0',
                                         24, 60, snr=(5, 12), reserve=25)):
            trial('v2_1m_low_snr_fine', lambda: fine(pre, 'campaign10h_v2_1m_low_snr_fine', gpu, selection='last'))


def route_controls(gpu):
    best_v2 = ROOT / 'experiments/campaign12h_physical_long_lr3e4/fold0'
    width_teacher = ROOT / 'experiments/real_team3_width32_sd_fp32_synthetic_finetune/fold0'
    for tag, config, initial, extra in [
        ('campaign10h_v2_rank', 'V2', best_v2, ['--criterion', 'rank', '--lr', '.0002']),
        ('campaign10h_v2_bce_rank01', 'V2', best_v2, ['--rank-weight', '.1', '--lr', '.0002']),
        ('campaign10h_width_rank', 'V-V2SD', width_teacher, ['--criterion', 'rank', '--lr', '.0002'])]:
        def execute(tag=tag, config=config, initial=initial, extra=extra):
            train(tag, gpu, config, 2, ['--checkpoint-dir', str(initial)]+extra, 35, reserve=15)
            predict(tag, gpu, config, mc=64)
        trial(tag, execute)
    tag = 'campaign10h_width_pseudo025'
    def execute_pseudo():
        train(tag, gpu, 'V-V2SD', 3, ['--checkpoint-dir', str(width_teacher), '--lr', '.0003',
              '--pseudo-label-dir', str(ROOT / 'data/pseudo_campaign10h_ensemble_fold0'), '--pseudo-weight', '.25'],
              60, reserve=20)
        predict(tag, gpu, 'V-V2SD', mc=128)
    trial(tag, execute_pseudo)
    tag = 'campaign10h_split_attention_fine'
    def execute_split():
        train(tag, gpu, 'M-1DS32', 6, ['--checkpoint-dir', str(ROOT / 'experiments/campaign12h_split_attention_pre4/fold0')], 75)
        predict(tag, gpu, 'M-1DS32')
        rank = 'campaign10h_split_attention_rank'
        train(rank, gpu, 'M-1DS32_adjust', 2, ['--checkpoint-dir', str(ck(tag).parent)], 35)
        predict(rank, gpu, 'M-1DS32_adjust')
    trial(tag, execute_split)
    # No training augmentation was used for these physical models. Explicitly
    # test whether detector-swapping TTA really helps their distributions.
    for source, prefix in [('campaign12h_physical_long_lr3e4', 'long'), ('campaign12h_low_snr_fine', 'low')]:
        for mode in ['none', 'sign']:
            tag = f'campaign10h_tta_{prefix}_{mode}'
            trial(tag, lambda tag=tag, source=source, mode=mode: predict(tag, gpu, 'V2', mode=mode, checkpoint=ck(source)))
    for interval, name in [((4, 10), '4_10'), ((6, 16), '6_16')]:
        pre, tag = f'campaign10h_snr_{name}_pre24', f'campaign10h_snr_{name}_fine'
        def execute(interval=interval, pre=pre, tag=tag):
            physical_pre(pre, gpu, 'V2', ROOT / 'experiments/campaign12h_low_snr_pre48/fold0',
                         24, 75, snr=interval, reserve=35)
            fine(pre, tag, gpu, selection='last')
        trial(tag, execute)
    if DEADLINE-time.time() > 60*60:
        tag = 'campaign10h_width_seed7148'
        trial(tag, lambda: (train(tag, gpu, 'V-V2SD', 6, ['--training-seed', '7148'], 50, reserve=15),
                             predict(tag, gpu, 'V-V2SD')))


def gpu_worker(gpu, work):
    lease_path = ROOT / f'reports/gpu_leases/gpu{gpu}.lock'
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with LOCK:
                STATUS['gpu_workers'][str(gpu)] = dict(state='running', started_at=now()); save()
            work(gpu)
            state, error = 'finished', None
        except Exception as problem:
            state, error = ('budget_exhausted' if isinstance(problem, TimeBudget) else 'failed'), str(problem)
        with LOCK:
            STATUS['gpu_workers'].setdefault(str(gpu), {}).update(state=state, error=error, ended_at=now()); save()


def calibrate_prior():
    for name in ['independent_noise_finetune', 'physical_finetune', 'width_finetune']:
        first = {'independent_noise_finetune': 'real_team3_kdl_independent_noise_finetune',
                 'physical_finetune': 'real_team3_kdl_physical_finetune',
                 'width_finetune': 'real_team3_width32_sd_fp32_synthetic_finetune'}[name]
        output = ROOT / f'experiments/campaign10h_calibrated_prior_{name}'
        sources = [ROOT / f'experiments/{first}/predictions']+[ROOT / f'experiments/fivefold_optimized_f{f}_{name}/predictions'
                                                                 for f in range(1, 5)]
        run('campaign10h_calibrate_prior_'+name, [TPY, 'scripts/calibrate_fivefold_family.py', '--sources',
                                                *[str(x) for x in sources], '--output', str(output)])
        add_candidate('campaign10h_calibrated_prior_'+name, output)


def finalize():
    with LOCK:
        STATUS['stage'] = 'final_ensemble_evaluation'; save()
    if STOP.is_set() or DEADLINE-time.time() < 180 or not CANDIDATES:
        return
    output = ROOT / 'experiments/campaign10h_final_ensemble'
    run('campaign10h_fit_final_ensemble', [TPY, 'scripts/ensemble_predictions.py', '--baseline', str(BASELINE),
                                         '--candidates', *CANDIDATES, '--output', str(output)])
    info = json.loads((output / 'metrics.json').read_text())
    eligible = info['evaluation_auc_gain'] > .0001 and info['evaluation_gain_95pct_interval'][0] > 0
    with LOCK:
        STATUS.update(final_ensemble=str(output), final_metrics=info, submission_eligible=eligible); save()
    records = [json.loads(p.read_text()) for p in (ROOT / 'reports/submissions').glob('*.json')]
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if eligible and sum(x['submitted_at'].startswith(day) for x in records) < 5 and DEADLINE-time.time() > 480:
        run('campaign10h_submit_final', [TPY, 'scripts/submit_checked.py', '--predictions', str(output),
            '--message', 'C33 10h deeper physics, low-SNR fold replication and ranking; calibrated held-out ensemble',
            '--wait-seconds', str(min(600, int(DEADLINE-time.time()-120)))])
        if DEADLINE-time.time() > 60:
            run('campaign10h_refresh_scores', [TPY, 'scripts/refresh_submissions.py', '--wait-seconds', '0'])
    else:
        with LOCK:
            STATUS['submission_skipped'] = 'Unsupported validation gain, project daily limit, or insufficient submission time'; save()


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=10)
    args = parser.parse_args()
    started = time.time(); DEADLINE = started+args.hours*3600
    if STATE.exists():
        raise FileExistsError('Refuse a duplicate campaign')
    lock = (ROOT / 'reports/campaign10h_manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    STATUS.update(started_at=now(), deadline_utc=dt.datetime.fromtimestamp(DEADLINE, dt.timezone.utc).isoformat(),
                  deadline_china=dt.datetime.fromtimestamp(DEADLINE, dt.timezone(dt.timedelta(hours=8))).isoformat(),
                  stage='four_gpu_training_and_parallel_data_generation', manager_pid=os.getpid(), planned_hours=args.hours,
                  jobs={}, gpu_workers={}, trial_outcomes={}, candidates={}, data_ready={}, data_errors={}, fold_replication={},
                  completed=False, checkpoint_loading='CPU deserialize then assigned GPU; no old-GPU allocation',
                  validation_policy='Per-fold excluded priors, matching checkpoints; fold-0 calibration/evaluation half seed 713 for final ensemble',
                  external_artifact_scope='Historical PSD and mass predictor training scope unknown',
                  selection_policy='Adaptive validation reuse; no hidden-score fitting; maximum one campaign submission')
    save()
    plan = dict(started_at=STATUS['started_at'], deadline_china=STATUS['deadline_china'], budget_hours=args.hours,
                hardware='Four A100 80GB, immediately assigned one worker per GPU', maximum_gpu_hours=args.hours*4,
                data={'fold0_physical_templates':1000000, 'fold1_to4_templates_each':300000, 'noise':2000000,
                      'prefix_reuse':'Reuse deterministic prefixes, generate new suffixes; bank counts are not all newly generated templates'},
                routes={'0':['Finish prior V2 continuation fine tuning', 'Width32 physical warm20 + up to100 epochs on 1m bank',
                              'Fine-tuning lr3e-4 vs1e-3', 'Optional Width32 low-SNR continuation'],
                        '1':['Continue KDL-inspired auxiliary model from24 epochs; warm12 then up to100 epochs on1m bank',
                              'Fine-tuning epoch count / learning-rate controls', 'Optional low-SNR adaptation'],
                        '2':['Finish mixed-noise fine tuning', 'Replicate SNR5-12 on folds1-4,43 epochs each + real fine tuning',
                              'Combine with existing fold0 and calibrate each fold using calibration rows only',
                              'Optional larger-bank fold0 continuation'],
                        '3':['Rank-only vs BCE+0.1rank', 'Width32 teacher pseudo labels weighted0.25',
                              'Finish split-attention fine tuning/rank', 'None/sign/channel-swap inference comparison',
                              'SNR4-10 vs6-16 controls', 'Optional seed diversity']},
                cpu_route='Calibrate old five-fold families on per-fold calibration halves, evaluate on separate halves',
                precision='BF16 convolution with FP32 GeM/parameters/validation for upstream 1D; KDL FP32 based on benchmark',
                priorities='Complete dependent fine tuning/inference; save both best and last checkpoints; isolate unrelated trial failures',
                submission_gate='Gain>0.0001 and paired95%CI lower>0; fresh API limits; project max5/day; campaign max1',
                experiments_are_hypotheses=True, target_champion_exceeded=False)
    atomic_json(ROOT / 'reports/campaign10h_plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(prepare_data), pool.submit(lambda: trial('prior_calibration', calibrate_prior))]
        futures += [pool.submit(gpu_worker, gpu, work) for gpu, work in enumerate(
            [route_width, route_kdl, route_folds, route_controls])]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    trial('finalization', finalize)
    with LOCK:
        failures = [key for key, value in STATUS['trial_outcomes'].items() if value['state'] == 'failed']
        STATUS.update(stage='stopped' if STOP.is_set() else ('finished_with_failures' if failures else 'finished'),
                      completed=not STOP.is_set() and not failures, finished_at=now(), actual_hours=(time.time()-started)/3600,
                      failed_trials=failures, completed_prediction_candidates=len(CANDIDATES))
        save()
    print(json.dumps(STATUS, indent=2), flush=True)


if __name__ == '__main__':
    main()
