"""Fourteen-hour campaign: reproduce the strongest physics model across folds.

Four GPUs train four independent held-out folds immediately while their signal
banks grow. Complete primary artifacts before SNR/averaging/control experiments.
All children belong to this manager, have budgets, and use exclusive GPU leases.
"""
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
import signal
import time
from pathlib import Path

import run_campaign10h as e

ROOT, PY, TPY, PPY = e.ROOT, e.PY, e.TPY, e.PPY
CURRENT = ROOT / 'experiments/campaign10h_final_ensemble'
CONTROL = ROOT / 'experiments/campaign10h_calibrated_prior_width_finetune'
NOISE = ROOT / 'data/synthetic_kdl_noise_2m_campaign10h'
FAMILIES = {
    'lr3e4': {0: ROOT / 'experiments/campaign10h_width_physical_lr3e4/predictions'},
    'lr1e3': {0: ROOT / 'experiments/campaign10h_width_physical_lr1e3/predictions'},
    'average_bn': {}, 'average_no_bn': {}, 'snr_mix': {}}


def bank(fold):
    if fold == 0:
        return ROOT / 'data/synthetic_kdl_1m_fold0_campaign10h'
    return ROOT / f'data/synthetic_kdl_1m_fold{fold}_campaign14h'


def prefix(fold):
    return ROOT / f'data/synthetic_kdl_300k_fold{fold}_campaign10h'


def save_field(**kwargs):
    with e.LOCK:
        e.STATUS.update(kwargs); e.save()


def prepare_data():
    def audit(fold):
        try:
            e.run(f'campaign14h_snr_audit_f{fold}', [PPY, 'scripts/prepare_physical_snr.py', '--bank', str(bank(fold))],
                  end=e.DEADLINE-6*3600, physics=True)
            metadata = json.loads((bank(fold) / 'metadata.json').read_text())
            evidence = json.loads((bank(fold) / 'snr_audit.json').read_text())
            if metadata['count'] != 1000000 or metadata['excluded_prior_fold'] != fold or not evidence['passed']:
                raise ValueError('New physical bank failed scope or SNR audit')
            e.READY[fold].set()
            with e.LOCK:
                e.STATUS['data_ready'][str(fold)] = dict(count=metadata['count'], bank=str(bank(fold)), at=e.now()); e.save()
        except Exception as error:
            with e.LOCK:
                e.STATUS['data_errors'][str(fold)] = str(error); e.save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        audits = []
        for fold in range(1, 5):
            try:
                e.run(f'campaign14h_generate_signals_f{fold}', [PPY, 'scripts/generate_kdl_bank.py', '--fold', str(fold),
                      '--count', '1000000', '--workers', '48', '--prefix-bank', str(prefix(fold)), '--output', str(bank(fold))],
                      end=e.DEADLINE-6*3600, physics=True)
                audits.append(pool.submit(audit, fold))
            except Exception as error:
                with e.LOCK:
                    e.STATUS['data_errors'][str(fold)] = str(error); e.save()
        for future in concurrent.futures.as_completed(audits):
            future.result()


def train(tag, gpu, config, epochs, extra=(), minutes=90, fold=0, reserve=60, precision='bf16'):
    end = min(time.time()+minutes*60, e.DEADLINE-max(reserve, 60)*60)
    if end-time.time() < 300:
        raise e.TimeBudget('Training would consume the inference/finalization reserve')
    command = [PY, 'scripts/run_team3.py', '--config', config, '--fold', str(fold), '--gpu', str(gpu),
               '--workers', '6', '--epochs', str(epochs), '--experiment-name', tag, '--'+precision,
               '--keep-last', '--stop-at', str(end)]+list(extra)
    e.run(tag, command, gpu, end=min(end+600, e.DEADLINE-50*60), checkpoint=e.ck(tag, fold))


def predict(tag, gpu, config='V-V2SD', fold=0, mc=64, mode='sign_swap', checkpoint=None, suffix=''):
    output = ROOT / f'experiments/{tag}/predictions{suffix}'
    e.run(tag+'_predict'+suffix, [PY, 'scripts/predict_upstream.py', '--family', 'team3', '--config', config,
          '--checkpoint', str(checkpoint or e.ck(tag, fold)), '--output', str(output), '--fold', str(fold),
          '--gpu', str(gpu), '--tta-mode', mode, '--mc-folds', str(mc)], gpu, end=e.DEADLINE-35*60)
    if fold == 0:
        e.add_candidate(tag+suffix, output)
    else:
        with e.LOCK:
            e.STATUS['fold_replication'][tag] = json.loads((output / 'metrics.json').read_text()); e.save()
    return output


def physical(tag, gpu, initial, epochs, minutes, fold=0, config='V-V2SD', selection='best',
             warm=False, snr=None, snr_probability=1., real_noise=0., lr='.001', reserve=100, schedule_epochs=None):
    if not warm:
        e.require_data(fold)
    extra = ['--checkpoint-dir', str(initial), '--checkpoint-selection', selection, '--lr', lr,
             '--no-augmentation', '--synthetic-bank', str(prefix(fold) if warm else bank(fold)),
             '--noise-bank', str(NOISE), '--noise-real-probability', str(real_noise)]
    if snr:
        extra += ['--synthetic-snr-range', *map(str, snr), '--synthetic-snr-probability', str(snr_probability)]
    if schedule_epochs is not None:
        extra += ['--max-local-epochs', str(epochs)]
    train(tag, gpu, config, schedule_epochs or epochs, extra, minutes=minutes, fold=fold, reserve=reserve)


def fine(pre, tag, gpu, fold=0, config='V-V2SD', lr='.0003', selection='best', epochs=3, extra=()):
    train(tag, gpu, config, epochs, ['--checkpoint-dir', str(e.ck(pre, fold).parent), '--checkpoint-selection', selection,
          '--lr', lr, '--no-augmentation', '--save-epoch-checkpoints']+list(extra), minutes=40, fold=fold)
    return predict(tag, gpu, config, fold, mc=64 if config == 'V-V2SD' else 0)


def register(family, fold, output):
    with e.LOCK:
        FAMILIES[family][fold] = output
        e.STATUS['families'][family] = {str(f):str(p) for f,p in FAMILIES[family].items()}; e.save()


def average(gpu, fold, inputs, tag, bn_samples=16384):
    output = e.ck(tag, fold).parent
    e.run(tag+'_average', [PY, 'scripts/average_team3_checkpoints.py', '--checkpoints', *map(str, inputs),
          '--output', str(output), '--gpu', str(gpu), '--bn-samples', str(bn_samples)], gpu, end=e.DEADLINE-50*60)
    return predict(tag, gpu, fold=fold)


def late_controls(gpu):
    remaining = (e.DEADLINE-time.time())/60
    if remaining < 110:
        return
    if gpu == 0:
        pre = 'campaign14h_width_f0_snr_mix_pre24'
        if e.trial(pre, lambda: physical(pre, gpu, ROOT / 'experiments/campaign10h_width_physical_1m_pre100/fold0',
                   24, min(125, remaining-110), snr=(6,16), snr_probability=.5)):
            tag = 'campaign14h_width_f0_snr_mix_fine'
            def execute():
                output = fine(pre, tag, gpu, selection='last')
                register('snr_mix', 0, output)
            e.trial(tag, execute)
    elif gpu == 1:
        pre = 'campaign14h_width_f0_continue48'
        if e.trial(pre, lambda: physical(pre, gpu, ROOT / 'experiments/campaign10h_width_physical_1m_pre100/fold0',
                   48, min(210, remaining-110), selection='last')):
            for lr, suffix in [('.0003','lr3e4'), ('.001','lr1e3')]:
                e.trial('f0_continue_'+suffix, lambda lr=lr,suffix=suffix: fine(pre, 'campaign14h_width_f0_continue_'+suffix,
                         gpu, lr=lr))
    elif gpu == 2:
        pre = 'campaign14h_v2_snr6_16_real_noise_pre40'
        if e.trial(pre, lambda: physical(pre, gpu, ROOT / 'experiments/campaign10h_snr_6_16_pre24/fold0',
                   40, min(140, remaining-100), config='V2', selection='last', snr=(6,16), real_noise=.15)):
            e.trial('v2_real_noise_fine', lambda: fine(pre, 'campaign14h_v2_snr6_16_real_noise_fine', gpu,
                    config='V2', selection='last'))
    else:
        initial = 'campaign10h_width_physical_lr1e3'
        e.trial('strong_student_sgd', lambda: fine(initial, 'campaign14h_width_sgd_fine', gpu,
                 lr='.001', extra=['--optimizer','SGD']))
        if e.DEADLINE-time.time() > 110*60:
            def pseudo():
                tag = 'campaign14h_width_strong_teacher_pseudo015'
                train(tag, gpu, 'V-V2SD', 3, ['--checkpoint-dir', str(e.ck(initial).parent), '--lr', '.0003',
                      '--pseudo-label-dir', str(ROOT / 'data/pseudo_campaign14h_ensemble_fold0'), '--pseudo-weight', '.15',
                      '--save-epoch-checkpoints'], minutes=55)
                predict(tag, gpu, mc=128)
            e.trial('strong_teacher_pseudo015', pseudo)
    # Cheap inference controls on the strongest existing student, with explicit
    # naming so an unchanged checkpoint is never reported as a training gain.
    if e.DEADLINE-time.time() > 90*60:
        mode = ['none','sign','sign_swap','sign_swap'][gpu]
        mc = [64,64,256,512][gpu]
        tag = f'campaign14h_width_inference_{mode}_mc{mc}'
        e.trial(tag, lambda: predict(tag, gpu, checkpoint=e.ck('campaign10h_width_physical_lr1e3'), mode=mode, mc=mc))


def route(gpu):
    fold = gpu+1
    warm = f'campaign14h_width_f{fold}_warm14'
    initial = ROOT / f'experiments/fivefold_optimized_f{fold}_width_pretrain/fold{fold}'
    if not e.trial(warm, lambda: physical(warm, gpu, initial, 14, 90, fold=fold, warm=True, schedule_epochs=20)):
        return
    pre = f'campaign14h_width_f{fold}_physical_pre66'
    if not e.trial(pre, lambda: physical(pre, gpu, e.ck(warm, fold).parent, 66, 350, fold=fold, selection='last', reserve=210, schedule_epochs=100)):
        return
    for lr, suffix in [('.0003','lr3e4'), ('.001','lr1e3')]:
        tag = f'campaign14h_width_f{fold}_{suffix}'
        def execute(tag=tag, lr=lr, suffix=suffix):
            output = fine(pre, tag, gpu, fold=fold, lr=lr)
            register(suffix, fold, output)
        e.trial(tag, execute)
    if all(fold in FAMILIES[name] for name in ['lr3e4','lr1e3']):
        tag = f'campaign14h_width_f{fold}_average_bn'
        def execute_average():
            output = average(gpu, fold, [e.ck(f'campaign14h_width_f{fold}_{suffix}', fold) for suffix in ['lr3e4','lr1e3']], tag)
            register('average_bn', fold, output)
        e.trial(tag, execute_average)
        if e.DEADLINE-time.time() > 260*60:
            tag = f'campaign14h_width_f{fold}_average_no_bn'
            def execute_no_bn():
                output = average(gpu, fold, [e.ck(f'campaign14h_width_f{fold}_{s}', fold) for s in ['lr3e4','lr1e3']], tag, bn_samples=0)
                register('average_no_bn', fold, output)
            e.trial(tag, execute_no_bn)
    if e.DEADLINE-time.time() > 230*60:
        mix = f'campaign14h_width_f{fold}_snr_mix_pre24'
        if e.trial(mix, lambda: physical(mix, gpu, e.ck(pre, fold).parent, 24, 125, fold=fold,
                    snr=(6,16), snr_probability=.5, reserve=100)):
            tag = f'campaign14h_width_f{fold}_snr_mix_fine'
            def execute_mix():
                output = fine(mix, tag, gpu, fold=fold, selection='last')
                register('snr_mix', fold, output)
            e.trial(tag, execute_mix)
    # The initial-vs-last control directly addresses inherited-best retention.
    if e.DEADLINE-time.time() > 210*60:
        tag = f'campaign14h_width_f{fold}_last_init_fine'
        e.trial(tag, lambda: fine(pre, tag, gpu, fold=fold, selection='last', epochs=2))
    # Fold-0 averaging makes the cross-fold averaging family complete. This uses
    # two old same-fold fine-tunes; no validation waveform is used for BN stats.
    if gpu == 3 and e.DEADLINE-time.time() > 85*60:
        tag = 'campaign14h_width_f0_average_bn'
        def execute_average0():
            output = average(gpu, 0, [e.ck('campaign10h_width_physical_'+s) for s in ['lr3e4','lr1e3']], tag)
            register('average_bn', 0, output)
        e.trial(tag, execute_average0)
        if e.DEADLINE-time.time() > 90*60:
            tag = 'campaign14h_width_f0_average_no_bn'
            def execute_average0_no_bn():
                output = average(gpu, 0, [e.ck('campaign10h_width_physical_'+s) for s in ['lr3e4','lr1e3']], tag, bn_samples=0)
                register('average_no_bn', 0, output)
            e.trial(tag, execute_average0_no_bn)
    late_controls(gpu)


def submit(output, name, message):
    records = [json.loads(p.read_text()) for p in (ROOT / 'reports/submissions').glob('*.json')]
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if sum(x['submitted_at'].startswith(day) for x in records) >= 5 or e.DEADLINE-time.time() < 500:
        save_field(submission_skipped='Project daily limit or insufficient submission reserve')
        return
    e.run('campaign14h_submit_'+name, [TPY, 'scripts/submit_checked.py', '--predictions', str(output),
          '--message', message, '--wait-seconds', str(min(600,int(e.DEADLINE-time.time()-120)))])


def finalize():
    save_field(stage='fivefold_calibration_and_final_evaluation')
    calibrated = {}
    for name, members in FAMILIES.items():
        if len(members) != 5:
            continue
        output = ROOT / f'experiments/campaign14h_calibrated_{name}_fivefold'
        def execute(name=name, members=members, output=output):
            e.run('campaign14h_calibrate_'+name, [TPY, 'scripts/calibrate_fivefold_family.py', '--sources',
                  *[str(members[f]) for f in range(5)], '--output', str(output)])
            calibrated[name] = output
            e.add_candidate('campaign14h_calibrated_'+name, output)
        e.trial('calibrate_'+name, execute)
    promoted = ROOT / 'experiments/campaign14h_promoted_fivefold'
    promoted_eligible = False
    if all(name in calibrated for name in ['lr3e4','lr1e3']):
        def promote():
            e.run('campaign14h_promote_width_families', [TPY, 'scripts/promote_width_fivefold.py', '--baseline', str(CURRENT),
                  '--families', str(calibrated['lr3e4']), str(calibrated['lr1e3']), '--control', str(CONTROL), '--output', str(promoted)])
        if e.trial('promote_fivefold', promote):
            info = json.loads((promoted / 'metrics.json').read_text())
            promoted_eligible = info['promotion_eligible']
            save_field(promotion_metrics=info, promotion_eligible=promoted_eligible)
    if calibrated:
        output = ROOT / 'experiments/campaign14h_full_fivefold_ensemble'
        def fit_fivefold():
            e.run('campaign14h_fit_fivefold_ensemble', [TPY, 'scripts/ensemble_predictions.py', '--baseline', str(CONTROL),
                  '--candidates', *map(str, calibrated.values()), '--per-fold-halves', '--output', str(output)])
            save_field(full_fivefold_metrics=json.loads((output / 'metrics.json').read_text()))
        e.trial('fit_full_fivefold', fit_fivefold)
    enhanced = ROOT / 'experiments/campaign14h_final_ensemble'
    enhanced_eligible = False
    if e.CANDIDATES and e.DEADLINE-time.time() > 15*60:
        def fit_final():
            e.run('campaign14h_fit_final_ensemble', [TPY, 'scripts/ensemble_predictions.py', '--baseline',
                  str(promoted if promoted_eligible else CURRENT), '--candidates', *e.CANDIDATES, '--output', str(enhanced)])
        if e.trial('fit_final_ensemble', fit_final):
            info = json.loads((enhanced / 'metrics.json').read_text())
            enhanced_eligible = info['evaluation_auc_gain'] > .0001 and info['evaluation_gain_95pct_interval'][0] > 0
            save_field(final_ensemble=str(enhanced), final_metrics=info, submission_eligible=enhanced_eligible)
    # Predeclare at most two recipes, with no weight refitting from Kaggle scores.
    if enhanced_eligible:
        e.trial('submit_enhanced', lambda: submit(enhanced, 'enhanced', 'C33 14h physics fold replication and held-out SNR/averaging ensemble'))
    if promoted_eligible:
        e.trial('submit_fivefold', lambda: submit(promoted, 'fivefold', 'C33 14h fixed-weight strongest-width fivefold replication; per-fold calibrated'))
    if e.DEADLINE-time.time() > 90:
        e.trial('refresh_scores', lambda: e.run('campaign14h_refresh_scores', [TPY,'scripts/refresh_submissions.py','--wait-seconds','0']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=14)
    args = parser.parse_args()
    e.STATE = ROOT / 'reports/campaign14h_status.json'
    if e.STATE.exists():
        raise FileExistsError('Refuse to overwrite an existing campaign')
    lock = (ROOT / 'reports/campaign14h_manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time(); e.DEADLINE = started+args.hours*3600
    e.BASELINE = CURRENT; e.NOISE = NOISE; e.bank = bank; e.train = train; e.predict = predict
    e.READY[0].set(); e.NOISE_READY.set()
    signal.signal(signal.SIGTERM, e.stop); signal.signal(signal.SIGINT, e.stop)
    e.STATUS.update(started_at=e.now(), deadline_utc=dt.datetime.fromtimestamp(e.DEADLINE,dt.timezone.utc).isoformat(),
        deadline_china=dt.datetime.fromtimestamp(e.DEADLINE,dt.timezone(dt.timedelta(hours=8))).isoformat(),
        stage='four_gpu_fold_replication_and_parallel_data_generation', manager_pid=os.getpid(), planned_hours=args.hours,
        jobs={}, gpu_workers={}, trial_outcomes={}, candidates={}, data_ready={'0':{'bank':str(bank(0)),'reused':True}},
        data_errors={}, fold_replication={}, families={}, completed=False,
        validation_policy='Same-fold initialization, excluded mass priors and whitening; seed713 per-fold calibration/evaluation halves',
        external_artifact_scope='Published historical PSD and mass estimator fitting scope unknown',
        selection_policy='Adaptive validation reuse; frozen replication weights; no hidden-score fitting; maximum two campaign submissions',
        precision='BF16 convolutions and FP32 GeM/validation', maximum_new_physical_templates=2800000)
    e.save()
    plan = dict(budget_hours=args.hours, started_at=e.STATUS['started_at'], deadline_china=e.STATUS['deadline_china'],
        gpu_primary={str(g):{'fold':g+1,'warm_epochs':14,'natural_physical_epochs':66,'fine_lrs':[.0003,.001],
                           'fine_epochs_each':3,'signal_bank':1000000,'noise_bank':2000000} for g in range(4)},
        secondary=['SNR6-16 and natural-amplitude 50/50 mixture, up to24epochs, same-fold real fine-tuning',
                   'Average two fine-tune checkpoints and refresh BN using16384 own-training-fold rows',
                   'Averaging without BN refresh as a matched control when budget permits',
                   'Best-vs-last initialization control; fine epoch snapshots'],
        late_routes={'0':'Fold0 Width32 SNR mixture', '1':'Fold0 Width32 longer natural pretraining',
                     '2':'Fold0 proven V2 SNR6-16 continuation with15% real training noise',
                     '3':'Strong-student SGD and new-strong-teacher pseudo labels at15% weight'},
        inference_controls='None/sign/sign+swap with MC64/256/512; checkpoint change separately audited',
        data_reuse='Fold1-4 existing300k prefixes; generate700k new signals each; reuse audited2m noise',
        finalization_reserve_minutes=50, maximum_submissions=2,
        submission_gates='New fold0 additions: eval gain>0.0001 and pairedCI>0; fivefold promotion: independent fold1-4 replication gain>0.0003 and CI>0, no collapsed fold',
        target={'champion_public':.88645,'champion_private':.88538,'current_public':.88539,'current_private':.88493},
        experiments_are_hypotheses=True, secondary_and_late_jobs_budget_dependent=True)
    e.atomic_json(ROOT / 'reports/campaign14h_plan.json',plan)
    print(json.dumps(plan,indent=2),flush=True)
    def teacher():
        e.run('campaign14h_prepare_teacher', [TPY,'scripts/prepare_ensemble_pseudo.py', '--sources',
              str(ROOT / 'experiments/campaign10h_width_physical_lr3e4/predictions'),
              str(ROOT / 'experiments/campaign10h_width_physical_lr1e3/predictions'),
              str(ROOT / 'experiments/campaign10h_snr_6_16_fine/predictions'),
              '--weights','.1720776606764062','.40559315662578116','.2526150446608133',
              '--output',str(ROOT / 'data/pseudo_campaign14h_ensemble_fold0')])
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(prepare_data), pool.submit(lambda:e.trial('prepare_teacher',teacher))]
        futures += [pool.submit(e.gpu_worker,gpu,route) for gpu in range(4)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    e.trial('finalization',finalize)
    failures = [name for name, outcome in e.STATUS['trial_outcomes'].items() if outcome['state']=='failed']
    workers_failed = [g for g,w in e.STATUS['gpu_workers'].items() if w['state']=='failed']
    save_field(stage='stopped' if e.STOP.is_set() else ('finished_with_failures' if failures or workers_failed else 'finished'),
               completed=not e.STOP.is_set() and not failures and not workers_failed, finished_at=e.now(),
               actual_hours=(time.time()-started)/3600, failed_trials=failures, failed_gpu_workers=workers_failed,
               completed_prediction_candidates=len(e.CANDIDATES))
    print(json.dumps(e.STATUS,indent=2),flush=True)


if __name__ == '__main__':
    main()
