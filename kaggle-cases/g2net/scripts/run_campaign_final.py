"""Last four-GPU attempt, bounded by an absolute China-time deadline.

Queue independent hypotheses, reuse audited data, enforce real optimizer updates,
reserve finalization time and retain all existing submissions/checkpoints.
"""
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
import queue
import signal
import time
from pathlib import Path
import run_campaign10h as e

ROOT, PY, TPY = e.ROOT, e.PY, e.TPY
CURRENT = ROOT / 'experiments/campaign14h_promoted_fivefold'
NOISE = ROOT / 'data/synthetic_kdl_noise_2m_campaign10h'
FAMILIES = {}
QUEUE = queue.Queue()


def save(**fields):
    with e.LOCK:
        e.STATUS.update(fields); e.save()


def bank(f):
    return ROOT / ('data/synthetic_kdl_1m_fold0_campaign10h' if f == 0 else
                   'data/synthetic_kdl_1m_fold%d_campaign14h' % f)


def natural(f, last=False):
    if f == 0:
        tag = 'campaign14h_width_f0_continue48' if last else 'campaign10h_width_physical_1m_pre100'
    else:
        tag = 'campaign14h_width_f%d_physical_pre66' % f
    return e.ck(tag, f).parent


def snr(f):
    return e.ck('campaign14h_width_f%d_snr_mix_pre24' % f, f).parent


def strong(f):
    return e.ck('campaign10h_width_physical_lr3e4' if f == 0 else 'campaign14h_width_f%d_lr3e4' % f, f).parent


def train(tag, gpu, fold, initial, epochs=3, lr='.0003', selection='best', config='V-V2SD',
          extra=(), minutes=55, save_epochs=True):
    end = min(time.time()+minutes*60, e.DEADLINE-85*60)
    if end-time.time() < 5*60: raise e.TimeBudget('Training reserve reached')
    command = [PY,'scripts/run_team3.py','--config',config,'--fold',str(fold),'--gpu',str(gpu),
               '--workers','6','--epochs',str(epochs),'--experiment-name',tag,'--bf16',
               '--stable-gem','--audited-training','--gradient-clip','10','--keep-last',
               '--no-augmentation','--lr',lr,'--checkpoint-dir',str(initial),
               '--checkpoint-selection',selection,'--stop-at',str(end)]
    if save_epochs: command += ['--save-epoch-checkpoints']
    command += list(extra)
    e.run(tag,command,gpu,end=min(end+8*60,e.DEADLINE-75*60),checkpoint=e.ck(tag,fold))
    audit=json.loads((e.ck(tag,fold).parent/'optimizer_audit.json').read_text())
    if audit['actual_optimizer_updates'] <= 0 or audit['state']=='failed':
        raise ValueError('Training made no valid optimizer updates')
    with e.LOCK:
        e.STATUS['optimizer_audits'][tag]=audit;e.save()


def predict(tag,gpu,fold,config='V-V2SD'):
    output=ROOT/('experiments/'+tag+'/predictions')
    e.run(tag+'_predict',[PY,'scripts/predict_upstream.py','--family','team3','--config',config,
          '--checkpoint',str(e.ck(tag,fold)),'--output',str(output),'--fold',str(fold),
          '--gpu',str(gpu),'--tta-mode','sign_swap','--mc-folds','64' if config=='V-V2SD' else '0'],
          gpu,end=e.DEADLINE-65*60)
    info=json.loads((output/'metrics.json').read_text())
    with e.LOCK:
        e.STATUS['predictions'][tag]=dict(path=str(output),fold=fold,auc=info['competition_validation_auc'])
        if fold==0:
            e.CANDIDATES.append(str(output))
            e.STATUS['candidates'][tag]=dict(directory=str(output),validation_auc=info['competition_validation_auc'])
        e.save()
    return output


def register(family,fold,path):
    with e.LOCK:
        FAMILIES.setdefault(family,{})[fold]=path
        e.STATUS['families'][family]={str(f):str(p) for f,p in FAMILIES[family].items()};e.save()


def fine(family,fold,gpu,initial,epochs=3,lr='.0003',selection='best',extra=(),config='V-V2SD',average=False):
    tag='campaign_final_%s_f%d' % (family,fold)
    train(tag,gpu,fold,initial,epochs,lr,selection,config,extra,minutes=80 if '--sam-rho' in extra else 55)
    result=predict(tag,gpu,fold,config);register(family,fold,result)
    if average and e.DEADLINE-time.time()>130*60:
        files=sorted(e.ck(tag,fold).parent.glob('Fold_%d_epoch_*_model.pth' % fold))[-3:]
        if len(files)>=2:
            def soup():
                name=tag+'_soup';dest=e.ck(name,fold).parent
                e.run(name+'_average',[PY,'scripts/average_team3_checkpoints.py','--checkpoints',*map(str,files),
                      '--output',str(dest),'--gpu',str(gpu),'--bn-samples','16384'],gpu,end=e.DEADLINE-80*60)
                register(family+'_soup',fold,predict(name,gpu,fold,config))
            e.trial(tag+'_soup',soup)
    return result


def long_chain(kind,fold,gpu):
    if e.DEADLINE-time.time()<150*60: raise e.TimeBudget('Insufficient time for physical training and real fine-tuning')
    config='V2' if kind=='v2_snr' else 'V-V2SD'
    epochs=32 if kind!='width_mix' else 24
    tag='campaign_final_%s_pre_f%d' % (kind,fold)
    extra=['--synthetic-bank',str(bank(fold)),'--noise-bank',str(NOISE)]
    if kind=='v2_snr':
        initial=(e.ck('campaign14h_v2_snr6_16_real_noise_pre40').parent if fold==0 else
                 e.ck('fivefold_optimized_f%d_physical_finetune' % fold,fold).parent)
        selection='last' if fold==0 else 'best'
        extra+=['--synthetic-snr-range','6','16','--noise-real-probability','.15']
    elif kind=='width_mix':
        initial=snr(fold);selection='last'
        extra+=['--synthetic-snr-range','6','16','--synthetic-snr-probability','.5',
                '--noise-real-probability','.15','--resume-optimizer-moments']
    else:
        initial=natural(fold,last=True);selection='last'
        extra+=['--resume-optimizer-moments']
    # Always reserve sufficient time for this chain's fine-tuning and inference.
    available_minutes=(e.DEADLINE-time.time())/60-160
    train(tag,gpu,fold,initial,epochs,'.0003',selection,config,extra,
          minutes=min(270,max(25,available_minutes)),save_epochs=False)
    family=kind+'_continued'
    fine(family,fold,gpu,e.ck(tag,fold).parent,epochs=3,selection='last',config=config,
         average=(kind=='width_natural'))
    if kind=='width_natural' and e.DEADLINE-time.time()>160*60:
        fine(family+'_low_lr',fold,gpu,e.ck(tag,fold).parent,epochs=3,lr='.0001',selection='last',config=config)


def execute(task,gpu):
    kind,fold=task['kind'],task['fold']
    if kind in ['v2_snr','width_natural','width_mix']:
        return long_chain(kind,fold,gpu)
    if kind=='rescue_snr': return fine(kind,fold,gpu,snr(fold),selection='last')
    if kind=='rescue_natural': return fine(kind,fold,gpu,natural(fold,True),selection='last')
    if kind=='best_control': return fine(kind,fold,gpu,natural(fold))
    if kind=='moment_control': return fine(kind,fold,gpu,natural(fold),extra=['--resume-optimizer-moments'])
    if kind=='v2_rescue':
        return fine(kind,fold,gpu,e.ck('campaign14h_v2_snr6_16_real_noise_pre40').parent,
                    selection='last',config='V2')
    options={
        'adam5':[], 'sam003':['--sam-rho','.03'], 'sam001':['--sam-rho','.01'],
        'bn_freeze':['--freeze-bn'], 'gain003':['--gain-jitter','.03'],
        'rank01':['--rank-weight','.1'],
    }
    return fine(kind,fold,gpu,strong(fold),epochs=5,extra=options[kind],average=(kind=='sam003'))


def make_tasks():
    tasks=[]
    # Restore the two affected hypotheses first; fold2 is the known failed fold.
    for kind in ['rescue_snr','rescue_natural','best_control']:
        for fold in [2,0,1,3,4]: tasks.append(dict(kind=kind,fold=fold))
    tasks.append(dict(kind='v2_rescue',fold=0))
    # Secure a complementary narrow-network family before expensive late work.
    for kind in ['v2_snr','sam003','width_natural','adam5','moment_control',
                 'bn_freeze','gain003','rank01','sam001','width_mix']:
        for fold in range(5): tasks.append(dict(kind=kind,fold=fold))
    return tasks


def worker(gpu):
    while not e.STOP.is_set() and e.DEADLINE-time.time()>110*60:
        try: task=QUEUE.get_nowait()
        except queue.Empty: break
        name='%s_f%d' % (task['kind'],task['fold'])
        with e.LOCK:
            e.STATUS['active_pipelines'][str(gpu)]=name
            e.STATUS['queued_pipelines']=QUEUE.qsize();e.save()
        ok=e.trial(name,lambda:execute(task,gpu))
        with e.LOCK:
            if ok:e.STATUS['trial_outcomes'][name]=dict(state='completed',at=e.now())
            e.STATUS['queued_pipelines']=QUEUE.qsize();e.save()
        QUEUE.task_done()
    # Use any genuine remaining budget for new-seed tests on existing strong
    # models. These are optional and cannot delay final artifact collection.
    count=0
    while not e.STOP.is_set() and e.DEADLINE-time.time()>150*60 and count<6:
        count+=1;family='late_seed%d_%d' % (gpu,count)
        e.trial(family,lambda: fine(family,0,gpu,strong(0),epochs=5,lr='.0001',
                extra=['--training-seed',str(20026+100*gpu+count)]))
    with e.LOCK:
        e.STATUS['active_pipelines'].pop(str(gpu),None);e.save()


def submit(output,name):
    records=[json.loads(p.read_text()) for p in (ROOT/'reports/submissions').glob('*.json')]
    day=dt.datetime.now(dt.timezone.utc).date().isoformat()
    if sum(x['submitted_at'].startswith(day) for x in records)>=5 or e.DEADLINE-time.time()<500:
        save(submission_skipped='Project daily limit or final reserve');return
    e.run('campaign_final_submit_'+name,[TPY,'scripts/submit_checked.py','--predictions',str(output),
          '--message','C33 final stable-gradient campaign: '+name+'; validation-selected, frozen before test feedback',
          '--wait-seconds',str(min(480,int(e.DEADLINE-time.time()-120)))])


def finalize():
    save(stage='final_calibration_ensemble_and_submission')
    calibrated=[]
    for family,members in list(FAMILIES.items()):
        if len(members)!=5: continue
        output=ROOT/('experiments/campaign_final_calibrated_'+family)
        if e.trial('calibrate_'+family,lambda members=members,output=output,family=family:
            e.run('campaign_final_calibrate_'+family,[TPY,'scripts/calibrate_fivefold_family.py',
                  '--sources',*[str(members[f]) for f in range(5)],'--output',str(output)])):
            calibrated.append(output)
    save(calibrated_families=[str(p) for p in calibrated])
    promoted=ROOT/'experiments/campaign_final_width_promoted'
    promoted_ok=False
    if calibrated and e.DEADLINE-time.time()>25*60:
        def width():
            ref=ROOT/'experiments/campaign_final_width_reference'
            ensemble=ROOT/'experiments/campaign_final_full_fivefold_ensemble'
            if not ref.exists():
                e.run('campaign_final_prepare_width_reference',[TPY,'scripts/final_width_replacement.py','--prepare'])
            e.run('campaign_final_fit_fivefold',[TPY,'scripts/ensemble_predictions.py','--baseline',str(ref),
                  '--candidates',*map(str,calibrated),'--per-fold-halves','--output',str(ensemble)])
            e.run('campaign_final_promote_width',[TPY,'scripts/final_width_replacement.py',
                  '--replacement',str(ensemble),'--output',str(promoted)])
        if e.trial('fivefold_width_replacement',width):
            metrics=json.loads((promoted/'metrics.json').read_text());promoted_ok=metrics['submission_eligible']
            save(width_replacement_metrics=metrics)
    enhanced=ROOT/'experiments/campaign_final_final_ensemble';enhanced_ok=False
    # All choices and both recipes are finalized before either Kaggle upload.
    candidates=e.CANDIDATES+[str(p) for p in calibrated]
    if candidates and e.DEADLINE-time.time()>15*60:
        baseline=promoted if promoted_ok else CURRENT
        if e.trial('final_fold0_ensemble',lambda:e.run('campaign_final_fit_final',
            [TPY,'scripts/ensemble_predictions.py','--baseline',str(baseline),'--candidates',*candidates,'--output',str(enhanced)])):
            metrics=json.loads((enhanced/'metrics.json').read_text())
            enhanced_ok=metrics['evaluation_auc_gain']>.0001 and metrics['evaluation_gain_95pct_interval'][0]>0
            save(final_ensemble=str(enhanced),final_ensemble_metrics=metrics,enhanced_submission_eligible=enhanced_ok)
    save(recipes_frozen_at=e.now(),submission_plan=[str(p) for p,ok in [(promoted,promoted_ok),(enhanced,enhanced_ok)] if ok])
    if promoted_ok:e.trial('submit_width',lambda:submit(promoted,'fivefold_width_replacement'))
    if enhanced_ok:e.trial('submit_enhanced',lambda:submit(enhanced,'final_diverse_ensemble'))
    if e.DEADLINE-time.time()>90:
        e.trial('refresh_scores',lambda:e.run('campaign_final_refresh_scores',[TPY,'scripts/refresh_submissions.py','--wait-seconds','0']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deadline',default='2026-10-14T10:00:00+08:00')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();tasks=make_tasks()
    deadline=dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:raise ValueError('Deadline must include timezone')
    e.DEADLINE=deadline.timestamp()
    if args.dry_run:
        print(json.dumps(dict(deadline=args.deadline,pipelines=len(tasks),tasks=tasks),indent=2));return
    if e.DEADLINE-time.time()<2*3600:raise ValueError('Insufficient campaign budget')
    e.STATE=ROOT/'reports/campaign_final_status.json'
    if e.STATE.exists():raise FileExistsError('Existing final campaign state; refuse duplicate launch')
    lock=(ROOT/'reports/campaign_final_manager.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=json.loads((ROOT/'reports/campaign_final_gem_audit.json').read_text())
    if audit['real_checkpoint_log']['nonfinite_gradient_tensors'] or audit['real_checkpoint_log']['changed_parameters']==0:
        raise ValueError('Numerical repair audit failed')
    for fold in range(5):
        info=json.loads((bank(fold)/'metadata.json').read_text())
        snr_info=json.loads((bank(fold)/'snr_audit.json').read_text())
        if info['excluded_prior_fold']!=fold or info['fold_seed']!=48 or not snr_info['passed']:
            raise ValueError('Invalid physical data scope')
    started=time.time();e.BASELINE=CURRENT
    signal.signal(signal.SIGTERM,e.stop);signal.signal(signal.SIGINT,e.stop)
    e.STATUS.update(started_at=e.now(),deadline_utc=deadline.astimezone(dt.timezone.utc).isoformat(),
        deadline_china=deadline.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        manager_pid=os.getpid(),stage='final_four_gpu_exploration',planned_hours=(e.DEADLINE-started)/3600,
        jobs={},gpu_workers={},trial_outcomes={},candidates={},predictions={},families={},optimizer_audits={},
        active_pipelines={},queued_pipelines=len(tasks),completed=False,
        numerical_audit='reports/campaign_final_gem_audit.json',research='reports/research_final.txt',
        no_automatic_extension=True,training_cutoff_china=dt.datetime.fromtimestamp(e.DEADLINE-85*60,deadline.tzinfo).isoformat(),
        validation_policy='Own-fold exclusions; calibration/evaluation halves seed713; adaptive reuse acknowledged; no hidden-score weight fitting')
    e.save()
    plan=dict(deadline_china=e.STATUS['deadline_china'],wall_clock_budget_hours=e.STATUS['planned_hours'],
        GPUs=4,pipelines=len(tasks),tasks=tasks,optional_seed_pipelines_per_gpu=6,
        training='BF16 convolutions; FP32 log-GeM and parameters; no loss scaler; every backward finite-checked; actual optimizer counts',
        methods=['Stalled natural/SNR checkpoint recovery','Matched best/last and optimizer-moment controls',
                 'V2 complementary SNR6-16 fivefold replication','SAM rho0.01/0.03 and matched AdamW5',
                 'BN freeze','Detector gain jitter sigma0.03','BCE plus pairwise rank0.1',
                 'Stable natural and SNR/real-noise physical continuation','Snapshot parameter soups with train-only BN'],
        data='Reuse5 million per-fold physical templates and2 million independent noise waveforms',
        budget_policy='Priority queue; long chains capped; optional jobs may not start; train stops85min early, GPU jobs65min early',
        finalization_reserve_minutes=65,maximum_submissions=2,
        submission_gates={'fivefold_replacement':'280k eval gain>0.0001, pairedCI>0, no fold gain<-0.00015, actual fold0 blend gain>=0',
                          'enhanced':'56k eval gain>0.0001 and pairedCI>0'},
        target=dict(champion_private=.88538,current_private=.88525,champion_public=.88645,current_public=.88563))
    e.atomic_json(ROOT/'reports/campaign_final_plan.json',plan)
    for task in tasks:QUEUE.put(task)
    print(json.dumps(plan,indent=2),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(e.gpu_worker,gpu,worker) for gpu in range(4)]
        for future in concurrent.futures.as_completed(futures):future.result()
    pending=[]
    while not QUEUE.empty():pending.append(QUEUE.get_nowait())
    save(unstarted_budget_dependent_pipelines=pending)
    e.trial('finalization',finalize)
    failures=[n for n,v in e.STATUS['trial_outcomes'].items() if v['state']=='failed']
    failed_workers=[n for n,v in e.STATUS['gpu_workers'].items() if v['state']=='failed']
    save(stage='stopped' if e.STOP.is_set() else ('finished_with_failures' if failures or failed_workers else 'finished'),
         completed=not e.STOP.is_set() and not failures and not failed_workers,finished_at=e.now(),
         actual_hours=(time.time()-started)/3600,failed_trials=failures,failed_gpu_workers=failed_workers)


if __name__=='__main__':main()
