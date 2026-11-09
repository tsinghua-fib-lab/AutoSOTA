"""Complete five folds of the two V2 initializations already validated in fold 0."""
import concurrent.futures
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

import run_phase1 as runner
from compare_predictions import digest, load_predictions


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/five_fold_baseline_status.json'
    runner.status.update(stage='waiting_for_current_round',
                         validation_policy='Each sample predicted only by models excluding its fold',
                         initialization_logit_weight=0.5,
                         weight_scope='Fixed from fold 0 calibration; no test-score weight fitting')
    runner.save()
    while True:
        previous = json.loads((root / 'reports/optimization_round_status.json').read_text())
        if previous['stage'] in ['finished', 'failed']:
            break
        time.sleep(15)
    project = json.loads((root / 'reports/project_status.json').read_text())
    if project.get('champion_exceeded'):
        runner.status.update(stage='finished', completed=True, skipped='Both historical score targets already exceeded')
        runner.save()
        return
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    runner.status['stage'] = 'training_remaining_folds'
    runner.save()

    def fold_job(fold, gpu):
        common = [python, 'scripts/run_team3.py', '--config', 'V2', '--fold', str(fold), '--gpu', str(gpu),
                  '--workers', '8']
        scratch = f'fivefold_v2_scratch_f{fold}'
        pre = f'fivefold_v2_synthetic_pretrain_f{fold}'
        fine = f'fivefold_v2_synthetic_finetune_f{fold}'

        def predict(tag):
            runner.run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', 'V2',
                    '--checkpoint', f'experiments/{tag}/fold{fold}/Fold_{fold}_best_model.pth',
                    '--output', f'experiments/{tag}/predictions', '--fold', str(fold), '--gpu', str(gpu), '--tta'], env)

        runner.run(scratch, common+['--epochs', '6', '--experiment-name', scratch], env)
        predict(scratch)
        runner.run(pre, common+['--epochs', '2', '--synthetic-bank', str(root / 'data/synthetic_team3_300k'),
                               '--experiment-name', pre], env)
        runner.run(fine, common+['--epochs', '6', '--checkpoint-dir', f'experiments/{pre}/fold{fold}',
                                '--experiment-name', fine], env)
        predict(fine)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(fold_job, fold, fold-1) for fold in range(1, 5)]
            errors = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError('; '.join(errors))
        train = pd.read_csv(root / 'data/packed/train.csv', dtype={'id': str})
        test = pd.read_csv(root / 'data/packed/test.csv', dtype={'id': str})
        oof = np.full(len(train), np.nan, dtype=np.float32)
        test_by_fold, aucs, sources = [], [], []
        for fold in range(5):
            tags = (['real_team3_v2_scratch', 'real_team3_v2_synthetic_finetune'] if fold == 0 else
                    [f'fivefold_v2_scratch_f{fold}', f'fivefold_v2_synthetic_finetune_f{fold}'])
            loaded = [load_predictions(root / 'experiments' / tag / 'predictions', train, test) for tag in tags]
            mask = train.fold_48.values == fold
            for values, _, evidence in loaded:
                if evidence['fold'] != fold or evidence['fold_seed'] != 48 or not np.array_equal(np.isfinite(values), mask):
                    raise ValueError('Incorrect fold coverage or source split')
            if np.isfinite(oof[mask]).any():
                raise ValueError('Repeated OOF assignment')
            oof[mask] = expit(np.mean([logit(np.clip(value[0][mask], 1e-6, 1-1e-6)) for value in loaded], axis=0))
            test_by_fold.append(expit(np.mean([logit(np.clip(value[1], 1e-6, 1-1e-6)) for value in loaded], axis=0)))
            aucs.append(float(roc_auc_score(train.target.values[mask], oof[mask])))
            sources.extend(tags)
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete five-fold OOF coverage')
        output = root / 'experiments/fivefold_v2_two_initializations'
        output.mkdir(parents=True, exist_ok=True)
        predictions = np.mean(test_by_fold, axis=0)
        np.save(output / 'oof.npy', oof)
        np.save(output / 'test_predictions.npy', predictions)
        pd.DataFrame({'id': test.id, 'target': predictions}).to_csv(output / 'submission.csv', index=False)
        evidence = dict(family='five_fold_two_initialization_ensemble', fixture=False,
                        data_scope='full official training data', competition_validation_auc=float(roc_auc_score(train.target, oof)),
                        validation_samples=len(train), test_samples=len(test), train_id_sha256=digest(train.id),
                        test_id_sha256=digest(test.id), unpredicted_oof_rows=0, fold_aucs=aucs, sources=sources,
                        test_aggregation='Arithmetic mean of five fold-wise, equally logit-weighted initialization ensembles',
                        validation_scope='OOF includes two models per row; test averages ten models',
                        full_competition_score_reproduction=False, kaggle_public_auc=None, kaggle_private_auc=None)
        (output / 'metrics.json').write_text(json.dumps(evidence, indent=2))
        runner.status.update(stage='submission_validation', output=str(output))
        runner.save()
        records = [json.loads(p.read_text()) for p in (root / 'reports/submissions').glob('*.json')]
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if sum(r['submitted_at'].startswith(today) for r in records) < 5:
            tools_python = str(root / 'envs/tools/bin/python')
            runner.run('submit_complete_fivefold_baseline', [tools_python, 'scripts/submit_checked.py', '--predictions', str(output),
                       '--message', 'C33 full five-fold V2 two-initialization ensemble; complete genuine OOF'], env)
            runner.run('refresh_complete_fivefold_baseline', [tools_python, 'scripts/refresh_submissions.py',
                       '--wait-seconds', '600'], env)
        else:
            runner.status['submission_skipped'] = 'Conservative daily submission budget reached; file ready'
        runner.status.update(stage='finished', completed=True, oof_auc=evidence['competition_validation_auc'])
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
