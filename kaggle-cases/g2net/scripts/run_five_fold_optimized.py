"""Expand the validated width and physical-data branches to complete five folds."""
import concurrent.futures
import datetime
import json
import os
import threading
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_phase1 as runner
from compare_predictions import digest, load_predictions


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/five_fold_optimized_status.json'
    runner.status.update(stage='training_remaining_folds',
                         validation_policy='Original fold 48 split; each model and mass prior excludes its own validation fold',
                         historical_prior_scope='Published mass-estimation artifact has unknown original training scope',
                         selection_basis='Round 1 held-out width/physical ensemble gains, not hidden test-score fitting')
    runner.save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1',
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'),
               NUMBA_CACHE_DIR=str(root / '.cache/numba'), MPLCONFIGDIR=str(root / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(root / '.cache'))
    python = str(root / 'envs/reproduce/bin/python')
    tools_python = str(root / 'envs/tools/bin/python')
    physics_python = str(root / 'envs/physics/bin/python')
    physics_env = dict(env, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    bank_failed = threading.Event()

    def ensure_banks():
        # Generate sequentially with 32 CPU workers while all GPUs train width models.
        try:
            for fold in range(1, 5):
                bank = root / f'data/synthetic_kdl_50k_fold{fold}'
                if not (bank / 'metadata.json').exists():
                    runner.run(f'generate_physical_prior_f{fold}', [physics_python, 'scripts/generate_kdl_bank.py',
                               '--fold', str(fold), '--count', '50000', '--workers', '32', '--output', str(bank)], physics_env)
                info = json.loads((bank / 'metadata.json').read_text())
                if info['excluded_prior_fold'] != fold or info['fold_seed'] != 48 or info['count'] != 50000:
                    raise ValueError('Incorrect fold-specific physical bank')
        except Exception:
            bank_failed.set()
            raise

    def fold_job(fold, gpu):
        prefix = f'fivefold_optimized_f{fold}'

        def train(tag, config, extra):
            runner.run(tag, [python, 'scripts/run_team3.py', '--config', config, '--fold', str(fold),
                       '--gpu', str(gpu), '--workers', '8', '--experiment-name', tag]+extra, env)

        def predict(tag, config, checkpoint=None, mc=0):
            checkpoint = checkpoint or f'experiments/{tag}/fold{fold}/Fold_{fold}_best_model.pth'
            runner.run(tag+'_predict', [python, 'scripts/predict_upstream.py', '--family', 'team3', '--config', config,
                       '--checkpoint', checkpoint, '--output', f'experiments/{tag}/predictions',
                       '--fold', str(fold), '--gpu', str(gpu), '--tta', '--mc-folds', str(mc)], env)

        tag = prefix+'_width_scratch'
        train(tag, 'V-V2SD', ['--epochs', '6'])
        predict(tag, 'V-V2SD')
        pre, fine = prefix+'_width_pretrain', prefix+'_width_finetune'
        train(pre, 'V-V2SD', ['--epochs', '2', '--synthetic-bank', str(root / 'data/synthetic_team3_300k')])
        train(fine, 'V-V2SD', ['--epochs', '6', '--checkpoint-dir', f'experiments/{pre}/fold{fold}'])
        predict(fine, 'V-V2SD')
        bank = root / f'data/synthetic_kdl_50k_fold{fold}'
        while not (bank / 'metadata.json').exists():
            if bank_failed.is_set():
                raise RuntimeError('Physical bank generation failed; do not wait indefinitely')
            time.sleep(15)
        for mode in ['physical', 'independent_noise']:
            pre, fine = prefix+'_'+mode+'_pretrain', prefix+'_'+mode+'_finetune'
            extra = ['--epochs', '24', '--lr', '0.003', '--no-augmentation', '--synthetic-bank', str(bank)]
            if mode == 'independent_noise':
                extra += ['--noise-bank', str(root / 'data/synthetic_kdl_noise_250k')]
            train(pre, 'V2', extra)
            train(fine, 'V2', ['--epochs', '2', '--lr', '0.0001', '--no-augmentation',
                              '--checkpoint-dir', f'experiments/{pre}/fold{fold}'])
            predict(fine, 'V2')
        # Round-1 pseudo training kept the exact teacher checkpoint. Replicate
        # its MC64 inference directly, without repeating an ineffective stage.
        predict(prefix+'_teacher_mc64', 'V2',
                checkpoint=f'experiments/fivefold_v2_synthetic_finetune_f{fold}/fold{fold}/Fold_{fold}_best_model.pth', mc=64)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ensure_banks)]
            futures += [executor.submit(fold_job, fold, fold-1) for fold in range(1, 5)]
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
        families = {
            'width_scratch': 'real_team3_width32_sd_fp32_scratch',
            'width_finetune': 'real_team3_width32_sd_fp32_synthetic_finetune',
            'physical_finetune': 'real_team3_kdl_physical_finetune',
            'independent_noise_finetune': 'real_team3_kdl_independent_noise_finetune',
            'teacher_mc64': 'real_team3_v2_pseudo_low_lr',
        }
        candidates = []
        for family, first_tag in families.items():
            oof = np.full(len(train), np.nan, dtype=np.float32)
            test_values, fold_aucs, sources = [], [], []
            for fold in range(5):
                tag = first_tag if fold == 0 else f'fivefold_optimized_f{fold}_{family}'
                directory = root / 'experiments' / tag / 'predictions'
                values, predicted, info = load_predictions(directory, train, test)
                mask = train.fold_48.values == fold
                if info['fold'] != fold or info['fold_seed'] != 48 or not np.array_equal(np.isfinite(values), mask):
                    raise ValueError('Family predictions have an incorrect held-out fold')
                oof[mask] = values[mask]
                test_values.append(predicted)
                fold_aucs.append(float(roc_auc_score(train.target.values[mask], values[mask])))
                sources.append(str(directory))
            if not np.isfinite(oof).all():
                raise ValueError('Incomplete family OOF coverage')
            output = root / 'experiments' / ('fivefold_family_'+family)
            output.mkdir(parents=True, exist_ok=True)
            predicted = np.mean(test_values, axis=0)
            np.save(output / 'oof.npy', oof)
            np.save(output / 'test_predictions.npy', predicted)
            pd.DataFrame({'id': test.id, 'target': predicted}).to_csv(output / 'submission.csv', index=False)
            evidence = dict(family='five_fold_'+family, fixture=False, data_scope='full official training data',
                            competition_validation_auc=float(roc_auc_score(train.target, oof)),
                            validation_samples=len(train), test_samples=len(test), train_id_sha256=digest(train.id),
                            test_id_sha256=digest(test.id), unpredicted_oof_rows=0, fold_aucs=fold_aucs, sources=sources,
                            full_competition_score_reproduction=False, kaggle_public_auc=None, kaggle_private_auc=None)
            (output / 'metrics.json').write_text(json.dumps(evidence, indent=2))
            candidates.append(str(output))
        output = root / 'experiments/fivefold_optimized_1d_ensemble'
        runner.status['stage'] = 'fitting_complete_oof_ensemble'
        runner.save()
        runner.run('fit_complete_oof_optimized_ensemble', [tools_python, 'scripts/ensemble_predictions.py',
                   '--baseline', str(root / 'experiments/fivefold_v2_two_initializations'),
                   '--candidates', *candidates, '--output', str(output)], env)
        info = json.loads((output / 'metrics.json').read_text())
        eligible = (info['evaluation_auc_gain'] > 0.0001 and info['evaluation_gain_95pct_interval'][0] > 0)
        runner.status.update(submission_eligible=eligible, output=str(output))
        runner.save()
        records = [json.loads(p.read_text()) for p in (root / 'reports/submissions').glob('*.json')]
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if eligible and sum(r['submitted_at'].startswith(today) for r in records) < 5:
            runner.run('submit_complete_fivefold_optimized', [tools_python, 'scripts/submit_checked.py',
                       '--predictions', str(output), '--message',
                       'C33 full five-fold width and physical-pretraining ensemble; complete OOF weights'], env)
            runner.run('refresh_complete_fivefold_optimized', [tools_python, 'scripts/refresh_submissions.py',
                       '--wait-seconds', '600'], env)
        else:
            runner.status['submission_skipped'] = 'Validation gain unsupported or conservative daily budget reached'
        runner.status.update(stage='finished', completed=True)
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
