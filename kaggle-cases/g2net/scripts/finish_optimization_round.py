"""Finish the queued ablations, fit validation weights and submit proven gains."""
import datetime
import hashlib
import json
import os
import time

import run_phase1 as runner


def wait_for_status(filename):
    path = runner.ROOT / 'reports' / filename
    while True:
        status = json.loads(path.read_text())
        if status['stage'] in ['finished', 'failed', 'finished_with_failures']:
            return status
        time.sleep(15)


def main():
    root = runner.ROOT
    runner.STATE = root / 'reports/optimization_round_status.json'
    runner.status.update(stage='waiting_for_1d_experiments',
                         submission_policy='Validation-selected improvements only; at most 5 local submissions per UTC day',
                         target_public_auc=0.88645, target_private_auc=0.88538)
    runner.save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    python = str(root / 'envs/tools/bin/python')

    def predictions(tag):
        return root / 'experiments' / tag / 'predictions'

    def valid_artifacts(paths):
        return [p for p in paths if (p / 'metrics.json').exists()]

    def submit_if_improved(output, tag):
        info = json.loads((output / 'metrics.json').read_text())
        good = (info['evaluation_auc_gain'] > 0.0001
                and info['evaluation_gain_95pct_interval'][0] > 0
                and info['competition_validation_auc'] > info['baseline_full_validation_auc']+0.00005)
        decision = dict(artifact=str(output), eligible=good,
                        evaluation_gain=info['evaluation_auc_gain'], interval=info['evaluation_gain_95pct_interval'])
        if not good:
            decision['reason'] = 'No statistically supported held-out gain over the existing ensemble'
        else:
            records = [json.loads(p.read_text()) for p in (root / 'reports/submissions').glob('*.json')]
            today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
            daily = [r for r in records if r['submitted_at'].startswith(today)]
            digest = hashlib.sha256((output / 'submission.csv').read_bytes()).hexdigest()
            if len(daily) >= 5:
                decision.update(submitted=False, reason='Conservative daily submission budget reached')
            elif any(r.get('file_sha256') == digest for r in records):
                decision.update(submitted=False, reason='Identical file already submitted')
            else:
                runner.run('submit_'+tag, [python, 'scripts/submit_checked.py', '--predictions', str(output),
                    '--message', 'C33 '+tag+' validation-weighted ensemble; calibration/evaluation split'], env)
                runner.run('refresh_'+tag, [python, 'scripts/refresh_submissions.py', '--wait-seconds', '600'], env)
                decision['submitted'] = True
        runner.status.setdefault('decisions', {})[tag] = decision
        runner.save()
        return good

    try:
        for filename in ['width_trial_status.json', 'physics_trial_status.json',
                         'pseudo_trial_status.json', 'independent_noise_trial_status.json']:
            result = wait_for_status(filename)
            runner.status.setdefault('dependency_results', {})[filename] = result['stage']
            runner.save()
        baseline = root / 'experiments/comparison_normalized_synthetic'
        candidates = valid_artifacts([predictions(tag) for tag in [
            'real_team3_width32_sd_fp32_scratch', 'real_team3_width32_sd_fp32_synthetic_finetune',
            'real_team3_kdl_physical_finetune', 'real_team3_kdl_independent_noise_finetune',
            'real_team3_v2_pseudo_low_lr']])
        runner.status['stage'] = 'fitting_1d_ensemble'
        runner.save()
        if candidates:
            output = root / 'experiments/optimized_1d_ensemble'
            runner.run('fit_optimized_1d_ensemble', [python, 'scripts/ensemble_predictions.py',
                       '--baseline', str(baseline), '--candidates', *map(str, candidates), '--output', str(output)], env)
            if submit_if_improved(output, 'optimized_1d'):
                baseline = output
        runner.status.update(stage='waiting_for_wavegram', best_1d_artifact=str(baseline),
                             full_1d_candidate_artifacts=[str(p) for p in candidates])
        runner.save()
        wait_for_status('real_event_wavegram_gpu2_status.json')
        candidates = valid_artifacts([predictions('real_event_wavegram_gpu2'), predictions('real_event_seq03')])
        if candidates:
            output = root / 'experiments/optimized_1d_2d_ensemble'
            runner.run('fit_optimized_1d_2d_ensemble', [python, 'scripts/ensemble_predictions.py',
                       '--baseline', str(baseline), '--candidates', *map(str, candidates), '--output', str(output)], env)
            if submit_if_improved(output, 'optimized_1d_2d'):
                baseline = output
        runner.status.update(stage='finished', completed=True, best_artifact=str(baseline))
    except Exception as error:
        runner.status.update(stage='failed', completed=False, error=str(error))
        raise
    finally:
        runner.save()


if __name__ == '__main__':
    main()
