"""
Job: Wall-Clock Convergence Time measurement.
Extracts convergence point (95% of final performance) from TensorBoard logs.
Reports convergence time + total time for Base vs PAVE, TD3 + SAC, 6 envs.

Run from PAVE_Merge root: python td3/tests/eval_convergence_time.py
"""
import os, csv
import numpy as np
from scipy.ndimage import uniform_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ──
TD3_TB_ROOT = "/home/airlab1tb/Project/PAVE/td3/results/tensorboard_logs/"
SAC_TB_ROOT = "/home/airlab1tb/Project/PAVE/sac/results/tensorboard_logs/"

ENVS = ['lunar', 'pendulum', 'reacher', 'ant', 'hopper', 'walker']
OUTPUT_DIR = "./Full/td3/pths/eval_convergence_time/"

TD3_PREFIX = {'BASE': 'Base', 'PAVE': 'PAVE'}
SAC_PREFIX = {'Vanilla': 'Base', 'PAVE': 'PAVE'}

CONVERGENCE_THRESHOLD = 1.0  # 최종 성능(마지막 10% 평균)에 처음 도달한 시점
SMOOTHING_WINDOW = 10


def extract_curve(tb_dir):
    """Extract (wall_time, step, reward) from TB log."""
    try:
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        tag = 'rollout/ep_rew_mean'
        if tag not in ea.Tags().get('scalars', []):
            return None
        events = ea.Scalars(tag)
        if len(events) < 20:
            return None
        return {
            'wall_times': np.array([e.wall_time for e in events]),
            'steps': np.array([e.step for e in events]),
            'values': np.array([e.value for e in events]),
        }
    except:
        return None


def compute_convergence(data, threshold_pct=0.95, smooth_window=10):
    """Find convergence point: first time smoothed curve reaches threshold_pct of final."""
    values = data['values']
    wall_times = data['wall_times']
    steps = data['steps']

    # Smooth
    smoothed = uniform_filter1d(values, size=smooth_window)

    # Final performance: last 10%
    n = len(smoothed)
    final_perf = np.mean(smoothed[int(0.9 * n):])

    # Total time
    total_sec = wall_times[-1] - wall_times[0]
    total_min = total_sec / 60

    # Convergence threshold
    # Handle positive and negative returns
    if final_perf >= 0:
        threshold = threshold_pct * final_perf
        converged_mask = smoothed >= threshold
    else:
        # For negative returns (e.g., Pendulum), "95% of final" means closer to 0
        # We want |return| to be within 5% of |final|
        # i.e., return >= final_perf * (2 - threshold_pct) when final < 0
        # Simpler: threshold = final_perf + (1 - threshold_pct) * abs(final_perf)
        threshold = final_perf + (1 - threshold_pct) * abs(final_perf)
        converged_mask = smoothed >= threshold

    converged_indices = np.where(converged_mask)[0]

    if len(converged_indices) > 0:
        conv_idx = converged_indices[0]
        conv_sec = wall_times[conv_idx] - wall_times[0]
        conv_min = conv_sec / 60
        conv_step = int(steps[conv_idx])
    else:
        conv_min = total_min  # never converged → use total
        conv_step = int(steps[-1])

    return {
        'conv_min': conv_min,
        'conv_step': conv_step,
        'total_min': total_min,
        'final_perf': final_perf,
        'threshold': threshold,
    }


def process_framework(tb_root, prefix_map, fw_name):
    """Process Base vs PAVE for one framework."""
    print(f"\n{'=' * 60}")
    print(f"  {fw_name}")
    print(f"{'=' * 60}")

    all_results = []

    for env_name in ENVS:
        env_dir = os.path.join(tb_root, env_name)
        if not os.path.isdir(env_dir):
            continue

        for run_dir in sorted(os.listdir(env_dir)):
            path = os.path.join(env_dir, run_dir)
            if not os.path.isdir(path):
                continue

            raw_name = run_dir.split('_')[0]
            if raw_name not in prefix_map:
                continue
            method = prefix_map[raw_name]

            data = extract_curve(path)
            if data is None:
                continue

            result = compute_convergence(data, CONVERGENCE_THRESHOLD, SMOOTHING_WINDOW)

            # Parse seed from dirname
            parts = run_dir.split('_')
            seed = 'unknown'
            for p in parts:
                if p.isdigit() and len(p) > 3:
                    seed = p
                    break

            row = {
                'algo': fw_name,
                'env': env_name,
                'method': method,
                'seed': seed,
                'conv_step': result['conv_step'],
                'conv_minutes': round(result['conv_min'], 1),
                'total_minutes': round(result['total_min'], 1),
                'final_perf': round(result['final_perf'], 1),
                'threshold': round(result['threshold'], 1),
            }
            all_results.append(row)

    # Print summary: method × env → mean ± std of conv_minutes
    print(f"\n  {'Method':<8} {'Env':<12} {'Conv(min)':<15} {'Total(min)':<15} {'Conv Step':<15} {'Seeds'}")
    for method in ['Base', 'PAVE']:
        for env_name in ENVS:
            rows = [r for r in all_results if r['method'] == method and r['env'] == env_name]
            if rows:
                conv = [r['conv_minutes'] for r in rows]
                total = [r['total_minutes'] for r in rows]
                steps = [r['conv_step'] for r in rows]
                print(f"  {method:<8} {env_name:<12} "
                      f"{np.mean(conv):>6.1f}±{np.std(conv):>4.1f}  "
                      f"{np.mean(total):>6.1f}±{np.std(total):>4.1f}  "
                      f"{int(np.mean(steps)):>10}  "
                      f"{len(rows)}")

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    td3_results = process_framework(TD3_TB_ROOT, TD3_PREFIX, "TD3")
    sac_results = process_framework(SAC_TB_ROOT, SAC_PREFIX, "SAC")

    # Save CSV
    all_results = td3_results + sac_results
    if all_results:
        csv_path = os.path.join(OUTPUT_DIR, "convergence_time_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
