"""
Wall-clock convergence time from TensorBoard logs.
Based on learning_curves.ipynb structure.
Reports total training time (minutes) per method × env × seed.

Run from PAVE_Merge root: python td3/tests/eval_wallclock.py
"""
import os, csv
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── TB log roots (same as learning_curves.ipynb) ──
TD3_TB_ROOT = "/home/airlab1tb/Project/PAVE/td3/results/tensorboard_logs/"
SAC_TB_ROOT = "/home/airlab1tb/Project/PAVE/sac/results/tensorboard_logs/"

ENVS = ['lunar', 'pendulum', 'reacher', 'ant', 'hopper', 'walker']
OUTPUT_DIR = "./Full/td3/pths/eval_wallclock/"

# Method name mapping (folder prefix → display name)
TD3_PREFIX_MAP = {'BASE': 'Base', 'CAPS': 'CAPS', 'GRAD': 'GRAD', 'ASAP': 'ASAP', 'PAVE': 'PAVE'}
SAC_PREFIX_MAP = {'Vanilla': 'Base', 'CAPS': 'CAPS', 'GRAD': 'GRAD', 'ASAP': 'ASAP', 'PAVE': 'PAVE'}

def extract_wallclock(log_dir, tag='rollout/ep_rew_mean'):
    """Extract wall_time and reward from a single TB run."""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            return None
        events = ea.Scalars(tag)
        if len(events) < 2:
            return None
        return {
            'wall_times': [e.wall_time for e in events],
            'rewards': [e.value for e in events],
            'steps': [e.step for e in events],
        }
    except Exception as e:
        print(f"  Error reading {log_dir}: {e}")
        return None

def process_framework(tb_root, prefix_map, fw_name):
    """Process all methods × envs. Returns dict[method][env] = {total_min_mean, total_min_std, ...}"""
    print(f"\n{'='*60}")
    print(f"  {fw_name}")
    print(f"{'='*60}")

    results = {}

    for env_name in ENVS:
        env_dir = os.path.join(tb_root, env_name)
        if not os.path.isdir(env_dir):
            continue

        for run_dir in sorted(os.listdir(env_dir)):
            path = os.path.join(env_dir, run_dir)
            if not os.path.isdir(path):
                continue

            # Parse method from folder name (same logic as notebook)
            raw_name = run_dir.split('_')[0]
            if raw_name not in prefix_map:
                continue
            method = prefix_map[raw_name]

            if method not in results:
                results[method] = {}
            if env_name not in results[method]:
                results[method][env_name] = []

            data = extract_wallclock(path)
            if data:
                total_sec = data['wall_times'][-1] - data['wall_times'][0]
                results[method][env_name].append(total_sec / 60)  # minutes

    # Print summary
    for method in ['Base', 'CAPS', 'GRAD', 'ASAP', 'PAVE']:
        if method not in results:
            continue
        for env_name in ENVS:
            if env_name not in results[method]:
                continue
            times = results[method][env_name]
            print(f"  {method:6s} {env_name:10s}: {np.mean(times):.1f} ± {np.std(times):.1f} min ({len(times)} seeds)")

    return results

def write_csv(results, fw_name):
    """Write total training time CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    headers = ["method"] + ENVS

    rows = [headers]
    for method in ['Base', 'CAPS', 'GRAD', 'ASAP', 'PAVE']:
        row = [method]
        for env in ENVS:
            if method in results and env in results[method]:
                times = results[method][env]
                row.append(f"{np.mean(times):.1f}")
            else:
                row.append("")
        rows.append(row)

    path = os.path.join(OUTPUT_DIR, f"wallclock_{fw_name}_total_min.csv")
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved: {path}")

def main():
    td3_results = process_framework(TD3_TB_ROOT, TD3_PREFIX_MAP, "TD3")
    sac_results = process_framework(SAC_TB_ROOT, SAC_PREFIX_MAP, "SAC")

    write_csv(td3_results, "td3")
    write_csv(sac_results, "sac")

if __name__ == "__main__":
    main()
