#!/usr/bin/env python3
"""Evaluate Jigsaw TF-IDF baseline across 10 seeds and compute aggregate NEC and Error Rate."""
import subprocess, sys
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats

SEEDS = [0, 1, 7, 13, 42, 99, 123, 314, 456, 2024]
RESULTS_DIR = Path("results/jigsaw")

def run_seed(seed):
    f = RESULTS_DIR / f"tfidf_classification_none_s{seed}.csv"
    if f.exists():
        print(f"Seed {seed}: result exists, skipping")
        return True
    cmd = ["python", "-m", "src.runners.run_experiment",
           "--dataset", "jigsaw", "--model", "tfidf",
           "--method", "classification", "--weighting", "none",
           "--seed", str(seed)]
    print(f"Seed {seed}: running...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Seed {seed}: FAILED\n{r.stderr[-500:]}")
        return False
    return True

def compute_aggregate():
    rows = []
    for seed in SEEDS:
        f = RESULTS_DIR / f"tfidf_classification_none_s{seed}.csv"
        if not f.exists():
            print(f"MISSING: seed {seed}")
            continue
        df = pd.read_csv(f)
        row = df.iloc[0]
        err = (1 - row['test_accuracy']) * 100
        nec = (1 - row['test_weighted_accuracy']) * 100
        rows.append({'seed': seed, 'error_rate': err, 'nec': nec})

    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        print("No results found!")
        return

    print(f"\n{'='*60}")
    print(f"Jigsaw TF-IDF Standard CE Baseline (n={n} seeds)")
    print(f"{'='*60}")
    for _, r in df.iterrows():
        print(f"  Seed {int(r.seed):4d}: Error={r.error_rate:.4f}%  NEC={r.nec:.4f}%")

    for metric, paper_val in [('nec', 1.76), ('error_rate', 5.34)]:
        vals = df[metric]
        mean = vals.mean()
        std = vals.std(ddof=1)
        se = std / np.sqrt(n)
        t_val = stats.t.ppf(0.975, n - 1)
        ci_low = mean - t_val * se
        ci_high = mean + t_val * se
        name = "NEC" if metric == 'nec' else "Error Rate"
        print(f"\n{name}: {mean:.2f}%  (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
        print(f"  Paper: {paper_val}%")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', help='Run experiments (default: only aggregate)')
    args = ap.parse_args()

    if args.run:
        for seed in SEEDS:
            if not run_seed(seed):
                sys.exit(1)
    compute_aggregate()
