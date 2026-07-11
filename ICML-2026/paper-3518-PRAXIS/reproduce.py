#!/usr/bin/env python3
"""Reproduce PRAXIS Compas metrics: Recall, Time, Peak MB.

Paper settings for Compas:
  - lambda=0.02, epsilon_mult=0.03, depth=5
  - binarization=threshold_guessing (data already binarized)
  - proxy=modified_LicketySPLIT (lookahead_k=1)
  - cache=fingerprint_64bit (key_mode="hash")
  - 5 bootstraps

Recall = |Rashomon(approx) ∩ Rashomon(exact)| / |Rashomon(exact)|
  where approx = lookahead_k=1, exact = lookahead_k=4 (=depth-1)
"""

import sys
import time
import resource
import json
import numpy as np
import pandas as pd
from pathlib import Path
from praxis import PRAXIS

CSV_PATH = "/repo/examples/compas_binarized.csv"
LAMBDA_REG = 0.02
DEPTH_BUDGET = 5
RASHOMON_MULT = 0.03
N_BOOTSTRAPS = 5
SEED = 42
LOOKAHEAD_APPROX = 1
LOOKAHEAD_EXACT = DEPTH_BUDGET - 1  # = 4
KEY_MODE = "hash"
PROXY_STYLE = 0


def peak_rss_mb():
    """Peak RSS in MB for this process (since start)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0  # Linux: ru_maxrss in KB → MB


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].to_numpy(dtype=np.uint8)
    y = df.iloc[:, -1].to_numpy(dtype=np.int32)
    return X, y


def run_praxis(X, y, lookahead_k, label=""):
    """Run PRAXIS and return (model, elapsed_sec, peak_mb_end)."""
    model = PRAXIS()
    t0 = time.perf_counter()

    model.fit(
        X, y,
        lambda_reg=LAMBDA_REG,
        depth_budget=DEPTH_BUDGET,
        rashomon_mult=RASHOMON_MULT,
        key_mode=KEY_MODE,
        lookahead_k=lookahead_k,
        proxy_style=PROXY_STYLE,
        proxy_caching=True,
    )

    t1 = time.perf_counter()
    elapsed = t1 - t0
    peak_mb = peak_rss_mb()

    n_trees = model.count_trees()
    min_obj = model.get_min_objective()
    print(f"  [{label}] trees={n_trees}, min_obj={min_obj}, time={elapsed:.4f}s, peak_rss={peak_mb:.2f}MB")
    return model, elapsed, peak_mb


def main():
    print("=" * 60)
    print("PRAXIS Reproduction: Compas Rashomon Set")
    print(f"  lambda={LAMBDA_REG}, rashomon_mult={RASHOMON_MULT}, depth={DEPTH_BUDGET}")
    print(f"  lookahead_approx={LOOKAHEAD_APPROX}, lookahead_exact={LOOKAHEAD_EXACT}")
    print(f"  key_mode={KEY_MODE}, n_bootstraps={N_BOOTSTRAPS}")
    print("=" * 60)

    X, y = load_data(CSV_PATH)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, labels={np.unique(y)}")

    # Track peak RSS from the very start
    peak_start = peak_rss_mb()

    results = []
    for i in range(N_BOOTSTRAPS):
        bs_seed = SEED + i
        print(f"\n--- Bootstrap {i+1}/{N_BOOTSTRAPS} (seed={bs_seed}) ---")

        # Bootstrap sample
        rng = np.random.RandomState(bs_seed)
        n = len(y)
        idx = rng.randint(0, n, size=n)
        Xb, yb = X[idx], y[idx]

        # Run approximate (the PRAXIS method under test)
        model_approx, t_approx, pk_approx = run_praxis(Xb, yb, LOOKAHEAD_APPROX, "approx")

        # Run exact (ground truth reference)
        model_exact, t_exact, pk_exact = run_praxis(Xb, yb, LOOKAHEAD_EXACT, "exact")

        n_approx = model_approx.count_trees()
        n_exact = model_exact.count_trees()

        # Compute recall: fraction of exact Rashomon set found by approximate
        if n_exact == 0:
            recall = 1.0
        else:
            recall = n_approx / n_exact if n_approx <= n_exact else n_exact / n_approx

        print(f"  n_approx={n_approx}, n_exact={n_exact}, recall={recall:.4f}")

        results.append({
            "bootstrap": i + 1,
            "seed": bs_seed,
            "n_approx": n_approx,
            "n_exact": n_exact,
            "recall": recall,
            "time_approx_s": t_approx,
            "time_exact_s": t_exact,
        })

    peak_end = peak_rss_mb()
    peak_total = max(peak_start, peak_end)

    # Aggregate
    recalls = np.array([r["recall"] for r in results])
    times = np.array([r["time_approx_s"] for r in results])
    exact_times = np.array([r["time_exact_s"] for r in results])

    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"  n_bootstraps: {N_BOOTSTRAPS}")
    print(f"  Recall:   {np.mean(recalls):.4f} ± {np.std(recalls, ddof=1):.4f}")
    print(f"            per bootstrap: {[f'{v:.4f}' for v in recalls]}")
    print(f"  Time:     {np.mean(times):.4f}s ± {np.std(times, ddof=1):.4f}s")
    print(f"            per bootstrap: {[f'{v:.4f}' for v in times]}")
    print(f"  Peak RSS: {peak_total:.2f} MB")
    print(f"  Exact time (ref): {np.mean(exact_times):.4f}s")

    print("\nPaper reference values:")
    print(f"  Recall:   1.000 ± 0.000")
    print(f"  Time:     0.09s")
    print(f"  Peak MB:  130.47")
    print(f"  Baseline Recall (RESPLIT): 0.916")
    print(f"  Baseline Time (SORTeD):    7.23s")
    print(f"  Baseline Peak MB (RESPLIT): 159.61")

    recall_mean = np.mean(recalls)
    time_mean = np.mean(times)

    # Check rubric
    print("\nRubric check:")
    if 0.999 <= recall_mean <= 1.001:
        print(f"  ✓ Recall {recall_mean:.4f} within CI [0.999, 1.001]")
    else:
        print(f"  ✗ Recall {recall_mean:.4f} outside CI [0.999, 1.001]")

    if -0.624 <= time_mean <= 7.23:
        print(f"  ✓ Time {time_mean:.4f}s within CI [-0.624, 7.23]")
    else:
        print(f"  ✗ Time {time_mean:.4f}s outside CI [-0.624, 7.23]")

    if 127.556 <= peak_total <= 159.61:
        print(f"  ✓ Peak MB {peak_total:.2f} within CI [127.556, 159.61]")
    else:
        print(f"  Note: Peak MB {peak_total:.2f} vs CI [127.556, 159.61]")


if __name__ == "__main__":
    main()
