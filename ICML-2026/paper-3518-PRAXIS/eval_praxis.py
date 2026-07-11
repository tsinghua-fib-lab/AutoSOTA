#!/usr/bin/env python3
"""PRAXIS eval for Compas Rashomon set.

Reproduces paper metrics for Compas benchmark:
  lambda=0.02, epsilon_mult=0.03, depth=5
  proxy=modified_LicketySPLIT (lookahead_k=1)
  cache=fingerprint_64bit (key_mode="hash")

Outputs parseable metrics on stdout.
"""

import time
import resource
import numpy as np
import pandas as pd
from praxis import PRAXIS

CSV_PATH = "examples/compas_binarized.csv"
LAMBDA_REG = 0.02
DEPTH_BUDGET = 5
RASHOMON_MULT = 0.03
LOOKAHEAD_K = 1
KEY_MODE = "hash"

# Optimization parameters
CACHE_EARLY_EXITS = True   # IDEA-09
TRIE_CACHE_ENABLED = True  # IDEA-13


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    peak_start = peak_rss_mb()

    # Load data
    df = pd.read_csv(CSV_PATH)
    X = df.iloc[:, :-1].to_numpy(dtype=np.uint8)
    y = df.iloc[:, -1].to_numpy(dtype=np.int32)

    # Run PRAXIS
    model = PRAXIS()
    t0 = time.perf_counter()
    model.fit(
        X, y,
        lambda_reg=LAMBDA_REG,
        depth_budget=DEPTH_BUDGET,
        rashomon_mult=RASHOMON_MULT,
        key_mode=KEY_MODE,
        lookahead_k=LOOKAHEAD_K,
        proxy_style=0,
        proxy_caching=True,
        cache_early_exits=CACHE_EARLY_EXITS,
        trie_cache_enabled=TRIE_CACHE_ENABLED,
    )
    t1 = time.perf_counter()
    peak_end = peak_rss_mb()

    elapsed = t1 - t0
    n_trees = model.count_trees()
    min_obj = model.get_min_objective()

    # Compute recall via exact reference
    model_exact = PRAXIS()
    model_exact.fit(
        X, y,
        lambda_reg=LAMBDA_REG,
        depth_budget=DEPTH_BUDGET,
        rashomon_mult=RASHOMON_MULT,
        key_mode=KEY_MODE,
        lookahead_k=DEPTH_BUDGET - 1,
        proxy_style=0,
        proxy_caching=True,
        cache_early_exits=CACHE_EARLY_EXITS,
        trie_cache_enabled=TRIE_CACHE_ENABLED,
    )
    n_exact = model_exact.count_trees()
    recall = n_trees / n_exact if n_exact > 0 else 1.0
    peak_final = peak_rss_mb()

    # Print metrics
    print("=== PRAXIS METRICS ===")
    print(f"recall: {recall:.4f}")
    print(f"time_s: {elapsed:.6f}")
    print(f"peak_mb: {peak_final:.2f}")
    print(f"n_trees: {n_trees}")
    print(f"n_trees_exact: {n_exact}")
    print(f"min_objective: {min_obj}")


if __name__ == "__main__":
    main()
