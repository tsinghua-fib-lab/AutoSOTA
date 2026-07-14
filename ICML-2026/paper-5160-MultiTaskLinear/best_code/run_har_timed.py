"""HAR experiment with per-split timing and progress."""
from __future__ import annotations
import time, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import MTL
import preprocessing
from ARMUL import ARMUL, Baselines
from path_setup import find_code_dir, find_har_dataset

# Apply the real_data_har patches
from real_data_har import (
    safe_mtl_preprocessing, _ORIGINAL_MTL_PREPROCESSING,
    _apply_local_preprocessing_patch,
    load_raw_minmax, nll_and_grad_theta, reg_sqrt_value_and_grad,
    fit_ours_bfgs, evaluate_ours, run_cv_selection, run_experiment,
    Q_GRID, N_SPLITS, N_FOLD, ETA, MAXITER, POSITIVE_LABELS
)

def run_with_progress():
    path = find_code_dir(mount_drive=False)
    data = load_raw_minmax(path=path, label_list=POSITIVE_LABELS)
    m = len(data[0])
    d = data[0][0].shape[1]
    print(f"Loaded HAR data with m={m} tasks and d={d} raw features.", flush=True)

    results = []
    total_start = time.time()
    print(f"Starting experiment: {N_SPLITS} random splits with {N_FOLD}-fold CV.", flush=True)

    for split_idx in range(N_SPLITS):
        split_start = time.time()
        seed = split_idx * 100
        prop = 0.2 * np.ones(m)
        data_train, data_test = preprocessing.split(data, prop=prop, seed=seed)

        # DP
        t0 = time.time()
        baseline = Baselines(link="logistic", n_class=2)
        baseline.DP_train(data_train, eta=ETA, T=MAXITER, standardization=False, intercept=True)
        dp_res = baseline.evaluate(data_test, model="DP")
        dp_time = time.time() - t0
        print(f"  [{split_idx}] DP: {dp_res['average error']:.4f} ({dp_time:.1f}s)", flush=True)

        # ITL
        t0 = time.time()
        baseline.STL_train(data_train, eta=ETA, T=MAXITER, standardization=False, intercept=True)
        itl_res = baseline.evaluate(data_test, model="STL")
        itl_time = time.time() - t0
        print(f"  [{split_idx}] ITL: {itl_res['average error']:.4f} ({itl_time:.1f}s)", flush=True)

        # ARMUL CV
        t0 = time.time()
        armul_model, best_q_armul = run_cv_selection(
            data_train, "ARMUL", Q_GRID, d, n_fold=N_FOLD, eta=ETA, maxiter=MAXITER)
        armul_res = armul_model.evaluate(data_test, model="vanilla")
        armul_time = time.time() - t0
        print(f"  [{split_idx}] ARMUL: {armul_res['average error']:.4f} q={best_q_armul} ({armul_time:.1f}s)", flush=True)

        # OURS CV
        t0 = time.time()
        ours_thetas, best_q_ours = run_cv_selection(
            data_train, "OURS", Q_GRID, d, n_fold=N_FOLD, eta=ETA, maxiter=MAXITER)
        X_test_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_test[0]]
        ours_err = evaluate_ours([X_test_int, data_test[1]], ours_thetas)
        ours_time = time.time() - t0
        print(f"  [{split_idx}] OURS: {ours_err:.4f} q={best_q_ours} ({ours_time:.1f}s)", flush=True)

        results.append({
            "split": split_idx,
            "DP": dp_res["average error"],
            "ITL": itl_res["average error"],
            "ARMUL": armul_res["average error"],
            "q_ARMUL": best_q_armul,
            "OURS": ours_err,
            "q_OURS": best_q_ours,
        })

        elapsed = time.time() - split_start
        total_elapsed = time.time() - total_start
        eta = (total_elapsed / (split_idx + 1)) * (N_SPLITS - split_idx - 1)
        print(f"  --- Split {split_idx} done in {elapsed:.1f}s, ETA {eta/60:.1f}min remaining ---", flush=True)

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Held-out Test Error)")
    print("=" * 60)
    for method in ["DP", "ITL", "ARMUL", "OURS"]:
        print(f"{method:<10} | mean={df[method].mean():.4f} | std={df[method].std():.4f}")
    print("=" * 60)

    total_time = time.time() - total_start
    print(f"Total time: {total_time/60:.1f} min")
    df.to_csv("/repo/har_results.csv", index=False)
    return df

if __name__ == "__main__":
    run_with_progress()
