"""Test different CV maxiter settings to verify q selection is consistent."""
import sys, time, numpy as np
sys.path.insert(0, "/repo")
from real_data_har import load_raw_minmax, POSITIVE_LABELS, _apply_local_preprocessing_patch
from preprocessing import split
from run_har_optimized import run_cv_selection, Q_GRID, N_FOLD, ETA
_apply_local_preprocessing_patch()

np.random.seed(42)
data = load_raw_minmax(label_list=POSITIVE_LABELS)
m, d = len(data[0]), data[0][0].shape[1]

prop = 0.2 * np.ones(m)
data_train, data_test = split(data, prop=prop, seed=0)

# Test different CV maxiter settings
for cv_iter in [10, 20, 30, 50]:
    t0 = time.time()
    _, best_q = run_cv_selection(
        data_train, "OURS", Q_GRID, d, n_fold=N_FOLD, eta=ETA,
        maxiter_cv=cv_iter, maxiter_full=200)
    elapsed = time.time() - t0
    print(f"CV maxiter={cv_iter:3d}: best_q={best_q:.2f}, time={elapsed:.1f}s")

print("\nTesting ARMUL CV speed:")
for cv_iter in [10, 20, 50]:
    t0 = time.time()
    _, best_q = run_cv_selection(
        data_train, "ARMUL", Q_GRID, d, n_fold=N_FOLD, eta=ETA,
        maxiter_cv=cv_iter, maxiter_full=200)
    elapsed = time.time() - t0
    print(f"CV maxiter={cv_iter:3d}: best_q={best_q:.2f}, time={elapsed:.1f}s")
