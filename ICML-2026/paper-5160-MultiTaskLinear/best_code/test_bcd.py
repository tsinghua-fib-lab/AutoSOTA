"""Test BCD vs original BFGS approach."""
import numpy as np
import time
from pathlib import Path
import sys
sys.path.insert(0, "/repo")

from real_data_har import (
    load_raw_minmax, POSITIVE_LABELS,
    fit_ours_bfgs, evaluate_ours as evaluate_ours_orig,
    _apply_local_preprocessing_patch
)
from ARMUL import Baselines
from preprocessing import split
from fast_ours import fit_ours_bcd, evaluate_ours

_apply_local_preprocessing_patch()

np.random.seed(42)
data = load_raw_minmax(label_list=POSITIVE_LABELS)
m = len(data[0])
d = data[0][0].shape[1]

# Single split
prop = 0.2 * np.ones(m)
data_train, data_test = split(data, prop=prop, seed=0)

# Add intercept
X_train_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_train[0]]
X_test_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_test[0]]

# Test DP and ITL baselines
baseline = Baselines(link="logistic", n_class=2)
baseline.DP_train(data_train, eta=0.1, T=200, standardization=False, intercept=True)
dp_res = baseline.evaluate(data_test, model="DP")
print(f"DP: {dp_res['average error']:.4f}")

baseline.STL_train(data_train, eta=0.1, T=200, standardization=False, intercept=True)
itl_res = baseline.evaluate(data_test, model="STL")
print(f"ITL: {itl_res['average error']:.4f}")

# Test BCD with different q values
for q in [0.05, 0.10, 0.20, 0.30, 0.50]:
    t0 = time.time()
    thetas_bcd = fit_ours_bcd([X_train_int, data_train[1]], q=q, maxiter=30, inner_maxiter=100)
    err_bcd = evaluate_ours([X_test_int, data_test[1]], thetas_bcd)
    elapsed = time.time() - t0
    print(f"q={q:.2f} BCD error={err_bcd:.4f} time={elapsed:.1f}s")

# Test BFGS for comparison (with reduced maxiter)
print("\nComparing with BFGS (reduced maxiter)...")
t0 = time.time()
thetas_bfgs = fit_ours_bfgs([X_train_int, data_train[1]], q=0.05, maxiter=50)
err_bfgs = evaluate_ours_orig([X_test_int, data_test[1]], thetas_bfgs)
print(f"q=0.05 BFGS error={err_bfgs:.4f} time={time.time()-t0:.1f}s")

