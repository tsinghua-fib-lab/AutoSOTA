import numpy as np
from real_data_har import (
    load_raw_minmax, run_cv_selection, evaluate_ours,
    N_SPLITS, N_FOLD, Q_GRID, ETA, MAXITER, POSITIVE_LABELS
)
from ARMUL import Baselines
from preprocessing import split

np.random.seed(42)

data = load_raw_minmax()
m = len(data[0])
d = data[0][0].shape[1]
print(f"Loaded HAR: m={m} tasks, d={d} features")

# Single split test
prop = 0.2 * np.ones(m)
data_train, data_test = split(data, prop=prop, seed=0)

# Test DP
baseline = Baselines(link="logistic", n_class=2)
baseline.DP_train(data_train, eta=ETA, T=MAXITER, standardization=False, intercept=True)
dp_res = baseline.evaluate(data_test, model="DP")
print(f"DP error: {dp_res['average error']:.4f}")

# Test ITL
baseline.STL_train(data_train, eta=ETA, T=MAXITER, standardization=False, intercept=True)
itl_res = baseline.evaluate(data_test, model="STL")
print(f"ITL error: {itl_res['average error']:.4f}")

# Test ARMUL (with CV)
print("Running ARMUL CV...")
armul_model, best_q_armul = run_cv_selection(data_train, "ARMUL", Q_GRID, d, n_fold=N_FOLD, eta=ETA, maxiter=MAXITER)
armul_res = armul_model.evaluate(data_test, model="vanilla")
print(f"ARMUL error: {armul_res['average error']:.4f} (best q={best_q_armul})")

# Test OURS (with CV)
print("Running OURS CV...")
ours_thetas, best_q_ours = run_cv_selection(data_train, "OURS", Q_GRID, d, n_fold=N_FOLD, eta=ETA, maxiter=MAXITER)
X_test_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_test[0]]
ours_err = evaluate_ours([X_test_int, data_test[1]], ours_thetas)
print(f"OURS error: {ours_err:.4f} (best q={best_q_ours})")

print("Single split test PASSED!")
