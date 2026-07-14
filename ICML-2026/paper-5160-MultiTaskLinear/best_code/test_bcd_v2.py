import numpy as np
import time, sys
sys.path.insert(0, "/repo")
from real_data_har import load_raw_minmax, POSITIVE_LABELS, _apply_local_preprocessing_patch, fit_ours_bfgs, evaluate_ours as eval_orig
from preprocessing import split
from fast_ours_v2 import fit_ours_bcd_v2, evaluate_ours

_apply_local_preprocessing_patch()
np.random.seed(42)
data = load_raw_minmax(label_list=POSITIVE_LABELS)
m, d = len(data[0]), data[0][0].shape[1]
prop = 0.2 * np.ones(m)
data_train, data_test = split(data, prop=prop, seed=0)
X_tr = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_train[0]]
X_te = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_test[0]]

print("Testing BCD v2 vs original BFGS (single q=0.05):")
t0 = time.time()
thetas_bfgs = fit_ours_bfgs([X_tr, data_train[1]], q=0.05, maxiter=200)
err_bfgs = eval_orig([X_te, data_test[1]], thetas_bfgs)
print(f"  BFGS (maxiter=200): error={err_bfgs:.4f} time={time.time()-t0:.1f}s")

t0 = time.time()
thetas_bcd = fit_ours_bcd_v2([X_tr, data_train[1]], q=0.05, max_outer=20, max_inner=100)
err_bcd = evaluate_ours([X_te, data_test[1]], thetas_bcd)
print(f"  BCD v2: error={err_bcd:.4f} time={time.time()-t0:.1f}s")

# Grid test
print("\nGrid test (BCD v2):")
for q in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    t0 = time.time()
    thetas = fit_ours_bcd_v2([X_tr, data_train[1]], q=q, max_outer=20, max_inner=100)
    err = evaluate_ours([X_te, data_test[1]], thetas)
    print(f"  q={q:.2f}: error={err:.4f} time={time.time()-t0:.1f}s")

