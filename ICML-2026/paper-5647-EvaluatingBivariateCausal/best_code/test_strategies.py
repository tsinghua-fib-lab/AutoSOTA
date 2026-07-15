import sys
sys.path.insert(0, "/repo")
from sensitivity_experiment import _estimate_bivariate_matrix
from experiments_llm_linear import compute_correlation_matrix, VARIABLES
from synthetic_experiments_linear import compatibility_score
import numpy as np

corr_df = compute_correlation_matrix()
corr = corr_df.values
n = len(VARIABLES)
empirical_std = float(np.std(corr[np.tril_indices(n, k=-1)]))
rng = np.random.RandomState(42)

# 1. Estimated A score
A_est = _estimate_bivariate_matrix(corr)
score_est = compatibility_score(A_est, corr)
print("1. Estimated A (bivariate regression):", score_est)

# 2. Sign-constrained random
scores_sign = []
for _ in range(5000):
    A = np.eye(n)
    for i in range(n):
        for j in range(i):
            A[i, j] = np.sign(corr[i, j]) * abs(rng.normal(0.0, empirical_std))
    scores_sign.append(compatibility_score(A, corr))
print("2. Sign-constrained random (5000): mean={:.4f} max={:.4f}".format(np.mean(scores_sign), np.max(scores_sign)))

# 3. A = corr lower triangle directly
A_corr = np.eye(n)
for i in range(n):
    for j in range(i):
        A_corr[i, j] = corr[i, j]
score_corr_direct = compatibility_score(A_corr, corr)
print("3. A = corr_lower:", score_corr_direct)

# 4. Best of many random samples
best = -float("inf")
for _ in range(50000):
    A = np.eye(n)
    for i in range(n):
        for j in range(i):
            A[i, j] = rng.normal(0.0, empirical_std)
    s = compatibility_score(A, corr)
    if s > best:
        best = s
print("4. Best of 50000 random:", best)

# 5. Estimated A + noise sweep
print("5. Estimated A + noise:")
for sigma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
    s_list = []
    for _ in range(100):
        A_pert = A_est + sigma * np.tril(rng.randn(n, n), -1)
        s_list.append(compatibility_score(A_pert, corr))
    print("   sigma={:.3f}: mean={:.4f} max={:.4f} min={:.4f}".format(sigma, np.mean(s_list), np.max(s_list), np.min(s_list)))
