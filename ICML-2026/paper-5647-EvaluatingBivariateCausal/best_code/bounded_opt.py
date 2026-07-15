import sys; sys.path.insert(0, "/repo")
import numpy as np
from experiments_llm_linear import compute_correlation_matrix, VARIABLES
from synthetic_experiments_linear import compatibility_score
from sensitivity_experiment import _estimate_bivariate_matrix

corr_df = compute_correlation_matrix()
corr = corr_df.values
n = len(VARIABLES)
A_est = _estimate_bivariate_matrix(corr)
base_score = compatibility_score(A_est, corr)
print(f"Base estimated A score: {base_score:.6f}")
print(f"Estimated A coef range: [{np.min(np.tril(A_est,-1)):.3f}, {np.max(np.tril(A_est,-1)):.3f}]")

# Test different coefficient bounds for meaningful optimization
for bound in [2.0, 3.0, 4.0, 5.0]:
    rng = np.random.RandomState(42)
    best_score = -float("inf")
    current_A = A_est.copy()
    current_score = base_score
    for it in range(10000):
        candidate_A = current_A.copy()
        for i in range(n):
            for j in range(i):
                delta = rng.normal(0.0, 0.05)
                candidate_A[i, j] = np.clip(candidate_A[i, j] + delta, -bound, bound)
        try:
            cs = compatibility_score(candidate_A, corr)
            if cs > current_score:
                current_score = cs
                current_A = candidate_A.copy()
            if cs > best_score:
                best_score = cs
        except:
            continue
    coef_range = f"[{np.min(np.tril(current_A,-1)):.2f}, {np.max(np.tril(current_A,-1)):.2f}]"
    print(f"Bound={bound}: best={best_score:.4f}, coef_range={coef_range}")

# Now try L2 regularization approach with Adam-like step
print("\nRegularized hill-climbing (penalizing large coefficients):")
for reg in [0.01, 0.05, 0.1, 0.5]:
    rng = np.random.RandomState(42)
    current_A = A_est.copy()
    current_score = compatibility_score(current_A, corr) - reg * np.sum(np.tril(current_A, -1)**2)
    best_reg_score = current_score
    best_raw_score = base_score
    for it in range(10000):
        candidate_A = current_A.copy()
        for i in range(n):
            for j in range(i):
                delta = rng.normal(0.0, 0.05)
                candidate_A[i, j] += delta
        try:
            raw = compatibility_score(candidate_A, corr)
            reg_pen = reg * np.sum(np.tril(candidate_A, -1)**2)
            cs = raw - reg_pen
            if cs > current_score:
                current_score = cs
                current_A = candidate_A.copy()
            if raw > best_raw_score:
                best_raw_score = raw
        except:
            continue
    coef_range = f"[{np.min(np.tril(current_A,-1)):.2f}, {np.max(np.tril(current_A,-1)):.2f}]"
    print(f"reg={reg}: best_reg={current_score:.4f}, best_raw={best_raw_score:.4f}, coef_range={coef_range}")
