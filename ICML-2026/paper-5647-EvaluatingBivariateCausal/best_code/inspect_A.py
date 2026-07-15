import sys; sys.path.insert(0, "/repo")
import numpy as np
from experiments_llm_linear import compute_correlation_matrix, VARIABLES
from synthetic_experiments_linear import compatibility_score
from sensitivity_experiment import _estimate_bivariate_matrix

corr_df = compute_correlation_matrix()
corr = corr_df.values
n = len(VARIABLES)
A_est = _estimate_bivariate_matrix(corr)

# Quick hill-climb
rng = np.random.RandomState(42)
current_A = A_est + 0.01 * np.tril(rng.randn(n, n), -1)
current_score = compatibility_score(current_A, corr)
best_score = current_score
best_A = current_A.copy()

for it in range(5000):
    candidate_A = current_A.copy()
    for i in range(n):
        for j in range(i):
            candidate_A[i, j] += rng.normal(0.0, 0.05)
    try:
        candidate_score = compatibility_score(candidate_A, corr)
        if candidate_score > current_score:
            current_score = candidate_score
            current_A = candidate_A.copy()
            if candidate_score > best_score:
                best_score = candidate_score
                best_A = candidate_A.copy()
    except:
        continue

print("Best score:", best_score)
print("Best A lower tri:")
print(np.array2string(np.tril(best_A, -1), precision=4, suppress_small=True))
print("Estimated A lower tri:")
print(np.array2string(np.tril(A_est, -1), precision=4, suppress_small=True))
print("A range:", np.min(np.tril(best_A,-1)), "to", np.max(np.tril(best_A,-1)))
print("A_est range:", np.min(np.tril(A_est,-1)), "to", np.max(np.tril(A_est,-1)))
evals = np.linalg.eigvals(best_A)
print("Eigenvalues min abs:", np.min(np.abs(evals)))
print("Condition number:", np.max(np.abs(evals))/np.min(np.abs(evals)))
