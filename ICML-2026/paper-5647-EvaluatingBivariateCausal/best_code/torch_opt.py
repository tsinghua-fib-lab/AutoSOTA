import sys; sys.path.insert(0, "/repo")
import numpy as np
import torch
from experiments_llm_linear import compute_correlation_matrix, VARIABLES
from synthetic_experiments_linear import compatibility_score
from sensitivity_experiment import _estimate_bivariate_matrix

corr_df = compute_correlation_matrix()
corr = corr_df.values
n = len(VARIABLES)

# Convert corr to torch
corr_t = torch.tensor(corr, dtype=torch.float64)

def torch_compatibility_score(A, cov):
    """PyTorch implementation of compatibility_score."""
    n = cov.shape[0]
    # bivariate score
    bs = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diff = cov[i,j] - A[j,i] * cov[i,i]
            bs += diff * diff
    
    # multivariate score - need C = I - inv(A) and backdoor paths
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    C = I - torch.linalg.inv(A)
    
    ms = 0.0
    for i in range(n):
        for j in range(i+1, n):
            conf = cov[i,j] - A[j,i] * cov[i,i]
            # backdoor paths from k to i and j
            for k in range(i):
                conf -= backdoor_torch(cov, C, k, i, j)
            ms += conf * conf
    
    return bs - ms

def backdoor_torch(cov, C, k, i, j):
    """PyTorch backdoor paths computation."""
    n = cov.shape[0]
    # Vertices between k and i, and between k and j
    verts_i = list(range(k+1, i))
    verts_j = [v for v in range(k+1, j) if v != i]
    
    from itertools import combinations
    
    total = 0.0
    for size_i in range(len(verts_i) + 1):
        for subset_i in combinations(verts_i, size_i):
            prod_i = 1.0
            path_i = [k] + sorted(subset_i) + [i]
            for idx in range(len(path_i) - 1):
                prod_i *= C[path_i[idx+1], path_i[idx]]
            
            for size_j in range(len(verts_j) + 1):
                for subset_j in combinations(verts_j, size_j):
                    # Check disjointness
                    if set(subset_i) & set(subset_j):
                        continue
                    prod_j = 1.0
                    path_j = [k] + sorted(subset_j) + [j]
                    for idx in range(len(path_j) - 1):
                        prod_j *= C[path_j[idx+1], path_j[idx]]
                    
                    total += prod_i * prod_j
    
    return total * cov[k, k]

# Test that our torch version matches numpy
A_np = _estimate_bivariate_matrix(corr)
score_np = compatibility_score(A_np, corr)
A_t = torch.tensor(A_np, dtype=torch.float64)
score_torch = torch_compatibility_score(A_t, corr_t)
print(f"Numpy score: {score_np:.6f}")
print(f"Torch score: {score_torch.item():.6f}")
print(f"Match: {abs(score_np - score_torch.item()) < 1e-6}")

# Now gradient-based optimization
print("\nGradient-based optimization...")
A_param = torch.tensor(A_np, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([A_param], lr=0.01)
coef_bound = 5.0

best_score = -float("inf")
best_A = None
scores_history = []

for iteration in range(5000):
    optimizer.zero_grad()
    
    # Constrain lower triangle to be within bounds (via projection)
    with torch.no_grad():
        A_param.data = A_param.data.clamp(-coef_bound, coef_bound)
        # Keep diagonal = 1 and upper triangle = 0
        for i in range(n):
            A_param.data[i, i] = 1.0
            for j in range(i+1, n):
                A_param.data[i, j] = 0.0
    
    score = torch_compatibility_score(A_param, corr_t)
    loss = -score  # maximize score = minimize -score
    loss.backward()
    
    # Zero out gradients for diagonal (fixed) and upper triangle (fixed)
    with torch.no_grad():
        for i in range(n):
            A_param.grad[i, i] = 0.0
            for j in range(i+1, n):
                A_param.grad[i, j] = 0.0
    
    optimizer.step()
    
    with torch.no_grad():
        current_score = score.item()
        scores_history.append(current_score)
        if current_score > best_score:
            best_score = current_score
            best_A = A_param.detach().clone().numpy()
    
    if iteration % 500 == 0:
        print(f"  iter {iteration}: score={current_score:.4f}")

print(f"\nBest score: {best_score:.6f}")
print("Best A lower tri:")
print(np.array2string(np.tril(best_A, -1), precision=3, suppress_small=True))
print(f"A range: [{np.min(np.tril(best_A,-1)):.2f}, {np.max(np.tril(best_A,-1)):.2f}]")
