import numpy as np
import torch

def topk_variance(matrix, k):
    k = min(k, matrix.shape[1])
    
    variances = np.var(matrix, axis=0, ddof=1)  
    top_k_indices = np.argsort(variances)[-k:][::-1]

    return top_k_indices


def topk_entropy(matrix, k):
    k = min(k, matrix.shape[1])
    
    entropy = [np.sum(matrix[i,:]*np.log(matrix[i,:])) for i in range(matrix.shape[0])]
    top_k_indices = np.argsort(entropy)[-k:][::-1]

    return top_k_indices

def global_variance_vary(X: torch.Tensor):
    N, M = X.shape
    var_before = X.var(unbiased=False)

    deltas = torch.empty(M, device=X.device, dtype=X.dtype)

    for j in range(M):
        mask = torch.ones(M, dtype=torch.bool, device=X.device)
        mask[j] = False
        var_after = X[:, mask].var(unbiased=False)
        deltas[j] = var_after - var_before

    return deltas

def greedy_var(X, k=None):
    N, M = X.shape
    if k is None:
        k = M

    selected = []
    remaining = list(range(M))
    total_var = []

    for _ in range(k):
        best_var = -np.inf
        best_feat = None

        for j in remaining:
            candidate = selected + [j]
            current_var = X[:, candidate].var(axis=0).sum()  # 总方差
            if current_var > best_var:
                best_var = current_var
                best_feat = j

        selected.append(best_feat)
        remaining.remove(best_feat)
        total_var.append(best_var)

    return selected, total_var