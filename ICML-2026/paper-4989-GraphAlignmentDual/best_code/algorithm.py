import numpy as np
import torch
from munkres import Munkres
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment


def greedy_match(X):
    X = X.cpu().numpy()
    m, n = X.shape
    minSize = min(m, n)
    usedRows = np.zeros(m, dtype=bool)
    usedCols = np.zeros(n, dtype=bool)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize, dtype=int)
    col = np.zeros(minSize, dtype=int)
    x = X.flatten()
    ix = np.argsort(-x)
    matched = 0
    index = 0
    while matched < minSize:
        ipos = ix[index]
        jc = ipos // m
        ic = ipos % m
        if not usedRows[ic] and not usedCols[jc]:
            row[matched] = ic
            col[matched] = jc
            maxList[matched] = x[ipos]
            usedRows[ic] = True
            usedCols[jc] = True
            matched += 1
        index += 1
    data = np.ones(minSize)
    M = csr_matrix((data, (row, col)), shape=(m, n))
    return M

def get_match(D,device):
    P = torch.zeros_like(D)
    size = D.shape[0]
    index_S = [i for i in range(size)]
    index_S_hat = [i for i in range(size)]
    
    # D = D.clone().detach().cpu()  # avoid modifying computation graph

    for i in range(size):
        cur_size = D.shape[0]
        argmin = torch.argmin(D.to(device)).item()
        r = argmin // cur_size
        c = argmin % cur_size
        P[index_S[r]][index_S_hat[c]] = 1
        index_S.remove(index_S[r])
        index_S_hat.remove(index_S_hat[c])
        D = D[torch.arange(D.size(0)) != r]
        D = D.t()[torch.arange(D.t().size(0)) != c].t()
    return P.t()

def hungarian(D):
    P = torch.zeros_like(D)
    matrix = D.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    for r,c in indexes:
        P[r][c] = 1
        total += matrix[r][c]
    return P.t()




def hungarian_alignment(E, E_hat):
    """
    Perform network alignment using the Hungarian algorithm.
    
    Solves: min_{P in P} ||E - P * E_hat||_F^2
    
    Args:
        E: torch.Tensor of shape (N, F) - embeddings from first graph
        E_hat: torch.Tensor of shape (N, F) - embeddings from second graph
        
    Returns:
        P: torch.Tensor of shape (N, N) - permutation matrix
    """
    # Move to CPU and convert to numpy for scipy
    E_cpu = E.detach().cpu().numpy()
    E_hat_cpu = E_hat.detach().cpu().numpy()
    
    N, F = E_cpu.shape
    
    # Compute cost matrix: C[i,j] = ||E[i,:] - E_hat[j,:]||^2
    # This is equivalent to minimizing ||E - P * E_hat||_F^2
    cost_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            cost_matrix[i, j] = np.sum((E_cpu[i, :] - E_hat_cpu[j, :]) ** 2)
    
    # Solve using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create permutation matrix
    P = np.zeros((N, N))
    P[row_ind, col_ind] = 1
    
    # Convert back to torch tensor
    P_torch = torch.from_numpy(P).float().to(E.device)
        
    return P_torch



def network_alignment_hungarian(E, E_hat):
    """
    Perform network alignment using the Hungarian algorithm.
    
    Solves: min_{P in P} ||E - P * E_hat||_F^2
    
    Args:
        E: torch.Tensor of shape (N, F) - embeddings from first graph
        E_hat: torch.Tensor of shape (N, F) - embeddings from second graph
        
    Returns:
        P: torch.Tensor of shape (N, N) - permutation matrix
    """
    # Move to CPU and convert to numpy for scipy
    E_cpu = E.detach().cpu().numpy()
    E_hat_cpu = E_hat.detach().cpu().numpy()
    
    N, F = E_cpu.shape
    
    # Compute cost matrix more efficiently using broadcasting
    # C[i,j] = ||E[i,:] - E_hat[j,:]||^2
    # Expanding: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    E_sq = np.sum(E_cpu ** 2, axis=1, keepdims=True)  # (N, 1)
    E_hat_sq = np.sum(E_hat_cpu ** 2, axis=1, keepdims=True).T  # (1, N)
    cross_term = 2 * E_cpu @ E_hat_cpu.T  # (N, N)
    
    cost_matrix = E_sq + E_hat_sq - cross_term  # (N, N)
    
    # Solve using Hungarian algorithm (optimal O(N^3) solution)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create permutation matrix
    P = np.zeros((N, N))
    P[row_ind, col_ind] = 1
    
    # Convert back to torch tensor
    P_torch = torch.from_numpy(P).float().to(E.device)
    
    # Calculate final cost (verify our solution)
    aligned_E_hat = torch.matmul(P_torch, E_hat)
    final_cost = torch.norm(E - aligned_E_hat, p='fro') ** 2
    
    return P_torch, final_cost

