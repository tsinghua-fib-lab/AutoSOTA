import numpy as np
import torch
import torch.nn.functional as F
import scanpy as sc
from scipy import sparse


def initialize(n_factors, n_cells, n_genes, device):
    factors = torch.rand(n_genes, n_factors, device=device, requires_grad=True)
    embeddings = torch.rand(n_factors, n_cells, device=device, requires_grad=True)

    return factors, embeddings

def normalize(x: torch.Tensor, axis=0) -> torch.Tensor:
    return torch.softmax(x, dim=axis)
 
def get_knn(dist_matrix, k=15):
    """
    Input : dist_matrix (m, m) fisher_rao_dis_matrix
    Output: indices (m, k) indices of neighbors
            distances (m, k) distance of neighbors
    """
    # 1. find top minor k+1 values use topk
    # largest=False indicate find mini values
    # k+1 Exclude the diagonal elements
    m = dist_matrix.shape[0]
    vals, inds = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
    
    # 2. Exclude the diagonal elements
    knn_inds = inds[:, 1:].detach().cpu().numpy()
    knn_dists = vals[:, 1:].detach().cpu().numpy()

    row_indices = np.repeat(np.arange(m), k)
    col_indices = knn_inds.flatten()
    data_dists = knn_dists.flatten()
    
    # 4. conduct distances sparse matrix (unsymmetry)
    distances_matrix = sparse.csr_matrix(
        (data_dists, (row_indices, col_indices)), 
        shape=(m, m)
    )

    # --- conduct connectivities matrix ---
    # 5. connectivity matrix (Binary)
    data_conn = np.ones(m * k)
    A = sparse.csr_matrix(
        (data_conn, (row_indices, col_indices)), 
        shape=(m, m)
    )
    
    # # 6. Symmetrization
    # connectivities_matrix = A + A.transpose()
    A_coo = A.tocoo()
    AT_coo = A.transpose().tocoo()

    # 找到 A 和 A^T 的最大值 (Max)，然后转回 CSR 格式
    connectivities_matrix = sparse.coo_matrix((A_coo.data, (A_coo.row, A_coo.col)), shape=(m, m))
    connectivities_matrix.data = np.maximum(connectivities_matrix.data, AT_coo.data)
    connectivities_matrix = connectivities_matrix.tocsr()



    return distances_matrix, connectivities_matrix