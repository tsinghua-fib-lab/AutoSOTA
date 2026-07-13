import scipy
from algorithm import *
import random
from scipy.io import loadmat
try:
    from netrd.distance import netsimile
except ImportError:
    pass
import os.path as osp
from torch_geometric.utils import add_self_loops

from scipy.sparse.linalg import eigsh
from geomloss import SamplesLoss
from scipy.sparse import csgraph
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
import networkx as nx
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import expm  

def adj_to_edge_index(adj):
    # Ensure it's a NumPy array
    if torch.is_tensor(adj):
        adj = adj.cpu().numpy()
    edge_index = np.array(adj.nonzero())
    return torch.tensor(edge_index, dtype=torch.long)

def compute_edge_indices(A, num_scales):
    edge_index_list = []
    for scale in range(num_scales):
        if scale == 0:
            edge_index_list.append(A)  # 1-hop neighbors (original edges)
        else:
            # Compute neighbors at higher scales (e.g., 2-hop, 3-hop)
            edge_index_list.append(A @ A)  # Example: A^2 for 2-hop, A^3 for 3-hop, etc.
    return edge_index_list


def scipy_sparse_to_torch(sparse_mx):
    """Convert a SciPy sparse CSR matrix to a PyTorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()



def get_match_from_similarity(S, device):
    P = torch.zeros_like(S)
    size = S.shape[0]
    index_S = [i for i in range(size)]
    index_S_hat = [i for i in range(size)]

    for _ in range(size):
        cur_size = S.shape[0]
        argmax = torch.argmax(S.to(device)).item()
        r = argmax // cur_size
        c = argmax % cur_size
        P[index_S[r]][index_S_hat[c]] = 1

        index_S.pop(r)
        index_S_hat.pop(c)

        S = S[torch.arange(S.size(0)) != r]
        S = S.t()[torch.arange(S.t().size(0)) != c].t()
        
    return P.t()


def compute_diffusion_matrix(A, eigvals, eigvecs, method='ppr', alpha=0.2, t=5.0):
    """
    Compute diffusion-based similarity matrix.
    
    Args:
        A: [N, N] adjacency matrix (binary, symmetric)
        method: 'ppr' or 'heat'
        alpha: teleport probability for PPR
        t: time parameter for heat kernel
    
    Returns:
        D: [N, N] diffusion similarity matrix
    """
    N = A.size(0)
    A = A.float()
    A = A + torch.eye(N, device=A.device)  # add self-loops

    D = torch.diag(A.sum(1).pow(-0.5))
    L = torch.eye(N, device=A.device) - D @ A @ D  # normalized Laplacian

    if method == 'ppr':
        # Personalized PageRank approximation: (1 - alpha)(I - alpha * A_hat)^(-1)
        A_hat = D @ A @ D
        I = torch.eye(N, device=A.device)
        PPR = alpha * torch.inverse(I - (1 - alpha) * A_hat)
        return PPR
    elif method == 'heat':
        exp_eigvals = torch.exp(-t * eigvals)
        H = eigvecs @ torch.diag(exp_eigvals) @ eigvecs.T
        return H
    else:
        raise ValueError("Method must be 'ppr' or 'heat'.")
    


def compute_tversky_matrix(A1: torch.Tensor, A2: torch.Tensor, alpha=0.5, beta=0.5, eps=1e-8):
    """
    Compute the Tversky similarity between all node pairs across graphs A1 and A2.
    A1, A2: Binary adjacency matrices [N, N]
    Returns: [N, N] Tversky similarity matrix
    """
    A1_bin = (A1 > 0).float()  # [N, N]
    A2_bin = (A2 > 0).float()  # [N, N]

    # Neighborhood sizes
    deg1 = A1_bin.sum(dim=1, keepdim=True)  # [N, 1]
    deg2 = A2_bin.sum(dim=1, keepdim=True)  # [N, 1]

    # Intersection counts: A1[i] @ A2[j].T for all i, j
    inter = A1_bin @ A2_bin.T  # [N, N]

    # Compute only_i and only_j
    only_i = deg1 - inter      
    only_j = deg2.T - inter    

    # Tversky index
    denom = inter + alpha * only_i + beta * only_j + eps
    T = inter / denom

    return T


def tversky_similarity_matrix(X: torch.Tensor, Y: torch.Tensor, alpha=0.5, beta=0.5, eps=1e-8):
    """
    Compute Tversky similarity between two sets of binary vectors.
    
    Args:
        X: [N, D] binary vectors for nodes in graph 1 (e.g., adjacency or features)
        Y: [M, D] binary vectors for nodes in graph 2
        alpha, beta: weights for asymmetry
    Returns:
        S: [N, M] similarity matrix
    """
    # Ensure binary
    X_bin = (X > 0).float()
    Y_bin = (Y > 0).float()

    # Set sizes
    X_sum = X_bin.sum(dim=1, keepdim=True)  # [N, 1]
    Y_sum = Y_bin.sum(dim=1, keepdim=True)  # [M, 1]

    # Intersection: X[i] ∩ Y[j]
    intersection = X_bin @ Y_bin.T  # [N, M]

    # Only X and only Y
    only_X = X_sum - intersection  # [N, M] via broadcast
    only_Y = Y_sum.T - intersection  # [N, M] via broadcast

    # Tversky index
    denominator = intersection + alpha * only_X + beta * only_Y + eps
    S = intersection / denominator

    return S



def gen_test_set(device,S, no_samples_each_level, perturbation_levels,method):
    S_hat_samples = {}
    S_prime_samples = {}
    p_samples = {}
    for level in perturbation_levels:
        S_hat_samples[str(level)] = []
        S_prime_samples[str(level)] = []
        p_samples[str(level)] = []
    for level in perturbation_levels:
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(num_edges*level)
        if(method == "degree"):
            print("Preprocessing degree probability distribution")
            S = torch.triu(S, diagonal=0)
            ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
            ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
            ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
            try:
                D = S @ ones_long
            except:
                try:
                    D = S @ ones_int
                except:
                    D = S @ ones_float
            sum = torch.sum(torch.mul(D @ D.T, S))
            edge_index = S.nonzero().t().contiguous()
            edge_index = np.array(edge_index)
            prob = []
            for i in range(edge_index.shape[1]):
                d1 = edge_index[0, i]
                d2 = edge_index[1, i]
                prob.append(D[d1] * D[d2] / sum)
            prob = np.array(prob, dtype='float64')
            prob = np.squeeze(prob)
        for i in range(no_samples_each_level):
            if(method == "uniform"):
                add_edge = random.randint(0, total_purturbations)
                delete_edge = total_purturbations - add_edge
                S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
                
            elif(method == "degree"):
                edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations, False, prob)
                edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
                edges_index = edge_index[:, edges_remain]
                S_prime = torch.zeros_like(S)
                for j in range(edges_index.shape[1]):
                    n1 = edges_index[:, j][0]
                    n2 = edges_index[:, j][1]
                    S_prime[n1][n2] = 1
                    if (S_prime[n2][n1] == 0):
                        S_prime[n2][n1] = 1
                SIZE = S_prime.shape[0]
                permutator = torch.randperm(SIZE)
                S_hat = S_prime[permutator]
                S_hat = S_hat.t()[permutator].t()
                P = torch.zeros(SIZE, SIZE)
                for i in range(permutator.shape[0]):
                    P[i, permutator[i]] = 1
            else:
                print("Probability model not defined")
                exit()
            S_hat_samples[str(level)].append(S_hat)
            p_samples[str(level)].append(P)
            S_prime_samples[str(level)].append(S_prime)
    return S_hat_samples, S_prime_samples, p_samples

def generate_features(purturbated_S):
    features = []
    for S in purturbated_S:
        feature = gen_netsmile(S)
        features.append(feature)
    return features

def gen_dataset(S, NUM_TO_ADD, NUM_TO_DELETE):
    SIZE = S.shape[0]
    num_added = 0
    num_deleted = 0
    E = torch.zeros(S.shape[0], S.shape[0])
    edge_indexes = (S == 1).nonzero(as_tuple=False).cpu()
    blank_indexes = (S == 0).nonzero(as_tuple=False).cpu()
    """
    delete edges
    """
    while(num_deleted < NUM_TO_DELETE):

        delete_index = random.randint(0, edge_indexes.shape[0]-1)
        index = edge_indexes[delete_index]
        E[index[0]][index[1]] = -1
        E[index[1]][index[0]] = -1
        num_deleted += 1

    """
    add edges
    """
    while (num_added < NUM_TO_ADD):

        add_index = random.randint(0, blank_indexes.shape[0] - 1)
        index = blank_indexes[add_index]
        E[index[0]][index[1]] = 1
        E[index[1]][index[0]] = 1
        num_added += 1

    S_prime = torch.add(S.cpu(),E.cpu())
    permutator = torch.randperm(SIZE)
    S_hat = S_prime[permutator]
    S_hat = S_hat.t()[permutator].t()
    P = torch.zeros(SIZE, SIZE)
    for i in range(permutator.shape[0]):
        P[i, permutator[i]] = 1
    return S, S_prime, S_hat, P

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def generate_purturbations(device, S, perturbation_level, no_samples, method):
    purturbated_samples = []
    if(method == "uniform"):
        for i in range(no_samples):
            num_edges = int(torch.count_nonzero(S).item()/2)
            total_purturbations = int(perturbation_level * num_edges)
            add_edge = random.randint(0,total_purturbations)
            delete_edge = total_purturbations - add_edge
            S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
            purturbated_samples.append(S_prime)
    elif(method == "degree"):
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(perturbation_level * num_edges)
        S = torch.triu(S, diagonal=0)
        ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
        ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
        ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
        try:
            D = S @ ones_long
        except:
            try:
                D = S @ ones_int
            except:
                D = S @ ones_float

        sum = torch.sum(torch.mul(D@D.T,S))
        edge_index = S.nonzero().t().contiguous()
        edge_index = np.array(edge_index)
        prob = []
        for i in range(edge_index.shape[1]):
            d1 = edge_index[0,i]
            d2 = edge_index[1,i]
            prob.append(D[d1]*D[d2]/sum)
        prob = np.array(prob,dtype='float64')
        prob = np.squeeze(prob)
        for i in range(no_samples):
            edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations,False,p=prob)
            edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
            edges_index = edge_index[:,edges_remain]
            S_prime = torch.zeros_like(S)
            for j in range(edges_index.shape[1]):
                n1 = edges_index[:,j][0]
                n2 = edges_index[:,j][1]
                S_prime[n1][n2] = 1
                S_prime[n2][n1] = 1
            purturbated_samples.append(S_prime)
    else:
        print("Probability model not defined.")
        exit()
    return purturbated_samples

def gen_netsmile(S):
    np_S = S.numpy()
    G = nx.from_numpy_array(np_S)
    feat = netsimile.feature_extraction(G)
    feat = torch.tensor(feat, dtype=torch.float)
    return feat



def compute_bipartite_graph(x1,x2):
    
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    x = torch.cat([x1, x2], dim=0)
    
    V1_indices = torch.repeat_interleave(torch.arange(n1), n2)

    V2_indices = torch.arange(n1, n1 + n2).repeat(n1)

    S = torch.stack([V1_indices, V2_indices])

    A = torch.zeros(n1+n2,n1+n2)
    A[V1_indices,V2_indices]=1
    
    return x,A


def perturb_feature(X_list, p, mode='gaussian', noise_std=0.1, dropout_val=0.0):
    """
    Applies perturbation to a list of feature tensors.

    Args:
        X_list (list of torch.Tensor): Each tensor has shape (N_i, D)
        p (float): Perturbation level (e.g., 0.01 for 1%)
        mode (str): 'gaussian', 'dropout', or 'swap'
        noise_std (float): Std of Gaussian noise (for 'gaussian' mode)
        dropout_val (float): Value to assign when dropping (for 'dropout' mode)

    Returns:
        list of torch.Tensor: Perturbed tensors
    """
    perturbed_list = []

    for X in X_list:
        X = X.clone()
        N, D = X.shape
        total_entries = N * D
        num_perturb = int(p * total_entries)

        # Choose random indices to perturb
        indices = torch.randperm(total_entries)[:num_perturb]
        rows = indices // D
        cols = indices % D

        if mode == 'gaussian':
            noise = torch.randn(num_perturb) * noise_std
            X[rows, cols] += noise

        elif mode == 'dropout':
            X[rows, cols] = dropout_val

        elif mode == 'swap':
            # Swap values across the tensor
            swap_rows = torch.randint(0, N, (num_perturb,))
            swap_cols = torch.randint(0, D, (num_perturb,))
            X[rows, cols] = X[swap_rows, swap_cols]

        else:
            raise ValueError(f"Unknown perturbation mode: {mode}")

        perturbed_list.append(X)

    return perturbed_list


## Laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.utils import get_laplacian
from scipy import sparse

def get_laplacian_selfloop(edge_index, edge_weight, num_nodes, n_type):
    
    if n_type=='with_sl':
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                              fill_value=1., num_nodes=num_nodes)
    edge_index, edge_weight = edge_index, edge_weight
    edge_weight = torch.ones(edge_index.size(1),device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # Compute A_norm = -D^{-1/2} A D^{-1/2}.
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    L = (edge_index, edge_weight)
    
    return L


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def laplacian_aug(edge_index, edge_weight, num_nodes ,lap_type,k, device):
    
    if edge_index.dim() == 2 and edge_index.size(0) == edge_index.size(1):
        edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()  # [2, num_edges]
                
    ## compute laplacian
    if lap_type in ['with_sl','without_sl']:
        L = get_laplacian_selfloop(edge_index, edge_weight, num_nodes, lap_type)
    else:
        L = get_laplacian(edge_index, edge_weight, num_nodes=num_nodes, normalization=lap_type)
            
    L_sparse = sparse.coo_matrix((L[1].cpu().numpy(), (L[0][0, :].cpu().numpy(), L[0][1, :].cpu().numpy())), shape=(num_nodes, num_nodes))
    
    
    evals, evecs = scipy.sparse.linalg.eigs(L_sparse, k=k , M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)
    
    evals=torch.tensor(evals.real)
    evecs=torch.tensor(evecs.real)

    mass=torch.ones(num_nodes).to(device)

    L = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(L_sparse)).to(device)

    return L, mass, evals, evecs



def compute_bipartite_graph(x1,x2):
    
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    x = torch.cat([x1, x2], dim=0)
    
    V1_indices = torch.repeat_interleave(torch.arange(n1), n2)

    V2_indices = torch.arange(n1, n1 + n2).repeat(n1)

    S = torch.stack([V1_indices, V2_indices])

    A = torch.zeros(n1+n2,n1+n2)
    A[V1_indices,V2_indices]=1
    
    return x,A

def wasserstein_loss(emb1, emb2, p=2, blur=0.05):
    """
    Computes the Wasserstein distance between two sets of embeddings using Sinkhorn divergence.

    Parameters:
    - emb1: Tensor of shape (N, d) -> Embeddings from Graph 1
    - emb2: Tensor of shape (M, d) -> Embeddings from Graph 2
    - p: int (default=2) -> Order of the Wasserstein distance (p=2 means Euclidean distance)
    - blur: float (default=0.05) -> Controls the smoothness of the Sinkhorn distance

    Returns:
    - loss: Wasserstein distance between emb1 and emb2
    """

    # Sinkhorn divergence for approximating Wasserstein distance
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    
    return loss(emb1, emb2)

def normalize_features(features: torch.Tensor, method: str = 'min-max') -> torch.Tensor:
    if method == 'min-max':
        min_vals = features.min(dim=0, keepdim=True).values
        max_vals = features.max(dim=0, keepdim=True).values
        normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == 'z-score':
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        normalized = (features - mean) / (std + 1e-8)
    else:
        raise ValueError("Normalization method must be either 'min-max' or 'z-score'.")

    return normalized


def compute_laplacian_positional_encodings(S, X, k=10):
    """
    S: (N x N) torch.Tensor, Adjacency matrix (assumed dense here)
    X: (N x F) torch.Tensor, Node feature matrix
    k: int, number of top eigenvectors to use
    """
    N = S.shape[0]
    S = S.float()  
    
    deg = torch.sum(S, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0  # Handle divide-by-zero

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    I = torch.eye(N, device=S.device)
    A_norm = D_inv_sqrt @ S @ D_inv_sqrt
    L = I - A_norm
    L_np = L.cpu().numpy()

    eigvals, eigvecs = eigsh(L_np, k=k, which='SM')  # Smallest magnitude eigenvalues

    eigvecs = torch.from_numpy(eigvecs).float().to(X.device)

    X_aug = torch.cat([X, eigvecs], dim=1)  # New shape: (N, F + k)

    return X_aug

def extract_structural_features(S, X):
    """
    S: torch.Tensor (N x N), adjacency matrix (dense, binary/int)
    Returns: torch.Tensor (N x num_structural_features)
    """
    S = S.float()
    N = S.shape[0]
    
    degree = S.sum(dim=1)  # (N,)

    deg_matrix = degree.unsqueeze(0).repeat(N, 1)  # (N, N)
    avg_neighbor_deg = (S * deg_matrix).sum(dim=1) / degree.clamp(min=1)

    S_sq = S @ S  # Count of 2-hop paths
    triangles = (S_sq * S).sum(dim=1) / 2  # Shared neighbors forming triangles
    possible_triplets = degree * (degree - 1) / 2
    clustering_coeff = triangles / possible_triplets.clamp(min=1)
    two_hop_neighbors = ((S @ S) > 0).float().sum(dim=1) - degree

    ego_size = degree + 1

    features = torch.stack([
        degree,
        avg_neighbor_deg,
        clustering_coeff,
        two_hop_neighbors,
        ego_size
    ], dim=1)

    X_augmented = torch.cat([X.to('cpu'), features], dim=1)

    return X_augmented





def compute_normalized_laplacian(adj):
    
    deg = np.array(adj.sum(1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L = sp.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt
    return L


def generate_wavelet_bases(adj_matrix, scales):
    wavelet_bases = []
    L = compute_normalized_laplacian(adj_matrix)
    for s in scales:
        exp_L = expm(-s * L)  # heat kernel: exp(-sL), returns NumPy ndarray
        wavelet_bases.append(torch.tensor(exp_L, dtype=torch.float32))  # No .toarray()
    return wavelet_bases








def compute_eigen_decomposition(A, mode='sym', add_self_loops=False, k=None):
    """
    Args:
        A (torch.Tensor or torch.sparse_coo_tensor): [N x N] adjacency matrix.
        mode (str): 'sym' for symmetric (D^-1/2 A D^-1/2),
                    'rw' for random-walk (D^-1 A)
        add_self_loops (bool): Whether to use A + I
        k (int or None): If set, return top-k eigenpairs; otherwise all.

    Returns:
        eigvals (torch.Tensor): [k] or [N] eigenvalues (ascending)
        eigvecs (torch.Tensor): [N x k] or [N x N] eigenvectors (columns)
    """
    assert mode in ['sym', 'rw'], "mode must be 'sym' or 'rw'"

    if A.is_sparse:
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        N = A.size(0)
        device = A.device
    else:
        assert A.ndim == 2 and A.size(0) == A.size(1), "Input A must be square"
        N = A.size(0)
        device = A.device
        indices = A.nonzero(as_tuple=False).T
        values = A[indices[0], indices[1]]

    # Add self-loops
    if add_self_loops:
        loop_idx = torch.arange(N, device=device)
        identity_indices = torch.stack([loop_idx, loop_idx])
        identity_values = torch.ones(N, device=device)

        indices = torch.cat([indices, identity_indices], dim=1)
        values = torch.cat([values, identity_values], dim=0)

    degrees = torch.zeros(N, device=device).scatter_add(0, indices[0], values)

    # Normalize adjacency
    if mode == 'sym':
        deg_inv_sqrt = torch.pow(degrees + 1e-10, -0.5)
        norm_vals = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
    else:  # 'rw'
        deg_inv = torch.pow(degrees + 1e-10, -1.0)
        norm_vals = deg_inv[indices[0]] * values

    # Create normalized dense matrix
    A_norm = torch.sparse_coo_tensor(indices, norm_vals, (N, N), device=device).to_dense()

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(A_norm)

    if k is not None and k < N:
        eigvals = eigvals[-k:]
        eigvecs = eigvecs[:, -k:]

    return eigvals, eigvecs






def compute_graph_basis(A, k=30):
    try:
        if A.layout == torch.sparse_coo:
            A = A.to_dense()  
        A = A.float()

        eigval, eigvec = torch.linalg.eigh(A)  

        eigval = eigval[:k]  
        eigvec = eigvec[:, :k]  
    except RuntimeError as e:
        raise RuntimeError(f"Eigen decomposition failed: {e}")
    return eigval, eigvec


def compute_laplacian_eigenbasis(A_torch, k=30, norm_type=None):
    # norm_type (str): 'sym', 'rw', or None for unnormalized
    device = A_torch.device


    if A_torch.layout != torch.sparse_coo:
        A_torch = A_torch.to_sparse_coo()

    A_torch = A_torch.coalesce()

    A_torch = A_torch.cpu()

    indices = A_torch.indices().numpy()
    values = A_torch.values().numpy()
    size = A_torch.size()

    A_scipy = coo_matrix((values, (indices[0], indices[1])), shape=size)
    
    # Compute normalized Laplacian matrix
    if norm_type == "sym":
        L = csgraph.laplacian(A_scipy, normed=True)
    elif norm_type == "rw":
        L = csgraph.laplacian(A_scipy, normed=False)
        # Convert to random-walk Laplacian: L_rw = D^-1 A → L_rw = I - D^-1 A = D^-1 (D - A)
        D_inv = csgraph.laplacian(A_scipy, normed=False).diagonal()
        D_inv[D_inv != 0] = 1.0 / D_inv[D_inv != 0]
        D_inv_mat = coo_matrix((D_inv, (range(size[0]), range(size[0]))), shape=size)
        L = D_inv_mat.dot(L)
        L = L.astype(np.float32)  
    elif norm_type is None:
        L = csgraph.laplacian(A_scipy, normed=False)
        L = L.astype(np.float32)  
    else:
        raise ValueError("norm_type must be one of {'sym', 'rw', None}")
    
    eigvals, eigvecs = eigsh(L, k=k, which='SM')
    eigvals = torch.from_numpy(eigvals).float().to(device)
    eigvecs = torch.from_numpy(eigvecs).float().to(device)
    
    return L, eigvals, eigvecs




def compute_full_laplacian_spectrum(A_torch, norm_type="sym"):
    device = A_torch.device

    # Ensure sparse COO
    if A_torch.layout != torch.sparse_coo:
        A_torch = A_torch.to_sparse_coo()
    A_torch = A_torch.coalesce().cpu()

    # Convert to scipy sparse
    indices = A_torch.indices().numpy()
    values = A_torch.values().numpy()
    size = A_torch.size()

    A_scipy = coo_matrix((values, (indices[0], indices[1])), shape=size)

    # Build Laplacian
    if norm_type == "sym":
        L = csgraph.laplacian(A_scipy, normed=True)
    elif norm_type == "rw":
        L = csgraph.laplacian(A_scipy, normed=False)
        deg = L.diagonal()
        deg[deg != 0] = 1.0 / deg[deg != 0]
        D_inv = coo_matrix((deg, (range(size[0]), range(size[0]))), shape=size)
        L = D_inv.dot(L)
    elif norm_type is None:
        L = csgraph.laplacian(A_scipy, normed=False)
    else:
        raise ValueError("norm_type must be one of {'sym', 'rw', None}")

    # Convert to dense and compute ALL eigenvalues
    L_dense = L.toarray()
    # eigvals, eigvecs = np.linalg.eigvalsh(L_dense)  # symmetric solver
    eigvals, eigvecs = np.linalg.eigh(L_dense) 

    # Convert back to torch
    eigvals = torch.from_numpy(eigvals).float().to(device)
    eigvecs = torch.from_numpy(eigvecs).float().to(device)

    indices = (eigvals > 0.00001).nonzero(as_tuple=True)[0]
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:, indices]
    return eigvals, eigvecs




def compute_hks(eigenvecs, eigenvals, time_scales=40, eps=1e-6):
    """
    Compute the Heat Kernel Signature (HKS) given eigenvectors and eigenvalues.
    - eigenvecs: [N, k]
    - eigenvals: [k] (assumed sorted in ascending order)
    Returns:
    - HKS: [N, T]
    """
    N, k = eigenvecs.shape

    # Ensure eigenvalues are non-negative
    eigenvals = torch.clamp(eigenvals, min=eps)

    # Select a valid range of eigenvalues for t scale
    l_min = eigenvals[1].item()
    l_max = eigenvals[-1].item()
    if l_max <= 0 or l_min <= 0:
        raise ValueError("Eigenvalues must be positive for HKS.")

    t_min = 4.0 / l_max
    t_max = 4.0 / l_min
    ts = torch.logspace(torch.log10(torch.tensor(t_min)), torch.log10(torch.tensor(t_max)), steps=time_scales, device=eigenvals.device)

    # Compute squared eigenvectors once
    phi_sq = eigenvecs ** 2  # [N, k]

    # Compute HKS matrix
    hks = torch.stack([torch.matmul(phi_sq, torch.exp(-eigenvals * t)) for t in ts], dim=1)  # [N, T]

    return hks




def MNC_regulizer(z1_h, z1_l, z2_h, z2_l, adj1, adj2, tau=0.1):
    device = z1_h.device

    ss1 = torch.cat([z1_h, z1_l], dim=1)
    ss2 = torch.cat([z2_h, z2_l], dim=1)
    
    S_emb1 = F.normalize(ss1, p=2, dim=1)
    S_emb2 = F.normalize(ss2, p=2, dim=1)

    D = torch.cdist(S_emb1, S_emb2, p=2)  

    P_soft = torch.softmax(-D / tau, dim=1) 

    N1, N2 = D.shape

    adj1 = adj1.coalesce()
    adj2 = adj2.coalesce()

    idx1 = adj1.indices()
    idx2 = adj2.indices()

    mnc_scores = []

    for i in range(N1):
        # Neighbors of node i in G1
        neighbors_i = idx1[1][idx1[0] == i]  
        if neighbors_i.numel() == 0:
            mnc_scores.append(torch.tensor(1.0, device=device))
            continue
        
        soft_matched_neighbors = P_soft[neighbors_i]  
        avg_soft_match = soft_matched_neighbors.mean(dim=0)  

        soft_match_i = P_soft[i]  

        adj2_binary = torch.sparse_coo_tensor(idx2, torch.ones_like(idx2[0], dtype=torch.float32, device=device),
                size=adj2.shape, device=device).coalesce()

        soft_match_i_row = soft_match_i.unsqueeze(0)
        soft_neighbors_i = torch.sparse.mm(adj2_binary.t(), soft_match_i_row.t()).squeeze(1)
        soft_neighbors_i = soft_neighbors_i / (soft_neighbors_i.sum() + 1e-8)
        
        soft_neighbors_i = soft_neighbors_i / (soft_neighbors_i.sum() + 1e-8)

        consistency = torch.sum(avg_soft_match * soft_neighbors_i)

        mnc_scores.append(1.0 - consistency)

    return torch.stack(mnc_scores).mean()



def sinkhorn_normalization(S, n_iters=20, eps=1e-8):
    """
    Sinkhorn normalization to get a doubly stochastic matrix from similarity matrix S.
    """
    Q = torch.exp(-S)  # Higher similarity -> lower cost
    for _ in range(n_iters):
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        Q = Q / (Q.sum(dim=0, keepdim=True) + eps)
    return Q

def normalize_adjacency(adj):
    """
    Symmetric normalization of adjacency matrix: A_norm = D^(-1/2) A D^(-1/2)
    """
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    N = adj.size(0)

    deg = torch.zeros(N, device=adj.device).scatter_add_(0, indices[0], values)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    
    norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
    adj_norm = torch.sparse_coo_tensor(indices, norm_values, adj.size(), device=adj.device)
    
    return adj_norm

def spectral_mnc_loss(z1_h, z1_l, z2_h, z2_l, adj1, adj2):
    device = adj1.device
    z1 = torch.cat([z1_h, z1_l], dim=1)  
    z2 = torch.cat([z2_h, z2_l], dim=1)  

    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    S = torch.cdist(z1.double(), z2.double(), p=2)  

    M = sinkhorn_normalization(S)  

    A1 = normalize_adjacency(adj1).to_dense()  
    A2 = normalize_adjacency(adj2).to_dense()  

    M = M.float()
    A1 = A1.float()
    A2 = A2.float()

    MA2 = torch.matmul(M, A2)     
    A1M = torch.matmul(A1, M)     

    loss = torch.norm(MA2 - A1M, p='fro') ** 2
    return loss.float()




def load_gt(path, id2idx_src, id2idx_trg, format='matrix', convert=False):    
    conversion_src = type(list(id2idx_src.keys())[0])
    conversion_trg = type(list(id2idx_trg.keys())[0])
    if format == 'matrix':
        gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()                
                gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
        return gt
    else:
        gt = {}
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                if convert:
                    gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
                else:
                    gt[conversion_src(src)] = conversion_trg(trg)
        return gt
    
    
    
    


def normalize_adjacency_filter(adj, high=True):
    """
    Symmetric normalization of adjacency matrix: A_norm = D^(-1/2) A D^(-1/2)
    high-pass filter: A = (D_tilde - A_tilde) / 2 
    low-pass filter: A = (D_tilde + A_tilde) / 2  
    """
    
    num_nodes = adj.shape[0]
    identity = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1), values=torch.ones(num_nodes, device=adj.device),size=(num_nodes, num_nodes))
    A_tilde = adj + identity
    deg_values = torch.sparse.sum(A_tilde, dim=1).to_dense()  
    D_tilde = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1), values=deg_values,size=(num_nodes, num_nodes))
    
    if high:
        adj = (D_tilde - A_tilde) / 2
    else:
        adj = (D_tilde + A_tilde) / 2
        
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    N = adj.size(0)

    deg = torch.zeros(N, device=adj.device).scatter_add_(0, indices[0], values)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    
    norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
    adj_norm = torch.sparse_coo_tensor(indices, norm_values, adj.size(), device=adj.device)
    
    return adj_norm


    
            
def orthonormalize_basis(phi):
    # phi: [N, k] — eigenvectors as columns
    Q, _ = torch.linalg.qr(phi)  # QR decomposition
    return Q




def k_hop_mask(A, k):
    """
    Compute binary reachability mask up to k hops.
    A: [N, N] adjacency matrix (0/1), with self-loops
    Returns: [N, N] binary mask where mask[i, j] = 1 if j is reachable from i within k hops
    """
    A = A.float()  
    N = A.size(0)
    A_power = torch.eye(N, device=A.device)
    A_k = torch.eye(N, device=A.device)

    for _ in range(k):
        A_power = A_power @ A
        A_k = ((A_power > 0) | (A_k > 0)).float()

    return A_k



def compute_alignment_metrics(vis_emb, lang_emb, ground_truth_pairs=None):
    """
    Compute alignment metrics
    """
    # Normalize embeddings
    vis_emb = F.normalize(vis_emb, p=2, dim=1)
    lang_emb = F.normalize(lang_emb, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(vis_emb, lang_emb.t())
    
    # Compute retrieval accuracy
    vis_to_lang_acc = (torch.argmax(sim_matrix, dim=1) == torch.arange(sim_matrix.size(0)).to(sim_matrix.device)).float().mean()
    lang_to_vis_acc = (torch.argmax(sim_matrix, dim=0) == torch.arange(sim_matrix.size(1)).to(sim_matrix.device)).float().mean()
    
    # Compute top-k accuracy
    top5_vis_to_lang = (torch.topk(sim_matrix, k=5, dim=1)[1] == torch.arange(sim_matrix.size(0)).to(sim_matrix.device).unsqueeze(1)).any(dim=1).float().mean()
    top5_lang_to_vis = (torch.topk(sim_matrix, k=5, dim=0)[1] == torch.arange(sim_matrix.size(1)).to(sim_matrix.device).unsqueeze(0)).any(dim=0).float().mean()
    
    metrics = {
        'vis_to_lang_acc': vis_to_lang_acc.item(),
        'lang_to_vis_acc': lang_to_vis_acc.item(),
        'top5_vis_to_lang': top5_vis_to_lang.item(),
        'top5_lang_to_vis': top5_lang_to_vis.item(),
        'avg_accuracy': (vis_to_lang_acc + lang_to_vis_acc).item() / 2
    }
    
    return metrics


def scipy_to_torch_sparse(scipy_sparse):
    """Convert scipy sparse matrix to torch sparse tensor"""
    scipy_coo = scipy_sparse.tocoo()
    indices = torch.from_numpy(np.vstack((scipy_coo.row, scipy_coo.col))).long()
    values = torch.from_numpy(scipy_coo.data).float()
    shape = scipy_coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))   


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # Add self-connections
    adj = adj + sp.eye(adj.shape[0])
    
    # Compute degree: D = sum of rows
    rowsum = np.array(adj.sum(1)).flatten()
    
    # Compute D^{-1/2}
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # handle inf
    
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Compute D^{-1/2} * A * D^{-1/2}
    adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device='cuda:0'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def normalize_adj_torch(adj):
    """
    Symmetrically normalize a sparse adjacency matrix in PyTorch.
    adj: torch.sparse_coo_tensor of shape (N, N)
    Returns: normalized torch.sparse_coo_tensor on same device
    """
    device = adj.device
    N = adj.size(0)

    adj = adj.coalesce()

    indices = adj.indices()
    values = adj.values()

    eye_indices = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    eye_values = torch.ones(N, device=device)
    eye = torch.sparse_coo_tensor(eye_indices, eye_values, (N, N), device=device).coalesce()

    # Add self-loops (A + I)
    new_indices = torch.cat([indices, eye.indices()], dim=1)
    new_values = torch.cat([values, eye.values()])
    adj_hat = torch.sparse_coo_tensor(new_indices, new_values, (N, N), device=device).coalesce()

    # degree 
    deg = torch.sparse.sum(adj_hat, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    # D^{-1/2} * A * D^{-1/2}
    row, col = adj_hat.indices()
    norm_values = adj_hat.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Rebuild normalized sparse tensor
    adj_norm = torch.sparse_coo_tensor(
        adj_hat.indices(), norm_values, adj_hat.size(), device=device
    ).coalesce()
    
    return adj_norm.transpose(0, 1)



def normalize_adj_I(adj_hat, deg):
    """
    inputs:
        adj_hat = Adj + I
        deg
    """
    device = adj_hat.device
    adj_hat = adj_hat.coalesce()

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    # D^{-1/2} * A * D^{-1/2}
    row, col = adj_hat.indices()
    norm_values = adj_hat.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Rebuild normalized sparse tensor
    adj_norm = torch.sparse_coo_tensor(
        adj_hat.indices(), norm_values, adj_hat.size(), device=device
    ).coalesce()
    
    return adj_norm.transpose(0, 1)





def extract_test_data(test_pairs, data):
    """
    Extract test indices and labels based on dataset format.
    
    Args:
        test_pairs: Test entity pairs
        data: Dataset name
        
    Returns:
        Tuple of (test_idx, labels)
    """
    if data == "ACM_DBLP":
        test_idx = test_pairs[:, 0].astype(np.int32)
        labels = test_pairs[:, 1].astype(np.int32)
    elif data in ["Douban Online_Offline", "phone_email", "cora_cora"]:
        test_idx = test_pairs[0, :].astype(np.int32)
        labels = test_pairs[1, :].astype(np.int32)
    elif data in ['DBP15K-fr', 'DBP15K-zh', 'DBP15K-ja']:
        test_idx = test_pairs[:, 0]
        labels = test_pairs[:, 1]
    elif data in ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']:
        test_idx = test_pairs[:, 0]
        labels = test_pairs[:, 1]
    elif data in ['flickr-lastfm', 'flickr-myspace', 'foursquare-twitter']:
        test_idx = test_pairs[:, 0]
        labels = test_pairs[:, 1]
    elif data == 'facebook_twitter':
        test_idx = test_pairs[1, :]
        labels = test_pairs[0, :]
    else:
        raise ValueError(f"Unknown dataset format: {data}")
        
    return test_idx, labels


def compute_hits(distance_matrix, test_idx, labels):
    """
    Compute Hit@K metrics for entity alignment.
    
    Args:
        distance_matrix: Distance matrix between entity embeddings
        test_idx: Test entity indices
        labels: Ground truth labels
        
    Returns:
        Dictionary of Hit@K scores
    """
    hits = {1: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    epsilon = 1e-8  # Floating point tolerance
    
    for i in range(len(test_idx)):
        dist_list = distance_matrix[test_idx[i]]
        label = labels[i]
        label_dist = dist_list[label].item()
        rank = torch.sum(dist_list < (label_dist - epsilon)).item()
        
        # Update hit counters
        if rank == 0:
            hits[1] += 1
            hits[5] += 1
            hits[10] += 1
            hits[50] += 1
            hits[100] += 1
        elif rank <= 4:
            hits[5] += 1
            hits[10] += 1
            hits[50] += 1
            hits[100] += 1
        elif rank <= 9:
            hits[10] += 1
            hits[50] += 1
            hits[100] += 1
        elif rank <= 49:
            hits[50] += 1
            hits[100] += 1
        elif rank <= 99:
            hits[100] += 1
    
    # Normalize by number of test samples
    n_test = len(test_idx)
    return {k: v / n_test for k, v in hits.items()}