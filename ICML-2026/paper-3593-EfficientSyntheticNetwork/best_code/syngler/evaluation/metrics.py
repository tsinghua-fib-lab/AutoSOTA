import numpy as np
import torch
try:
    import pandas as pd
except ImportError:
    pd = None
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def rmse_of_means(data, x_fake):
    real_mean = data.mean(axis=0)
    fake_mean = x_fake.mean(axis=0)
    return np.sqrt(np.mean((real_mean - fake_mean) ** 2))

def rmse_of_covariances(data, x_fake):
    real_cov = np.cov(data, rowvar=False)
    fake_cov = np.cov(x_fake, rowvar=False)
    return np.sqrt(np.mean((real_cov - fake_cov) ** 2))

# Network density function removed as requested

def triangle_density(A_batch: torch.Tensor, device: str | torch.device | None = None, return_numpy = None) -> torch.Tensor:
    """
    A_batch: (B, n, n) or (n, n) 0/1 symmetric adjacency matrices.
    device: Target device for computation; defaults to CUDA if available, otherwise the input's device.
    Returns: Triangle densities on the input's original device.
    """
    if A_batch.dim() == 2:
        A_batch = A_batch.unsqueeze(0)
    B, n, _ = A_batch.shape
    orig_device = A_batch.device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else orig_device.type)
    else:
        target_device = torch.device(device)
    if n < 3:
        result = torch.zeros(B, device=orig_device, dtype=torch.float32)
    else:
        A = A_batch.to(target_device, non_blocking=True).float()
        A_cube = A @ A @ A
        tri_counts = torch.diagonal(A_cube, dim1=-2, dim2=-1).sum(dim=-1) / 6.0
        denom = n * (n - 1) * (n - 2)/6.0
        density = (tri_counts) / denom
        result = density.to(orig_device)
    
    if return_numpy:
        if result.dim() == 0:
            return float(result.detach().cpu())
        return result.detach().cpu().numpy()
    return result

def local_clustering_coefficients(A_batch: torch.Tensor,
                                  device: str | torch.device | None = None,
                                  zero_diagonal: bool = True) -> torch.Tensor:
    """
    Compute local clustering coefficients C_i for undirected simple graphs.

    Args:
        A_batch: Adjacency tensor of shape (B, n, n) or (n, n), 0/1, symmetric.
        device:  Preferred device for computation; defaults to 'cuda' if available,
                else keeps the input device. Results are returned on the input device.
        zero_diagonal: If True, force zero diagonal before computing.

    Returns:
        C: Tensor of shape (B, n) or (n,) with local clustering coefficients.
           Formula: C_i = diag(A^3)_i / (k_i * (k_i - 1)) for k_i >= 2, else 0.
    """
    orig_device = A_batch.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else orig_device
    A = A_batch.to(device, non_blocking=True)
    if A.dim() == 2:
        A = A.unsqueeze(0)  # -> (1, n, n)

    B, n, m = A.shape
    if n != m:
        raise ValueError(f"Adjacency must be square, got {(n, m)}")

    A = A.float()
    if zero_diagonal:
        eye = torch.eye(n, device=A.device).unsqueeze(0)  # (1,n,n)
        A = A * (1.0 - eye)

    deg = A.sum(dim=-1)  # (B,n)

    A2 = torch.bmm(A, A)             # (B,n,n)
    diagA3 = (A2 * A).sum(dim=-1)    # (B,n)

    denom = deg * (deg - 1.0)
    C = torch.zeros_like(deg)
    mask = denom > 0
    C[mask] = diagA3[mask] / denom[mask]

    C = C.clamp_(0.0, 1.0)

    if A_batch.dim() == 2:
        C = C.squeeze(0)  # (n,)
    return C.to(orig_device)

def global_clustering_coefficient(A_batch, device=None, zero_diagonal=True, return_numpy=None):
    """
    Global clustering coefficient (transitivity) for undirected simple graphs.
    C = tr(A^3) / sum_i k_i (k_i - 1)
    Supports numpy or torch; shapes (n,n) or (B,n,n).
    """
    is_numpy = isinstance(A_batch, np.ndarray)
    if return_numpy is None:
        return_numpy = is_numpy
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.as_tensor(A_batch, dtype=torch.float32, device=device)
    if A.dim() == 2:
        A = A.unsqueeze(0)   # -> (1,n,n)
    elif A.dim() != 3:
        raise ValueError(f"Adjacency must be (n,n) or (B,n,n), got {tuple(A.shape)}")

    B, n, m = A.shape
    if n != m:
        raise ValueError(f"Adjacency must be square, got {(n, m)}")

    if zero_diagonal:
        eye = torch.eye(n, device=A.device).unsqueeze(0)
        A = A * (1.0 - eye)  # simple graph

    deg = A.sum(dim=-1)                  # (B,n)
    A2  = torch.bmm(A, A)                # (B,n,n)
    num = (A2 * A).sum(dim=(-2, -1))     # tr(A^3) for symmetric A, shape (B,)
    den = (deg * (deg - 1.0)).sum(dim=-1)

    C = torch.zeros_like(num)
    mask = den > 0
    C[mask] = num[mask] / den[mask]
    C.clamp_(0.0, 1.0)

    if A_batch.ndim == 2:
        C = C.squeeze(0)  # scalar
    if return_numpy:
        if C.ndim == 0:
            return float(C.detach().cpu())
        return C.detach().cpu().numpy()
    return C

def global_clustering_coefficient_prob(P_batch, device=None, zero_diagonal=True, return_numpy=None):
    """
    Expected global clustering coefficient for independent-edge graphs (probability matrix P).
    C = tr(P^3) / sum_i [ (sum_j P_ij)^2 - sum_j P_ij^2 ]
    Same I/O semantics as global_clustering_coefficient(...).
    Supports numpy or torch; shapes (n,n) or (B,n,n).
    """
    is_numpy = isinstance(P_batch, np.ndarray)
    if return_numpy is None:
        return_numpy = is_numpy
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P = torch.as_tensor(P_batch, dtype=torch.float32, device=device)
    if P.dim() == 2:
        P = P.unsqueeze(0)   # -> (1,n,n)
    elif P.dim() != 3:
        raise ValueError(f"Adjacency/probability must be (n,n) or (B,n,n), got {tuple(P.shape)}")

    B, n, m = P.shape
    if n != m:
        raise ValueError(f"Matrix must be square, got {(n, m)}")

    if zero_diagonal:
        eye = torch.eye(n, device=P.device).unsqueeze(0)
        P = P * (1.0 - eye)  # zero diagonal

    # Numerator: E[triangles] = tr(P^3)
    P2  = torch.bmm(P, P)
    num = (P2 * P).sum(dim=(-2, -1))  # shape (B,)

    # Denominator: E[connected triples] = sum_i [ (sum_j p_ij)^2 - sum_j p_ij^2 ]
    s1  = P.sum(dim=-1)               # (B,n) row sums
    s2  = (P * P).sum(dim=-1)         # (B,n) row squared sums
    den = (s1 * s1 - s2).sum(dim=-1)  # (B,)

    C = torch.zeros_like(num)
    mask = den > 0
    C[mask] = num[mask] / den[mask]
    C.clamp_(0.0, 1.0)

    if P_batch.ndim == 2:
        C = C.squeeze(0)  # scalar

    if return_numpy:
        if C.ndim == 0:
            return float(C.detach().cpu())
        return C.detach().cpu().numpy()
    return C

def degree_centrality(A_batch, device=None):
    """
    Compute degree centrality for a batch of adjacency matrices.

    Args:
        A_batch (torch.Tensor or numpy.ndarray): 
            Batch of adjacency matrices with shape (B, n, n), where B is batch size, n is number of nodes.
        device (torch.device, optional): 
            Device for computation (e.g., 'cuda' or 'cpu'). If None, automatically select
            available GPU. Default is None.

    Returns:
        torch.Tensor: 
            Tensor with shape (B, n) containing degree centrality for each node in each network.
    """
    is_numpy = isinstance(A_batch, torch.Tensor) is False
    if is_numpy:
        A_batch = torch.as_tensor(A_batch, dtype=torch.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    A_batch = A_batch.to(device)
    
    original_dim = A_batch.dim()
    if original_dim == 2:
        A_batch = A_batch.unsqueeze(0)  # Convert from (n, n) to (1, n, n)
    elif original_dim != 3:
        raise ValueError(
            f"Expected (B, n, n) or (n, n), got {A_batch.shape}."
        )
    
    B, n, _ = A_batch.shape
    
    # Degree centrality computation
    # In adjacency matrix, degree of a node is the sum of its row (or column)
    # Undirected graph: A_ij = A_ji
    # Here we simply sum each row, which works for both directed (out-degree) and undirected graphs
    # sum(dim=-1) sums over the last dimension (columns), resulting in shape (B, n)
    degrees = torch.sum(A_batch, dim=-1)
    
    if n > 1:
        degree_centrality = degrees / (n - 1)
    else:
        degree_centrality = torch.zeros_like(degrees)
        
    return degree_centrality

def energy_distance_cpu(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    
    if n == 0 or m == 0:
        return 0.0
    
    xy_dist = np.mean(np.abs(np.subtract.outer(x, y)))
    xx_dist = np.mean(np.abs(np.subtract.outer(x, x)))
    yy_dist = np.mean(np.abs(np.subtract.outer(y, y)))
    
    return 2 * xy_dist - xx_dist - yy_dist

def energy_distance_gpu(x, y, device=None):
    """
    Compute energy distance using PyTorch on GPU or CPU.

    Args:
        x (torch.Tensor or numpy.ndarray): Sample vector of first distribution.
        y (torch.Tensor or numpy.ndarray): Sample vector of second distribution.
        device (torch.device, optional): Device for computation. If None, automatically select
            available GPU. Default is None.

    Returns:
        float: Energy distance between the two distributions.
    """
    is_numpy_x = isinstance(x, np.ndarray)
    is_numpy_y = isinstance(y, np.ndarray)

    if device is None:
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    if is_numpy_x:
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    
    if is_numpy_y:
        y = torch.from_numpy(y).to(device)
    else:
        y = y.to(device)

    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0

    # Core computation using PyTorch tensor operations
    # Use unsqueeze and broadcasting to implement outer operation
    # (n, 1) - (1, m) -> (n, m)
    xy_dist = torch.mean(torch.abs(x.unsqueeze(1) - y.unsqueeze(0)))
    
    # (n, 1) - (1, n) -> (n, n)
    xx_dist = torch.mean(torch.abs(x.unsqueeze(1) - x.unsqueeze(0)))
    
    # (m, 1) - (1, m) -> (m, m)
    yy_dist = torch.mean(torch.abs(y.unsqueeze(1) - y.unsqueeze(0)))

    return (2 * xy_dist - xx_dist - yy_dist).item()

def energy_distance(x, y, device="cpu", dtype=torch.float64):
    # Convert to 1D vectors
    x = torch.as_tensor(x, dtype=dtype, device=device).flatten()
    y = torch.as_tensor(y, dtype=dtype, device=device).flatten()
    n, m = x.numel(), y.numel()
    if n == 0 or m == 0:
        return 0.0

    xs, _ = torch.sort(x)  # Ascending order
    idx_x = torch.arange(1, n + 1, device=device, dtype=dtype)
    S_xx = 2.0 * torch.sum((2 * idx_x - n - 1) * xs)   # = \sum_{i,j}|x_i-x_j|
    xx_mean = S_xx / (n * n)

    ys, _ = torch.sort(y)
    idx_y = torch.arange(1, m + 1, device=device, dtype=dtype)
    S_yy = 2.0 * torch.sum((2 * idx_y - m - 1) * ys)
    yy_mean = S_yy / (m * m)

    # ---- E||X-Y|| part: sorting + prefix sum + binary search (O(n log m)) ----
    pref_y = torch.cumsum(ys, dim=0)                 # Prefix sum
    sum_y = pref_y[-1]
    # k_i = #{ y_j <= x_i }
    k = torch.searchsorted(ys, x, right=True)        # int64, on device
    # sum_{y_j <= x_i} y_j
    sum_le = torch.where(k > 0, pref_y[k - 1], torch.zeros_like(x, dtype=dtype, device=device))
    S_xy = torch.sum(x * (2 * k.to(dtype) - m) - (2 * sum_le - sum_y))  # = \sum_{i,j}|x_i - y_j|
    xy_mean = S_xy / (n * m)

    return float(2 * xy_mean - xx_mean - yy_mean)

def _rbf_median_bandwidth(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    z = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=0)
    d2 = ((z[:,None,:] - z[None,:,:])**2).sum(axis=2)
    tri = d2[np.triu_indices_from(d2, k=1)]
    if tri.size == 0 or not np.any(tri > 0):
        return 1e-8
    med = np.median(tri[tri > 0])
    return max(med/2.0, 1e-8)

def _mmd2_unbiased(x, y, sigma2):
    x = np.asarray(x).ravel().reshape(-1,1)
    y = np.asarray(y).ravel().reshape(-1,1)

    def k_mat(a, b):
        d2 = ((a[:,None,:]-b[None,:,:])**2).sum(axis=2)
        return np.exp(-d2/(2.0*sigma2))

    Kxx = k_mat(x,x); Kyy = k_mat(y,y); Kxy = k_mat(x,y)
    n = Kxx.shape[0]; m = Kyy.shape[0]

    if n > 1:
        np.fill_diagonal(Kxx, 0.0)
        term_x = Kxx.sum()/(n*(n-1))
    else:
        term_x = 0.0
    if m > 1:
        np.fill_diagonal(Kyy, 0.0)
        term_y = Kyy.sum()/(m*(m-1))
    else:
        term_y = 0.0
    term_xy = Kxy.mean()
    mmd2 = term_x + term_y - 2.0*term_xy
    return float(max(mmd2, 0.0))

def compute_mmd(x, y, subsample=None, seed=None):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if subsample is not None and subsample > 0:
        rng = np.random.default_rng(seed)
        if x.size > subsample:
            x = rng.choice(x, size=subsample, replace=False)
        if y.size > subsample:
            y = rng.choice(y, size=subsample, replace=False)
    sigma2 = _rbf_median_bandwidth(x, y)
    mmd2 = _mmd2_unbiased(x, y, sigma2)
    return float(np.sqrt(mmd2))

def eigenvalues(A_batch, device=None, return_fiedler=False, return_eigen = True,laplacian: str = 'combinatorial'):
    """
    Compute eigenvalues of a batch of matrices, optionally return Fiedler value (algebraic connectivity).
    
    Args:
        A_batch: (B, n, n) or (n, n) adjacency/probability matrices (recommended undirected symmetric).
        device:  Computation device; None then automatically select GPU first.
        return_fiedler: When True, additionally return Fiedler value for each sample.
        laplacian: 'combinatorial' uses L = D - A; 'normalized' uses L_sym = I - D^{-1/2} A D^{-1/2}.
    
    Returns:
        - If return_fiedler=False:
            eigenvals_sorted: (B, n) or (n,)  —— |λ(A)| sorted in descending order (float32)
        - If return_fiedler=True:
            (eigenvals_sorted, fiedler): where fiedler has shape (B,) or scalar
    """
    is_numpy = not isinstance(A_batch, torch.Tensor)
    if is_numpy:
        A_batch = torch.as_tensor(A_batch, dtype=torch.float32)
    else:
        A_batch = A_batch.to(torch.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_batch = A_batch.to(device)

    original_dim = A_batch.dim()
    if original_dim == 2:
        A_batch = A_batch.unsqueeze(0)  # -> (1, n, n)
    elif original_dim != 3:
        raise ValueError(f"Expected (B, n, n) or (n, n), got {A_batch.shape}.")

    B, n, _ = A_batch.shape
    if return_eigen:
        # --- 1) Adjacency eigenvalues (original behavior) ---
        eig_adj = torch.linalg.eigvalsh(A_batch).to(torch.float32)      # Symmetric case -> real
        eigenvals_sorted, _ = torch.sort(torch.abs(eig_adj), dim=-1, descending=True)

        if not return_fiedler:
            return eigenvals_sorted.squeeze(0) if original_dim == 2 else eigenvals_sorted

    # --- 2) Fiedler value from Laplacian ---
    A_sym = 0.5 * (A_batch + A_batch.transpose(-1, -2))

    deg = A_sym.sum(dim=-1)  # (B, n)

    if laplacian == 'combinatorial':
        L = torch.diag_embed(deg) - A_sym
    elif laplacian == 'normalized':
        # L_sym = I - D^{-1/2} A D^{-1/2}
        inv_sqrt_deg = torch.zeros_like(deg)
        positive = deg > 0
        inv_sqrt_deg[positive] = deg[positive].pow(-0.5)
        D_inv_sqrt = torch.diag_embed(inv_sqrt_deg)
        I = torch.eye(n, device=device, dtype=A_batch.dtype).unsqueeze(0).expand(B, n, n)
        L = I - D_inv_sqrt @ A_sym @ D_inv_sqrt
    else:
        raise ValueError("laplacian must be 'combinatorial' or 'normalized'.")

    # Laplacian is symmetric positive semi-definite, use eigvalsh (ascending)
    lap_eigs = torch.linalg.eigvalsh(L)                   # (B, n), ascending
    lap_eigs = torch.clamp(lap_eigs, min=0.0)
    if n < 2:
        fiedler = lap_eigs[..., 0]
    else:
        fiedler = lap_eigs[..., 1]

    if original_dim == 2:
        if return_eigen:
            return eigenvals_sorted.squeeze(0), fiedler.squeeze(0)
        return fiedler.squeeze(0)
    if return_eigen:
        return eigenvals_sorted, fiedler
    return fiedler

