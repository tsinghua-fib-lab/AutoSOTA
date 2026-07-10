"""
Geometric Embeddings

Advanced Euclidean embedding methods including diffusion maps and geodesic embeddings for meshes.
"""

import numpy as np
import potpourri3d as pp3d
import torch
from pykeops.numpy import LazyTensor as NumpyLazyTensor
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, eigsh
from sklearn.manifold import MDS

from utils.data.meshes.preprocess import decimate_mesh
from utils.math.meshes import lumped_mass


def diffusion_embedding(X, dim=10, sigma=0.1, t=0.1):
    """
    Compute diffusion map embedding.
    
    Embeds points into lower-dimensional space using the diffusion of a gaussian operator.
    It is a point-cloud adaptation of diffusion geometry for meshes, where the graph laplacian is replaced with a Gaussian kernel
    
    Args:
        X: Input points (torch.Tensor), shape (N, D)
        dim: Target embedding dimension
        sigma: Bandwidth for Gaussian kernel
        t: Diffusion time parameter
    
    Returns:
        torch.Tensor: Embedded points, shape (N, dim)
    """
    X_i = NumpyLazyTensor(X[:, None, :].cpu().numpy().astype(np.float32))
    X_j = NumpyLazyTensor(X[None, :, :].cpu().numpy().astype(np.float32))

    D_ij = ((X_i - X_j) ** 2).sum(dim=-1)
    K_ij = (-D_ij / (2 * sigma ** 2)).exp()

    K = aslinearoperator(K_ij)
    D = K @ np.ones(X.shape[0], dtype=np.float32)

    D_2 = aslinearoperator(diags(1 / np.sqrt(D)))
    L_norm = aslinearoperator(diags(np.ones_like(D))) - D_2 @ K @ D_2
    L_norm.dtype = np.dtype(np.float32)

    eigenvalues, eigenvectors = eigsh(L_norm, k=dim, which="SM")
    U = eigenvectors * np.exp((-t * eigenvalues) / 2)

    return torch.tensor(U, dtype=torch.float32, device=X.device).contiguous()


def geodesic_euclidean_embedding(mesh, dim=8, n_vertices_sub=500, n_jobs=-1):
    """
    Embed mesh into Euclidean space preserving geodesic distances.
    
    It decimates the initial mesh, computes exact geodesic distance on the coarse mesh, apply MDS on the coarse geodesic matrix, and interpolates
    euclidean embedding on the fine mesh using quadratic interpolation.
    
    Args:
        mesh: Dictionary with 'pos' (vertices, torch.tensor of shape (N,3)) and 'face' (triangles, torch.tensor of of shape (F,3))
        dim: Target embedding dimension
        n_vertices_sub: Number of vertices to subsample for MDS
        n_jobs: Number of parallel jobs for MDS (-1 for all cores)
    
    Returns:
        torch.Tensor: Embedded vertex positions, shape (N, dim)
    """
    def solve_quad_with_fixed(Q, inds, vals):
        """Solve the quadratic problem with fixed values:   min_x x^T Q x s.t. x[inds] = vals"""
        N = inds.size
        subsample_mat = sparse.csc_matrix((np.ones(N), (np.arange(N), inds)), shape=(N, Q.shape[0]))
        Q_aug = sparse.block_array([[Q, subsample_mat.T], [subsample_mat, None]], format="csc")
        b_aug = np.concatenate([np.zeros((Q.shape[0], vals.shape[1])), vals], axis=0)

        return sparse.linalg.spsolve(Q_aug, b_aug)[: Q.shape[0]]

    sub_inds, sub_mesh = decimate_mesh(mesh, n_vertices_sub)

    unique_subs, unique_indices = np.unique(sub_inds, return_index=True)
    n_subs = unique_subs.shape[0]

    pos_np, face_np = sub_mesh['pos'].cpu().numpy(), sub_mesh['face'].cpu().numpy()
    heat_solver = pp3d.MeshHeatMethodDistanceSolver(pos_np, face_np)

    geod_sub = np.array([heat_solver.compute_distance(unique_indices[i])[unique_indices] for i in range(n_subs)])
    geod_sub = 0.5 * (geod_sub + geod_sub.T)


    myMDS = MDS(n_components=dim, n_init=4, dissimilarity="precomputed", max_iter=1000, n_jobs=n_jobs)
    emb1 = myMDS.fit_transform(geod_sub)

    L = pp3d.cotan_laplacian(mesh['pos'].cpu().numpy().astype('float64'), mesh['face'].cpu().numpy())
    Minv = sparse.diags(1 / lumped_mass(mesh).cpu().numpy())

    emb_final = solve_quad_with_fixed(L @ Minv @ L, sub_inds[unique_indices].cpu().numpy(), emb1, )
    return torch.tensor(emb_final, dtype=torch.float32).contiguous()
