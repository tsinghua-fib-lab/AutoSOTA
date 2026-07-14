#!/usr/bin/env python3
"""
Synthetic ellipsoid dataset generator for testing equivariant vs non-equivariant GNNs.

This script creates datasets where:
- Training set: ellipsoids biased toward having diameter along x-axis
- Test set: ellipsoids with diameter along different axes (y or z)

An equivariant model should generalize well, while non-equivariant models
that treat coordinates as separate scalar features should fail.

Example run:
! python3.11 /home/davejohnson/Research/vdw/code/theory_verification/ellipsoid_dataset.py \
    --save_dir "/home/davejohnson/Research/vdw/data/ellipsoids/" \
    --config /home/davejohnson/Research/vdw/code/config/yaml_files/ellipsoid_vdw_modular_minimal.yaml \
    --pq_h5_name pq_tensor_data.h5 \
    --random_seed 457892 \
    --num_samples 512 \
    --num_oversample_points 1024 \
    --abc_means 3.0 1.0 1.0 \
    --abc_stdevs 0.5 0.2 0.2 \
    --local_pca_kernel_fn gaussian \
    --laplacian_type sym_norm \
    --k_laplacian 10 \
    --num_nodes_per_graph 128 \
    --knn_graph_k 5 \
    --harmonic_bands 1-3,6-8,10-12 \
    --harmonic_band_weights 1.0 1.5 2.0 \
    --global_band_index 1 \
    --random_harmonic_k 16 \
    --random_harmonic_coeff_bounds 1.0 2.0 \
    --random_harmonic_smoothing_window 1 \
    --modulation_scale 0.9 \
    --sing_vect_align_method column_dot

----------------------------------------------------------------
LaTeX summary of the ellipsoid dataset and regression tasks
----------------------------------------------------------------
\section*{Graph-Level Regression Target}

\subsection*{Diameter}

For points $\{\mathbf{x}_i\}_{i=1}^N$ sampled from an ellipsoid surface, the diameter is
\[
\mathrm{diam} = \max_{i,j} \lVert \mathbf{x}_i - \mathbf{x}_j \rVert.
\]


\section*{Node-Level Regression Targets}

Let $f(\mathbf{x}) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1$ define the ellipsoid with semi-axes $(a,b,c)$. Let $\{\phi_k\}_{k\ge 0}$ be Laplacian eigenvectors of the $k$-NN graph built on the (optionally over-sampled) surface points. We use either the symmetric normalized or the unnormalized Laplacian.

\subsection*{Base normals}
Outward unit normals are
\[
\mathbf{n}_i = \frac{\nabla f(\mathbf{x}_i)}{\lVert \nabla f(\mathbf{x}_i) \rVert},\qquad \nabla f(\mathbf{x}) = \Big(\tfrac{x}{a^2},\tfrac{y}{b^2},\tfrac{z}{c^2}\Big).
\]

\subsection*{Global-harmonic modulated normals}
Let $k_g$ be a user-selected eigenvector index (typically the first non-trivial mode). Define the raw scalar modulation and its rescaled version
\[
g_i^{(\mathrm{glob})} = \phi_{k_g}(i),\quad \hat g_i^{(\mathrm{glob})} = a\,\frac{g_i^{(\mathrm{glob})}}{\max_j |g_j^{(\mathrm{glob})}|},\quad \mathbf{n}_i^{(\mathrm{glob})} = \big(1 + \hat g_i^{(\mathrm{glob})}\big)\,\mathbf{n}_i.
\]
Choosing $a\in(0,1]$ (default $a=0.9$) guarantees $1 + \hat g_i^{(\mathrm{glob})} \ge 1 - a > 0$, preventing sign flips while allowing magnitude variation.

\subsection*{Multiscale-harmonic modulated normals}
For band ranges $\mathcal{B}_b = \{k: \ell_b \le k \le h_b\}$, define per-band averages, optional positive weights $w_b>0$, and the combined modulation
\[
e_i^{(b)} = \frac{1}{|\mathcal{B}_b|}\sum_{k\in\mathcal{B}_b} \phi_k(i),\qquad g_i^{(\mathrm{ms})} = \sum_b w_b\, e_i^{(b)},\qquad \hat g_i^{(\mathrm{ms})} = a\,\frac{g_i^{(\mathrm{ms})}}{\max_j |g_j^{(\mathrm{ms})}|},\qquad \mathbf{n}_i^{(\mathrm{ms})} = \big(1 + \hat g_i^{(\mathrm{ms})}\big)\,\mathbf{n}_i.
\]
If no weights are provided, $w_b$ defaults to 1 for all bands. The rescaling with $a\in(0,1]$ (default $a=0.9$) ensures $1 + \hat g_i^{(\mathrm{ms})} > 0$.

\subsection*{Random-harmonic modulated normals}
Pick $K$ nontrivial eigenvectors $\{\phi_{k}\}_{k=1}^{K}$ (omitting the trivial constant mode). Sample coefficients $c_j \sim \mathcal{U}(L,U)$ and optionally apply a moving-average smoothing with window size $w$ (average across $2w+1$ neighboring indices, truncated at boundaries) to obtain $\tilde{c}_j$. Define
\[
g_i^{(\mathrm{rand})} = \sum_{j=1}^{K} \tilde{c}_j\, \phi_{j}(i),\qquad \hat g_i^{(\mathrm{rand})} = a\,\frac{g_i^{(\mathrm{rand})}}{\max_j |g_j^{(\mathrm{rand})}|},\qquad \mathbf{n}_i^{(\mathrm{rand})} = \big(1 + \hat g_i^{(\mathrm{rand})}\big)\,\mathbf{n}_i.
\]
The $+1$ keeps directions aligned with base normals, and the rescaling avoids direction inversions and zero vectors when $a<1$. This target mirrors the multiscale construction but uses randomly sampled, smoothed coefficients over the first $K$ nontrivial modes. The modulation scale $a\in(0,1]$ (default $a=0.9$) is configurable via command-line argument.

\subsection*{Spectral vector field}
Given vector coefficients $\{\mathbf{a}_k\}_{k=1}^K\subset\mathbb{R}^3$,
\[
\mathbf{v}_i = \sum_{k=1}^K \phi_k(i)\,\mathbf{a}_k \in \mathbb{R}^3.
\]
By default, coefficients are drawn by sampling scalar weights and multiplying a randomly rotated reference direction; optionally, users may provide an explicit $(K,3)$ matrix of coefficients.

\subsection*{Equivariance and eigenvector ambiguities}
Under a rigid rotation $R\in\mathrm{SO}(3)$, normals transform equivariantly $\mathbf{n}_i' = R\mathbf{n}_i$. The scalar modulations $g_i$ are intrinsic to the graph geometry. Hence the modulated normals rotate as $\mathbf{n}_i^{(\cdot)\,\prime} = R\,\mathbf{n}_i^{(\cdot)}$ provided the same scalar fields are reused. Individual eigenvectors are defined up to a sign (and arbitrary mixing within degenerate subspaces), so band-mean modulations are sign-sensitive. Energy-style, sign-invariant alternatives (e.g., RMS within a band) can be used if strict rotation-consistent magnitudes are required.

\subsection*{Summary table (node-level)}
\begin{center}
\begin{tabular}{l l l}
\toprule
Target Type & Symbol & Dimensionality \\
\midrule
Base normal & $\mathbf{n}_i$ & $\mathbb{R}^3$ \\
Global-harmonic modulated normal & $\mathbf{n}_i^{(\mathrm{glob})}$ & $\mathbb{R}^3$ \\
Multiscale-harmonic modulated normal & $\mathbf{n}_i^{(\mathrm{ms})}$ & $\mathbb{R}^3$ \\
Random-harmonic modulated normal & $\mathbf{n}_i^{(\mathrm{rand})}$ & $\mathbb{R}^3$ \\
Spectral vector field & $\mathbf{v}_i$ & $\mathbb{R}^3$ \\
\bottomrule
\end{tabular}
\end{center}

\subsection*{Intuition}
These targets combine geometric information (surface normals) with spectral structure from the graph Laplacian, yielding direction-preserving fields whose magnitudes vary with global and multiscale harmonic content, and a smooth spectral vector field. Rescaling the scalar modulations to a bounded range ensures non-inverting normal magnitudes while preserving relative variation, supporting evaluation of frequency-aware representations and rotational equivariance in GNNs.

"""
import sys
sys.path.append('../')
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from typing import Dict, Tuple, List, Optional, Callable, Literal
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import pickle
import yaml
import h5py
# Efficient pairwise distance function
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
# Custom classes
from data_processing.ellipsoid_data_classes import EllipsoidDataset, EllipsoidPointCloudGenerator


def compute_graph_laplacian(
    coords: np.ndarray,
    k: int = 5,
    laplacian_type: Literal["sym_norm", "unnorm"] = "sym_norm",
) -> sp.spmatrix:
    """
    Build the graph Laplacian for a k-NN graph from coords and 
    return a sparse matrix.
    Args:
        coords: (N, 3) array of coordinates
        k: number of nearest neighbors
        laplacian_type: type of Laplacian to compute
    Returns:
        L: (N, N) sparse Laplacian matrix
    """
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k+1)  # +1 to include self
    rows, cols = [], []
    for i, neigh in enumerate(idxs):
        for j in neigh[1:]:  # skip self
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows))
    W = sp.coo_matrix((data, (rows, cols)), shape=(len(coords), len(coords)))
    W = ((W + W.T) > 0).astype(float)  # symmetrize

    deg = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(deg)

    if laplacian_type == "sym_norm":
        # Symmetric normalized Laplacian
        deg_inv_sqrt = sp.diags(1.0 / np.sqrt(deg + 1e-10))
        L = sp.eye(W.shape[0]) - deg_inv_sqrt @ W @ deg_inv_sqrt
    else:
        # Unnormalized L = D - A
        L = D - W

    return L.tocsr()


def compute_ellipsoid_normals(
    coords: np.ndarray, 
    a: float, 
    b: float, 
    c: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute normalized outward normals for points on an ellipsoid.
    Equation: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    Gradient (= unnormalized normal vector) = (x/a^2, y/b^2, z/c^2).
    Args:
        coords: (N, 3) array of coordinates
        a, b, c: Ellipsoid axis scale parameters
        normalize: whether to normalize the normals to unit vectors
    Returns:
        normals: (N, 3) array of outward unit normal vectors
    """
    grads = torch.stack([
        coords[:, 0] / (a ** 2),
        coords[:, 1] / (b ** 2),
        coords[:, 2] / (c ** 2)
    ], dim=-1)
    if normalize:
        normals = F.normalize(grads, dim=-1)  # outward
    else:
        normals = grads
    return normals


def rescale_modulation_to_bounded_range(
    modulation: Tensor,
    scale_a: float = 0.9,
) -> Tensor:
    """
    Rescale a modulation tensor to m' = a * m / max(abs(m)).

    With a in (0, 1], adding 1 ensures 1 + m' >= 1 - a > 0, preventing sign flips.
    For example, if a = 0.9, then 1 + m' \in [1 - 0.9, 1 + 0.9] = [0.1, 1.9].

    Args:
        modulation: Tensor of shape (N, 1) or broadcastable to that.
        scale_a: Maximum absolute magnitude after scaling.

    Returns:
        Tensor with the same shape as input, rescaled to have max absolute value <= a.
    """
    max_abs = torch.amax(torch.abs(modulation))
    if torch.isclose(max_abs, torch.tensor(0.0, dtype=modulation.dtype, device=modulation.device)):
        return torch.zeros_like(modulation)
    return (float(scale_a) * modulation) / max_abs


def generate_ellipsoid_vector_targets(
    list_of_coords: List[Tensor],
    ellipsoid_params: List[Tuple[float, float, float]],
    k_neighbors: int = 5,
    laplacian_type: Literal["sym_norm", "unnorm"] = "sym_norm",
    harmonic_bands: Optional[List[Tuple[int, int]]] = [(1, 3), (4, 6)],  # bands = (low_idx, high_idx) inclusive
    global_band_index: int = 1, # lowest-frequency band for global modulation
    spectral_coefficients: Optional[List[float]] = None, # list of floats for spectral vector field
    max_num_eigenvectors: Optional[int] = None,
    harmonic_band_weights: Optional[List[float]] = None,
    random_harmonic_num_nontriv_evecs: int = 16,
    random_harmonic_coeff_bounds: Tuple[float, float] = (1.0, 2.0),
    random_harmonic_smoothing_window: int = 1,
    modulation_scale: float = 0.9,
) -> List[Dict[str, Tensor]]:
    """
    Generate multiple regression targets for each ellipsoid's nodes.
    
    Args:
        list_of_coords: list of tensors [N_i, 3], each ellipsoid's 
            points.
        ellipsoid_params: list of (a, b, c) tuples matching 
            list_of_coords.
        k_neighbors: k in k-NN for Laplacian construction.
        laplacian_type: 'sym_norm' or 'unnorm'.
        harmonic_bands: list of (low, high) index ranges for 
            multiscale modulation.
        harmonic_band_weights: optional positive weights per band; if
            provided, must have the same length as 'harmonic_bands'.
        global_band_index: index for global harmonic modulation.
        spectral_coefficients: list of floats for spectral vector 
            field. If None, random.
        max_num_eigenvectors: if provided, compute only this many 
            smallest-magnitude eigenpairs using sparse eigensolver. 
            Otherwise, infer minimal count needed from bands and 
            global index. Falls back to dense eigh if k >= N.
        modulation_scale: scale parameter 'a' for normal vector modulation.
            Controls the maximum absolute magnitude of modulation while
            guaranteeing no direction flips. Must be in (0, 1].

    Returns:
        results: list of dicts with keys:
            'base_normals' : [N, 3]
            'global_harmonic_normals' : [N, 3]
            'multiscale_harmonic_normals' : [N, 3]
            'random_harmonic_normals' : [N, 3]
            'spectral_vector_field' : [N, 3]
    """
    # Validate modulation_scale parameter
    if not (0 < modulation_scale <= 1):
        raise ValueError(f"modulation_scale must be in (0, 1], got {modulation_scale}")
    
    results = []

    for coords, (a, b, c) in zip(list_of_coords, ellipsoid_params):
        # Compute base normals on ellipsoid surface
        base_normals = compute_ellipsoid_normals(coords, a, b, c)

        # Compute Laplacian
        L = compute_graph_laplacian(
            coords.numpy(), 
            k=k_neighbors, 
            laplacian_type=laplacian_type
        ) 
        num_nodes = L.shape[0]
        # Determine how many eigenpairs we need
        if max_num_eigenvectors is not None:
            k_needed = int(max_num_eigenvectors)
        else:
            # Determine needs from bands (highest index), global index, and random-harmonic K (uses indices 1..K)
            max_band = max((high for (_, high) in (harmonic_bands or [])), default=0)
            highest_index_needed = max(max_band, int(global_band_index), int(random_harmonic_num_nontriv_evecs))
            # Need count up to and including highest_index_needed, plus the trivial/zero-mode
            k_needed = highest_index_needed + 1
        k_needed = max(1, min(k_needed, num_nodes))

        # Compute eigenpairs efficiently
        if k_needed < num_nodes:
            try:
                eigvals, eigvecs = eigsh(L, k=k_needed, which='SM')
            except Exception:
                # Fallback to dense if sparse fails
                evals, evecs = np.linalg.eigh(L.toarray())
                eigvals, eigvecs = evals[:k_needed], evecs[:, :k_needed]
        else:
            evals, evecs = np.linalg.eigh(L.toarray())
            eigvals, eigvecs = evals, evecs

        # Ensure ascending order
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # --- Global harmonic modulation ---
        global_mode = torch.tensor(
            eigvecs[:, global_band_index:global_band_index+1], 
            dtype=torch.float32
        )  # (N, 1)
        global_mode_scaled = rescale_modulation_to_bounded_range(global_mode, scale_a=modulation_scale)
        global_harmonic_normals = base_normals * (1 + global_mode_scaled)

        # --- Multiscale harmonic modulation ---
        multiscale_modulation = torch.zeros(coords.shape[0], 1)
        # Validate and prepare weights
        if harmonic_band_weights is not None:
            if len(harmonic_band_weights) != len(harmonic_bands):
                raise ValueError(
                    f"harmonic_band_weights length {len(harmonic_band_weights)} must match number of bands {len(harmonic_bands)}"
                )
            weights_tensor = torch.tensor(harmonic_band_weights, dtype=torch.float32)
            if torch.any(weights_tensor <= 0):
                raise ValueError("All harmonic_band_weights must be positive")
        else:
            weights_tensor = None

        if harmonic_bands is not None:
            for band_idx, (low, high) in enumerate(harmonic_bands):
                band_energy = torch.tensor(
                    eigvecs[:, low:high+1], dtype=torch.float32
                ).mean(dim=1, keepdim=True)
                if weights_tensor is not None:
                    multiscale_modulation += weights_tensor[band_idx] * band_energy
                else:
                    multiscale_modulation += band_energy
        multiscale_modulation_scaled = rescale_modulation_to_bounded_range(multiscale_modulation, scale_a=modulation_scale)
        multiscale_harmonic_normals = base_normals * (1 + multiscale_modulation_scaled)

        # --- Random-harmonic modulated normals ---
        # Use first K nontrivial eigenvectors (skip index 0)
        num_modes_total = eigvecs.shape[1]
        num_nontrivial_available = max(0, num_modes_total - 1)
        k_rand = min(max(0, int(random_harmonic_num_nontriv_evecs)), num_nontrivial_available)
        if k_rand > 0:
            nontriv = torch.tensor(eigvecs[:, 1:1 + k_rand], dtype=torch.float32)  # (N, k_rand)
            low, high = float(random_harmonic_coeff_bounds[0]), float(random_harmonic_coeff_bounds[1])
            coeffs = low + (high - low) * torch.rand(k_rand, dtype=torch.float32)
            # Moving-average smoothing with window w: average across [j-w, j+w] truncated to [0, k_rand-1]
            w = max(0, int(random_harmonic_smoothing_window))
            if w > 0:
                smoothed = torch.empty_like(coeffs)
                for j in range(k_rand):
                    lo = max(0, j - w)
                    hi = min(k_rand - 1, j + w)
                    smoothed[j] = coeffs[lo:hi + 1].mean()
                coeffs = smoothed
            rand_modulation = nontriv @ coeffs.view(-1, 1)  # (N, 1)
        else:
            rand_modulation = torch.zeros(coords.shape[0], 1, dtype=torch.float32)
        rand_modulation_scaled = rescale_modulation_to_bounded_range(rand_modulation, scale_a=modulation_scale)
        random_harmonic_normals = base_normals * (1 + rand_modulation_scaled)

        # --- Spectral vector field ---
        num_nodes = coords.shape[0]
        num_modes = eigvecs.shape[1]
        
        if spectral_coefficients is None:
            # Start with base coefficients of shape (num_modes,)
            base_coeffs = torch.rand(num_modes, dtype=torch.float32)
            
            # Create a reference direction [1, 0, 0] and apply random rotation
            # to avoid trivial alignment with coordinate axes
            ref_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            
            # Generate a random rotation matrix (3x3 orthogonal matrix)
            # Using QR decomposition of random matrix to ensure orthogonality
            random_matrix = torch.randn(3, 3, dtype=torch.float32)
            Q, _ = torch.linalg.qr(random_matrix)
            # Ensure proper rotation (det = 1) by flipping sign if needed
            if torch.det(Q) < 0:
                Q[:, -1] *= -1
            
            # Apply rotation to reference direction
            rotated_direction = Q @ ref_direction
            
            # Use outer product to create 3D coefficients: (num_modes, 3)
            coeffs = torch.outer(base_coeffs, rotated_direction)
        else:
            # User provided coefficients should be (num_modes, 3)
            coeffs = torch.tensor(spectral_coefficients, dtype=torch.float32)
            if coeffs.shape != (num_modes, 3):
                raise ValueError(f"Expected spectral_coefficients of shape ({num_modes}, 3), got {coeffs.shape}")

        # Compute vector field: (num_nodes, num_modes) @ (num_modes, 3) = (num_nodes, 3)
        spectral_vector_field = torch.tensor(eigvecs, dtype=torch.float32) @ coeffs

        results.append({
            "base_normals": base_normals,
            "global_harmonic_normals": global_harmonic_normals,
            "multiscale_harmonic_normals": multiscale_harmonic_normals,
            "random_harmonic_normals": random_harmonic_normals,
            "spectral_vector_field": spectral_vector_field
        })

    return results


def create_biased_ellipsoid_datasets(
    num_samples: int = 1024,
    num_points: int = 64,
    k: int = 5,
    bias_axis: str = 'x',
    long_axis_params: Tuple[float, float] = (3.0, 0.5),
    short_axis_params: Tuple[float, float] = (1.0, 0.2),
    random_state_seed: int = 457892,
    oversample_factor: float = 1.0,
    num_oversample_points: Optional[int] = None,
    k_laplacian: Optional[int] = None,
    laplacian_type: Literal["sym_norm", "unnorm"] = "sym_norm",
    add_vector_targets: bool = True,
    harmonic_bands: Optional[List[Tuple[int, int]]] = [(1, 3), (4, 6)],
    harmonic_band_weights: Optional[List[float]] = None,
    global_band_index: int = 1,
    spectral_coefficients: Optional[List[float]] = None,
    max_num_eigenvectors: Optional[int] = None,
    node_target_prefix: str = 'y_node_',
    abc_means: Optional[Tuple[float, float, float]] = None,
    abc_stdevs: Optional[Tuple[float, float, float]] = None,
    random_harmonic_k: int = 16,
    random_harmonic_coeff_bounds: Tuple[float, float] = (1.0, 2.0),
    random_harmonic_smoothing_window: int = 1,
    modulation_scale: float = 0.9,
    dirac_types: Optional[List[str]] = None,
) -> EllipsoidDataset:
    """
    Create a single ellipsoid dataset with biased shapes.
    
    Args:
        num_samples: Total number of ellipsoid samples to create
        num_points: Number of points to sample per ellipsoid
        k: Number of nearest neighbors for k-NN graph construction
        bias_axis: Axis to bias ellipsoids toward ('x', 'y', or 'z')
        long_axis_params: Parameters for the long axis (mean, std)
        short_axis_params: Parameters for the short axes (mean, std)
        random_state_seed: Random seed for reproducibility
        oversample_factor: Factor to oversample points before subsampling
        num_oversample_points: Explicit number of oversampled points per ellipsoid
        k_laplacian: k for Laplacian eigen computation on oversampled graphs
        laplacian_type: Type of Laplacian for spectral computations
        add_vector_targets: Whether to add spectral vector targets
        harmonic_bands: List of (low, high) index ranges for multiscale modulation
        harmonic_band_weights: Optional positive weights per band
        global_band_index: Index for global harmonic modulation
        spectral_coefficients: List of floats for spectral vector field
        max_num_eigenvectors: Number of eigenpairs to compute
        node_target_prefix: Prefix for node vector target keys
        abc_means: Explicit means for (a, b, c) semi-axes
        abc_stdevs: Explicit standard deviations for (a, b, c) semi-axes
        random_harmonic_k: Number of nontrivial eigenvectors for random harmonic modulation
        random_harmonic_coeff_bounds: Bounds for random harmonic coefficients
        random_harmonic_smoothing_window: Smoothing window for random coefficients over modes (averages over 2w+1 window, truncated at boundaries). Default: 1
        modulation_scale: Scale parameter 'a' for normal vector modulation.
            Controls the maximum absolute magnitude of modulation while
            guaranteeing no direction flips. Must be in (0, 1].
        
    Returns:
        EllipsoidDataset containing all samples
    """
    # Set random seed for reproducibility
    np.random.seed(random_state_seed)
    torch.manual_seed(random_state_seed)
    
    # Initialize point cloud generator
    generator = EllipsoidPointCloudGenerator(random_state_seed=random_state_seed)
    
    # Define shape parameters: prefer explicit abc means/stdevs if provided
    def get_shape_dict(
        bias_axis: str,
        abc_means: Optional[Tuple[float, float, float]],
        abc_stdevs: Optional[Tuple[float, float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        if abc_means is not None or abc_stdevs is not None:
            if abc_means is None or abc_stdevs is None:
                raise ValueError("Both --abc_means and --abc_stdevs must be provided together")
            if len(abc_means) != 3 or len(abc_stdevs) != 3:
                raise ValueError("--abc_means and --abc_stdevs must each provide exactly three values: a b c")
            return {
                'a': (float(abc_means[0]), float(abc_stdevs[0])),
                'b': (float(abc_means[1]), float(abc_stdevs[1])),
                'c': (float(abc_means[2]), float(abc_stdevs[2])),
            }
        # Fall back to biased long/short axis sampling
        if bias_axis == 'x':
            return {'a': long_axis_params, 'b': short_axis_params, 'c': short_axis_params}
        if bias_axis == 'y':
            return {'a': short_axis_params, 'b': long_axis_params, 'c': short_axis_params}
        if bias_axis == 'z':
            return {'a': short_axis_params, 'b': short_axis_params, 'c': long_axis_params}
        raise ValueError(f"Invalid bias_axis: {bias_axis}. Must be 'x', 'y', or 'z'.")

    # Generate ellipsoid point clouds
    shape_dict = get_shape_dict(bias_axis, abc_means, abc_stdevs)

    # Determine oversampled size
    if num_oversample_points is not None:
        num_points_oversampled = int(num_oversample_points)
    else:
        num_points_oversampled = int(max(1, round(num_points * oversample_factor)))

    # Use separate k for Laplacian if provided
    k_for_laplacian = k_laplacian if k_laplacian is not None else k

    # Generate oversampled point clouds and keep true parameters
    oversampled_point_clouds, ellipsoid_params = generator.generate_ellipsoid_points_with_params(
        num_ellipsoids=num_samples,
        num_points=num_points_oversampled,
        shape_dict=shape_dict,
        random_state_seed=random_state_seed
    )
    
    # Compute graph-level targets (e.g., diameter)
    graph_level_targets: List[Dict[str, Tensor]] = []
    for points in oversampled_point_clouds:
        diameter_value = generator.compute_ellipsoid_diameter(points)
        graph_level_targets.append(
            {'diameter': torch.tensor([diameter_value], dtype=torch.float32)}
            )

    # Optionally compute spectral vector targets on oversampled sets
    precomputed_node_vector_targets: Optional[List[Dict[str, Tensor]]] = None
    if add_vector_targets:
        list_of_coords = [torch.tensor(pc, dtype=torch.float32) for pc in oversampled_point_clouds]
        vector_targets_all = generate_ellipsoid_vector_targets(
            list_of_coords=list_of_coords,
            ellipsoid_params=ellipsoid_params,
            k_neighbors=k_for_laplacian,
            laplacian_type=laplacian_type,
            harmonic_bands=harmonic_bands,
            harmonic_band_weights=harmonic_band_weights,
            global_band_index=global_band_index,
            spectral_coefficients=spectral_coefficients,
            max_num_eigenvectors=max_num_eigenvectors,
            random_harmonic_num_nontriv_evecs=random_harmonic_k,
            random_harmonic_coeff_bounds=random_harmonic_coeff_bounds,
            random_harmonic_smoothing_window=random_harmonic_smoothing_window,
            modulation_scale=modulation_scale,
        )
    else:
        vector_targets_all = None

    # Subsample to create final graphs, and carry corresponding vector targets
    rng = np.random.RandomState(random_state_seed)
    point_clouds_final: List[np.ndarray] = []
    precomputed_list: List[Dict[str, Tensor]] = []
    for idx in range(num_samples):
        pc_over = oversampled_point_clouds[idx]
        n_over = pc_over.shape[0]
        if num_points > n_over:
            raise ValueError(f"Requested final num_points={num_points} exceeds oversampled size={n_over}")
        sel_idx = rng.choice(n_over, size=num_points, replace=False)
        pc_final = pc_over[sel_idx]
        point_clouds_final.append(pc_final)

        if add_vector_targets and vector_targets_all is not None:
            vt = vector_targets_all[idx]
            precomputed_list.append({
                'base_normals': vt['base_normals'][sel_idx],
                'global_harmonic_normals': vt['global_harmonic_normals'][sel_idx],
                'multiscale_harmonic_normals': vt['multiscale_harmonic_normals'][sel_idx],
                'random_harmonic_normals': vt['random_harmonic_normals'][sel_idx],
                'spectral_vector_field': vt['spectral_vector_field'][sel_idx],
            })

    if add_vector_targets:
        precomputed_node_vector_targets = precomputed_list
    else:
        precomputed_node_vector_targets = None
    
    # Create dataset
    dataset = EllipsoidDataset(
        point_clouds=point_clouds_final,
        graph_level_targets=graph_level_targets,
        k=k,
        random_seed=random_state_seed,
        node_vector_targets=precomputed_node_vector_targets,
        node_target_prefix=node_target_prefix,
        dirac_types=dirac_types,
    )
    return dataset


def visualize_ellipsoid_dataset(
    dataset: EllipsoidDataset,
    num_samples: int = 5,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize sample ellipsoids from the dataset.
    
    Args:
        dataset: EllipsoidDataset to visualize
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    for i in range(min(num_samples, len(dataset))):
        data = dataset[i]
        points = data.pos.numpy()
        diameter = data.y.item()
        
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.6)
        ax.set_title(f'Sample {i+1}\nDiameter: {diameter:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()


def average_neighbor_distance(data: Data) -> float:
    # Get positions of source and target nodes
    pos = data.pos  # shape (N, d)
    row, col = data.edge_index  # edge_index is [2, E]
    
    src_pos = pos[row]  # shape (E, d)
    dst_pos = pos[col]  # shape (E, d)
    
    # Compute Euclidean distances between each pair
    distances = torch.norm(src_pos - dst_pos, dim=1)  # shape (E,)
    
    # Take mean distance
    return distances.mean().item()


def analyze_dataset_statistics(
    dataset: EllipsoidDataset, 
    name: str = "Dataset",
    return_mean_of_mean_distances: bool = True
) -> Optional[float]:
    """
    Analyze and print statistics about the dataset.
    
    Args:
        dataset: EllipsoidDataset to analyze
        name: Name of the dataset for printing
    """
    diameters = [data.y.item() for data in dataset]
    diameters = np.array(diameters)
    
    print(f"\n{name} Statistics:")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Number of points per sample: {dataset[0].pos.shape[0]}")
    print(f"  k-NN parameter: {dataset.k}")
    print(f"  Diameter statistics:")
    print(f"    Mean: {diameters.mean():.3f}")
    print(f"    Std:  {diameters.std():.3f}")
    print(f"    Min:  {diameters.min():.3f}")
    print(f"    Max:  {diameters.max():.3f}")

    # Analyze distance distributions
    mean_distances = [average_neighbor_distance(data) for data in dataset]
    mean_of_mean_distances = np.mean(mean_distances)
    print(f"  Mean distance statistics:")
    print(f"    Mean of means: {mean_of_mean_distances}")
    print(f"    Std of means:  {np.std(mean_distances):.3f}")
    print(f"    Min of means:  {np.min(mean_distances):.3f}")
    print(f"    Max of means:  {np.max(mean_distances):.3f}")
    
    # Analyze coordinate distributions
    all_points = torch.cat([data.pos for data in dataset], dim=0).numpy()
    print(f"  Coordinate ranges:")
    print(f"    X: [{all_points[:, 0].min():.2f}, {all_points[:, 0].max():.2f}]")
    print(f"    Y: [{all_points[:, 1].min():.2f}, {all_points[:, 1].max():.2f}]")
    print(f"    Z: [{all_points[:, 2].min():.2f}, {all_points[:, 2].max():.2f}]")

    if return_mean_of_mean_distances:
        return mean_of_mean_distances
    else:
        return None


def parse_args():
    """Parse command line arguments for dataset creation."""
    parser = argparse.ArgumentParser(
        description="Create biased ellipsoid dataset for equivariance testing"
    )
    # Dataset size parameters
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1024,
        help="Total number of samples to create (default: 1024)"
    )
    parser.add_argument(
        "--num_nodes_per_graph", 
        type=int, 
        default=64,
        help="Number of points per ellipsoid (default: 64)"
    )

    # Explicit (a,b,c) sampling controls
    parser.add_argument(
        "--abc_means",
        type=float,
        nargs=3,
        default=None,
        help="Means for (a, b, c) semi-axes; overrides bias presets when provided"
    )
    parser.add_argument(
        "--abc_stdevs",
        type=float,
        nargs=3,
        default=None,
        help="Standard deviations for (a, b, c) semi-axes; overrides bias presets when provided"
    )
    # Graph construction parameters
    parser.add_argument(
        "--knn_graph_k", 
        type=int, 
        default=5,
        help="Number of nearest neighbors for k-NN graph (default: 5)"
    )
    # Laplacian graph k (can differ from final graph k)
    parser.add_argument(
        "--k_laplacian",
        type=int,
        default=None,
        help="k for Laplacian eigen computation on oversampled graphs (default: same as --k)"
    )
    # Oversampling controls
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=1.0,
        help="Factor to oversample points before subsampling to --num_nodes_per_graph (default: 1.0 = no oversampling)"
    )
    parser.add_argument(
        "--num_oversample_points",
        type=int,
        default=None,
        help="Explicit number of oversampled points per ellipsoid (overrides --oversample_factor if set)"
    )
    # Laplacian type
    parser.add_argument(
        "--laplacian_type",
        type=str,
        choices=["sym_norm", "unnorm"],
        default="sym_norm",
        help="Type of Laplacian for spectral computations (default: sym_norm)"
    )
    parser.add_argument(
        "--column_normalize_P",
        action='store_true',
        help="Column-normalize the P diffusion operator"
    )
    # Local PCA kernel parameters
    parser.add_argument(
        "--local_pca_kernel_fn",
        type=str,
        default='gaussian',
        help="Local PCA kernel function (default: gaussian)"
    )
    parser.add_argument(
        "--local_pca_kernel_scale_param",
        type=float,
        default=None,
        help="Scale parameter for local PCA kernel function"
    )
    parser.add_argument(
        "--use_mean_recentering",
        action='store_true',
        help="Use mean-recentering for local PCA kernel (exclude this flag to use reference point recentering)"
    )
    parser.add_argument(
        "--sing_vect_align_method",
        type=str,
        default='column_dot',
        choices=['column_dot', 'procrustes'],
        help="Method to align the singular vectors of O_i and O_j"
    )
    # Bias parameters
    parser.add_argument(
        "--bias_axis", 
        type=str, 
        default='x',
        choices=['x', 'y', 'z'],
        help="Axis to bias ellipsoids toward (default: x)"
    )
    # Vector targets from spectral fields
    parser.add_argument(
        "--no_vector_targets",
        action='store_true',
        help="Disable adding new spectral/normal vector node targets"
    )
    parser.add_argument(
        "--node_target_prefix",
        type=str,
        default='y_',
        help="Prefix for keys of added node vector targets (default: y_)"
    )
    parser.add_argument(
        "--harmonic_bands",
        type=str,
        default=None,
        help="Comma-separated band ranges like '1-3,4-6' for multiscale modulation (default: internal)"
    )
    parser.add_argument(
        "--harmonic_band_weights",
        type=float,
        nargs='+',
        default=None,
        help="Optional positive weights (one per band) to scale each band's contribution in multiscale modulation; must match number of bands"
    )
    # Random-harmonic normals parameters
    parser.add_argument(
        "--random_harmonic_k",
        type=int,
        default=16,
        help="Number of nontrivial eigenvectors to use for random harmonic-modulated normals (default: 16)"
    )
    parser.add_argument(
        "--random_harmonic_coeff_bounds",
        type=float,
        nargs=2,
        default=[1.0, 2.0],
        help="Lower and upper bounds for uniform sampling of coefficients for random harmonic-modulated normals (default: 1.0 2.0)"
    )
    parser.add_argument(
        "--random_harmonic_smoothing_window",
        type=int,
        default=1,
        help="Smoothing window w for moving-average of random coefficients over modes (averages over 2w+1 window, truncated at boundaries). Default: 1"
    )
    parser.add_argument(
        "--global_band_index",
        type=int,
        default=1,
        help="Eigenvector index for global harmonic modulation (default: 1, for the first non-trivial eigenvector)"
    )
    parser.add_argument(
        "--spectral_coefficients",
        type=float,
        nargs='+',
        default=None,
 help="Optional list of spectral coefficients for the spectral vector field; should provide 3*num_modes values that will be reshaped to (num_modes, 3) for 3D vector field generation. Example: --spectral_coefficients 1.0 0.5 -0.3 0.8 0.2 0.1 for 2 modes × 3 dimensions"
    )
    parser.add_argument(
        "--max_num_eigenvectors",
        type=int,
        default=32,
        help="Number of smallest-magnitude (lowest-frequency) Laplacian eigenpairs to compute for spectral targets (default: 32; set to None to compute all)"
    )
    parser.add_argument(
        "--modulation_scale",
        type=float,
        default=0.9,
        help="Scale parameter 'a' for normal vector modulation. Controls the maximum absolute magnitude of modulation while guaranteeing no direction flips. Must be in (0, 1] (default: 0.9)"
    )
    # Dirac node types
    parser.add_argument(
        "--dirac_types",
        type=str,
        nargs='+',
        choices=['max', 'min'],
        default=['max', 'min'],
        help="Which Dirac node types to compute/store as indices per graph. Choices from {max,min}. Default: max min"
    )
    # Random seed
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=589175,
        help="Random seed for reproducibility (default: 589175)"
    )
    # Visualization options
    parser.add_argument(
        "--visualize", 
        action='store_true',
        help="Show 3D visualization of sample ellipsoids"
    )
    parser.add_argument(
        "--num_viz_samples", 
        type=int, 
        default=6,
        help="Number of samples to visualize (default: 6)"
    )
    # Save options
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save dataset (if provided, saves as pickle file)"
    )
    # Optional flag to disable P/Q computation (enabled by default if a
    # save_dir is provided). Useful for quick debugging.
    parser.add_argument(
        "--no_compute_pq",
        action='store_true',
        help="Skip computation of P and Q diffusion operators"
    )
    # Customise HDF5 filename if desired
    parser.add_argument(
        "--pq_h5_name",
        type=str,
        default="pq_tensor_data.h5",
        help="Filename for the HDF5 file that stores P and Q tensors"
    )
    # Optional path to a YAML config to update with effective kernel params
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config to overwrite dataset.local_pca_distance_kernel[_scale],"
             " and related parameters with the effective values used here"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    print("Creating biased ellipsoid dataset for equivariance testing...")
    print(f"Parameters:")
    print(f"  Total samples: {args.num_samples}")
    print(f"  Points per ellipsoid (final): {args.num_nodes_per_graph}")
    print(f"  k-NN parameter (final graph): {args.knn_graph_k}")
    print(f"  Bias axis: {args.bias_axis}")
    print(f"  Random seed: {args.random_seed}")
    if args.oversample_factor != 1.0 or args.num_oversample_points is not None:
        print(f"  Oversampling: factor={args.oversample_factor}, num_oversample_points={args.num_oversample_points}")
        print(f"  Laplacian k: {args.k_laplacian if args.k_laplacian is not None else args.knn_graph_k}")
        print(f"  Laplacian type: {args.laplacian_type}")
    if not args.no_vector_targets:
        print(f"  Vector node targets: enabled; prefix='{args.node_target_prefix}'")
        if args.harmonic_bands is not None:
            print(f"  Harmonic bands: {args.harmonic_bands}")
        print(f"  Global band index: {args.global_band_index}")
        print(f"  Modulation scale: {args.modulation_scale}")
    
    # Parse harmonic bands string -> List[Tuple[int, int]] if provided
    def _parse_bands(bands_str: Optional[str]) -> Optional[List[Tuple[int, int]]]:
        if bands_str is None:
            return None
        ranges = []
        for part in bands_str.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                lo_str, hi_str = part.split('-', 1)
                ranges.append((int(lo_str), int(hi_str)))
            else:
                idx = int(part)
                ranges.append((idx, idx))
        return ranges

    harmonic_bands = _parse_bands(args.harmonic_bands)

    # Validate harmonic_band_weights against harmonic_bands
    harmonic_band_weights = None
    if args.harmonic_band_weights is not None:
        if harmonic_bands is None:
            raise ValueError("--harmonic_band_weights provided but --harmonic_bands is empty")
        if len(args.harmonic_band_weights) != len(harmonic_bands):
            raise ValueError(
                f"--harmonic_band_weights has {len(args.harmonic_band_weights)} values but --harmonic_bands defines {len(harmonic_bands)} bands"
            )
        if any(w <= 0 for w in args.harmonic_band_weights):
            raise ValueError("All --harmonic_band_weights must be positive")
        harmonic_band_weights = list(map(float, args.harmonic_band_weights))

    # Handle spectral coefficients format conversion
    # New format: provide 3*num_modes values that get reshaped to (num_modes, 3)
    # This allows for proper 3D vector field generation with non-axis-aligned directions
    spectral_coeffs = None
    if args.spectral_coefficients is not None:
        if len(args.spectral_coefficients) % 3 != 0:
            raise ValueError(f"spectral_coefficients must have length divisible by 3, got {len(args.spectral_coefficients)}")
        # Reshape to (num_modes, 3) format for 3D vector field generation
        num_modes = len(args.spectral_coefficients) // 3
        spectral_coeffs = np.array(args.spectral_coefficients).reshape(num_modes, 3).tolist()
        print(f"Reshaped spectral coefficients to {num_modes} modes × 3 dimensions")

    # Create dataset with specified parameters
    dataset = create_biased_ellipsoid_datasets(
        num_samples=args.num_samples,
        num_points=args.num_nodes_per_graph,
        k=args.knn_graph_k,
        bias_axis=args.bias_axis,
        random_state_seed=args.random_seed,
        oversample_factor=args.oversample_factor,
        num_oversample_points=args.num_oversample_points,
        k_laplacian=args.k_laplacian,
        laplacian_type=args.laplacian_type,
        add_vector_targets=(not args.no_vector_targets),
        harmonic_bands=harmonic_bands,
        harmonic_band_weights=harmonic_band_weights,
        global_band_index=args.global_band_index,
        spectral_coefficients=spectral_coeffs,
        max_num_eigenvectors=args.max_num_eigenvectors,
        node_target_prefix=args.node_target_prefix,
        abc_means=tuple(args.abc_means) if args.abc_means is not None else None,
        abc_stdevs=tuple(args.abc_stdevs) if args.abc_stdevs is not None else None,
        random_harmonic_k=int(args.random_harmonic_k),
        random_harmonic_coeff_bounds=(float(args.random_harmonic_coeff_bounds[0]), float(args.random_harmonic_coeff_bounds[1])),
        random_harmonic_smoothing_window=int(args.random_harmonic_smoothing_window),
        modulation_scale=float(args.modulation_scale),
        dirac_types=args.dirac_types,
    )
    
    # Analyze dataset, and use the mean of mean distances to set the 
    # Gaussian kernel bandwidth.
    use_mean_of_mean_dists_for_kernel_scale = args.local_pca_kernel_scale_param is None
    mean_of_mean_distances = analyze_dataset_statistics(
        dataset, 
        "Ellipsoid Dataset", 
        return_mean_of_mean_distances=use_mean_of_mean_dists_for_kernel_scale
    )
    if use_mean_of_mean_dists_for_kernel_scale:
        print(f"Mean of mean distances: {mean_of_mean_distances}")
    
    # --------------------------------------------------------------
    # Optionally update a YAML config with the effective kernel params
    # --------------------------------------------------------------
    def _update_yaml_dataset_kernel_config(
        config_path: str,
        kernel_name: str,
        kernel_scale: float,
        use_mean_recentering_flag: bool,
        sing_vect_align_method_name: str,
        column_normalize_P_flag: bool,
        dirac_types: Optional[List[str]] = None,
    ) -> None:
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"WARNING: YAML config not found at '{config_path}'; skipping update.")
            return

        if not isinstance(data, dict):
            data = {}
        if 'dataset' not in data or not isinstance(data['dataset'], dict):
            data['dataset'] = {}
        if 'model' not in data or not isinstance(data['model'], dict):
            data['model'] = {}

        data['dataset']['local_pca_distance_kernel'] = str(kernel_name)
        # store numeric scale explicitly as float
        try:
            data['dataset']['local_pca_distance_kernel_scale'] = float(kernel_scale)
        except Exception:
            # Fallback to string if conversion fails
            data['dataset']['local_pca_distance_kernel_scale'] = kernel_scale
        data['dataset']['use_mean_recentering'] = bool(use_mean_recentering_flag)
        data['dataset']['sing_vect_align_method'] = str(sing_vect_align_method_name)
        data['dataset']['column_normalize_P'] = bool(column_normalize_P_flag)
        # Also write model-level dirac_types for later use
        if dirac_types is not None:
            data['model']['dirac_types'] = list(map(str, dirac_types))

        with open(config_path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"Updated YAML config: {config_path}")

    # Determine effective kernel parameters used in process_pyg_data
    effective_kernel_name = args.local_pca_kernel_fn
    if args.local_pca_kernel_scale_param is None:
        # Match the gaussian_eps used in process_pyg_data below
        effective_kernel_scale = float(mean_of_mean_distances ** 2)
    else:
        effective_kernel_scale = float(args.local_pca_kernel_scale_param)

    if args.config is not None:
        _update_yaml_dataset_kernel_config(
            config_path=args.config,
            kernel_name=effective_kernel_name,
            kernel_scale=effective_kernel_scale,
            use_mean_recentering_flag=bool(args.use_mean_recentering),
            sing_vect_align_method_name=str(args.sing_vect_align_method),
            column_normalize_P_flag=bool(args.column_normalize_P),
            dirac_types=args.dirac_types,
        )

    # Visualize samples if requested
    if args.visualize:
        print(f"\nVisualizing {args.num_viz_samples} samples...")
        visualize_ellipsoid_dataset(dataset, num_samples=args.num_viz_samples)
    
    print("\nDataset creation complete!")
    print(f"Created dataset with ellipsoids biased toward {args.bias_axis}-axis (diameter along {args.bias_axis})")
    # if args.eigenvector_indices is not None:
    #     print(f"Added node targets using eigenvectors: {args.eigenvector_indices}")
    
    # Save dataset if save_dir is provided
    if args.save_dir is not None:
        print(f"\nSaving dataset to {args.save_dir}...")
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Save the dataset
        save_path = os.path.join(args.save_dir, f'ellipsoid_dataset_{args.num_samples}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {save_path}")
        print(f"Dataset size: {len(dataset)} samples")

        # --------------------------------------------------------------
        # Compute and save P / Q sparse diffusion operators -> HDF5
        # --------------------------------------------------------------
        if not args.no_compute_pq:
            print("\nComputing P and Q diffusion operators (this may take a few minutes)...")

            # Lazy import to avoid the heavyweight dependency if skipped
            from data_processing.process_pyg_data import process_pyg_data

            # Determine output path
            h5_path = os.path.join(args.save_dir, args.pq_h5_name)

            # Overwrite existing file only after successful computation. We
            # therefore write to a temporary file first and rename at the end.
            tmp_h5_path = h5_path + ".tmp"

            # Ensure we start fresh
            if os.path.exists(tmp_h5_path):
                os.remove(tmp_h5_path)

            with h5py.File(tmp_h5_path, 'w') as h5f:
                for idx, data in enumerate(dataset):
                    tensor_dict = process_pyg_data(
                        data,
                        data_i=idx,
                        row_normalize=not args.column_normalize_P,
                        return_data_object=False,
                        geom_feat_key='pos',
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        graph_construction=None,
                        use_mean_recentering=args.use_mean_recentering,
                        local_pca_kernel_fn_kwargs={
                            'kernel': args.local_pca_kernel_fn,
                            'gaussian_eps': mean_of_mean_distances**2 \
                                if use_mean_of_mean_dists_for_kernel_scale \
                                else args.local_pca_kernel_scale_param,
                        },
                        sing_vect_align_method=args.sing_vect_align_method
                    )

                    for op_key in ("P", "Q"):
                        op_data = tensor_dict[op_key]

                        grp = h5f.require_group(f"{op_key}/{idx}")

                        # Store sparse COO components separately;
                        # indices shape: (2, nnz), values shape: (nnz,)
                        grp.create_dataset(
                            'indices',
                            data=op_data['indices'].cpu().numpy(),
                            compression='gzip'
                        )
                        grp.create_dataset(
                            'values',
                            data=op_data['values'].cpu().numpy(),
                            compression='gzip'
                        )
                        # Size is a simple tuple -> turn into int64 vector
                        grp.create_dataset(
                            'size',
                            data=np.array(op_data['size'], dtype='int64')
                        )

                    # Progress logging every 100 samples
                    if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
                        print(f"  Processed {idx + 1}/{len(dataset)} samples", end='\r')

            # Atomically move temp to final destination
            os.replace(tmp_h5_path, h5_path)
            print(f"\nP and Q tensors saved to {h5_path}")
        else:
            print("\nSkipping P/Q computation as requested via --no_compute_pq.")
    
    