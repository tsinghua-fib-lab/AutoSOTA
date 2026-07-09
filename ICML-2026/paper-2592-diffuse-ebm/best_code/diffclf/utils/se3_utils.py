# SE(3)-equivariance helpers for particle systems (mean removal, RMSD, distances)

# Libraries
import torch
from typing import Tuple


def remove_mean(samples):
    """Remove the centre‑of‑mass from a configuration so it is mean‑free.

    Args:
        * samples (torch.Tensor of arbitrary shape ending in (..., n_particles, n_dimensions)):
            Particle

    Returns:
        * samples_mean_free (torch.Tensor of the same shape as samples):
            Positions translated so that the arithmetic mean over the particle axis is zero for every
            configuration in the batch.
    """
    if isinstance(samples, torch.Tensor):
        samples = samples - torch.mean(samples, dim=-2, keepdim=True)
    else:
        samples = samples - samples.mean(axis=-2, keepdims=True)
    return samples


def interatomic_dist(samples, return_sq_norms=False, return_displacements=False,
                     keep_only_upper_tri=True, mask=None):
    """Pair‑wise Euclidean distances between all atoms inside each configuration.

    Args:
        * samples (torch.Tensor of shape (batch_size, n_particles, n_dimensions)): Particles
        * return_sq_norms (bool): Whether to return the squared norm (default is False)
        * return_displacements (bool): Whether to return the displacements (default is False)
        * keep_only_upper_tri (bool): Whether to not return the full distance matrix (default is True)
        * mask (torch.Tensor): Precomputed mask (default is None)
            mask = torch.triu(
                    torch.ones((n_particles, n_particles), dtype=torch.bool, device=samples.device),
                    diagonal=1
                )

    Returns:
        * distances (torch.Tensor of shape (batch_size, n_particles (n_particles‑1) / 2)): Distances
            If not keep_only_upper_tri, the shape is (batch_size, n_particles, n_particles)
    """
    n_particles = samples.shape[-2]

    # Compute pair‑wise displacement vectors (broadcasted subtraction)
    displacements = samples[:, None, :, :] - samples[:, :, None, :]

    # Keep only the strictly upper‑triangular entries (i < j)
    if keep_only_upper_tri:
        if mask is None:
            mask = torch.triu(
                torch.ones((n_particles, n_particles), dtype=torch.bool, device=samples.device),
                diagonal=1
            )
        displacements = displacements[:, mask]

    # Euclidean norm along the coordinate axis
    if return_sq_norms:
        sq_norms = torch.sum(torch.square(displacements), dim=-1)
        if return_displacements:
            return sq_norms, displacements
        else:
            return sq_norms
    else:
        distances = torch.linalg.norm(displacements, dim=-1)
        if return_displacements:
            return distances, displacements
        else:
            return distances


def find_alignment_kabsch(P, Q) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimal rigid alignment of two point clouds via the Kabsch algorithm.

    Args:
        * P (torch.Tensor of shape (batch_size, dim)): Reference coordinates (to be rotated).
        * Q (torch.Tensor of shape (batch_size, dim)): Target coordinates (stay fixed).

    Returns:
        * R (torch.Tensor of shape (dim, dim)):
            Orthogonal rotation matrix that best superimposes P onto Q in the least‑squares sense.
        * t (torch.Tensor of shape (dim,)):
            Translation vector such that R @ P + t is optimally aligned with Q.
    """
    # Shift points w.r.t centroid
    centroid_P, centroid_Q = P.mean(dim=0), Q.mean(dim=0)
    P_c, Q_c = P - centroid_P, Q - centroid_Q
    # Find rotation matrix by Kabsch algorithm
    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    # ensure right-handedness
    d = torch.sign(torch.linalg.det(V @ U.T))
    # Trick for torch.vmap
    if P.shape[-1] == 3:
        diag_values = torch.cat(
            [
                torch.ones(1, dtype=P.dtype, device=P.device),
                torch.ones(1, dtype=P.dtype, device=P.device),
                d * torch.ones(1, dtype=P.dtype, device=P.device),
            ]
        )
    elif P.shape[-1] == 2:
        diag_values = torch.cat(
            [
                torch.ones(1, dtype=P.dtype, device=P.device),
                d * torch.ones(1, dtype=P.dtype, device=P.device),
            ]
        )
    else:
        print("unsupport dim for kabsch")
        raise ValueError
    # This is only [[1,0,0],[0,1,0],[0,0,d]]
    M = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device) * diag_values
    R = V @ M @ U.T
    # Find translation vectors
    t = centroid_Q[None, :] - (R @ centroid_P[None, :].T).T
    t = t.T
    return R, t.squeeze()


def calculate_rmsd(pos, ref):
    """Root‑Mean‑Square Deviation (RMSD) after optimal superposition.

    Args:
        * pos (torch.Tensor of shape (batch_size, dim)): Coordinates of the structure to be compared.
        * ref (torch.Tensor of shape (batch_size, dim)): Reference coordinates to align against.

    Returns:
        * rmsd (torch.Tensor scalar):
            RMSD between the optimally aligned structures. Lower values indicate higher similarity.
    """
    if pos.shape[0] != ref.shape[0]:
        raise ValueError("pos and ref must have the same number of points")

    R, t = find_alignment_kabsch(ref, pos)  # rotate *ref* onto *pos*
    ref_aligned = (R @ ref.T).T + t  # apply rigid transform

    rmsd = torch.linalg.norm(ref_aligned - pos, dim=1).mean()
    return rmsd


def calculate_rmsd_matrix(R_ref, R):
    """Pairwise RMSD matrix between two batches of structures.

    Args:
        * R_ref (torch.Tensor of shape (batch_size, n_particles, n_dimensions)): Reference particles.
        * R (torch.Tensor of shape (batch_size, n_particles, n_dimensions)): Query particles.

    Returns:
        * rmsd_matrix (torch.Tensor of shape (batch_size, batch_size)):
            Entry [i, j] contains the RMSD between the i‑th query and the j‑th reference after
            optimal alignment.
    """
    fn_vmap_row = torch.vmap(calculate_rmsd, in_dims=(0, None))
    fn_vmap_row_col = torch.vmap(fn_vmap_row, in_dims=(None, 0))
    return fn_vmap_row_col(R, R_ref)


def avg_distance_to_origin(samples, n_particles, n_dimensions):
    """Average radial distance of particles from the origin after centring.

    Args:
        * samples (torch.Tensor of arbitrary shape ending in (..., n_particles, n_dimensions)): Particles
        * n_particles (int): Number of particles
        * n_dimensions (int): Spatial dimensionality

    Returns:
        * avg_dist (torch.Tensor of shape (B,) where *B* is the batch size):
            Mean Euclidean distance of particles to the origin for every configuration in the batch.
    """
    # First remove the centre‑of‑mass so that the origin is the centroid
    samples = remove_mean(samples)
    batch_size = samples.shape[0]  # assume leading dimension is the batch size
    samples = samples.view(batch_size, n_particles, n_dimensions)
    return torch.mean(torch.linalg.norm(samples, dim=-1), dim=1)


def compute_intersection(h1, h2):
    """Intersection between two histograms

    Args:
            h1 (torch.Tensor of shape (bins,)): First histogram
            h2 (torch.Tensor of shape (bins,)): Second histogram

    Returns:
            inter (float): Metric
    """

    return torch.sum(torch.minimum(h1, h2))


def compute_correlation(h1, h2):
    """Correlation between two histograms

    Args:
            h1 (torch.Tensor of shape (bins,)): First histogram
            h2 (torch.Tensor of shape (bins,)): Second histogram

    Returns:
            corr (float): Metric
    """

    h1_norm = h1 - h1.mean()
    h2_norm = h2 - h2.mean()
    return torch.sum(h1_norm * h2_norm) / torch.sqrt(torch.sum(torch.square(h1_norm))
                                                     * torch.sum(torch.square(h2_norm)))
