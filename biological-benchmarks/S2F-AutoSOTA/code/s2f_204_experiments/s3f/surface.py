import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster
import robust_laplacian
from scipy.sparse.linalg import eigsh


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    if 'mps' in str(batch.device):
        Ns = batch.to('cpu').bincount()
    else:
        Ns = batch.bincount()
        
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""
    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def knn_atoms(x, y, k, batch_x=None, batch_y=None):
    k = k+1
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    if batch_x is not None:
        pairwise_distance_ij.ranges = diagonal_ranges(batch_x, batch_y)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def compute_eigens(n_verts,  xyz,  min_n_eigs,  eigs_ratio):
    n_eigs = max(min_n_eigs, int(eigs_ratio * n_verts)+1)
    assert n_eigs < n_verts
    Laplacian, Mass = robust_laplacian.point_cloud_laplacian(xyz)
    Laplacian.data += 1e-8
    
    eig_vals, eig_vecs = eigsh(Laplacian, n_eigs, Mass, sigma=0)
    eig_vals[0] = 0.
    eig_vecs_inv = eig_vecs.T @ Mass
    
    return eig_vals, eig_vecs, eig_vecs_inv
    

def compute_HKS(eigen_vecs, eigen_vals, num_t, t_min=0.1, t_max=1000, scale=1000):
    eigen_vals = eigen_vals.flatten()
    assert eigen_vals[1] > 0
    assert np.min(eigen_vals) > -1E-6
    assert np.array_equal(eigen_vals, sorted(eigen_vals))

    t_list = np.geomspace(t_min, t_max, num_t)
    phase = np.exp(-np.outer(t_list, eigen_vals[1:]), dtype='f')
    wphi = phase[:, None, :] * eigen_vecs[None, :, 1:] 
    HKS = np.einsum('tnk,nk->nt', wphi, eigen_vecs[:, 1:]) * scale
    heat_trace = np.sum(phase, axis=1)
    HKS /= heat_trace

    return HKS


def compute_curvatures(xyz, normals, batch, curvature_scales):
    cur = curvatures(
                      vertices=xyz,
                      triangles=None,
                      normals=normals,
                      scales=curvature_scales,
                      batch=batch,
                      reg=1e-10,
                      )
    cur[torch.isnan(cur)] = 0.
    return cur
    
    
# On-the-fly generation of the surfaces ========================================
# Code taken from https://github.com/FreyrS/dMaSIF/blob/master/geometry_processing.py
def subsample(x, batch=None, scale=1.0):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
            Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        (M,3): sub-sampled point cloud, with M <= N.
    """

    if batch is None:  # Single protein case:
        if True:  # Use a fast scatter_add_ implementation
            labels = grid_cluster(x, scale).long()
            C = labels.max() + 1

            # We append a "1" to the input vectors, in order to
            # compute both the numerator and denominator of the "average"
            #  fraction in one pass through the data.
            x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
            D = x_1.shape[1]
            points = torch.zeros_like(x_1[:C])
            points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
            return (points[:, :-1] / points[:, -1:]).contiguous()

        else:  # Older implementation;
            points = scatter(points * weights[:, None], labels, dim=0)
            weights = scatter(weights, labels, dim=0)
            points = points / weights[:, None]

    else:  # We process proteins using a for loop.
        # This is probably sub-optimal, but I don't really know
        # how to do more elegantly (this type of computation is
        # not super well supported by PyTorch).
        batch_size = torch.max(batch).item() + 1  # Typically, =32
        points, batches = [], []
        for b in range(batch_size):
            p = subsample(x[batch == b], scale=scale)
            points.append(p)
            batches.append(b * torch.ones_like(batch[: len(p)]))

    return torch.cat(points, dim=0), torch.cat(batches, dim=0)


def soft_distances(x, y, batch_x, batch_y, smoothness=0.01, atomtypes=None):
    """Computes a soft distance function to the atom centers of a protein.

    Implements Eq. (1) of the paper in a fast and numerically stable way.

    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.

    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    x_i = LazyTensor(x[:, None, :])  # (N, 1, 3) atoms
    y_j = LazyTensor(y[None, :, :])  # (1, M, 3) sampling points
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) squared distances

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    D_ij.ranges = diagonal_ranges(batch_x, batch_y)

    if atomtypes is not None:
        # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i":
        # (N, 6)  -> (N, 1, 1)  (There are 6 atom types)
        atomic_radii = torch.tensor(
            [170, 110, 152, 155, 180, 190], dtype=torch.float, device=x.device
        )
        
        atomic_radii = atomic_radii / atomic_radii.min()
        atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes
        # smoothness = atomtypes @ atomic_radii  # (N, 6) @ (6,) = (N,)
        smoothness = torch.sum(
            smoothness * atomtype_radii, dim=1, keepdim=False
        )  # n_atoms, 1
        smoothness_i = LazyTensor(smoothness[:, None, None])

        # Compute an estimation of the mean smoothness in a neighborhood
        # of each sampling point:
        # density = (-D_ij.sqrt()).exp().sum(0).view(-1)  # (M,) local density of atoms
        # smooth = (smoothness_i * (-D_ij.sqrt()).exp()).sum(0).view(-1)  # (M,)
        # mean_smoothness = smooth / density  # (M,)

        # soft_dists = -mean_smoothness * (
        #    (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        # ).view(-1)
        mean_smoothness = (-D_ij.sqrt()).exp()
        mean_smoothness = mean_smoothness.sum(0)
        mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
        mean_smoothness = (
            smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j
        )  # n_atoms, n_points, 1
        mean_smoothness = mean_smoothness.sum(0).view(-1)
        soft_dists = -mean_smoothness * (
            (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        ).view(-1)

    else:
        soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(
            -1
        )

    return soft_dists


def atoms_to_points_normals(
    atoms,
    batch,
    distance=1.05,
    smoothness=0.5,
    resolution=1.0,
    nits=5,
    atomtypes=None,
    sup_sampling=20,
    variance=0.5,
):
    """Turns a collection of atoms into an oriented point cloud.

    Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

    Args:
        atoms (Tensor): (N,3) coordinates of the atom centers `a_k`.
        batch (integer Tensor): (N,) batch vector, as in PyTorch_geometric.
        distance (float, optional): value of the level set to sample from
            the smooth distance function. Defaults to 1.05.
        smoothness (float, optional): radii of the atoms, if atom types are
            not provided. Defaults to 0.5.
        resolution (float, optional): side length of the cubic cells in
            the final sub-sampling pass. Defaults to 1.0.
        nits (int, optional): number of iterations . Defaults to 5.
        atomtypes (Tensor, optional): (N,6) one-hot encoding of the atom
            chemical types. Defaults to None.

    Returns:
        (Tensor): (M,3) coordinates for the surface points `x_i`.
        (Tensor): (M,3) unit normals `n_i`.
        (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
    """
    # a) Parameters for the soft distance function and its level set:
    T = distance

    N, D = atoms.shape
    B = sup_sampling  # Sup-sampling ratio

    # Batch vectors:
    batch_atoms = batch
    batch_z = batch[:, None].repeat(1, B).view(N * B)

    # b) Draw N*B points at random in the neighborhood of our atoms
    z = atoms[:, None, :] + 10 * T * torch.randn(N, B, D).type_as(atoms)
    z = z.view(-1, D)  # (N*B, D)

    # We don't want to backprop through a full network here!
    atoms = atoms.detach().contiguous()
    z = z.detach().contiguous()

    # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
    with torch.enable_grad():
        if z.is_leaf:
            z.requires_grad = True

        # c) Iterative loop: gradient descent along the potential
        # ".5 * (dist - T)^2" with respect to the positions z of our samples
        for it in range(nits):
            dists = soft_distances(
                atoms,
                z,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = ((dists - T) ** 2).sum()
            g = torch.autograd.grad(Loss, z)[0]
            z.data -= 0.5 * g

        # d) Only keep the points which are reasonably close to the level set:
        dists = soft_distances(
            atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
        )
        margin = (dists - T).abs()
        mask = margin < variance * T

        # d') And remove the points that are trapped *inside* the protein:
        zz = z.detach()
        zz.requires_grad = True
        for it in range(nits):
            dists = soft_distances(
                atoms,
                zz,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, zz)[0]
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
            zz = zz + 1.0 * T * normals

        dists = soft_distances(
            atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
        )
        mask = mask & (dists > 1.5 * T)

        z = z[mask].contiguous().detach()
        batch_z = batch_z[mask].contiguous().detach()

        # e) Subsample the point cloud:
        points, batch_points = subsample(z, batch_z, scale=resolution)

        # f) Compute the normals on this smaller point cloud:
        p = points.detach()
        p.requires_grad = True
        dists = soft_distances(
            atoms,
            p,
            batch_atoms,
            batch_points,
            smoothness=smoothness,
            atomtypes=atomtypes,
        )
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, p)[0]
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
    points = points - 0.5 * normals
    return points.detach(), normals.detach(), batch_points.detach()


def mesh_normals_areas(vertices, triangles=None, scale=[1.0], batch=None, normals=None):
    # Single- or Multi-scale mode:
    if hasattr(scale, "__len__"):
        scales, single_scale = scale, False
    else:
        scales, single_scale = [scale], True
    scales = torch.Tensor(scales).type_as(vertices)  # (S,)

    # Compute the "raw" field of normals:
    if triangles is not None:
        # Vertices of all triangles in the mesh:
        A = vertices[triangles[0, :]]  # (N, 3)
        B = vertices[triangles[1, :]]  # (N, 3)
        C = vertices[triangles[2, :]]  # (N, 3)

        # Triangle centers and normals (length = surface area):
        centers = (A + B + C) / 3  # (N, 3)
        V = (B - A).cross(C - A)  # (N, 3)

        # Vertice areas:
        S = (V ** 2).sum(-1).sqrt() / 6  # (N,) 1/3 of a triangle area
        areas = torch.zeros(len(vertices)).type_as(vertices)  # (N,)
        areas.scatter_add_(0, triangles[0, :], S)  # Aggregate from "A's"
        areas.scatter_add_(0, triangles[1, :], S)  # Aggregate from "B's"
        areas.scatter_add_(0, triangles[2, :], S)  # Aggregate from "C's"

    else:  # Use "normals" instead
        areas = None
        V = normals
        centers = vertices

    # Normal of a vertex = average of all normals in a ball of size "scale":
    x_i = LazyTensor(vertices[:, None, :])  # (N, 1, 3)
    y_j = LazyTensor(centers[None, :, :])  # (1, M, 3)
    v_j = LazyTensor(V[None, :, :])  # (1, M, 3)
    s = LazyTensor(scales[None, None, :])  # (1, 1, S)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  #  (N, M, 1)
    K_ij = (-D_ij / (2 * s ** 2)).exp()  # (N, M, S)

    # Support for heterogeneous batch processing:
    if batch is not None:
        batch_vertices = batch
        batch_centers = batch[triangles[0, :]] if triangles is not None else batch
        #K_ij.ranges = diagonal_ranges(batch_vertices, batch_centers)

    if single_scale:
        U = (K_ij * v_j).sum(dim=1)  # (N, 3)
    else:
        U = (K_ij.tensorprod(v_j)).sum(dim=1)  # (N, S*3)
        U = U.view(-1, len(scales), 3)  # (N, S, 3)

    normals = F.normalize(U, p=2, dim=-1)  # (N, 3) or (N, S, 3)

    return normals, areas
    

def tangent_vectors(normals):
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv


def curvatures(
    vertices, triangles=None, scales=[1.0], batch=None, normals=None, reg=1e-10
):
    # Number of points, number of scales:
    N, S = vertices.shape[0], len(scales)
    #ranges = diagonal_ranges(batch)

    # Compute the normals at different scales + vertice areas:
    normals_s, _ = mesh_normals_areas(
        vertices, triangles=triangles, normals=normals, scale=scales, batch=batch
    )  # (N, S, 3), (N,)

    # Local tangent bases:
    uv_s = tangent_vectors(normals_s)  # (N, S, 2, 3)

    features = []

    for s, scale in enumerate(scales):
        # Extract the relevant descriptors at the current scale:
        normals = normals_s[:, s, :].contiguous()  #  (N, 3)
        uv = uv_s[:, s, :, :].contiguous()  # (N, 2, 3)

        # Encode as symbolic tensors:
        # Points:
        x_i = LazyTensor(vertices.view(N, 1, 3))
        x_j = LazyTensor(vertices.view(1, N, 3))
        # Normals:
        n_i = LazyTensor(normals.view(N, 1, 3))
        n_j = LazyTensor(normals.view(1, N, 3))
        # Tangent bases:
        uv_i = LazyTensor(uv.view(N, 1, 6))

        # Pseudo-geodesic squared distance:
        d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-d2_ij / (2 * (scale ** 2))).exp()  # (N, N, 1)

        # Project on the tangent plane:
        P_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)
        Q_ij = uv_i.matvecmult(n_j - n_i)  # (N, N, 2)
        # Concatenate:
        PQ_ij = P_ij.concat(Q_ij)  # (N, N, 2+2)

        # Covariances, with a scale-dependent weight:
        PPt_PQt_ij = P_ij.tensorprod(PQ_ij)  # (N, N, 2*(2+2))
        PPt_PQt_ij = window_ij * PPt_PQt_ij  #  (N, N, 2*(2+2))

        # Reduction - with batch support:
        #PPt_PQt_ij.ranges = ranges
        PPt_PQt = PPt_PQt_ij.sum(1)  # (N, 2*(2+2))

        # Reshape to get the two covariance matrices:
        PPt_PQt = PPt_PQt.view(N, 2, 2, 2)
        PPt, PQt = PPt_PQt[:, :, 0, :], PPt_PQt[:, :, 1, :]  # (N, 2, 2), (N, 2, 2)

        # Add a small ridge regression:
        PQt[:, 0, 0] += reg
        PQt[:, 1, 1] += reg

        PPt[:, 0, 0] += reg
        PPt[:, 1, 1] += reg

        # (minus) Shape operator, i.e. the differential of the Gauss map:
        # = (PPt^-1 @ PQt) : simple estimation through linear regression
        #S = torch.linalg.solve(PQt, PPt)
        S = torch.linalg.lstsq(PQt, PPt).solution
        a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]  # (N,)

        # Normalization
        mean_curvature = a + d
        gauss_curvature = a * d - b * c
        features += [mean_curvature.clamp(-1, 1), gauss_curvature.clamp(-1, 1)]

    features = torch.stack(features, dim=-1)
    return features
