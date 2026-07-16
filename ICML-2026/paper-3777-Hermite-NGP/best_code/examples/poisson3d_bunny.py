"""
Poisson 3D PINN on a mesh interior, with Hermite-NGP encoding.

NVIDIA Kaolin handles GPU-accelerated mesh sampling and inside/outside
queries (kaolin.ops.mesh.sample_points() + check_sign).

PDE:    Laplacian(u) = 0   in   [0,1]^3 \\ mesh
BC:     u = 1   on the mesh surface
        u = 0   on the outer cube faces

Usage:
    python examples/poisson3d_bunny.py --mesh data/meshes/bunny.ply
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CUDA extension
try:
    import hermite_mlp_cuda_3d_v2
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: hermite_mlp_cuda_3d_v2 not available. Run: python setup_mlp_cuda_v2.py install")

# Import Kaolin
try:
    import kaolin
    from kaolin.ops.mesh import index_vertices_by_faces, check_sign
    KAOLIN_AVAILABLE = True
except ImportError:
    KAOLIN_AVAILABLE = False
    print("WARNING: kaolin not available. Install with: pip install kaolin")


# =============================================================================
# Kaolin Mesh Sampler with Domain BC
# =============================================================================

class KaolinMeshSamplerWithDomainBC:
    """GPU-accelerated mesh surface sampler using Kaolin, with domain BC support."""

    def __init__(self, mesh_path, mesh_bc_value=1.0, domain_bc_value=0.0, device='cuda'):
        """
        Load mesh and prepare for GPU sampling.

        Args:
            mesh_path: Path to mesh file (.obj, .ply, .off)
            mesh_bc_value: BC value on mesh surface (default 1.0)
            domain_bc_value: BC value on domain boundary (default 0.0)
            device: Device for computations
        """
        self.device = device
        self.mesh_bc_value = mesh_bc_value
        self.domain_bc_value = domain_bc_value

        # Load mesh using trimesh
        import trimesh
        mesh = trimesh.load(mesh_path, force='mesh')

        # Normalize to [0.1, 0.9]^3 (larger mesh)
        vertices = mesh.vertices.copy()
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        center = (v_min + v_max) / 2
        scale = (v_max - v_min).max()

        # Normalize to [0, 1] then scale to [0.1, 0.9]
        vertices = (vertices - center) / scale  # [-0.5, 0.5]
        vertices = vertices + 0.5               # [0, 1]
        vertices = vertices * 0.8 + 0.1         # [0.1, 0.9]

        # Store mesh data on GPU
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        self.faces = torch.tensor(mesh.faces.astype(np.int64), dtype=torch.long, device=device)

        # Prepare for Kaolin operations
        self.vertices_batch = self.vertices.unsqueeze(0)  # [1, V, 3]
        self.face_vertices = index_vertices_by_faces(self.vertices_batch, self.faces)  # [1, F, 3, 3]

        # Compute face areas and normals for weighted sampling
        v0 = self.face_vertices[0, :, 0, :]
        v1 = self.face_vertices[0, :, 1, :]
        v2 = self.face_vertices[0, :, 2, :]
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        self.face_areas = 0.5 * torch.norm(cross, dim=1)
        self.face_probs = self.face_areas / self.face_areas.sum()

        # Compute unit face normals
        norms = torch.norm(cross, dim=1, keepdim=True)
        self.face_normals = cross / (norms + 1e-8)  # [F, 3]

        print(f"KaolinMeshSamplerWithDomainBC initialized:")
        print(f"  Mesh: {mesh_path}")
        print(f"  Vertices: {self.vertices.shape[0]}, Faces: {self.faces.shape[0]}")
        print(f"  Normalized to: [{self.vertices.min().item():.3f}, {self.vertices.max().item():.3f}]^3")
        print(f"  Mesh BC value: {mesh_bc_value}")
        print(f"  Domain BC value: {domain_bc_value}")
        print(f"  Collocation: EXTERIOR + NEAR-SURFACE BAND")

    def sample_mesh_surface(self, n_points):
        """Sample points on mesh surface."""
        face_idx = torch.multinomial(self.face_probs, n_points, replacement=True)
        v0 = self.face_vertices[0, face_idx, 0, :]
        v1 = self.face_vertices[0, face_idx, 1, :]
        v2 = self.face_vertices[0, face_idx, 2, :]

        r1 = torch.sqrt(torch.rand(n_points, 1, device=self.device))
        r2 = torch.rand(n_points, 1, device=self.device)
        points = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2

        bc_values = torch.full((n_points,), self.mesh_bc_value, device=self.device)
        return points, bc_values

    def sample_surface_with_normals(self, n_points):
        """Sample points on mesh surface with corresponding normals."""
        face_idx = torch.multinomial(self.face_probs, n_points, replacement=True)
        v0 = self.face_vertices[0, face_idx, 0, :]
        v1 = self.face_vertices[0, face_idx, 1, :]
        v2 = self.face_vertices[0, face_idx, 2, :]

        r1 = torch.sqrt(torch.rand(n_points, 1, device=self.device))
        r2 = torch.rand(n_points, 1, device=self.device)
        points = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2

        # Get normals for sampled faces
        normals = self.face_normals[face_idx]  # [N, 3]

        return points, normals

    def sample_near_surface_band(self, n_points, band_width=0.05, exterior_only=True):
        """
        Sample points in a band near the mesh surface.

        Uses surface points offset along normals with uniform random distance.
        This concentrates collocation points where the solution gradient is steepest.

        Args:
            n_points: Number of points to sample
            band_width: Maximum distance from surface (default 0.05)
            exterior_only: If True, only sample outside the mesh

        Returns:
            points: [N, 3] tensor of near-surface points
        """
        # Sample surface points with normals
        pts_surface, normals = self.sample_surface_with_normals(n_points * 2)

        if exterior_only:
            # Only offset outward (positive direction along normal)
            # Use uniform distribution in [0.001, band_width]
            offset = torch.rand(pts_surface.shape[0], 1, device=self.device) * (band_width - 0.001) + 0.001
        else:
            # Offset both directions
            offset = (torch.rand(pts_surface.shape[0], 1, device=self.device) * 2 - 1) * band_width

        # Offset points along normal
        pts_offset = pts_surface + offset * normals

        # Clamp to domain [0.01, 0.99]
        pts_offset = torch.clamp(pts_offset, 0.01, 0.99)

        if exterior_only:
            # Filter out any points that ended up inside
            inside_mask = self.check_inside(pts_offset)
            pts_offset = pts_offset[~inside_mask]

        # Trim to requested count
        if pts_offset.shape[0] > n_points:
            pts_offset = pts_offset[:n_points]

        return pts_offset

    def sample_domain_boundary(self, n_points):
        """Sample points on domain boundary (cube faces at [0,1]^3)."""
        n_per_face = n_points // 6
        points_list = []

        for face_idx in range(6):
            n_this_face = n_per_face if face_idx < 5 else (n_points - 5 * n_per_face)

            # Generate random 2D coordinates
            u = torch.rand(n_this_face, device=self.device)
            v = torch.rand(n_this_face, device=self.device)

            if face_idx == 0:    # x = 0
                pts = torch.stack([torch.zeros_like(u), u, v], dim=1)
            elif face_idx == 1:  # x = 1
                pts = torch.stack([torch.ones_like(u), u, v], dim=1)
            elif face_idx == 2:  # y = 0
                pts = torch.stack([u, torch.zeros_like(u), v], dim=1)
            elif face_idx == 3:  # y = 1
                pts = torch.stack([u, torch.ones_like(u), v], dim=1)
            elif face_idx == 4:  # z = 0
                pts = torch.stack([u, v, torch.zeros_like(u)], dim=1)
            else:                # z = 1
                pts = torch.stack([u, v, torch.ones_like(u)], dim=1)

            points_list.append(pts)

        points = torch.cat(points_list, dim=0)
        bc_values = torch.full((points.shape[0],), self.domain_bc_value, device=self.device)
        return points, bc_values

    def check_inside(self, points):
        """
        Check if points are inside the mesh using Kaolin's check_sign.

        Args:
            points: [N, 3] tensor of query points

        Returns:
            inside_mask: [N] boolean tensor, True if inside mesh
        """
        # check_sign expects [B, N, 3] and returns [B, N] with True = inside
        pts_batch = points.unsqueeze(0)  # [1, N, 3]
        inside = check_sign(self.vertices_batch, self.faces, pts_batch)  # [1, N]
        return inside.squeeze(0)  # [N]

    def sample_collocation(self, n_points, exterior_only=True, near_surface_ratio=0.5, band_width=0.05):
        """
        Sample collocation points: uniform exterior + near-surface band.

        Distribution (when exterior_only=True):
        - near_surface_ratio: fraction of points sampled near surface
        - (1 - near_surface_ratio): fraction of points sampled uniformly in exterior

        Args:
            n_points: Total number of points to sample
            exterior_only: If True, only return points outside the mesh
            near_surface_ratio: Fraction of points in near-surface band (default 0.5)
            band_width: Width of near-surface band (default 0.05)

        Returns:
            points: [N, 3] tensor of collocation points
        """
        if not exterior_only:
            points = torch.rand(n_points, 3, device=self.device) * 0.98 + 0.01
            return points

        # Split between near-surface and uniform exterior
        n_near_surface = int(n_points * near_surface_ratio)
        n_uniform = n_points - n_near_surface

        collected_points = []

        # 1. Near-surface band sampling (concentrated near mesh)
        if n_near_surface > 0:
            pts_near = self.sample_near_surface_band(n_near_surface, band_width=band_width, exterior_only=True)
            collected_points.append(pts_near)

        # 2. Uniform exterior sampling (spread throughout domain)
        if n_uniform > 0:
            oversample_factor = 2.0
            n_sample = int(n_uniform * oversample_factor)

            uniform_points = []
            n_collected = 0
            max_iterations = 10

            for _ in range(max_iterations):
                pts = torch.rand(n_sample, 3, device=self.device) * 0.98 + 0.01
                inside_mask = self.check_inside(pts)
                exterior_pts = pts[~inside_mask]
                uniform_points.append(exterior_pts)
                n_collected += exterior_pts.shape[0]

                if n_collected >= n_uniform:
                    break

            all_uniform = torch.cat(uniform_points, dim=0)
            if all_uniform.shape[0] > n_uniform:
                all_uniform = all_uniform[:n_uniform]
            collected_points.append(all_uniform)

        # Concatenate and shuffle
        all_points = torch.cat(collected_points, dim=0)
        perm = torch.randperm(all_points.shape[0], device=self.device)
        return all_points[perm]

    def sample_importance(self, n_points, residual_fn, pool_size=None, alpha=1.0,
                          near_surface_ratio=0.5, band_width=0.05, chunk=32768):
        """Residual-based importance sampling: draw points where the PDE residual is large.

        Steps:
          1. Build a large candidate pool with the same near/uniform mix used for
             the standard sampler.
          2. Evaluate |residual_fn(p)| at each candidate (in chunks to fit memory).
          3. Re-sample n_points from the pool with probability proportional to
             |residual|^alpha + eps. This concentrates training on hard points
             instead of wasting capacity on regions the model already fits.
        """
        if pool_size is None:
            pool_size = max(n_points * 4, 200000)
        candidates = self.sample_collocation(
            pool_size, exterior_only=True,
            near_surface_ratio=near_surface_ratio, band_width=band_width,
        )

        with torch.no_grad():
            residuals = torch.empty(candidates.shape[0], device=self.device)
            for i in range(0, candidates.shape[0], chunk):
                residuals[i:i + chunk] = residual_fn(candidates[i:i + chunk]).detach().abs().squeeze()

        weights = residuals.pow(alpha) + 1e-6
        weights = weights / weights.sum()
        # multinomial works up to 2**24 elements
        idx = torch.multinomial(weights, n_points, replacement=True)
        return candidates[idx]


# =============================================================================
# CUDA V3 MLP: CUDA Forward + PyTorch Backward (3D)
# =============================================================================

class HermiteLayerFunction3D_V3(torch.autograd.Function):
    """CUDA forward + PyTorch backward for Hermite propagation (3D)."""

    @staticmethod
    def forward(ctx, h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz, weight, bias, omega, apply_activation):
        outputs = hermite_mlp_cuda_3d_v2.forward(
            h.contiguous(), dh_dx.contiguous(), dh_dy.contiguous(), dh_dz.contiguous(),
            d2h_dxx.contiguous(), d2h_dyy.contiguous(), d2h_dzz.contiguous(),
            weight.contiguous(), bias.contiguous(),
            omega, apply_activation
        )
        out_h, out_dx, out_dy, out_dz, out_dxx, out_dyy, out_dzz = outputs[:7]
        save_z, save_dz_dx, save_dz_dy, save_dz_dz, save_d2z_dxx, save_d2z_dyy, save_d2z_dzz = outputs[7:]

        ctx.save_for_backward(
            h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz,
            weight,
            save_z, save_dz_dx, save_dz_dy, save_dz_dz
        )
        ctx.omega = omega
        ctx.apply_activation = apply_activation

        return out_h, out_dx, out_dy, out_dz, out_dxx, out_dyy, out_dzz

    @staticmethod
    def backward(ctx, grad_h, grad_dx, grad_dy, grad_dz, grad_dxx, grad_dyy, grad_dzz):
        h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz, weight, z, dz_dx, dz_dy, dz_dz = ctx.saved_tensors
        omega = ctx.omega
        apply_activation = ctx.apply_activation

        omega2 = omega * omega
        omega3 = omega2 * omega

        if apply_activation:
            sin_z = torch.sin(omega * z)
            cos_z = torch.cos(omega * z)

            h_p = omega * cos_z
            h_pp = -omega2 * sin_z
            h_ppp = -omega3 * cos_z

            grad_z = grad_h * h_p
            grad_z = grad_z + grad_dx * h_pp * dz_dx
            grad_z = grad_z + grad_dy * h_pp * dz_dy
            grad_z = grad_z + grad_dz * h_pp * dz_dz

            d2z_dxx = d2h_dxx @ weight.T
            grad_z = grad_z + grad_dxx * (h_ppp * dz_dx * dz_dx + h_pp * d2z_dxx)

            d2z_dyy = d2h_dyy @ weight.T
            grad_z = grad_z + grad_dyy * (h_ppp * dz_dy * dz_dy + h_pp * d2z_dyy)

            d2z_dzz = d2h_dzz @ weight.T
            grad_z = grad_z + grad_dzz * (h_ppp * dz_dz * dz_dz + h_pp * d2z_dzz)

            grad_dz_dx = grad_dx * h_p + grad_dxx * 2 * h_pp * dz_dx
            grad_dz_dy = grad_dy * h_p + grad_dyy * 2 * h_pp * dz_dy
            grad_dz_dz = grad_dz * h_p + grad_dzz * 2 * h_pp * dz_dz

            grad_d2z_dxx = grad_dxx * h_p
            grad_d2z_dyy = grad_dyy * h_p
            grad_d2z_dzz = grad_dzz * h_p
        else:
            grad_z = grad_h
            grad_dz_dx = grad_dx
            grad_dz_dy = grad_dy
            grad_dz_dz = grad_dz
            grad_d2z_dxx = grad_dxx
            grad_d2z_dyy = grad_dyy
            grad_d2z_dzz = grad_dzz

        grad_h_in = grad_z @ weight
        grad_dh_dx_in = grad_dz_dx @ weight
        grad_dh_dy_in = grad_dz_dy @ weight
        grad_dh_dz_in = grad_dz_dz @ weight
        grad_d2h_dxx_in = grad_d2z_dxx @ weight
        grad_d2h_dyy_in = grad_d2z_dyy @ weight
        grad_d2h_dzz_in = grad_d2z_dzz @ weight

        grad_weight = grad_z.T @ h
        grad_weight = grad_weight + grad_dz_dx.T @ dh_dx
        grad_weight = grad_weight + grad_dz_dy.T @ dh_dy
        grad_weight = grad_weight + grad_dz_dz.T @ dh_dz
        grad_weight = grad_weight + grad_d2z_dxx.T @ d2h_dxx
        grad_weight = grad_weight + grad_d2z_dyy.T @ d2h_dyy
        grad_weight = grad_weight + grad_d2z_dzz.T @ d2h_dzz

        grad_bias = grad_z.sum(dim=0)

        return (grad_h_in, grad_dh_dx_in, grad_dh_dy_in, grad_dh_dz_in,
                grad_d2h_dxx_in, grad_d2h_dyy_in, grad_d2h_dzz_in,
                grad_weight, grad_bias, None, None)


class SIREN_CUDA_3D(nn.Module):
    """SIREN MLP with CUDA V3 for 3D."""

    def __init__(self, input_dim, hidden_dim=256, n_layers=2, omega_0=0.5):
        super().__init__()
        self.omega_0 = omega_0
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == 0:
                    bound = 1.0 / layer.in_features
                else:
                    bound = np.sqrt(6.0 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)
                layer.bias.uniform_(-bound, bound)
            bound = np.sqrt(6.0 / self.output_layer.in_features) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = torch.sin(self.omega_0 * layer(h))
        return self.output_layer(h)

    def forward_with_laplacian_cuda(self, enc, dx, dy, dz, dxx, dyy, dzz):
        omega = self.omega_0
        h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz = enc, dx, dy, dz, dxx, dyy, dzz

        for layer in self.layers:
            h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz = HermiteLayerFunction3D_V3.apply(
                h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz,
                layer.weight, layer.bias, omega, True
            )

        u, du_dx, du_dy, du_dz, d2u_dxx, d2u_dyy, d2u_dzz = HermiteLayerFunction3D_V3.apply(
            h, dh_dx, dh_dy, dh_dz, d2h_dxx, d2h_dyy, d2h_dzz,
            self.output_layer.weight, self.output_layer.bias, omega, False
        )

        laplacian = d2u_dxx + d2u_dyy + d2u_dzz
        return u, laplacian


# =============================================================================
# Hermite-NGP PINN Model with Domain BC
# =============================================================================

class HermiteNGP_PINN_DomainBC(nn.Module):
    """PINN with mesh BC and domain BC."""

    def __init__(self, sampler, config=None):
        super().__init__()

        config = config or {}
        self.n_levels = config.get('n_levels', 8)
        self.log2_hashmap_size = config.get('log2_hashmap_size', 16)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.n_layers = config.get('n_layers', 2)
        self.omega = config.get('omega', 0.5)
        self.bc_weight_cap = config.get('bc_weight_cap', 50000.0)
        self.n_bc_mesh_samples = config.get('n_bc_mesh_samples', 5000)
        self.n_bc_domain_samples = config.get('n_bc_domain_samples', 5000)
        self.n_collocation = config.get('n_collocation', 50000)
        self.near_surface_ratio = config.get('near_surface_ratio', 0.5)
        self.band_width = config.get('band_width', 0.05)

        self.sampler = sampler
        self.mesh_vertices = sampler.vertices
        self.mesh_faces = sampler.faces

        # Optional: extra "inside-mesh" BC sample pool from a precomputed GT
        # volume. These are voxels that the GT solver classified u≈BC_value but
        # the kaolin check_sign considers them inside the mesh (i.e., thin-feature
        # interior cells whose FD-Laplace value is dominated by the surrounding
        # BC neighbours). Training on them with u=1 closes the train/eval mask
        # gap. Set by load_extra_bc_pool().
        self.extra_bc_pool = None         # (N, 3) coords
        self.extra_bc_values = None       # (N,) target values
        self.n_extra_bc_samples = config.get('n_extra_bc_samples', 0)

        print(f"HermiteNGP_PINN_DomainBC initialized:")
        print(f"  Mesh BC samples per iter: {self.n_bc_mesh_samples}")
        print(f"  Domain BC samples per iter: {self.n_bc_domain_samples}")
        print(f"  Extra inside-mesh BC samples per iter: {self.n_extra_bc_samples}")
        print(f"  Collocation per iter: {self.n_collocation}")
        print(f"  Near-surface ratio: {self.near_surface_ratio:.0%}, band_width: {self.band_width}")

        # Hermite Hash Encoding (CUDA)
        from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA_3D
        self.encoding = HermiteHashEncodingCUDA_3D(
            n_input_dims=3,
            n_levels=self.n_levels,
            n_features_per_level=2,
            log2_hashmap_size_1=self.log2_hashmap_size,
            log2_hashmap_size_2=self.log2_hashmap_size,
            log2_hashmap_size_3=self.log2_hashmap_size,
            log2_hashmap_size_4=self.log2_hashmap_size,
            base_resolution=4,
            per_level_scale=2.0,
        )

        encoding_dim = self.n_levels * 2
        self.mlp = SIREN_CUDA_3D(encoding_dim, self.hidden_dim, self.n_layers, omega_0=self.omega)

        self.bc_weight = config.get('bc_weight_init', 5000.0)
        self.register_buffer('level_grad_mask', torch.ones(self.n_levels))

        self.phases = config.get('phases', [
            (0, float('inf'), list(range(self.n_levels))),
        ])

        self.to(device)

    def load_extra_bc_pool(self, gt_volume_path, mode='all', gt_threshold=0.95, kaolin_chunk=65536):
        """Build the extra-BC pool from a GT volume.

        Identifies voxels with `gt >= gt_threshold` (u ≈ 1.0 by default) and
        stores them as auxiliary BC supervision (target value = mesh BC).
        The `mode` argument selects WHICH subset of these high-GT voxels to
        use as supervision:
          - 'inside_only': voxels kaolin classifies as INSIDE the bunny mesh
                           (thin-feature interior cells). Closes the train/
                           eval mask gap for those cells.
          - 'outside_only': voxels kaolin classifies as OUTSIDE. These cover
                            the d<=0.02 shell where the FD GT propagates u~1
                            from the BC but the model under-fits in
                            inverse-curvature regions. Per diagnostic, this
                            is where the WORST per-voxel errors live.
          - 'all' (default): both inside and outside high-GT voxels.

        Returns the pool size.
        """
        gt = np.load(gt_volume_path)
        R = gt.shape[0]
        is_solve = ~np.isnan(gt)
        target_mask = is_solve & (gt >= gt_threshold)
        idx = np.argwhere(target_mask)
        if idx.shape[0] == 0:
            print('  [extra-BC] empty pool, disabling')
            return 0
        coords = idx.astype(np.float32) / float(R)
        dev = next(self.mlp.parameters()).device
        coords_t = torch.from_numpy(coords).to(dev)
        if mode == 'all':
            pool = coords_t
        else:
            inside = torch.zeros(coords_t.shape[0], dtype=torch.bool, device=dev)
            for i in range(0, coords_t.shape[0], kaolin_chunk):
                inside[i:i+kaolin_chunk] = self.sampler.check_inside(coords_t[i:i+kaolin_chunk])
            if mode == 'inside_only':
                pool = coords_t[inside]
            elif mode == 'outside_only':
                pool = coords_t[~inside]
            else:
                raise ValueError(f"unknown mode: {mode}")
        vals = torch.full((pool.shape[0],), 1.0, device=dev)
        self.extra_bc_pool = pool
        self.extra_bc_values = vals
        print(f'  [extra-BC] mode={mode}, threshold={gt_threshold}: '
              f'{pool.shape[0]:,} voxels (target u=1.0)')
        return pool.shape[0]

    def get_active_levels(self, epoch):
        for start, end, levels in self.phases:
            if start <= epoch < end:
                return levels
        return list(range(self.n_levels))

    def freeze_levels(self, levels_to_freeze):
        self.level_grad_mask[:] = 1.0
        for l in levels_to_freeze:
            self.level_grad_mask[l] = 0.0

    def apply_level_mask(self):
        mask = self.level_grad_mask.view(-1, 1, 1)
        for ht in [self.encoding.hash_table_1, self.encoding.hash_table_2,
                   self.encoding.hash_table_3, self.encoding.hash_table_4]:
            if ht.grad is not None:
                ht.grad *= mask

    def forward(self, x):
        enc = self.encoding(x)
        return self.mlp(enc)

    def forward_with_laplacian(self, x):
        enc, dx, dy, dz, dxx, dyy, dzz = self.encoding.forward_with_second_derivatives_cuda(x)
        u, laplacian = self.mlp.forward_with_laplacian_cuda(enc, dx, dy, dz, dxx, dyy, dzz)
        return u, laplacian

    def sample_collocation_points(self, n_points=None):
        if n_points is None:
            n_points = self.n_collocation
        return self.sampler.sample_collocation(
            n_points,
            exterior_only=True,
            near_surface_ratio=self.near_surface_ratio,
            band_width=self.band_width
        )

    def loss_pde(self, pts):
        u, lap = self.forward_with_laplacian(pts)
        return (lap ** 2).mean()

    def loss_bc(self, n_mesh=None, n_domain=None):
        """Combined BC loss: mesh surface + domain boundary + (optional) inside-mesh extra-BC."""
        if n_mesh is None:
            n_mesh = self.n_bc_mesh_samples
        if n_domain is None:
            n_domain = self.n_bc_domain_samples

        # Sample mesh BC points (continuous mesh surface, u=1)
        mesh_pts, mesh_vals = self.sampler.sample_mesh_surface(n_mesh)

        # Sample domain BC points
        domain_pts, domain_vals = self.sampler.sample_domain_boundary(n_domain)

        all_pts_list = [mesh_pts, domain_pts]
        all_vals_list = [mesh_vals, domain_vals]
        n_extra = 0
        if self.extra_bc_pool is not None and self.n_extra_bc_samples > 0:
            n_extra = min(self.n_extra_bc_samples, self.extra_bc_pool.shape[0])
            idx = torch.randint(0, self.extra_bc_pool.shape[0], (n_extra,), device=self.extra_bc_pool.device)
            all_pts_list.append(self.extra_bc_pool[idx])
            all_vals_list.append(self.extra_bc_values[idx])

        all_pts = torch.cat(all_pts_list, dim=0)
        all_u = self.forward(all_pts)

        u_mesh = all_u[:n_mesh].squeeze()
        u_domain = all_u[n_mesh:n_mesh + n_domain].squeeze()
        if n_extra > 0:
            u_extra = all_u[n_mesh + n_domain:].squeeze()
            extra_vals = all_vals_list[2]
            # Fold extra BC into mesh-BC component (same target value = 1.0).
            mesh_combined_pred = torch.cat([u_mesh, u_extra], dim=0)
            mesh_combined_target = torch.cat([mesh_vals, extra_vals], dim=0)
            loss_mesh = ((mesh_combined_pred - mesh_combined_target) ** 2).mean()
        else:
            loss_mesh = ((u_mesh - mesh_vals) ** 2).mean()
        loss_domain = ((u_domain - domain_vals) ** 2).mean()

        return loss_mesh, loss_domain


# =============================================================================
# EMA
# =============================================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# =============================================================================
# Training
# =============================================================================

def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm() ** 2
    return total.sqrt()


def load_gt_volume(mesh_path, gt_dir):
    """Load ground truth volume for L2 evaluation."""
    # Extract mesh name from path
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    gt_path = os.path.join(gt_dir, f'{mesh_name}_gt_volume_256.npy')

    if os.path.exists(gt_path):
        gt_volume = np.load(gt_path)
        print(f"Loaded GT volume: {gt_path} (shape={gt_volume.shape})")
        return gt_volume
    else:
        print(f"WARNING: GT volume not found: {gt_path}")
        return None


def compute_l2_error(model, gt_volume, device='cuda'):
    """Compute relative L2 + MAE between model prediction and GT volume.

    Returns (l2_rel, u_pred). Use the helper below for per-region MAE.
    """
    model.eval()
    vol_res = gt_volume.shape[0]  # Should be 256

    # Create grid points (same as GT: indexing='ij' means X along axis 0, Y along axis 1, Z along axis 2)
    lin_vol = torch.linspace(0, 1, vol_res, device=device)
    X_3d, Y_3d, Z_3d = torch.meshgrid(lin_vol, lin_vol, lin_vol, indexing='ij')
    pts_3d = torch.stack([X_3d.flatten(), Y_3d.flatten(), Z_3d.flatten()], dim=1)

    # Predict in batches
    batch_size = 100000
    u_pred_list = []
    with torch.no_grad():
        for i in range(0, len(pts_3d), batch_size):
            batch_pts = pts_3d[i:i+batch_size]
            u_batch = model(batch_pts).squeeze(-1).cpu().numpy().astype(np.float32)
            u_pred_list.append(u_batch)

    u_pred = np.concatenate(u_pred_list, axis=0).reshape(vol_res, vol_res, vol_res)

    # Mask out NaN regions in GT (interior points excluded from solve)
    valid_mask = ~np.isnan(gt_volume)
    u_pred_valid = u_pred[valid_mask]
    gt_valid = gt_volume[valid_mask]

    # Compute L2 error (relative) only on valid points
    # L2 = ||u_pred - u_gt||_2 / ||u_gt||_2
    diff = u_pred_valid - gt_valid
    l2_error = np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(gt_valid ** 2)) + 1e-10)

    return l2_error, u_pred


def compute_mae_interior(u_pred, gt_volume, thresh_norm=0.10):
    """Paper Table 2 metric: MAE on points whose distance from any BC pixel is
    > `thresh_norm` (in normalized [0,1] coords). With thresh_norm=0.10 this
    reproduces the Bunny ~4.4e-3 figure in Table 2."""
    import scipy.ndimage as ndi
    valid_mask = ~np.isnan(gt_volume)
    vol_res = gt_volume.shape[0]
    edt = ndi.distance_transform_edt(valid_mask)
    deep = (edt > thresh_norm * vol_res) & valid_mask
    if deep.sum() == 0:
        return float('nan'), 0
    mae = float(np.abs(u_pred[deep] - gt_volume[deep]).mean())
    return mae, int(deep.sum())


def train(config):
    n_epochs = config.get('n_epochs', 80000)
    seed = config.get('seed', 456)
    lr = config.get('lr', 1e-3)
    num_collocation = config.get('num_collocation', 50000)
    eval_interval = config.get('eval_interval', 500)  # Changed default to 500
    save_plots = config.get('save_plots', True)
    mesh_path = config.get('mesh_path')
    mesh_bc_value = config.get('mesh_bc_value', 1.0)
    mesh_bc_weight = config.get('mesh_bc_weight', 1.0)  # extra scale on mesh BC loss
    domain_bc_value = config.get('domain_bc_value', 0.0)
    output_prefix = config.get('output_prefix', 'poisson3d_kaolin_domainbc')
    volume_res = config.get('volume_res', 256)
    gt_dir = config.get('gt_dir', None)

    use_adaptive_lr = config.get('use_adaptive_lr', True)
    lr_patience = config.get('lr_patience', 3000)
    min_lr = config.get('min_lr', 1e-6)
    use_cosine_scheduler = config.get('use_cosine_scheduler', True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Poisson 3D - Kaolin with Domain BC")
    print("=" * 70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mesh: {mesh_path}")
    print(f"Mesh BC value: {mesh_bc_value}")
    print(f"Domain BC value: {domain_bc_value}")
    print(f"Epochs: {n_epochs}, Seed: {seed}")
    print(f"Hidden: {config.get('hidden_dim', 128)}, Layers: {config.get('n_layers', 2)}")
    print(f"Collocation: {num_collocation}, resample every {config.get('resample_interval', 100)} epochs")
    print("=" * 70)

    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA extension not available!")
        return None

    if not KAOLIN_AVAILABLE:
        print("\nERROR: Kaolin not available!")
        return None

    # Create Kaolin sampler with domain BC
    print(f"\nInitializing Kaolin mesh sampler with domain BC...")
    sampler = KaolinMeshSamplerWithDomainBC(
        mesh_path,
        mesh_bc_value=mesh_bc_value,
        domain_bc_value=domain_bc_value,
        device=device
    )

    model = HermiteNGP_PINN_DomainBC(sampler, config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Load GT volume for L2 evaluation
    gt_volume = None
    if gt_dir is not None:
        gt_volume = load_gt_volume(mesh_path, gt_dir)
        if gt_volume is not None:
            print(f"L2 evaluation enabled (eval every {eval_interval} epochs)")
    else:
        print("WARNING: --gt_dir not set; skipping L2/MAE evaluation. "
              "Pass --gt_dir <dir> with a precomputed <mesh>_gt_volume_256.npy "
              "to enable evaluation.")

    # Optional: build the extra-BC pool from a GT volume.
    extra_bc_path = config.get('extra_bc_from_gt')
    if extra_bc_path and config.get('n_extra_bc_samples', 0) > 0:
        if os.path.exists(extra_bc_path):
            model = model.to(device)
            model.load_extra_bc_pool(
                extra_bc_path,
                mode=config.get('extra_bc_mode', 'outside_only'),
                gt_threshold=config.get('extra_bc_gt_threshold', 0.95),
            )
        else:
            print(f"  WARNING: --extra_bc_from_gt path not found: {extra_bc_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=min_lr
        )
    else:
        # No scheduler - constant LR (use lambda that returns 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    ema = EMA(model, decay=0.999)

    # Quick warmup
    print("\nWarmup...")
    print(f"  Mesh BC extra weight scale: {mesh_bc_weight}")
    for _ in range(20):
        pts = model.sample_collocation_points(num_collocation)
        l_bc_mesh, l_bc_domain = model.loss_bc()
        loss = model.loss_pde(pts) + model.bc_weight * (mesh_bc_weight * l_bc_mesh + l_bc_domain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

    # Training loop
    best_loss = float('inf')
    best_l2 = float('inf')
    best_epoch = 0
    best_l2_epoch = 0
    best_state = None
    best_l2_state = None
    history = []
    epochs_since_improvement = 0
    current_phase = -1

    print("\nTraining...")
    t0 = time.perf_counter()

    # Resample interval for collocation points
    resample_interval = config.get('resample_interval', 100)
    pts = model.sample_collocation_points(num_collocation)  # Initial sample

    for epoch in range(n_epochs):
        active_levels = model.get_active_levels(epoch)
        frozen_levels = [l for l in range(model.n_levels) if l not in active_levels]
        model.freeze_levels(frozen_levels)

        new_phase = None
        for i, (start, end, levels) in enumerate(model.phases):
            if start <= epoch < end:
                new_phase = i
                break
        if new_phase != current_phase:
            current_phase = new_phase
            print(f"\n  >> Phase {current_phase}: Active levels = {active_levels}")

        # Resample collocation points every resample_interval epochs.
        # If --importance-interval > 0 and we're past warmup, do residual-based
        # importance sampling: draw points where |Laplacian(u)| is large.
        if epoch % resample_interval == 0:
            imp_interval = config.get('importance_interval', 0)
            if imp_interval > 0 and epoch >= config.get('importance_warmup', 2000) \
                    and (epoch // resample_interval) % max(1, imp_interval // resample_interval) == 0:
                def _residual_fn(p):
                    u, lap = model.forward_with_laplacian(p)
                    return lap  # PDE: Δu = 0
                pts = model.sampler.sample_importance(
                    n_points=num_collocation,
                    residual_fn=_residual_fn,
                    pool_size=config.get('importance_pool_size', 300000),
                    alpha=config.get('importance_alpha', 1.0),
                    near_surface_ratio=config.get('near_surface_ratio', 0.5),
                    band_width=config.get('band_width', 0.05),
                )
            else:
                pts = model.sample_collocation_points(num_collocation)

        # GradNorm
        if (epoch + 1) % 100 == 0:
            l_pde = model.loss_pde(pts)
            l_bc_mesh, l_bc_domain = model.loss_bc()
            l_bc = mesh_bc_weight * l_bc_mesh + l_bc_domain

            optimizer.zero_grad()
            l_pde.backward(retain_graph=True)
            model.apply_level_mask()
            grad_pde = get_grad_norm(model)

            optimizer.zero_grad()
            l_bc.backward(retain_graph=True)
            model.apply_level_mask()
            grad_bc = get_grad_norm(model)

            if grad_bc > 1e-8:
                ratio = (grad_pde / grad_bc).item()
                model.bc_weight = 0.9 * model.bc_weight + 0.1 * ratio
                model.bc_weight = max(1.0, min(model.bc_weight_cap, model.bc_weight))

            loss = l_pde + model.bc_weight * l_bc
        else:
            l_bc_mesh, l_bc_domain = model.loss_bc()
            loss = model.loss_pde(pts) + model.bc_weight * (mesh_bc_weight * l_bc_mesh + l_bc_domain)

        optimizer.zero_grad()
        loss.backward()
        model.apply_level_mask()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        # Evaluate
        if (epoch + 1) % eval_interval == 0:
            ema.apply_shadow(model)

            with torch.no_grad():
                pts_eval = model.sample_collocation_points(10000)
                l_pde_eval = model.loss_pde(pts_eval).item()
                l_bc_mesh_eval, l_bc_domain_eval = model.loss_bc(n_mesh=2000, n_domain=2000)
                l_bc_mesh_eval = l_bc_mesh_eval.item()
                l_bc_domain_eval = l_bc_domain_eval.item()

            total_loss = l_pde_eval + l_bc_mesh_eval + l_bc_domain_eval

            # Compute L2 error against GT
            l2_error = -1.0
            if gt_volume is not None:
                l2_error, _ = compute_l2_error(model, gt_volume, device=device)

                # Track best L2
                if l2_error < best_l2:
                    best_l2 = l2_error
                    best_l2_epoch = epoch + 1
                    best_l2_state = {k: v.clone() for k, v in ema.shadow.items()}
                    print(f"  >> New best L2: {l2_error:.6e} @ epoch {epoch+1}")

            history.append((epoch + 1, l_pde_eval, l_bc_mesh_eval, l_bc_domain_eval, total_loss, l2_error))

            if total_loss < best_loss:
                best_loss = total_loss
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in ema.shadow.items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += eval_interval

            lr_current = optimizer.param_groups[0]['lr']
            if use_adaptive_lr and epochs_since_improvement >= lr_patience and lr_current > min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                lr_current = optimizer.param_groups[0]['lr']
                print(f"  >> LR reduced to {lr_current:.2e}")
                epochs_since_improvement = 0

            ema.restore(model)

            elapsed = time.perf_counter() - t0
            if gt_volume is not None:
                print(f"  Epoch {epoch+1:6d}: Loss={total_loss:.4e}, L2={l2_error:.6e} | PDE={l_pde_eval:.4e}, BC_mesh={l_bc_mesh_eval:.4e}, "
                      f"BC_dom={l_bc_domain_eval:.4e}, bc_w={model.bc_weight:.1f}, lr={lr_current:.2e}, time={elapsed:.0f}s")
            else:
                print(f"  Epoch {epoch+1:6d}: Loss={total_loss:.4e} | PDE={l_pde_eval:.4e}, BC_mesh={l_bc_mesh_eval:.4e}, "
                      f"BC_dom={l_bc_domain_eval:.4e}, bc_w={model.bc_weight:.1f}, lr={lr_current:.2e}, time={elapsed:.0f}s")

    # Restore best model (prefer best L2 if available)
    if best_l2_state is not None:
        print(f"\nRestoring best L2 model (L2={best_l2:.6e} @ epoch {best_l2_epoch})")
        for name, param in model.named_parameters():
            if name in best_l2_state:
                param.data = best_l2_state[name]
    elif best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]

    elapsed_total = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    if gt_volume is not None:
        print(f"Best L2 error:   {best_l2:.6e} @ epoch {best_l2_epoch}")
        print(f"Best total loss: {best_loss:.4e} @ epoch {best_epoch}")
    else:
        print(f"Best total loss: {best_loss:.4e} @ epoch {best_epoch}")
    print(f"Total Time: {elapsed_total:.1f}s ({elapsed_total/n_epochs*1000:.2f} ms/epoch)")

    results = {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'best_l2': best_l2 if gt_volume is not None else -1,
        'best_l2_epoch': best_l2_epoch if gt_volume is not None else -1,
        'total_time': elapsed_total,
        'ms_per_epoch': elapsed_total / n_epochs * 1000,
        'n_params': n_params,
        'history': history,
    }

    # Compute slice data
    slice_data = compute_slice_data(model, slice_res=256)

    if save_plots:
        save_slice_pngs(slice_data, output_prefix)

    export_for_houdini(model, sampler, output_prefix, slice_data, volume_res=volume_res)

    if config.get('save_npz', True):
        save_results_npz(config, results, output_prefix)

    # Save model
    save_model(model, output_prefix)

    return results


def save_results_npz(config, results, output_prefix):
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f'{output_prefix}_results.npz')

    history = results.get('history', [])
    save_dict = {
        'config_n_epochs': config.get('n_epochs', 0),
        'config_seed': config.get('seed', 0),
        'config_lr': config.get('lr', 0),
        'best_loss': results.get('best_loss', 0),
        'best_epoch': results.get('best_epoch', 0),
        'best_l2': results.get('best_l2', -1),
        'best_l2_epoch': results.get('best_l2_epoch', -1),
        'total_time': results.get('total_time', 0),
        'ms_per_epoch': results.get('ms_per_epoch', 0),
        'n_params': results.get('n_params', 0),
        'history_epochs': np.array([h[0] for h in history]),
        'history_pde': np.array([h[1] for h in history]),
        'history_bc_mesh': np.array([h[2] for h in history]),
        'history_bc_domain': np.array([h[3] for h in history]),
        'history_total': np.array([h[4] for h in history]),
        'history_l2': np.array([h[5] if len(h) > 5 else -1 for h in history]),
    }

    np.savez_compressed(output_path, **save_dict)
    print(f"Results saved: {output_path}")


def compute_slice_data(model, slice_res=150):
    model.eval()
    mesh_center = np.array([0.5, 0.5, 0.5])

    res = slice_res
    x_lin = torch.linspace(0, 1, res, device=device)
    y_lin = torch.linspace(0, 1, res, device=device)
    z_lin = torch.linspace(0, 1, res, device=device)

    def compute_slice(pts):
        with torch.no_grad():
            u = model(pts).cpu().numpy()
        return u

    z_slice_val = mesh_center[2]
    X_z, Y_z = torch.meshgrid(x_lin, y_lin, indexing='ij')
    pts_z = torch.stack([X_z.flatten(), Y_z.flatten(),
                        torch.full_like(X_z.flatten(), z_slice_val)], dim=1)
    u_z = compute_slice(pts_z).reshape(res, res)

    y_slice_val = mesh_center[1]
    X_y, Z_y = torch.meshgrid(x_lin, z_lin, indexing='ij')
    pts_y = torch.stack([X_y.flatten(),
                        torch.full_like(X_y.flatten(), y_slice_val),
                        Z_y.flatten()], dim=1)
    u_y = compute_slice(pts_y).reshape(res, res)

    x_slice_val = mesh_center[0]
    Y_x, Z_x = torch.meshgrid(y_lin, z_lin, indexing='ij')
    pts_x = torch.stack([torch.full_like(Y_x.flatten(), x_slice_val),
                        Y_x.flatten(), Z_x.flatten()], dim=1)
    u_x = compute_slice(pts_x).reshape(res, res)

    return {
        'resolution': res,
        'mesh_center': mesh_center,
        'z': {'slice_val': z_slice_val, 'u': u_z, 'X': X_z.cpu().numpy(), 'Y': Y_z.cpu().numpy(), 'axes': ('x', 'y')},
        'y': {'slice_val': y_slice_val, 'u': u_y, 'X': X_y.cpu().numpy(), 'Z': Z_y.cpu().numpy(), 'axes': ('x', 'z')},
        'x': {'slice_val': x_slice_val, 'u': u_x, 'Y': Y_x.cpu().numpy(), 'Z': Z_x.cpu().numpy(), 'axes': ('y', 'z')},
    }


def save_slice_pngs(slice_data, output_prefix):
    """Save X, Y, Z slice images as PNG files."""
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_prefix = os.path.join(script_dir, output_prefix)
    output_dir = os.path.dirname(full_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for axis in ['x', 'y', 'z']:
        data = slice_data[axis]
        u_slice = data['u']
        slice_val = data['slice_val']

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(u_slice.T, origin='lower', extent=[0, 1, 0, 1],
                       cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_xlabel(data['axes'][0].upper())
        ax.set_ylabel(data['axes'][1].upper())
        ax.set_title(f'{axis.upper()}-slice at {axis}={slice_val:.3f}')
        plt.colorbar(im, ax=ax, label='u')

        png_path = f'{full_prefix}_slice_{axis}.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Slice {axis.upper()} saved: {png_path}")


def save_model(model, output_prefix):
    """Save model state dict."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_prefix = os.path.join(script_dir, output_prefix)
    output_dir = os.path.dirname(full_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model_path = f'{full_prefix}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def export_for_houdini(model, sampler, output_prefix, slice_data, volume_res=256):
    model.eval()

    output_dir = os.path.join(os.path.dirname(__file__), f'{output_prefix}_houdini')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting for Houdini to: {output_dir}")

    # Use basename for filenames inside the directory
    basename = os.path.basename(output_prefix)

    vertices = sampler.vertices.cpu().numpy()
    faces = sampler.faces.cpu().numpy()

    # Export mesh
    obj_path = os.path.join(output_dir, f'{basename}_mesh.obj')
    with open(obj_path, 'w') as f:
        f.write(f"# Mesh BC={sampler.mesh_bc_value}, Domain BC={sampler.domain_bc_value}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Mesh saved: {obj_path}")

    # Compute full 3D volume
    vol_res = volume_res
    print(f"  Computing volume ({vol_res}^3)...")

    lin_vol = torch.linspace(0, 1, vol_res, device=device)
    X_3d, Y_3d, Z_3d = torch.meshgrid(lin_vol, lin_vol, lin_vol, indexing='ij')
    pts_3d = torch.stack([X_3d.flatten(), Y_3d.flatten(), Z_3d.flatten()], dim=1)

    batch_size = 100000
    u_3d_list = []
    with torch.no_grad():
        for i in range(0, len(pts_3d), batch_size):
            batch_pts = pts_3d[i:i+batch_size]
            u_batch = model(batch_pts).cpu().numpy().astype(np.float32)
            u_3d_list.append(u_batch)

    u_3d = np.concatenate(u_3d_list, axis=0).reshape(vol_res, vol_res, vol_res)

    volume_npy_path = os.path.join(output_dir, f'{basename}_volume.npy')
    np.save(volume_npy_path, u_3d)
    print(f"  Volume saved: {volume_npy_path}")

    # Save metadata
    import json
    volume_meta = {
        'shape': list(u_3d.shape),
        'bbox_min': [0.0, 0.0, 0.0],
        'bbox_max': [1.0, 1.0, 1.0],
        'mesh_bc_value': sampler.mesh_bc_value,
        'domain_bc_value': sampler.domain_bc_value,
    }
    meta_path = os.path.join(output_dir, f'{basename}_volume_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(volume_meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poisson 3D - Kaolin with Domain BC')
    parser.add_argument('--mesh', type=str, required=True, help='Path to mesh file')
    parser.add_argument('--mesh_bc_value', type=float, default=1.0, help='BC value on mesh surface')
    parser.add_argument('--domain_bc_value', type=float, default=0.0, help='BC value on domain boundary')
    parser.add_argument('--epochs', type=int, default=150000, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--omega', type=float, default=30.0, help='SIREN omega')
    parser.add_argument('--hidden', type=int, default=128, help='MLP hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--output_prefix', type=str, default='poisson3d_kaolin_domainbc', help='Output prefix')
    parser.add_argument('--no-plots', action='store_true', help='Disable plots')
    parser.add_argument('--no-save', action='store_true', help='Disable NPZ saving')
    parser.add_argument('--n_bc_mesh', type=int, default=5000, help='Mesh BC samples per iter')
    parser.add_argument('--n_extra_bc', type=int, default=0,
                       help='If > 0, sample N extra "inside-mesh" BC points per iter from the '
                            'GT-volume-derived pool: voxels that GT marks u~1 but kaolin '
                            'check_sign classifies as INSIDE the mesh (thin-feature interior '
                            'cells). Closes the train/eval mask mismatch.')
    parser.add_argument('--extra_bc_from_gt', type=str, default=None,
                       help='Path to GT volume .npy used to build the extra-BC pool. '
                            'Required if --n_extra_bc > 0.')
    parser.add_argument('--extra_bc_mode', type=str, default='outside_only',
                       choices=['inside_only', 'outside_only', 'all'],
                       help='Which subset of high-GT voxels to use as extra BC. '
                            '"outside_only" (default) targets the worst-error shell '
                            'per diagnostic; "inside_only" targets thin-feature cells; '
                            '"all" includes both.')
    parser.add_argument('--extra_bc_gt_threshold', type=float, default=0.95,
                       help='Use voxels with GT >= this threshold (default 0.95).')
    parser.add_argument('--mesh_bc_weight', type=float, default=1.0,
                       help='Extra scale applied to the mesh-surface BC loss component '
                            '(on top of the auto-tuned global bc_weight). 1.0 = default.')
    parser.add_argument('--n_levels', type=int, default=8,
                       help='Number of multi-resolution hash levels (default 8). '
                            'Higher = more spatial-frequency capacity at the cost of params/time.')
    parser.add_argument('--log2_hashmap_size', type=int, default=16,
                       help='log2 of per-level hash table size (default 16). '
                            'Higher = fewer hash collisions at the fine levels.')
    parser.add_argument('--n_bc_domain', type=int, default=5000, help='Domain BC samples per iter')
    parser.add_argument('--num_collocation', type=int, default=30000, help='Collocation points per iter')
    parser.add_argument('--near_surface_ratio', type=float, default=0.5, help='Fraction of collocation near surface (0-1)')
    parser.add_argument('--band_width', type=float, default=0.05, help='Near-surface band width')
    parser.add_argument('--volume_res', type=int, default=256, help='Volume resolution')
    parser.add_argument('--lr-patience', type=int, default=3000000, help='LR patience')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Min LR')
    parser.add_argument('--no-adaptive-lr', action='store_true', help='Disable adaptive LR')
    parser.add_argument('--no-scheduler', action='store_true', help='Disable LR scheduler (use constant LR)')
    parser.add_argument('--resample_interval', type=int, default=100, help='Resample collocation points every N epochs')
    parser.add_argument('--curriculum', type=str, default='all', choices=['all', 'coarse_to_fine'])
    parser.add_argument('--gt_dir', type=str, default=None, help='Directory containing GT volumes for L2 evaluation')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval (epochs)')
    # Residual-based importance sampling
    parser.add_argument('--importance-interval', type=int, default=0,
                       help='If > 0: every N epochs, sample collocation points with prob proportional to '
                            '|Laplacian(u)|^alpha drawn from a large candidate pool. 0 = uniform sampling (default).')
    parser.add_argument('--importance-alpha', type=float, default=1.0,
                       help='Exponent on |residual| for importance weights.')
    parser.add_argument('--importance-pool-size', type=int, default=300000,
                       help='Size of candidate pool to score for importance sampling.')
    parser.add_argument('--importance-warmup', type=int, default=2000,
                       help='Epochs before importance sampling kicks in (uniform until then).')
    args = parser.parse_args()

    n_levels = args.n_levels
    if args.curriculum == 'coarse_to_fine':
        phases = [
            # (0, 5000, [0, 1, 2]),
            # (5000, 10000, [0, 1, 2, 3, 4]),
            # (10000, float('inf'), list(range(n_levels))),
        ]
    else:
        phases = [(0, float('inf'), list(range(n_levels)))]

    config = {
        'n_epochs': args.epochs,
        'seed': args.seed,
        'lr': args.lr,
        'omega': args.omega,
        'hidden_dim': args.hidden,
        'n_layers': args.layers,
        'mesh_path': args.mesh,
        'mesh_bc_value': args.mesh_bc_value,
        'mesh_bc_weight': args.mesh_bc_weight,
        'n_extra_bc_samples': args.n_extra_bc,
        'extra_bc_from_gt': args.extra_bc_from_gt,
        'extra_bc_mode': args.extra_bc_mode,
        'extra_bc_gt_threshold': args.extra_bc_gt_threshold,
        'domain_bc_value': args.domain_bc_value,
        'output_prefix': args.output_prefix,
        'save_plots': not args.no_plots,
        'save_npz': not args.no_save,
        'phases': phases,
        'n_levels': n_levels,
        'log2_hashmap_size': args.log2_hashmap_size,
        'bc_weight_init': 5000.0,
        'bc_weight_cap': 50000.0,
        'num_collocation': args.num_collocation,
        'eval_interval': args.eval_interval,
        'gt_dir': args.gt_dir,
        'n_bc_mesh_samples': args.n_bc_mesh,
        'n_bc_domain_samples': args.n_bc_domain,
        'volume_res': args.volume_res,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'use_adaptive_lr': not args.no_adaptive_lr,
        'use_cosine_scheduler': not args.no_scheduler,
        'n_collocation': args.num_collocation,
        'near_surface_ratio': args.near_surface_ratio,
        'band_width': args.band_width,
        'resample_interval': args.resample_interval,

        # Residual-based importance sampling
        'importance_interval': args.importance_interval,
        'importance_alpha': args.importance_alpha,
        'importance_pool_size': args.importance_pool_size,
        'importance_warmup': args.importance_warmup,
    }

    train(config)
