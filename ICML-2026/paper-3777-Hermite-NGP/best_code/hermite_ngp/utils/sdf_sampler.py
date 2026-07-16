"""
CUDA-accelerated SDF Sampler using NVIDIA Kaolin.

Implements instant-ngp style sampling distribution:
- 50% exact surface points (SDF=0)
- 37.5% surface offset points (surface + logistic noise along normal)
- 12.5% uniform random points

Dependencies:
    pip install kaolin
    # or from source: pip install git+https://github.com/NVIDIAGameWorks/kaolin.git
"""

# Fix OpenMP library conflict on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np

try:
    import kaolin
    from kaolin.ops.mesh import sample_points, check_sign, index_vertices_by_faces
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    KAOLIN_AVAILABLE = True
except ImportError:
    KAOLIN_AVAILABLE = False
    print("Warning: Kaolin not available. Install with: pip install kaolin")


def sample_logistic(n_samples, loc=0.0, scale=1.0, device='cuda'):
    """
    Sample from logistic distribution using inverse CDF method.

    If U ~ Uniform(0, 1), then X = loc + scale * log(U / (1 - U)) ~ Logistic(loc, scale)

    Args:
        n_samples: Number of samples
        loc: Location parameter (mean)
        scale: Scale parameter
        device: Torch device

    Returns:
        Tensor of shape [n_samples] with logistic samples
    """
    u = torch.rand(n_samples, device=device)
    # Clamp to avoid log(0) or log(inf)
    u = torch.clamp(u, 1e-7, 1 - 1e-7)
    return loc + scale * torch.log(u / (1 - u))


class SDFSamplerCUDA:
    """
    CUDA-accelerated SDF sampler using NVIDIA Kaolin.

    Provides instant-ngp style sampling distribution for SDF training:
    - 50% exact surface points (SDF=0)
    - 37.5% surface offset points
    - 12.5% uniform random points

    All operations are performed on GPU for maximum efficiency.
    """

    def __init__(self, mesh_path, device='cuda', domain_min=0.1, domain_max=0.9):
        """
        Load mesh and prepare for CUDA sampling.

        Args:
            mesh_path: Path to mesh file (.obj, .ply, .stl)
            device: CUDA device
            domain_min: Minimum coordinate after normalization
            domain_max: Maximum coordinate after normalization
        """
        if not KAOLIN_AVAILABLE:
            raise ImportError("Kaolin is required for SDFSamplerCUDA. "
                            "Install with: pip install kaolin")

        import trimesh
        mesh = trimesh.load(mesh_path, force='mesh')

        # Store original mesh info
        self.mesh_path = mesh_path
        self.device = device
        self.domain_min = domain_min
        self.domain_max = domain_max

        # Normalize mesh vertices to [domain_min, domain_max]^3
        verts = self._normalize_mesh(mesh.vertices)

        # Store vertices and faces on GPU
        # Kaolin expects vertices as [B, V, 3] for batched operations
        self.vertices = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
        self.faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

        # Pre-compute face vertices for distance computation
        # index_vertices_by_faces returns [B, F, 3, 3] (batch, faces, vertices_per_face, xyz)
        self.face_vertices = index_vertices_by_faces(self.vertices, self.faces)

        # Compute and store face normals
        self.face_normals = self._compute_face_normals()

        # Compute face areas for weighted sampling
        self.face_areas = self._compute_face_areas()

        print(f"Loaded mesh from {mesh_path}")
        print(f"  Vertices: {self.vertices.shape[1]}")
        print(f"  Faces: {self.faces.shape[0]}")
        print(f"  Domain: [{domain_min}, {domain_max}]^3")

    def _normalize_mesh(self, vertices):
        """
        Normalize mesh vertices to [domain_min, domain_max]^3.

        Args:
            vertices: numpy array of shape [V, 3]

        Returns:
            Normalized vertices in [domain_min, domain_max]^3
        """
        verts = np.array(vertices, dtype=np.float32)

        # Compute bounding box
        verts_min = verts.min(axis=0)
        verts_max = verts.max(axis=0)
        center = (verts_min + verts_max) / 2
        extent = (verts_max - verts_min).max()

        # Normalize to [0, 1] then scale to [domain_min, domain_max]
        # with some margin (0.9 factor) to keep points away from boundary
        scale = (self.domain_max - self.domain_min) / extent * 0.9
        verts_normalized = (verts - center) * scale + 0.5

        return verts_normalized

    def _compute_face_normals(self):
        """
        Compute unit face normals.

        Returns:
            Face normals tensor of shape [F, 3]
        """
        # Get vertices of each face
        v0 = self.face_vertices[0, :, 0, :]  # [F, 3]
        v1 = self.face_vertices[0, :, 1, :]  # [F, 3]
        v2 = self.face_vertices[0, :, 2, :]  # [F, 3]

        # Compute edge vectors
        e1 = v1 - v0
        e2 = v2 - v0

        # Cross product for normal
        normals = torch.cross(e1, e2, dim=-1)

        # Normalize
        norms = torch.norm(normals, dim=-1, keepdim=True)
        normals = normals / (norms + 1e-8)

        return normals

    def _compute_face_areas(self):
        """
        Compute face areas for area-weighted sampling.

        Returns:
            Face areas tensor of shape [F]
        """
        v0 = self.face_vertices[0, :, 0, :]
        v1 = self.face_vertices[0, :, 1, :]
        v2 = self.face_vertices[0, :, 2, :]

        e1 = v1 - v0
        e2 = v2 - v0

        cross = torch.cross(e1, e2, dim=-1)
        areas = 0.5 * torch.norm(cross, dim=-1)

        return areas

    def compute_sdf_cuda(self, points):
        """
        CUDA-accelerated signed distance computation.

        Uses Kaolin's check_sign for inside/outside determination and
        point_to_mesh_distance for unsigned distance.

        Args:
            points: Tensor of shape [N, 3] with query points

        Returns:
            SDF values tensor of shape [N, 1]
            (negative inside, positive outside)
        """
        # Ensure points are on correct device
        points = points.to(self.device)

        # Add batch dimension: [N, 3] -> [1, N, 3]
        pts = points.unsqueeze(0)

        # CUDA: Get inside/outside sign
        # check_sign returns True for points inside the mesh
        signs = check_sign(self.vertices, self.faces, pts)  # [1, N]

        # CUDA: Get unsigned squared distance to mesh surface
        dist_sq, face_idx, dist_type = point_to_mesh_distance(pts, self.face_vertices)  # [1, N]

        # Take square root to get actual distance
        dist = torch.sqrt(dist_sq + 1e-10)

        # Signed distance: negative inside, positive outside
        # check_sign returns True for inside, so we negate for inside points
        sdf = torch.where(signs.squeeze(0), -dist.squeeze(0), dist.squeeze(0))

        return sdf.unsqueeze(-1)  # [N, 1]

    def sample_surface_cuda(self, n_points):
        """
        CUDA-accelerated surface sampling with normals.

        Uses area-weighted random sampling on mesh faces.

        Args:
            n_points: Number of points to sample

        Returns:
            points: Tensor of shape [N, 3] with surface points
            normals: Tensor of shape [N, 3] with corresponding normals
        """
        # Use Kaolin's sample_points for CUDA-accelerated sampling
        # Returns: points [B, N, 3], face_indices [B, N]
        pts, face_idx = sample_points(self.vertices, self.faces, n_points)

        # Get normals from face indices
        normals = self.face_normals[face_idx.squeeze(0)]  # [N, 3]

        return pts.squeeze(0), normals  # [N, 3], [N, 3]

    def sample_surface_with_barycentric(self, n_points):
        """
        Sample surface points using barycentric coordinates.

        Provides more control over the sampling process.

        Args:
            n_points: Number of points to sample

        Returns:
            points: Tensor of shape [N, 3]
            normals: Tensor of shape [N, 3]
        """
        # Area-weighted face selection
        probs = self.face_areas / self.face_areas.sum()
        face_idx = torch.multinomial(probs, n_points, replacement=True)

        # Random barycentric coordinates
        r1 = torch.sqrt(torch.rand(n_points, device=self.device))
        r2 = torch.rand(n_points, device=self.device)

        b0 = 1 - r1
        b1 = r1 * (1 - r2)
        b2 = r1 * r2

        # Get face vertices
        v0 = self.face_vertices[0, face_idx, 0, :]  # [N, 3]
        v1 = self.face_vertices[0, face_idx, 1, :]
        v2 = self.face_vertices[0, face_idx, 2, :]

        # Interpolate positions
        points = b0.unsqueeze(-1) * v0 + b1.unsqueeze(-1) * v1 + b2.unsqueeze(-1) * v2

        # Get normals
        normals = self.face_normals[face_idx]

        return points, normals

    def sample_ingp(self, n_total, offset_scale=0.01,
                    surface_ratio=0.50, offset_ratio=0.375):
        """
        instant-ngp style sampling distribution - ALL ON GPU.

        Distribution (configurable):
        - surface_ratio (default 50%) exact surface points (SDF=0)
        - offset_ratio (default 37.5%) surface offset points (surface + noise along normal)
        - (1 - surface_ratio - offset_ratio) (default 12.5%) uniform random points

        Args:
            n_total: Total number of points to sample
            offset_scale: Scale parameter for logistic noise (default 0.01)
            surface_ratio: Fraction of points on surface
            offset_ratio: Fraction of points near surface (offset)

        Returns:
            points: Tensor of shape [N, 3]
            sdf_values: Tensor of shape [N, 1]
        """
        # Calculate sample counts
        n_surface = int(n_total * surface_ratio)
        n_offset = int(n_total * offset_ratio)
        n_uniform = n_total - n_surface - n_offset

        # 1. Surface exact (CUDA): points exactly on mesh, SDF=0
        pts_surface, _ = self.sample_surface_cuda(n_surface)
        sdf_surface = torch.zeros(n_surface, 1, device=self.device)

        # 2. Surface offset (CUDA): surface + logistic noise along normal
        pts_base, normals = self.sample_surface_cuda(n_offset)

        # Logistic noise with scale 0.01 (instant-ngp uses ~leaf_size)
        # Logistic distribution has heavier tails than Gaussian
        noise = sample_logistic(n_offset, loc=0.0, scale=offset_scale, device=self.device)

        # Offset points along normal direction
        pts_offset = pts_base + noise.unsqueeze(-1) * normals

        # Clamp to keep within domain (with small margin)
        pts_offset = torch.clamp(pts_offset, self.domain_min + 0.01, self.domain_max - 0.01)

        # Compute SDF for offset points
        sdf_offset = self.compute_sdf_cuda(pts_offset)

        # 3. Uniform random (CUDA): random points in domain
        pts_uniform = torch.rand(n_uniform, 3, device=self.device)
        pts_uniform = pts_uniform * (self.domain_max - self.domain_min) + self.domain_min
        sdf_uniform = self.compute_sdf_cuda(pts_uniform)

        # Concatenate all samples
        pts = torch.cat([pts_surface, pts_offset, pts_uniform], dim=0)
        sdf = torch.cat([sdf_surface, sdf_offset, sdf_uniform], dim=0)

        # Shuffle to decorrelate sample types
        perm = torch.randperm(pts.shape[0], device=self.device)

        return pts[perm], sdf[perm]

    def sample_ingp_with_grad(self, n_total, offset_scale=0.01,
                              surface_ratio=0.50, offset_ratio=0.375):
        """
        instant-ngp style sampling with ground truth gradients for surface points.

        For surface points, the SDF gradient equals the surface normal.
        For other points, gradient is not available (use Eikonal loss instead).

        Returns:
            points: Tensor of shape [N, 3]
            sdf_values: Tensor of shape [N, 1]
            grad_gt: Tensor of shape [N, 3] - ground truth gradients (normals for surface pts)
            has_grad: Tensor of shape [N] - boolean mask for points with valid gradients
        """
        n_surface = int(n_total * surface_ratio)
        n_offset = int(n_total * offset_ratio)
        n_uniform = n_total - n_surface - n_offset

        # 1. Surface exact: SDF=0, gradient = normal
        pts_surface, normals_surface = self.sample_surface_cuda(n_surface)
        sdf_surface = torch.zeros(n_surface, 1, device=self.device)
        grad_surface = normals_surface  # gradient = outward normal at surface
        has_grad_surface = torch.ones(n_surface, dtype=torch.bool, device=self.device)

        # 2. Surface offset: gradient direction ~ normal (approximate)
        pts_base, normals_offset = self.sample_surface_cuda(n_offset)
        noise = sample_logistic(n_offset, loc=0.0, scale=offset_scale, device=self.device)
        pts_offset = pts_base + noise.unsqueeze(-1) * normals_offset
        pts_offset = torch.clamp(pts_offset, self.domain_min + 0.01, self.domain_max - 0.01)
        sdf_offset = self.compute_sdf_cuda(pts_offset)
        # For offset points, gradient direction is approximately the normal
        # but we only have Eikonal (magnitude) guarantee, so mark as no grad
        grad_offset = normals_offset  # approximate direction
        has_grad_offset = torch.zeros(n_offset, dtype=torch.bool, device=self.device)

        # 3. Uniform random: no gradient supervision
        pts_uniform = torch.rand(n_uniform, 3, device=self.device)
        pts_uniform = pts_uniform * (self.domain_max - self.domain_min) + self.domain_min
        sdf_uniform = self.compute_sdf_cuda(pts_uniform)
        grad_uniform = torch.zeros(n_uniform, 3, device=self.device)
        has_grad_uniform = torch.zeros(n_uniform, dtype=torch.bool, device=self.device)

        # Concatenate
        pts = torch.cat([pts_surface, pts_offset, pts_uniform], dim=0)
        sdf = torch.cat([sdf_surface, sdf_offset, sdf_uniform], dim=0)
        grad_gt = torch.cat([grad_surface, grad_offset, grad_uniform], dim=0)
        has_grad = torch.cat([has_grad_surface, has_grad_offset, has_grad_uniform], dim=0)

        # Shuffle
        perm = torch.randperm(pts.shape[0], device=self.device)
        return pts[perm], sdf[perm], grad_gt[perm], has_grad[perm]

    def sample_uniform(self, n_points):
        """
        Sample uniform random points within the domain.

        Args:
            n_points: Number of points to sample

        Returns:
            points: Tensor of shape [N, 3]
            sdf_values: Tensor of shape [N, 1]
        """
        pts = torch.rand(n_points, 3, device=self.device)
        pts = pts * (self.domain_max - self.domain_min) + self.domain_min
        sdf = self.compute_sdf_cuda(pts)
        return pts, sdf

    def sample_near_surface(self, n_points, max_dist=0.05):
        """
        Sample points near the mesh surface.

        Uses rejection sampling to ensure points are within max_dist of surface.

        Args:
            n_points: Number of points to sample
            max_dist: Maximum distance from surface

        Returns:
            points: Tensor of shape [N, 3]
            sdf_values: Tensor of shape [N, 1]
        """
        collected_pts = []
        collected_sdf = []

        while len(collected_pts) * (collected_pts[0].shape[0] if collected_pts else 1) < n_points:
            # Sample surface and offset
            pts_base, normals = self.sample_surface_cuda(n_points)

            # Random offset in [-max_dist, max_dist]
            offset = (torch.rand(n_points, device=self.device) * 2 - 1) * max_dist
            pts = pts_base + offset.unsqueeze(-1) * normals

            # Clamp to domain
            pts = torch.clamp(pts, self.domain_min + 0.01, self.domain_max - 0.01)

            # Compute SDF
            sdf = self.compute_sdf_cuda(pts)

            # Keep points within max_dist
            mask = torch.abs(sdf.squeeze()) < max_dist
            if mask.sum() > 0:
                collected_pts.append(pts[mask])
                collected_sdf.append(sdf[mask])

        pts = torch.cat(collected_pts, dim=0)[:n_points]
        sdf = torch.cat(collected_sdf, dim=0)[:n_points]

        return pts, sdf

    def get_mesh_bounds(self):
        """
        Get the bounding box of the normalized mesh.

        Returns:
            bounds_min: Tensor of shape [3]
            bounds_max: Tensor of shape [3]
        """
        verts = self.vertices.squeeze(0)  # [V, 3]
        bounds_min = verts.min(dim=0).values
        bounds_max = verts.max(dim=0).values
        return bounds_min, bounds_max


class SDFSamplerAnalytic:
    """
    Analytic SDF sampler for simple shapes (sphere, torus, box).

    Useful for testing and validation without requiring mesh files.
    """

    def __init__(self, shape='sphere', device='cuda', **kwargs):
        """
        Initialize analytic SDF sampler.

        Args:
            shape: 'sphere', 'torus', or 'box'
            device: CUDA device
            **kwargs: Shape-specific parameters
                sphere: center (default [0.5, 0.5, 0.5]), radius (default 0.3)
                torus: center, major_radius (default 0.3), minor_radius (default 0.1)
                box: center, half_extents (default [0.3, 0.2, 0.25])
        """
        self.shape = shape
        self.device = device
        self.domain_min = 0.1
        self.domain_max = 0.9

        # Shape parameters
        self.center = torch.tensor(
            kwargs.get('center', [0.5, 0.5, 0.5]),
            dtype=torch.float32, device=device
        )

        if shape == 'sphere':
            self.radius = kwargs.get('radius', 0.3)
        elif shape == 'torus':
            self.major_radius = kwargs.get('major_radius', 0.3)
            self.minor_radius = kwargs.get('minor_radius', 0.1)
        elif shape == 'box':
            self.half_extents = torch.tensor(
                kwargs.get('half_extents', [0.3, 0.2, 0.25]),
                dtype=torch.float32, device=device
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")

        print(f"Initialized analytic SDF sampler for {shape}")

    def compute_sdf_cuda(self, points):
        """
        Compute analytic SDF values.

        Args:
            points: Tensor of shape [N, 3]

        Returns:
            SDF values tensor of shape [N, 1]
        """
        points = points.to(self.device)
        p = points - self.center

        if self.shape == 'sphere':
            # SDF of sphere: |p| - r
            sdf = torch.norm(p, dim=-1, keepdim=True) - self.radius

        elif self.shape == 'torus':
            # SDF of torus centered at origin, axis along z
            # d = length(vec2(length(p.xy) - R, p.z)) - r
            q_xy = torch.norm(p[:, :2], dim=-1) - self.major_radius
            q = torch.stack([q_xy, p[:, 2]], dim=-1)
            sdf = (torch.norm(q, dim=-1) - self.minor_radius).unsqueeze(-1)

        elif self.shape == 'box':
            # SDF of box:
            # q = abs(p) - half_extents
            # length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0)
            q = torch.abs(p) - self.half_extents
            outside = torch.norm(torch.clamp(q, min=0), dim=-1)
            inside = torch.clamp(q.max(dim=-1).values, max=0)
            sdf = (outside + inside).unsqueeze(-1)

        return sdf

    def sample_surface_cuda(self, n_points):
        """
        Sample points on the analytic surface.

        Args:
            n_points: Number of points to sample

        Returns:
            points: Tensor of shape [N, 3]
            normals: Tensor of shape [N, 3]
        """
        if self.shape == 'sphere':
            # Random points on sphere surface
            phi = torch.rand(n_points, device=self.device) * 2 * np.pi
            cos_theta = torch.rand(n_points, device=self.device) * 2 - 1
            sin_theta = torch.sqrt(1 - cos_theta**2)

            x = sin_theta * torch.cos(phi)
            y = sin_theta * torch.sin(phi)
            z = cos_theta

            normals = torch.stack([x, y, z], dim=-1)
            points = self.center + self.radius * normals

        elif self.shape == 'torus':
            # Random points on torus surface
            u = torch.rand(n_points, device=self.device) * 2 * np.pi
            v = torch.rand(n_points, device=self.device) * 2 * np.pi

            R, r = self.major_radius, self.minor_radius

            x = (R + r * torch.cos(v)) * torch.cos(u)
            y = (R + r * torch.cos(v)) * torch.sin(u)
            z = r * torch.sin(v)

            points = self.center + torch.stack([x, y, z], dim=-1)

            # Normal direction
            nx = torch.cos(v) * torch.cos(u)
            ny = torch.cos(v) * torch.sin(u)
            nz = torch.sin(v)
            normals = torch.stack([nx, ny, nz], dim=-1)

        elif self.shape == 'box':
            # Sample on box faces
            n_per_face = n_points // 6 + 1
            all_points = []
            all_normals = []

            for axis in range(3):
                for sign in [-1, 1]:
                    # Random 2D coordinates on face
                    other_axes = [i for i in range(3) if i != axis]
                    coords = torch.rand(n_per_face, 2, device=self.device)
                    coords = coords * 2 * self.half_extents[other_axes] - self.half_extents[other_axes]

                    # Full 3D coordinates
                    pts = torch.zeros(n_per_face, 3, device=self.device)
                    pts[:, axis] = sign * self.half_extents[axis]
                    pts[:, other_axes[0]] = coords[:, 0]
                    pts[:, other_axes[1]] = coords[:, 1]

                    # Normal
                    normal = torch.zeros(3, device=self.device)
                    normal[axis] = sign
                    normals = normal.unsqueeze(0).expand(n_per_face, -1)

                    all_points.append(pts + self.center)
                    all_normals.append(normals)

            points = torch.cat(all_points, dim=0)[:n_points]
            normals = torch.cat(all_normals, dim=0)[:n_points]

        return points, normals

    def sample_ingp(self, n_total, offset_scale=0.01):
        """
        instant-ngp style sampling distribution.

        Args:
            n_total: Total number of points
            offset_scale: Scale for logistic noise

        Returns:
            points: Tensor of shape [N, 3]
            sdf_values: Tensor of shape [N, 1]
        """
        n_surface = int(n_total * 0.50)
        n_offset = int(n_total * 0.375)
        n_uniform = n_total - n_surface - n_offset

        # 1. Surface exact
        pts_surface, _ = self.sample_surface_cuda(n_surface)
        sdf_surface = torch.zeros(n_surface, 1, device=self.device)

        # 2. Surface offset
        pts_base, normals = self.sample_surface_cuda(n_offset)
        noise = sample_logistic(n_offset, loc=0.0, scale=offset_scale, device=self.device)
        pts_offset = pts_base + noise.unsqueeze(-1) * normals
        pts_offset = torch.clamp(pts_offset, self.domain_min + 0.01, self.domain_max - 0.01)
        sdf_offset = self.compute_sdf_cuda(pts_offset)

        # 3. Uniform random
        pts_uniform = torch.rand(n_uniform, 3, device=self.device)
        pts_uniform = pts_uniform * (self.domain_max - self.domain_min) + self.domain_min
        sdf_uniform = self.compute_sdf_cuda(pts_uniform)

        # Concatenate and shuffle
        pts = torch.cat([pts_surface, pts_offset, pts_uniform], dim=0)
        sdf = torch.cat([sdf_surface, sdf_offset, sdf_uniform], dim=0)
        perm = torch.randperm(pts.shape[0], device=self.device)

        return pts[perm], sdf[perm]

    def sample_ingp_with_grad(self, n_total, offset_scale=0.01):
        """
        instant-ngp style sampling with ground truth gradients for surface points.

        Returns:
            points: Tensor of shape [N, 3]
            sdf_values: Tensor of shape [N, 1]
            grad_gt: Tensor of shape [N, 3] - ground truth gradients
            has_grad: Tensor of shape [N] - boolean mask for points with valid gradients
        """
        n_surface = int(n_total * 0.50)
        n_offset = int(n_total * 0.375)
        n_uniform = n_total - n_surface - n_offset

        # 1. Surface exact: gradient = normal
        pts_surface, normals_surface = self.sample_surface_cuda(n_surface)
        sdf_surface = torch.zeros(n_surface, 1, device=self.device)
        grad_surface = normals_surface
        has_grad_surface = torch.ones(n_surface, dtype=torch.bool, device=self.device)

        # 2. Surface offset: no exact gradient
        pts_base, normals_offset = self.sample_surface_cuda(n_offset)
        noise = sample_logistic(n_offset, loc=0.0, scale=offset_scale, device=self.device)
        pts_offset = pts_base + noise.unsqueeze(-1) * normals_offset
        pts_offset = torch.clamp(pts_offset, self.domain_min + 0.01, self.domain_max - 0.01)
        sdf_offset = self.compute_sdf_cuda(pts_offset)
        grad_offset = normals_offset
        has_grad_offset = torch.zeros(n_offset, dtype=torch.bool, device=self.device)

        # 3. Uniform random: no gradient
        pts_uniform = torch.rand(n_uniform, 3, device=self.device)
        pts_uniform = pts_uniform * (self.domain_max - self.domain_min) + self.domain_min
        sdf_uniform = self.compute_sdf_cuda(pts_uniform)
        grad_uniform = torch.zeros(n_uniform, 3, device=self.device)
        has_grad_uniform = torch.zeros(n_uniform, dtype=torch.bool, device=self.device)

        # Concatenate and shuffle
        pts = torch.cat([pts_surface, pts_offset, pts_uniform], dim=0)
        sdf = torch.cat([sdf_surface, sdf_offset, sdf_uniform], dim=0)
        grad_gt = torch.cat([grad_surface, grad_offset, grad_uniform], dim=0)
        has_grad = torch.cat([has_grad_surface, has_grad_offset, has_grad_uniform], dim=0)

        perm = torch.randperm(pts.shape[0], device=self.device)
        return pts[perm], sdf[perm], grad_gt[perm], has_grad[perm]


def benchmark_sampler(sampler, n_points=50000, n_iterations=100):
    """
    Benchmark sampling speed.

    Args:
        sampler: SDFSamplerCUDA or SDFSamplerAnalytic instance
        n_points: Number of points per sample
        n_iterations: Number of iterations

    Returns:
        Average time per iteration in milliseconds
    """
    import time

    # Warmup
    for _ in range(10):
        pts, sdf = sampler.sample_ingp(n_points)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(n_iterations):
        pts, sdf = sampler.sample_ingp(n_points)

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    ms_per_iter = (elapsed / n_iterations) * 1000
    print(f"Sampling {n_points} points: {ms_per_iter:.2f} ms/iter")

    return ms_per_iter


if __name__ == "__main__":
    # Test with analytic shape
    print("Testing analytic SDF sampler (sphere)...")
    sampler_sphere = SDFSamplerAnalytic(shape='sphere', radius=0.3)
    pts, sdf = sampler_sphere.sample_ingp(10000)
    print(f"  Sample shape: {pts.shape}, SDF shape: {sdf.shape}")
    print(f"  SDF range: [{sdf.min().item():.4f}, {sdf.max().item():.4f}]")

    # Benchmark
    print("\nBenchmarking analytic sampler...")
    benchmark_sampler(sampler_sphere, n_points=50000, n_iterations=100)

    # Test with mesh if Kaolin is available
    if KAOLIN_AVAILABLE:
        import os
        test_mesh = "mesh/bunny.obj"
        if os.path.exists(test_mesh):
            print(f"\nTesting mesh SDF sampler ({test_mesh})...")
            sampler_mesh = SDFSamplerCUDA(test_mesh)
            pts, sdf = sampler_mesh.sample_ingp(10000)
            print(f"  Sample shape: {pts.shape}, SDF shape: {sdf.shape}")
            print(f"  SDF range: [{sdf.min().item():.4f}, {sdf.max().item():.4f}]")

            print("\nBenchmarking mesh sampler...")
            benchmark_sampler(sampler_mesh, n_points=50000, n_iterations=100)
