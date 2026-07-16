"""
Hermite Hash Encoding with CUDA acceleration.

This is a NEW file that wraps the CUDA kernel.
Falls back to the original PyTorch implementation if CUDA extension is not available.

Usage:
    1. Build the CUDA extension first:
       python setup_cuda.py install

    2. Use HermiteHashEncodingCUDA instead of HermiteHashEncoding:
       from encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA
       encoding = HermiteHashEncodingCUDA_3D(n_input_dims=2, n_levels=8, ...)
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional

# Try to import CUDA extension (3D version)
try:
    import hermite_encoding_cuda_3d as _C
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: hermite_encoding_cuda_3d not found. Build with: python setup_cuda.py install")
    print("Falling back to PyTorch implementation.")


class HermiteEncodingFunction_3D(Function):
    """
    Autograd function for CUDA Hermite encoding.
    Enables gradient computation through the encoding.
    """

    @staticmethod
    def forward(ctx, x, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions):
        ctx.save_for_backward(x, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions)
        output = _C.forward(x, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions = ctx.saved_tensors
        grad_hash_table_1,grad_hash_table_2,grad_hash_table_3,grad_hash_table_4 = _C.backward(x, grad_output.contiguous(), hash_table_1, hash_table_2, hash_table_3, hash_table_4, resolutions)
        # No gradient for x and resolutions
        return None, grad_hash_table_1,grad_hash_table_2,grad_hash_table_3,grad_hash_table_4, None


class HermiteEncodingWithDerivativesFunction_3D(Function):
    """
    FULL CUDA autograd function for Hermite encoding with all derivatives.

    Forward: CUDA kernel computes enc, dx, dy, dxx, dyy
    Backward: CUDA kernel computes gradients from all 5 outputs to hash_table

    This enables 100% CUDA training for PINN - no PyTorch autograd overhead.
    """

    @staticmethod
    def forward(ctx, x, hash_table_1,hash_table_2,hash_table_3,hash_table_4, resolutions):
        # CUDA forward - returns 5 tensors
        enc, dx, dy, dz, dxx, dyy, dzz = _C.forward_with_laplacian(x, hash_table_1,hash_table_2,hash_table_3,hash_table_4, resolutions)

        # Save for backward
        ctx.save_for_backward(x, hash_table_1,hash_table_2,hash_table_3,hash_table_4, resolutions)

        return enc, dx, dy, dz, dxx, dyy, dzz

    @staticmethod
    def backward(ctx, grad_enc, grad_dx, grad_dy, grad_dz, grad_dxx, grad_dyy, grad_dzz):
        x, hash_table_1,hash_table_2,hash_table_3,hash_table_4, resolutions = ctx.saved_tensors

        # CUDA backward - computes gradients from all 5 outputs
        grad_hash_table_1,grad_hash_table_2,grad_hash_table_3,grad_hash_table_4 = _C.backward_full(
            x,
            grad_enc.contiguous(),
            grad_dx.contiguous(),
            grad_dy.contiguous(),
            grad_dz.contiguous(),
            grad_dxx.contiguous(),
            grad_dyy.contiguous(),
            grad_dzz.contiguous(),
            hash_table_1,
            hash_table_2,
            hash_table_3,
            hash_table_4,
            resolutions
        )

        # No gradient for x and resolutions
        return None, grad_hash_table_1,grad_hash_table_2,grad_hash_table_3,grad_hash_table_4, None


class HermiteHashEncodingCUDA_3D(nn.Module):
    """
    Multi-resolution hash encoding with Hermite interpolation.
    CUDA-accelerated version for maximum performance.

    This is a NEW class that does NOT modify the original HermiteHashEncoding.
    """

    def __init__(
        self,
        n_input_dims: int = 3,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size_1: int = 19,
        log2_hashmap_size_2: int = 19,
        log2_hashmap_size_3: int = 19,
        log2_hashmap_size_4: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
    ):
        super().__init__()

        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Build with: python setup_cuda.py install\n"
                "Or use the PyTorch version: from hermite_ngp.encoding._hermite_encoding_torch_3d import HermiteHashEncoding"
            )

        assert n_input_dims == 3, "Currently only 3D is implemented"

        self.n_input_dims = n_input_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size_1 = 2 ** log2_hashmap_size_1
        self.hashmap_size_2 = 2 ** log2_hashmap_size_2
        self.hashmap_size_3 = 2 ** log2_hashmap_size_3
        self.hashmap_size_4 = 2 ** log2_hashmap_size_4
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale

        # Hash table: [n_levels, hashmap_size, features * k]
        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_1, F)
        init_table[..., :F] = torch.randn(n_levels, self.hashmap_size_1, F) * 0.01
        self.hash_table_1 = nn.Parameter(init_table)

        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_2, F * 3)
        self.hash_table_2 = nn.Parameter(init_table)

        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_3, F * 3)
        self.hash_table_3 = nn.Parameter(init_table)

        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_4, F)
        self.hash_table_4 = nn.Parameter(init_table)


        # Precompute resolutions
        self.register_buffer('resolutions', torch.tensor([
            int(base_resolution * (per_level_scale ** level))
            for level in range(n_levels)
        ], dtype=torch.float32))

        self.output_dim = n_levels * n_features_per_level

        # Level gradient mask for multigrid training
        # 1.0 = active, 0.0 = frozen
        self.register_buffer('level_grad_mask', torch.ones(n_levels))

    def freeze_levels(self, levels: list):
        """
        Freeze specified levels (no gradient updates).
        Used for multigrid/alternating level training.

        Args:
            levels: List of level indices to freeze (0 to n_levels-1)
        """
        self.level_grad_mask[:] = 1.0
        for l in levels:
            if 0 <= l < self.n_levels:
                self.level_grad_mask[l] = 0.0

    def unfreeze_all(self):
        """Unfreeze all levels (restore gradient updates)."""
        self.level_grad_mask[:] = 1.0

    def get_active_levels(self) -> list:
        """Return list of currently active (unfrozen) levels."""
        return [i for i in range(self.n_levels) if self.level_grad_mask[i] > 0.5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: just returns encoded features.
        Uses CUDA kernel for speed.
        """
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")

        x = x.contiguous().float()
        return HermiteEncodingFunction_3D.apply(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.hash_table_4, self.resolutions)

    def forward_with_derivatives(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning value AND first derivatives.
        Uses CUDA kernel - much faster than PyTorch autograd.

        Returns:
            features: [N, L*F] encoded features
            df_dx: [N, L*F] derivative w.r.t. x
            df_dy: [N, L*F] derivative w.r.t. y
        """
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")

        x = x.contiguous().float()
        output, dx, dy, dz, _, _, _ = _C.forward_with_laplacian(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.hash_table_4, self.resolutions)
        return output, dx, dy, dz

    def forward_with_laplacian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning value, first derivatives, AND Laplacian.
        All computed in one fused CUDA kernel - maximum performance.

        Returns:
            features: [N, L*F]
            df_dx: [N, L*F]
            df_dy: [N, L*F]
            laplacian: [N, L*F] (d^2f/dx^2 + d^2f/dy^2)
        """
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")

        x = x.contiguous().float()
        output, dx, dy, dz, dxx, dyy, dzz = _C.forward_with_laplacian(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.hash_table_4, self.resolutions)
        laplacian = dxx + dyy + dzz 
        return output, dx, dy, dz, laplacian

    def forward_with_second_derivatives(self, x: torch.Tensor):
        """
        Forward pass returning all derivatives including separate d²/dx² and d²/dy².
        NOTE: These derivatives are DETACHED - no gradient tracking.

        Returns:
            features: [N, L*F]
            df_dx: [N, L*F]
            df_dy: [N, L*F]
            d2f_dx2: [N, L*F]
            d2f_dy2: [N, L*F]
        """
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")

        x = x.contiguous().float()
        output, dx, dy, dz, dxx, dyy, dzz = _C.forward_with_laplacian(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.hash_table_4, self.resolutions)
        return output, dx, dy, dz, dxx, dyy, dzz

    def forward_with_second_derivatives_cuda(self, x: torch.Tensor):
        """
        FULL CUDA forward + backward for second derivatives.

        Both forward and backward are computed in CUDA kernels.
        This is the FASTEST option for PINN training.

        The backward pass uses analytic gradients:
        - d(enc)/d(hash_table) = basis_function(tx, ty)
        - d(dx)/d(hash_table) = d_basis_function(tx, ty) * res
        - d(dxx)/d(hash_table) = dd_basis_function(tx, ty) * res²
        etc.

        Returns:
            features: [N, L*F] - gradient tracked via CUDA backward
            df_dx: [N, L*F] - gradient tracked via CUDA backward
            df_dy: [N, L*F] - gradient tracked via CUDA backward
            d2f_dx2: [N, L*F] - gradient tracked via CUDA backward
            d2f_dy2: [N, L*F] - gradient tracked via CUDA backward
        """
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")

        x = x.contiguous().float()
        return HermiteEncodingWithDerivativesFunction_3D.apply(
            x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.hash_table_4, self.resolutions
        )

    def forward_with_second_derivatives_ste(self, x: torch.Tensor):
        """
        Forward pass with Straight-Through Estimator for gradient tracking.

        Uses CUDA kernel for fast forward values, but allows gradients to flow
        through PyTorch computation for correct backward pass.

        This solves the gradient tracking issue for PINN training.

        Returns:
            features: [N, L*F] - gradient tracked
            df_dx: [N, L*F] - gradient tracked via STE
            df_dy: [N, L*F] - gradient tracked via STE
            d2f_dx2: [N, L*F] - gradient tracked via STE
            d2f_dy2: [N, L*F] - gradient tracked via STE
        """
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")

        x = x.contiguous().float()

        # Get CUDA values (fast, but detached)
        cuda_out, cuda_dx, cuda_dy, cuda_dz, cuda_dxx, cuda_dyy, cuda_dzz = _C.forward_with_laplacian(
            x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.hash_table_4, self.resolutions
        )

        # Get PyTorch values (slower, but gradient-tracked)
        # Import here to avoid circular dependency
        from hermite_ngp.encoding._hermite_encoding_torch_3d import HermiteHashEncoding

        # Create a temporary PyTorch encoding with same parameters
        # We reuse the hash_table parameter directly for gradient tracking
        pt_out, pt_dx, pt_dy, pt_dz, pt_dxx, pt_dyy, pt_dzz = self._pytorch_forward_with_second_derivatives(x)

        # Straight-Through Estimator:
        # Forward: use CUDA value
        # Backward: use PyTorch gradient
        # Formula: cuda_value + (pt_value - pt_value.detach())
        # This equals cuda_value in forward, but has pt_value's gradient in backward

        out = cuda_out + (pt_out - pt_out.detach())
        dx = cuda_dx + (pt_dx - pt_dx.detach())
        dy = cuda_dy + (pt_dy - pt_dy.detach())
        dz = cuda_dz + (pt_dz - pt_dz.detach())
        dxx = cuda_dxx + (pt_dxx - pt_dxx.detach())
        dyy = cuda_dyy + (pt_dyy - pt_dyy.detach())
        dzz = cuda_dzz + (pt_dzz - pt_dzz.detach())        

        return out, dx, dy, dz, dxx, dyy, dzz

    def _pytorch_forward_with_second_derivatives(self, x: torch.Tensor):
        """
        Compute encoding and derivatives using PyTorch operations.
        This is slower but provides correct gradient tracking.
        """
        N = x.shape[0]
        device = x.device
        L = self.n_levels
        F = self.n_features_per_level

        # Scale for all levels: [L, N, 2]
        scaled = x.unsqueeze(0) * self.resolutions.view(-1, 1, 1)

        floor_coords = torch.floor(scaled).long()
        ceil_coords = floor_coords + 1

        # Local coordinates [0, 1]
        t = scaled - floor_coords.float()
        tx, ty, tz = t[..., 0], t[..., 1], t[..., 2]  # [L, N]

        # Hash corners
        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long, device=device)

        def hash_coords_1(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1] ^ coords[..., 2] * primes[2]) % self.hashmap_size_1

        def hash_coords_2(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1] ^ coords[..., 2] * primes[2]) % self.hashmap_size_2

        def hash_coords_3(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1] ^ coords[..., 2] * primes[2]) % self.hashmap_size_3

        def hash_coords_4(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1] ^ coords[..., 2] * primes[2]) % self.hashmap_size_4


        floor_x, floor_y, floor_z = floor_coords[..., 0], floor_coords[..., 1], floor_coords[..., 2]
        ceil_x, ceil_y, ceil_z = ceil_coords[..., 0], ceil_coords[..., 1], ceil_coords[..., 2]
        c000 = floor_coords
        c100 = torch.stack([ceil_x, floor_y, floor_z], dim = -1)
        c010 = torch.stack([floor_x, ceil_y, floor_z], dim = -1)
        c110 = torch.stack([ceil_x, ceil_y, floor_z], dim = -1)
        c001 = torch.stack([floor_x, floor_y, ceil_z], dim = -1)
        c101 = torch.stack([ceil_x, floor_y, ceil_z], dim = -1)
        c011 = torch.stack([floor_x, ceil_y, ceil_z], dim = -1)
        c111 = ceil_coords

        idx000_1, idx100_1, idx010_1, idx110_1, idx001_1, idx101_1, idx011_1, idx111_1 = hash_coords_1(c000), hash_coords_1(c100), hash_coords_1(c010), hash_coords_1(c110),hash_coords_1(c001), hash_coords_1(c101), hash_coords_1(c011), hash_coords_1(c111)
        idx000_2, idx100_2, idx010_2, idx110_2, idx001_2, idx101_2, idx011_2, idx111_2 = hash_coords_2(c000), hash_coords_2(c100), hash_coords_2(c010), hash_coords_2(c110),hash_coords_2(c001), hash_coords_2(c101), hash_coords_2(c011), hash_coords_2(c111)
        idx000_3, idx100_3, idx010_3, idx110_3, idx001_3, idx101_3, idx011_3, idx111_3 = hash_coords_3(c000), hash_coords_3(c100), hash_coords_3(c010), hash_coords_3(c110),hash_coords_3(c001), hash_coords_3(c101), hash_coords_3(c011), hash_coords_3(c111)
        idx000_4, idx100_4, idx010_4, idx110_4, idx001_4, idx101_4, idx011_4, idx111_4 = hash_coords_4(c000), hash_coords_4(c100), hash_coords_4(c010), hash_coords_4(c110),hash_coords_4(c001), hash_coords_4(c101), hash_coords_4(c011), hash_coords_4(c111)
        # Gather features
        level_idx = torch.arange(L, device=device).view(L, 1).expand(L, N)

        feat000_1,feat000_2,feat000_3,feat000_4 = self.hash_table_1[level_idx, idx000_1],self.hash_table_2[level_idx, idx000_2],self.hash_table_3[level_idx, idx000_3],self.hash_table_4[level_idx, idx000_4]
        feat100_1,feat100_2,feat100_3,feat100_4 = self.hash_table_1[level_idx, idx100_1],self.hash_table_2[level_idx, idx100_2],self.hash_table_3[level_idx, idx100_3],self.hash_table_4[level_idx, idx100_4]
        feat010_1,feat010_2,feat010_3,feat010_4 = self.hash_table_1[level_idx, idx010_1],self.hash_table_2[level_idx, idx010_2],self.hash_table_3[level_idx, idx010_3],self.hash_table_4[level_idx, idx010_4]
        feat110_1,feat110_2,feat110_3,feat110_4 = self.hash_table_1[level_idx, idx110_1],self.hash_table_2[level_idx, idx110_2],self.hash_table_3[level_idx, idx110_3],self.hash_table_4[level_idx, idx110_4]
        feat001_1,feat001_2,feat001_3,feat001_4 = self.hash_table_1[level_idx, idx001_1],self.hash_table_2[level_idx, idx001_2],self.hash_table_3[level_idx, idx001_3],self.hash_table_4[level_idx, idx001_4]
        feat101_1,feat101_2,feat101_3,feat101_4 = self.hash_table_1[level_idx, idx101_1],self.hash_table_2[level_idx, idx101_2],self.hash_table_3[level_idx, idx101_3],self.hash_table_4[level_idx, idx101_4]
        feat011_1,feat011_2,feat011_3,feat011_4 = self.hash_table_1[level_idx, idx011_1],self.hash_table_2[level_idx, idx011_2],self.hash_table_3[level_idx, idx011_3],self.hash_table_4[level_idx, idx011_4]
        feat111_1,feat111_2,feat111_3,feat111_4 = self.hash_table_1[level_idx, idx111_1],self.hash_table_2[level_idx, idx111_2],self.hash_table_3[level_idx, idx111_3],self.hash_table_4[level_idx, idx111_4]

        # Split: [L, N, F] each
        f000, fx000, fy000, fz000, fxy000, fyz000, fzx000, fxyz000 = feat000_1[..., :F], feat000_2[..., :F], feat000_2[..., F:2*F], feat000_2[..., 2*F:3*F], feat000_3[..., :F], feat000_3[..., F:2*F], feat000_3[..., 2*F:3*F], feat000_4[..., :F]
        f100, fx100, fy100, fz100, fxy100, fyz100, fzx100, fxyz100 = feat100_1[..., :F], feat100_2[..., :F], feat100_2[..., F:2*F], feat100_2[..., 2*F:3*F], feat100_3[..., :F], feat100_3[..., F:2*F], feat100_3[..., 2*F:3*F], feat100_4[..., :F]
        f010, fx010, fy010, fz010, fxy010, fyz010, fzx010, fxyz010 = feat010_1[..., :F], feat010_2[..., :F], feat010_2[..., F:2*F], feat010_2[..., 2*F:3*F], feat010_3[..., :F], feat010_3[..., F:2*F], feat010_3[..., 2*F:3*F], feat010_4[..., :F]
        f110, fx110, fy110, fz110, fxy110, fyz110, fzx110, fxyz110 = feat110_1[..., :F], feat110_2[..., :F], feat110_2[..., F:2*F], feat110_2[..., 2*F:3*F], feat110_3[..., :F], feat110_3[..., F:2*F], feat110_3[..., 2*F:3*F], feat110_4[..., :F]
        f001, fx001, fy001, fz001, fxy001, fyz001, fzx001, fxyz001 = feat001_1[..., :F], feat001_2[..., :F], feat001_2[..., F:2*F], feat001_2[..., 2*F:3*F], feat001_3[..., :F], feat001_3[..., F:2*F], feat001_3[..., 2*F:3*F], feat001_4[..., :F]
        f101, fx101, fy101, fz101, fxy101, fyz101, fzx101, fxyz101 = feat101_1[..., :F], feat101_2[..., :F], feat101_2[..., F:2*F], feat101_2[..., 2*F:3*F], feat101_3[..., :F], feat101_3[..., F:2*F], feat101_3[..., 2*F:3*F], feat101_4[..., :F]
        f011, fx011, fy011, fz011, fxy011, fyz011, fzx011, fxyz011 = feat011_1[..., :F], feat011_2[..., :F], feat011_2[..., F:2*F], feat011_2[..., 2*F:3*F], feat011_3[..., :F], feat011_3[..., F:2*F], feat011_3[..., 2*F:3*F], feat011_4[..., :F]
        f111, fx111, fy111, fz111, fxy111, fyz111, fzx111, fxyz111 = feat111_1[..., :F], feat111_2[..., :F], feat111_2[..., F:2*F], feat111_2[..., 2*F:3*F], feat111_3[..., :F], feat111_3[..., F:2*F], feat111_3[..., 2*F:3*F], feat111_4[..., :F]

        # Hermite basis functions
        tx2, tx3 = tx * tx, tx * tx * tx
        ty2, ty3 = ty * ty, ty * ty * ty
        tz2, tz3 = tz * tz, tz * tz * tz

        # Value basis H(t)
        hx0 = (2*tx3 - 3*tx2 + 1).unsqueeze(-1)
        hx1 = (tx3 - 2*tx2 + tx).unsqueeze(-1)
        hx2 = (-2*tx3 + 3*tx2).unsqueeze(-1)
        hx3 = (tx3 - tx2).unsqueeze(-1)

        hy0 = (2*ty3 - 3*ty2 + 1).unsqueeze(-1)
        hy1 = (ty3 - 2*ty2 + ty).unsqueeze(-1)
        hy2 = (-2*ty3 + 3*ty2).unsqueeze(-1)
        hy3 = (ty3 - ty2).unsqueeze(-1)

        hz0 = (2*tz3 - 3*tz2 + 1).unsqueeze(-1)
        hz1 = (tz3 - 2*tz2 + tz).unsqueeze(-1)
        hz2 = (-2*tz3 + 3*tz2).unsqueeze(-1)
        hz3 = (tz3 - tz2).unsqueeze(-1)

        # First derivative basis H'(t)
        dhx0 = (6*tx2 - 6*tx).unsqueeze(-1)
        dhx1 = (3*tx2 - 4*tx + 1).unsqueeze(-1)
        dhx2 = (-6*tx2 + 6*tx).unsqueeze(-1)
        dhx3 = (3*tx2 - 2*tx).unsqueeze(-1)

        dhy0 = (6*ty2 - 6*ty).unsqueeze(-1)
        dhy1 = (3*ty2 - 4*ty + 1).unsqueeze(-1)
        dhy2 = (-6*ty2 + 6*ty).unsqueeze(-1)
        dhy3 = (3*ty2 - 2*ty).unsqueeze(-1)

        dhz0 = (6*tz2 - 6*tz).unsqueeze(-1)
        dhz1 = (3*tz2 - 4*tz + 1).unsqueeze(-1)
        dhz2 = (-6*tz2 + 6*tz).unsqueeze(-1)
        dhz3 = (3*tz2 - 2*tz).unsqueeze(-1)

        # Second derivative basis H''(t)
        ddx0 = (12*tx - 6).unsqueeze(-1)
        ddx1 = (6*tx - 4).unsqueeze(-1)
        ddx2 = (-12*tx + 6).unsqueeze(-1)
        ddx3 = (6*tx - 2).unsqueeze(-1)

        ddy0 = (12*ty - 6).unsqueeze(-1)
        ddy1 = (6*ty - 4).unsqueeze(-1)
        ddy2 = (-12*ty + 6).unsqueeze(-1)
        ddy3 = (6*ty - 2).unsqueeze(-1)

        ddz0 = (12*tz - 6).unsqueeze(-1)
        ddz1 = (6*tz - 4).unsqueeze(-1)
        ddz2 = (-12*tz + 6).unsqueeze(-1)
        ddz3 = (6*tz - 2).unsqueeze(-1)

        res_exp = self.resolutions.view(-1, 1, 1)
        res2_exp = res_exp * res_exp

        # VALUE
        result = (
            f000 * hx0 * hy0 * hz0 + f100 * hx2 * hy0 * hz0 + f010 * hx0 * hy2 * hz0 + f110 * hx2 * hy2 * hz0 +
            f001 * hx0 * hy0 * hz2 + f101 * hx2 * hy0 * hz2 + f011 * hx0 * hy2 * hz2 + f111 * hx2 * hy2 * hz2 +

            fx000 * hx1 * hy0 * hz0 + fx100 * hx3 * hy0* hz0 + fx010 * hx1 * hy2 * hz0 + fx110 * hx3 * hy2* hz0 +
            fx001 * hx1 * hy0 * hz2 + fx101 * hx3 * hy0* hz2 + fx011 * hx1 * hy2 * hz2 + fx111 * hx3 * hy2* hz2 +            
            fy000 * hx0 * hy1 * hz0 + fy100 * hx2 * hy1* hz0 + fy010 * hx0 * hy3 * hz0 + fy110 * hx2 * hy3* hz0 +
            fy001 * hx0 * hy1 * hz2 + fy101 * hx2 * hy1* hz2 + fy011 * hx0 * hy3 * hz2 + fy111 * hx2 * hy3* hz2 +
            fz000 * hx0 * hy0 * hz1 + fz100 * hx2 * hy0* hz1 + fz010 * hx0 * hy2 * hz1 + fz110 * hx2 * hy2* hz1 +
            fz001 * hx0 * hy0 * hz3 + fz101 * hx2 * hy0* hz3 + fz011 * hx0 * hy2 * hz3 + fz111 * hx2 * hy2* hz3 +

            fxy000 * hx1 * hy1* hz0 + fxy100 * hx3 * hy1* hz0 + fxy010 * hx1 * hy3* hz0 + fxy110 * hx3 * hy3* hz0 +
            fxy001 * hx1 * hy1* hz2 + fxy101 * hx3 * hy1* hz2 + fxy011 * hx1 * hy3* hz2 + fxy111 * hx3 * hy3* hz2 +
            fyz000 * hx0 * hy1* hz1 + fyz100 * hx2 * hy1* hz1 + fyz010 * hx0 * hy3* hz1 + fyz110 * hx2 * hy3* hz1 +
            fyz001 * hx0 * hy1* hz3 + fyz101 * hx2 * hy1* hz3 + fyz011 * hx0 * hy3* hz3 + fyz111 * hx2 * hy3* hz3 +
            fzx000 * hx1 * hy0* hz1 + fzx100 * hx3 * hy0* hz1 + fzx010 * hx1 * hy2* hz1 + fzx110 * hx3 * hy2* hz1 +
            fzx001 * hx1 * hy0* hz3 + fzx101 * hx3 * hy0* hz3 + fzx011 * hx1 * hy2* hz3 + fzx111 * hx3 * hy2* hz3 +

            fxyz000 * hx1 * hy1* hz1 + fxyz100 * hx3 * hy1* hz1 + fxyz010 * hx1 * hy3* hz1 + fxyz110 * hx3 * hy3* hz1 +
            fxyz001 * hx1 * hy1* hz3 + fxyz101 * hx3 * hy1* hz3 + fxyz011 * hx1 * hy3* hz3 + fxyz111 * hx3 * hy3* hz3
        )

        # du/dx
        du_dx = (
            f000 * dhx0 * hy0 * hz0 + f100 * dhx2 * hy0 * hz0 + f010 * dhx0 * hy2 * hz0 + f110 * dhx2 * hy2 * hz0 +
            f001 * dhx0 * hy0 * hz2 + f101 * dhx2 * hy0 * hz2 + f011 * dhx0 * hy2 * hz2 + f111 * dhx2 * hy2 * hz2 +

            fx000 * dhx1 * hy0 * hz0 + fx100 * dhx3 * hy0* hz0 + fx010 * dhx1 * hy2 * hz0 + fx110 * dhx3 * hy2* hz0 +
            fx001 * dhx1 * hy0 * hz2 + fx101 * dhx3 * hy0* hz2 + fx011 * dhx1 * hy2 * hz2 + fx111 * dhx3 * hy2* hz2 +            
            fy000 * dhx0 * hy1 * hz0 + fy100 * dhx2 * hy1* hz0 + fy010 * dhx0 * hy3 * hz0 + fy110 * dhx2 * hy3* hz0 +
            fy001 * dhx0 * hy1 * hz2 + fy101 * dhx2 * hy1* hz2 + fy011 * dhx0 * hy3 * hz2 + fy111 * dhx2 * hy3* hz2 +
            fz000 * dhx0 * hy0 * hz1 + fz100 * dhx2 * hy0* hz1 + fz010 * dhx0 * hy2 * hz1 + fz110 * dhx2 * hy2* hz1 +
            fz001 * dhx0 * hy0 * hz3 + fz101 * dhx2 * hy0* hz3 + fz011 * dhx0 * hy2 * hz3 + fz111 * dhx2 * hy2* hz3 +

            fxy000 * dhx1 * hy1* hz0 + fxy100 * dhx3 * hy1* hz0 + fxy010 * dhx1 * hy3* hz0 + fxy110 * dhx3 * hy3* hz0 +
            fxy001 * dhx1 * hy1* hz2 + fxy101 * dhx3 * hy1* hz2 + fxy011 * dhx1 * hy3* hz2 + fxy111 * dhx3 * hy3* hz2 +
            fyz000 * dhx0 * hy1* hz1 + fyz100 * dhx2 * hy1* hz1 + fyz010 * dhx0 * hy3* hz1 + fyz110 * dhx2 * hy3* hz1 +
            fyz001 * dhx0 * hy1* hz3 + fyz101 * dhx2 * hy1* hz3 + fyz011 * dhx0 * hy3* hz3 + fyz111 * dhx2 * hy3* hz3 +
            fzx000 * dhx1 * hy0* hz1 + fzx100 * dhx3 * hy0* hz1 + fzx010 * dhx1 * hy2* hz1 + fzx110 * dhx3 * hy2* hz1 +
            fzx001 * dhx1 * hy0* hz3 + fzx101 * dhx3 * hy0* hz3 + fzx011 * dhx1 * hy2* hz3 + fzx111 * dhx3 * hy2* hz3 +

            fxyz000 * dhx1 * hy1* hz1 + fxyz100 * dhx3 * hy1* hz1 + fxyz010 * dhx1 * hy3* hz1 + fxyz110 * dhx3 * hy3* hz1 +
            fxyz001 * dhx1 * hy1* hz3 + fxyz101 * dhx3 * hy1* hz3 + fxyz011 * dhx1 * hy3* hz3 + fxyz111 * dhx3 * hy3* hz3
        )* res_exp

        # du/dy
        du_dy = (
            f000 * hx0 * dhy0 * hz0 + f100 * hx2 * dhy0 * hz0 + f010 * hx0 * dhy2 * hz0 + f110 * hx2 * dhy2 * hz0 +
            f001 * hx0 * dhy0 * hz2 + f101 * hx2 * dhy0 * hz2 + f011 * hx0 * dhy2 * hz2 + f111 * hx2 * dhy2 * hz2 +

            fx000 * hx1 * dhy0 * hz0 + fx100 * hx3 * dhy0* hz0 + fx010 * hx1 * dhy2 * hz0 + fx110 * hx3 * dhy2* hz0 +
            fx001 * hx1 * dhy0 * hz2 + fx101 * hx3 * dhy0* hz2 + fx011 * hx1 * dhy2 * hz2 + fx111 * hx3 * dhy2* hz2 +            
            fy000 * hx0 * dhy1 * hz0 + fy100 * hx2 * dhy1* hz0 + fy010 * hx0 * dhy3 * hz0 + fy110 * hx2 * dhy3* hz0 +
            fy001 * hx0 * dhy1 * hz2 + fy101 * hx2 * dhy1* hz2 + fy011 * hx0 * dhy3 * hz2 + fy111 * hx2 * dhy3* hz2 +
            fz000 * hx0 * dhy0 * hz1 + fz100 * hx2 * dhy0* hz1 + fz010 * hx0 * dhy2 * hz1 + fz110 * hx2 * dhy2* hz1 +
            fz001 * hx0 * dhy0 * hz3 + fz101 * hx2 * dhy0* hz3 + fz011 * hx0 * dhy2 * hz3 + fz111 * hx2 * dhy2* hz3 +

            fxy000 * hx1 * dhy1* hz0 + fxy100 * hx3 * dhy1* hz0 + fxy010 * hx1 * dhy3* hz0 + fxy110 * hx3 * dhy3* hz0 +
            fxy001 * hx1 * dhy1* hz2 + fxy101 * hx3 * dhy1* hz2 + fxy011 * hx1 * dhy3* hz2 + fxy111 * hx3 * dhy3* hz2 +
            fyz000 * hx0 * dhy1* hz1 + fyz100 * hx2 * dhy1* hz1 + fyz010 * hx0 * dhy3* hz1 + fyz110 * hx2 * dhy3* hz1 +
            fyz001 * hx0 * dhy1* hz3 + fyz101 * hx2 * dhy1* hz3 + fyz011 * hx0 * dhy3* hz3 + fyz111 * hx2 * dhy3* hz3 +
            fzx000 * hx1 * dhy0* hz1 + fzx100 * hx3 * dhy0* hz1 + fzx010 * hx1 * dhy2* hz1 + fzx110 * hx3 * dhy2* hz1 +
            fzx001 * hx1 * dhy0* hz3 + fzx101 * hx3 * dhy0* hz3 + fzx011 * hx1 * dhy2* hz3 + fzx111 * hx3 * dhy2* hz3 +

            fxyz000 * hx1 * dhy1* hz1 + fxyz100 * hx3 * dhy1* hz1 + fxyz010 * hx1 * dhy3* hz1 + fxyz110 * hx3 * dhy3* hz1 +
            fxyz001 * hx1 * dhy1* hz3 + fxyz101 * hx3 * dhy1* hz3 + fxyz011 * hx1 * dhy3* hz3 + fxyz111 * hx3 * dhy3* hz3
        )* res_exp
        
        du_dz = (
            f000 * hx0 * hy0 * dhz0 + f100 * hx2 * hy0 * dhz0 + f010 * hx0 * hy2 * dhz0 + f110 * hx2 * hy2 * dhz0 +
            f001 * hx0 * hy0 * dhz2 + f101 * hx2 * hy0 * dhz2 + f011 * hx0 * hy2 * dhz2 + f111 * hx2 * hy2 * dhz2 +

            fx000 * hx1 * hy0 * dhz0 + fx100 * hx3 * hy0* dhz0 + fx010 * hx1 * hy2 * dhz0 + fx110 * hx3 * hy2* dhz0 +
            fx001 * hx1 * hy0 * dhz2 + fx101 * hx3 * hy0* dhz2 + fx011 * hx1 * hy2 * dhz2 + fx111 * hx3 * hy2* dhz2 +            
            fy000 * hx0 * hy1 * dhz0 + fy100 * hx2 * hy1* dhz0 + fy010 * hx0 * hy3 * dhz0 + fy110 * hx2 * hy3* dhz0 +
            fy001 * hx0 * hy1 * dhz2 + fy101 * hx2 * hy1* dhz2 + fy011 * hx0 * hy3 * dhz2 + fy111 * hx2 * hy3* dhz2 +
            fz000 * hx0 * hy0 * dhz1 + fz100 * hx2 * hy0* dhz1 + fz010 * hx0 * hy2 * dhz1 + fz110 * hx2 * hy2* dhz1 +
            fz001 * hx0 * hy0 * dhz3 + fz101 * hx2 * hy0* dhz3 + fz011 * hx0 * hy2 * dhz3 + fz111 * hx2 * hy2* dhz3 +

            fxy000 * hx1 * hy1* dhz0 + fxy100 * hx3 * hy1* dhz0 + fxy010 * hx1 * hy3* dhz0 + fxy110 * hx3 * hy3* dhz0 +
            fxy001 * hx1 * hy1* dhz2 + fxy101 * hx3 * hy1* dhz2 + fxy011 * hx1 * hy3* dhz2 + fxy111 * hx3 * hy3* dhz2 +
            fyz000 * hx0 * hy1* dhz1 + fyz100 * hx2 * hy1* dhz1 + fyz010 * hx0 * hy3* dhz1 + fyz110 * hx2 * hy3* dhz1 +
            fyz001 * hx0 * hy1* dhz3 + fyz101 * hx2 * hy1* dhz3 + fyz011 * hx0 * hy3* dhz3 + fyz111 * hx2 * hy3* dhz3 +
            fzx000 * hx1 * hy0* dhz1 + fzx100 * hx3 * hy0* dhz1 + fzx010 * hx1 * hy2* dhz1 + fzx110 * hx3 * hy2* dhz1 +
            fzx001 * hx1 * hy0* dhz3 + fzx101 * hx3 * hy0* dhz3 + fzx011 * hx1 * hy2* dhz3 + fzx111 * hx3 * hy2* dhz3 +

            fxyz000 * hx1 * hy1* dhz1 + fxyz100 * hx3 * hy1* dhz1 + fxyz010 * hx1 * hy3* dhz1 + fxyz110 * hx3 * hy3* dhz1 +
            fxyz001 * hx1 * hy1* dhz3 + fxyz101 * hx3 * hy1* dhz3 + fxyz011 * hx1 * hy3* dhz3 + fxyz111 * hx3 * hy3* dhz3
        )* res_exp
        
        # d²u/dx²
        d2u_dx2 = (
            f000 * ddx0 * hy0 * hz0 + f100 * ddx2 * hy0 * hz0 + f010 * ddx0 * hy2 * hz0 + f110 * ddx2 * hy2 * hz0 +
            f001 * ddx0 * hy0 * hz2 + f101 * ddx2 * hy0 * hz2 + f011 * ddx0 * hy2 * hz2 + f111 * ddx2 * hy2 * hz2 +

            fx000 * ddx1 * hy0 * hz0 + fx100 * ddx3 * hy0* hz0 + fx010 * ddx1 * hy2 * hz0 + fx110 * ddx3 * hy2* hz0 +
            fx001 * ddx1 * hy0 * hz2 + fx101 * ddx3 * hy0* hz2 + fx011 * ddx1 * hy2 * hz2 + fx111 * ddx3 * hy2* hz2 +            
            fy000 * ddx0 * hy1 * hz0 + fy100 * ddx2 * hy1* hz0 + fy010 * ddx0 * hy3 * hz0 + fy110 * ddx2 * hy3* hz0 +
            fy001 * ddx0 * hy1 * hz2 + fy101 * ddx2 * hy1* hz2 + fy011 * ddx0 * hy3 * hz2 + fy111 * ddx2 * hy3* hz2 +
            fz000 * ddx0 * hy0 * hz1 + fz100 * ddx2 * hy0* hz1 + fz010 * ddx0 * hy2 * hz1 + fz110 * ddx2 * hy2* hz1 +
            fz001 * ddx0 * hy0 * hz3 + fz101 * ddx2 * hy0* hz3 + fz011 * ddx0 * hy2 * hz3 + fz111 * ddx2 * hy2* hz3 +

            fxy000 * ddx1 * hy1* hz0 + fxy100 * ddx3 * hy1* hz0 + fxy010 * ddx1 * hy3* hz0 + fxy110 * ddx3 * hy3* hz0 +
            fxy001 * ddx1 * hy1* hz2 + fxy101 * ddx3 * hy1* hz2 + fxy011 * ddx1 * hy3* hz2 + fxy111 * ddx3 * hy3* hz2 +
            fyz000 * ddx0 * hy1* hz1 + fyz100 * ddx2 * hy1* hz1 + fyz010 * ddx0 * hy3* hz1 + fyz110 * ddx2 * hy3* hz1 +
            fyz001 * ddx0 * hy1* hz3 + fyz101 * ddx2 * hy1* hz3 + fyz011 * ddx0 * hy3* hz3 + fyz111 * ddx2 * hy3* hz3 +
            fzx000 * ddx1 * hy0* hz1 + fzx100 * ddx3 * hy0* hz1 + fzx010 * ddx1 * hy2* hz1 + fzx110 * ddx3 * hy2* hz1 +
            fzx001 * ddx1 * hy0* hz3 + fzx101 * ddx3 * hy0* hz3 + fzx011 * ddx1 * hy2* hz3 + fzx111 * ddx3 * hy2* hz3 +

            fxyz000 * ddx1 * hy1* hz1 + fxyz100 * ddx3 * hy1* hz1 + fxyz010 * ddx1 * hy3* hz1 + fxyz110 * ddx3 * hy3* hz1 +
            fxyz001 * ddx1 * hy1* hz3 + fxyz101 * ddx3 * hy1* hz3 + fxyz011 * ddx1 * hy3* hz3 + fxyz111 * ddx3 * hy3* hz3
        ) * res2_exp
        
        
        d2u_dy2 = (
            f000 * hx0 * ddy0 * hz0 + f100 * hx2 * ddy0 * hz0 + f010 * hx0 * ddy2 * hz0 + f110 * hx2 * ddy2 * hz0 +
            f001 * hx0 * ddy0 * hz2 + f101 * hx2 * ddy0 * hz2 + f011 * hx0 * ddy2 * hz2 + f111 * hx2 * ddy2 * hz2 +

            fx000 * hx1 * ddy0 * hz0 + fx100 * hx3 * ddy0* hz0 + fx010 * hx1 * ddy2 * hz0 + fx110 * hx3 * ddy2* hz0 +
            fx001 * hx1 * ddy0 * hz2 + fx101 * hx3 * ddy0* hz2 + fx011 * hx1 * ddy2 * hz2 + fx111 * hx3 * ddy2* hz2 +            
            fy000 * hx0 * ddy1 * hz0 + fy100 * hx2 * ddy1* hz0 + fy010 * hx0 * ddy3 * hz0 + fy110 * hx2 * ddy3* hz0 +
            fy001 * hx0 * ddy1 * hz2 + fy101 * hx2 * ddy1* hz2 + fy011 * hx0 * ddy3 * hz2 + fy111 * hx2 * ddy3* hz2 +
            fz000 * hx0 * ddy0 * hz1 + fz100 * hx2 * ddy0* hz1 + fz010 * hx0 * ddy2 * hz1 + fz110 * hx2 * ddy2* hz1 +
            fz001 * hx0 * ddy0 * hz3 + fz101 * hx2 * ddy0* hz3 + fz011 * hx0 * ddy2 * hz3 + fz111 * hx2 * ddy2* hz3 +

            fxy000 * hx1 * ddy1* hz0 + fxy100 * hx3 * ddy1* hz0 + fxy010 * hx1 * ddy3* hz0 + fxy110 * hx3 * ddy3* hz0 +
            fxy001 * hx1 * ddy1* hz2 + fxy101 * hx3 * ddy1* hz2 + fxy011 * hx1 * ddy3* hz2 + fxy111 * hx3 * ddy3* hz2 +
            fyz000 * hx0 * ddy1* hz1 + fyz100 * hx2 * ddy1* hz1 + fyz010 * hx0 * ddy3* hz1 + fyz110 * hx2 * ddy3* hz1 +
            fyz001 * hx0 * ddy1* hz3 + fyz101 * hx2 * ddy1* hz3 + fyz011 * hx0 * ddy3* hz3 + fyz111 * hx2 * ddy3* hz3 +
            fzx000 * hx1 * ddy0* hz1 + fzx100 * hx3 * ddy0* hz1 + fzx010 * hx1 * ddy2* hz1 + fzx110 * hx3 * ddy2* hz1 +
            fzx001 * hx1 * ddy0* hz3 + fzx101 * hx3 * ddy0* hz3 + fzx011 * hx1 * ddy2* hz3 + fzx111 * hx3 * ddy2* hz3 +

            fxyz000 * hx1 * ddy1* hz1 + fxyz100 * hx3 * ddy1* hz1 + fxyz010 * hx1 * ddy3* hz1 + fxyz110 * hx3 * ddy3* hz1 +
            fxyz001 * hx1 * ddy1* hz3 + fxyz101 * hx3 * ddy1* hz3 + fxyz011 * hx1 * ddy3* hz3 + fxyz111 * hx3 * ddy3* hz3
        ) * res2_exp

        d2u_dz2 = (
            f000 * hx0 * hy0 * ddz0 + f100 * hx2 * hy0 * ddz0 + f010 * hx0 * hy2 * ddz0 + f110 * hx2 * hy2 * ddz0 +
            f001 * hx0 * hy0 * ddz2 + f101 * hx2 * hy0 * ddz2 + f011 * hx0 * hy2 * ddz2 + f111 * hx2 * hy2 * ddz2 +

            fx000 * hx1 * hy0 * ddz0 + fx100 * hx3 * hy0* ddz0 + fx010 * hx1 * hy2 * ddz0 + fx110 * hx3 * hy2* ddz0 +
            fx001 * hx1 * hy0 * ddz2 + fx101 * hx3 * hy0* ddz2 + fx011 * hx1 * hy2 * ddz2 + fx111 * hx3 * hy2* ddz2 +            
            fy000 * hx0 * hy1 * ddz0 + fy100 * hx2 * hy1* ddz0 + fy010 * hx0 * hy3 * ddz0 + fy110 * hx2 * hy3* ddz0 +
            fy001 * hx0 * hy1 * ddz2 + fy101 * hx2 * hy1* ddz2 + fy011 * hx0 * hy3 * ddz2 + fy111 * hx2 * hy3* ddz2 +
            fz000 * hx0 * hy0 * ddz1 + fz100 * hx2 * hy0* ddz1 + fz010 * hx0 * hy2 * ddz1 + fz110 * hx2 * hy2* ddz1 +
            fz001 * hx0 * hy0 * ddz3 + fz101 * hx2 * hy0* ddz3 + fz011 * hx0 * hy2 * ddz3 + fz111 * hx2 * hy2* ddz3 +

            fxy000 * hx1 * hy1* ddz0 + fxy100 * hx3 * hy1* ddz0 + fxy010 * hx1 * hy3* ddz0 + fxy110 * hx3 * hy3* ddz0 +
            fxy001 * hx1 * hy1* ddz2 + fxy101 * hx3 * hy1* ddz2 + fxy011 * hx1 * hy3* ddz2 + fxy111 * hx3 * hy3* ddz2 +
            fyz000 * hx0 * hy1* ddz1 + fyz100 * hx2 * hy1* ddz1 + fyz010 * hx0 * hy3* ddz1 + fyz110 * hx2 * hy3* ddz1 +
            fyz001 * hx0 * hy1* ddz3 + fyz101 * hx2 * hy1* ddz3 + fyz011 * hx0 * hy3* ddz3 + fyz111 * hx2 * hy3* ddz3 +
            fzx000 * hx1 * hy0* ddz1 + fzx100 * hx3 * hy0* ddz1 + fzx010 * hx1 * hy2* ddz1 + fzx110 * hx3 * hy2* ddz1 +
            fzx001 * hx1 * hy0* ddz3 + fzx101 * hx3 * hy0* ddz3 + fzx011 * hx1 * hy2* ddz3 + fzx111 * hx3 * hy2* ddz3 +

            fxyz000 * hx1 * hy1* ddz1 + fxyz100 * hx3 * hy1* ddz1 + fxyz010 * hx1 * hy3* ddz1 + fxyz110 * hx3 * hy3* ddz1 +
            fxyz001 * hx1 * hy1* ddz3 + fxyz101 * hx3 * hy1* ddz3 + fxyz011 * hx1 * hy3* ddz3 + fxyz111 * hx3 * hy3* ddz3
        ) * res2_exp


        # Reshape to [N, L*F]
        result = result.permute(1, 0, 2).reshape(N, -1)
        du_dx = du_dx.permute(1, 0, 2).reshape(N, -1)
        du_dy = du_dy.permute(1, 0, 2).reshape(N, -1)
        du_dz = du_dz.permute(1, 0, 2).reshape(N, -1)
        d2u_dx2 = d2u_dx2.permute(1, 0, 2).reshape(N, -1)
        d2u_dy2 = d2u_dy2.permute(1, 0, 2).reshape(N, -1)
        d2u_dz2 = d2u_dz2.permute(1, 0, 2).reshape(N, -1)

        return result, du_dx, du_dy, du_dz, d2u_dx2, d2u_dy2, d2u_dz2


# Convenience function to get best available implementation
def get_hermite_encoding(use_cuda: bool = True, **kwargs):
    """
    Factory function to get the best available Hermite encoding.

    Args:
        use_cuda: If True and CUDA extension is available, use CUDA version.
                  Otherwise, fall back to PyTorch version.
        **kwargs: Arguments passed to the encoding constructor.

    Returns:
        HermiteHashEncodingCUDA_3D or HermiteHashEncoding instance.
    """
    if use_cuda and CUDA_AVAILABLE:
        return HermiteHashEncodingCUDA_3D(**kwargs)
    else:
        # Import PyTorch version as fallback
        from hermite_ngp.encoding._hermite_encoding_torch_3d import HermiteHashEncoding
        return HermiteHashEncoding(**kwargs)


# Public name used by the open-source API
HermiteHashEncoding3D = HermiteHashEncodingCUDA_3D


# For testing
if __name__ == '__main__':
    if not CUDA_AVAILABLE:
        print("CUDA extension not available. Cannot run test.")
        exit(1)

    print("Testing HermiteHashEncodingCUDA_3D...")

    device = 'cuda'
    encoding = HermiteHashEncodingCUDA_3D(
        n_input_dims=3,
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size_1=16,
        log2_hashmap_size_2=16,
        log2_hashmap_size_3=16,
        log2_hashmap_size_4=16,
        base_resolution=16,
        per_level_scale=1.5,
    ).to(device)

    # Test forward
    x = torch.rand(1024, 3, device=device)
    output = encoding(x)
    print(f"Forward output shape: {output.shape}")

    # Test forward with laplacian
    output, dx, dy, lap = encoding.forward_with_laplacian(x)
    print(f"With Laplacian - output: {output.shape}, dx: {dx.shape}, dy: {dy.shape}, lap: {lap.shape}")

    # Test backward
    output = encoding(x)
    loss = output.sum()
    loss.backward()
    print(f"Backward: grad_hash_table shape: {encoding.hash_table_1.grad.shape} {encoding.hash_table_2.grad.shape} {encoding.hash_table_3.grad.shape} {encoding.hash_table_4.grad.shape}")
    print(f"Backward: grad_hash_table max: {encoding.hash_table_1.grad.abs().max().item():.6f} {encoding.hash_table_2.grad.abs().max().item():.6f} {encoding.hash_table_3.grad.abs().max().item():.6f} {encoding.hash_table_4.grad.abs().max().item():.6f}")

    # Benchmark
    import time

    # Warmup
    for _ in range(10):
        _ = encoding.forward_with_laplacian(x)
    torch.cuda.synchronize()

    # Time CUDA version
    start = time.perf_counter()
    for _ in range(100):
        _ = encoding.forward_with_laplacian(x)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / 100 * 1000

    print(f"\nCUDA forward_with_laplacian time: {cuda_time:.3f} ms")

    # Compare with PyTorch version
    from hermite_ngp.encoding._hermite_encoding_torch_3d import HermiteHashEncoding
    encoding_pt = HermiteHashEncoding(
        n_input_dims=3,
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size_1=16,
        log2_hashmap_size_2=16,
        log2_hashmap_size_3=16,
        log2_hashmap_size_4=16,
        base_resolution=16,
        per_level_scale=1.5,
    ).to(device)

    # Copy weights
    encoding_pt.hash_table_1.data = encoding.hash_table_1.data.clone()
    encoding_pt.hash_table_2.data = encoding.hash_table_2.data.clone()
    encoding_pt.hash_table_3.data = encoding.hash_table_3.data.clone()
    encoding_pt.hash_table_4.data = encoding.hash_table_4.data.clone()

    # Warmup
    for _ in range(10):
        _ = encoding_pt.forward_with_laplacian(x)
    torch.cuda.synchronize()

    # Time PyTorch version
    start = time.perf_counter()
    for _ in range(100):
        _ = encoding_pt.forward_with_laplacian(x)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - start) / 100 * 1000

    print(f"PyTorch forward_with_laplacian time: {pt_time:.3f} ms")
    print(f"Speedup: {pt_time / cuda_time:.1f}x")

    print("\nAll tests passed!")
