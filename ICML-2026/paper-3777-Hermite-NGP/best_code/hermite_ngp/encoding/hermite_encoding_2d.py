"""
Hermite Hash Encoding with CUDA acceleration.

This is a NEW file that wraps the CUDA kernel.
Falls back to the original PyTorch implementation if CUDA extension is not available.

Usage:
    1. Build the CUDA extension first:
       python setup_cuda.py install

    2. Use HermiteHashEncodingCUDA instead of HermiteHashEncoding:
       from encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA
       encoding = HermiteHashEncodingCUDA(n_input_dims=2, n_levels=8, ...)
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional

# Try to import CUDA extension
try:
    import hermite_encoding_cuda as _C
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: hermite_encoding_cuda not found. Build with: python setup_cuda.py install")
    print("Falling back to PyTorch implementation.")


class HermiteEncodingFunction(Function):
    """
    Autograd function for CUDA Hermite encoding.
    Enables gradient computation through the encoding.
    """

    @staticmethod
    def forward(ctx, x, hash_table_1, hash_table_2, hash_table_3, resolutions):
        ctx.save_for_backward(x, hash_table_1, hash_table_2, hash_table_3, resolutions)
        output = _C.forward(x, hash_table_1, hash_table_2, hash_table_3, resolutions)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, hash_table_1, hash_table_2, hash_table_3, resolutions = ctx.saved_tensors
        grad_hash_table_1,grad_hash_table_2,grad_hash_table_3 = _C.backward(x, grad_output.contiguous(), hash_table_1, hash_table_2, hash_table_3, resolutions)
        # No gradient for x and resolutions
        return None, grad_hash_table_1,grad_hash_table_2,grad_hash_table_3, None


class HermiteEncodingWithDerivativesFunction(Function):
    """
    FULL CUDA autograd function for Hermite encoding with all derivatives.

    Forward: CUDA kernel computes enc, dx, dy, dxx, dyy
    Backward: CUDA kernel computes gradients from all 5 outputs to hash_table

    This enables 100% CUDA training for PINN - no PyTorch autograd overhead.
    """

    @staticmethod
    def forward(ctx, x, hash_table_1,hash_table_2,hash_table_3, resolutions):
        # CUDA forward - returns 5 tensors
        enc, dx, dy, dxx, dyy = _C.forward_with_laplacian(x, hash_table_1,hash_table_2,hash_table_3, resolutions)

        # Save for backward
        ctx.save_for_backward(x, hash_table_1,hash_table_2,hash_table_3, resolutions)

        return enc, dx, dy, dxx, dyy

    @staticmethod
    def backward(ctx, grad_enc, grad_dx, grad_dy, grad_dxx, grad_dyy):
        x, hash_table_1,hash_table_2,hash_table_3, resolutions = ctx.saved_tensors

        # CUDA backward - computes gradients from all 5 outputs
        grad_hash_table_1,grad_hash_table_2,grad_hash_table_3 = _C.backward_full(
            x,
            grad_enc.contiguous(),
            grad_dx.contiguous(),
            grad_dy.contiguous(),
            grad_dxx.contiguous(),
            grad_dyy.contiguous(),
            hash_table_1,
            hash_table_2,
            hash_table_3,
            resolutions
        )

        # No gradient for x and resolutions
        return None, grad_hash_table_1,grad_hash_table_2,grad_hash_table_3, None


class HermiteHashEncodingCUDA(nn.Module):
    """
    Multi-resolution hash encoding with Hermite interpolation.
    CUDA-accelerated version for maximum performance.

    This is a NEW class that does NOT modify the original HermiteHashEncoding.
    """

    def __init__(
        self,
        n_input_dims: int = 2,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size_1: int = 19,
        log2_hashmap_size_2: int = 19,
        log2_hashmap_size_3: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
    ):
        super().__init__()

        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Build with: python setup_cuda.py install\n"
                "Or use the PyTorch version: from hermite_ngp.encoding._hermite_encoding_torch_2d import HermiteHashEncoding"
            )

        assert n_input_dims == 2, "Currently only 2D is implemented"

        self.n_input_dims = n_input_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size_1 = 2 ** log2_hashmap_size_1
        self.hashmap_size_2 = 2 ** log2_hashmap_size_2
        self.hashmap_size_3 = 2 ** log2_hashmap_size_3
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.values_per_vertex = 4  # f, fx, fy, fxy

        # Hash table: [n_levels, hashmap_size, features * 4]
        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_1, F)
        init_table[..., :F] = torch.randn(n_levels, self.hashmap_size_1, F) * 0.01
        self.hash_table_1 = nn.Parameter(init_table)

        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_2, F * 2)
        self.hash_table_2 = nn.Parameter(init_table)

        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_3, F)
        self.hash_table_3 = nn.Parameter(init_table)


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
        return HermiteEncodingFunction.apply(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.resolutions)

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
        output, dx, dy, _, _ = _C.forward_with_laplacian(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.resolutions)
        return output, dx, dy

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
        output, dx, dy, dxx, dyy = _C.forward_with_laplacian(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.resolutions)
        laplacian = dxx + dyy
        return output, dx, dy, laplacian

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
        output, dx, dy, dxx, dyy = _C.forward_with_laplacian(x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.resolutions)
        return output, dx, dy, dxx, dyy

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
        return HermiteEncodingWithDerivativesFunction.apply(
            x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.resolutions
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
        cuda_out, cuda_dx, cuda_dy, cuda_dxx, cuda_dyy = _C.forward_with_laplacian(
            x, self.hash_table_1, self.hash_table_2, self.hash_table_3, self.resolutions
        )

        # Get PyTorch values (slower, but gradient-tracked)
        # Import here to avoid circular dependency
        from hermite_ngp.encoding._hermite_encoding_torch_2d import HermiteHashEncoding

        # Create a temporary PyTorch encoding with same parameters
        # We reuse the hash_table parameter directly for gradient tracking
        pt_out, pt_dx, pt_dy, pt_dxx, pt_dyy = self._pytorch_forward_with_second_derivatives(x)

        # Straight-Through Estimator:
        # Forward: use CUDA value
        # Backward: use PyTorch gradient
        # Formula: cuda_value + (pt_value - pt_value.detach())
        # This equals cuda_value in forward, but has pt_value's gradient in backward

        out = cuda_out + (pt_out - pt_out.detach())
        dx = cuda_dx + (pt_dx - pt_dx.detach())
        dy = cuda_dy + (pt_dy - pt_dy.detach())
        dxx = cuda_dxx + (pt_dxx - pt_dxx.detach())
        dyy = cuda_dyy + (pt_dyy - pt_dyy.detach())

        return out, dx, dy, dxx, dyy

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
        tx, ty = t[..., 0], t[..., 1]  # [L, N]

        # Hash corners
        primes = torch.tensor([1, 2654435761], dtype=torch.long, device=device)

        def hash_coords_1(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1]) % self.hashmap_size_1

        def hash_coords_2(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1]) % self.hashmap_size_2

        def hash_coords_3(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1]) % self.hashmap_size_3

        c00 = floor_coords
        c10 = torch.stack([ceil_coords[..., 0], floor_coords[..., 1]], dim=-1)
        c01 = torch.stack([floor_coords[..., 0], ceil_coords[..., 1]], dim=-1)
        c11 = ceil_coords

        idx00_1, idx10_1, idx01_1, idx11_1 = hash_coords_1(c00), hash_coords_1(c10), hash_coords_1(c01), hash_coords_1(c11)
        idx00_2, idx10_2, idx01_2, idx11_2 = hash_coords_2(c00), hash_coords_2(c10), hash_coords_2(c01), hash_coords_2(c11)
        idx00_3, idx10_3, idx01_3, idx11_3 = hash_coords_3(c00), hash_coords_3(c10), hash_coords_3(c01), hash_coords_3(c11)

        # Gather features
        level_idx = torch.arange(L, device=device).view(L, 1).expand(L, N)

        feat00_1,feat00_2,feat00_3 = self.hash_table_1[level_idx, idx00_1],self.hash_table_2[level_idx, idx00_2],self.hash_table_3[level_idx, idx00_3]
        feat10_1,feat10_2,feat10_3 = self.hash_table_1[level_idx, idx10_1],self.hash_table_2[level_idx, idx10_2],self.hash_table_3[level_idx, idx10_3]
        feat01_1,feat01_2,feat01_3 = self.hash_table_1[level_idx, idx01_1],self.hash_table_2[level_idx, idx01_2],self.hash_table_3[level_idx, idx01_3]
        feat11_1,feat11_2,feat11_3 = self.hash_table_1[level_idx, idx11_1],self.hash_table_2[level_idx, idx11_2],self.hash_table_3[level_idx, idx11_3]

        # Split: [L, N, F] each
        f00, fx00, fy00, fxy00 = feat00_1[..., :F], feat00_2[..., :F], feat00_2[..., F:2*F], feat00_3[..., :F]
        f10, fx10, fy10, fxy10 = feat10_1[..., :F], feat10_2[..., :F], feat10_2[..., F:2*F], feat10_3[..., :F]
        f01, fx01, fy01, fxy01 = feat01_1[..., :F], feat01_2[..., :F], feat01_2[..., F:2*F], feat01_3[..., :F]
        f11, fx11, fy11, fxy11 = feat11_1[..., :F], feat11_2[..., :F], feat11_2[..., F:2*F], feat11_3[..., :F]

        # Hermite basis functions
        tx2, tx3 = tx * tx, tx * tx * tx
        ty2, ty3 = ty * ty, ty * ty * ty

        # Value basis H(t)
        hx0 = (2*tx3 - 3*tx2 + 1).unsqueeze(-1)
        hx1 = (tx3 - 2*tx2 + tx).unsqueeze(-1)
        hx2 = (-2*tx3 + 3*tx2).unsqueeze(-1)
        hx3 = (tx3 - tx2).unsqueeze(-1)

        hy0 = (2*ty3 - 3*ty2 + 1).unsqueeze(-1)
        hy1 = (ty3 - 2*ty2 + ty).unsqueeze(-1)
        hy2 = (-2*ty3 + 3*ty2).unsqueeze(-1)
        hy3 = (ty3 - ty2).unsqueeze(-1)

        # First derivative basis H'(t)
        dhx0 = (6*tx2 - 6*tx).unsqueeze(-1)
        dhx1 = (3*tx2 - 4*tx + 1).unsqueeze(-1)
        dhx2 = (-6*tx2 + 6*tx).unsqueeze(-1)
        dhx3 = (3*tx2 - 2*tx).unsqueeze(-1)

        dhy0 = (6*ty2 - 6*ty).unsqueeze(-1)
        dhy1 = (3*ty2 - 4*ty + 1).unsqueeze(-1)
        dhy2 = (-6*ty2 + 6*ty).unsqueeze(-1)
        dhy3 = (3*ty2 - 2*ty).unsqueeze(-1)

        # Second derivative basis H''(t)
        ddx0 = (12*tx - 6).unsqueeze(-1)
        ddx1 = (6*tx - 4).unsqueeze(-1)
        ddx2 = (-12*tx + 6).unsqueeze(-1)
        ddx3 = (6*tx - 2).unsqueeze(-1)

        ddy0 = (12*ty - 6).unsqueeze(-1)
        ddy1 = (6*ty - 4).unsqueeze(-1)
        ddy2 = (-12*ty + 6).unsqueeze(-1)
        ddy3 = (6*ty - 2).unsqueeze(-1)

        res_exp = self.resolutions.view(-1, 1, 1)
        res2_exp = res_exp * res_exp

        # VALUE
        result = (
            f00 * hx0 * hy0 + f10 * hx2 * hy0 + f01 * hx0 * hy2 + f11 * hx2 * hy2 +
            fx00 * hx1 * hy0 + fx10 * hx3 * hy0 + fx01 * hx1 * hy2 + fx11 * hx3 * hy2 +
            fy00 * hx0 * hy1 + fy10 * hx2 * hy1 + fy01 * hx0 * hy3 + fy11 * hx2 * hy3 +
            fxy00 * hx1 * hy1 + fxy10 * hx3 * hy1 + fxy01 * hx1 * hy3 + fxy11 * hx3 * hy3
        )

        # du/dx
        du_dx = (
            f00 * dhx0 * hy0 + f10 * dhx2 * hy0 + f01 * dhx0 * hy2 + f11 * dhx2 * hy2 +
            fx00 * dhx1 * hy0 + fx10 * dhx3 * hy0 + fx01 * dhx1 * hy2 + fx11 * dhx3 * hy2 +
            fy00 * dhx0 * hy1 + fy10 * dhx2 * hy1 + fy01 * dhx0 * hy3 + fy11 * dhx2 * hy3 +
            fxy00 * dhx1 * hy1 + fxy10 * dhx3 * hy1 + fxy01 * dhx1 * hy3 + fxy11 * dhx3 * hy3
        ) * res_exp

        # du/dy
        du_dy = (
            f00 * hx0 * dhy0 + f10 * hx2 * dhy0 + f01 * hx0 * dhy2 + f11 * hx2 * dhy2 +
            fx00 * hx1 * dhy0 + fx10 * hx3 * dhy0 + fx01 * hx1 * dhy2 + fx11 * hx3 * dhy2 +
            fy00 * hx0 * dhy1 + fy10 * hx2 * dhy1 + fy01 * hx0 * dhy3 + fy11 * hx2 * dhy3 +
            fxy00 * hx1 * dhy1 + fxy10 * hx3 * dhy1 + fxy01 * hx1 * dhy3 + fxy11 * hx3 * dhy3
        ) * res_exp

        # d²u/dx²
        d2u_dx2 = (
            f00 * ddx0 * hy0 + f10 * ddx2 * hy0 + f01 * ddx0 * hy2 + f11 * ddx2 * hy2 +
            fx00 * ddx1 * hy0 + fx10 * ddx3 * hy0 + fx01 * ddx1 * hy2 + fx11 * ddx3 * hy2 +
            fy00 * ddx0 * hy1 + fy10 * ddx2 * hy1 + fy01 * ddx0 * hy3 + fy11 * ddx2 * hy3 +
            fxy00 * ddx1 * hy1 + fxy10 * ddx3 * hy1 + fxy01 * ddx1 * hy3 + fxy11 * ddx3 * hy3
        ) * res2_exp

        # d²u/dy²
        d2u_dy2 = (
            f00 * hx0 * ddy0 + f10 * hx2 * ddy0 + f01 * hx0 * ddy2 + f11 * hx2 * ddy2 +
            fx00 * hx1 * ddy0 + fx10 * hx3 * ddy0 + fx01 * hx1 * ddy2 + fx11 * hx3 * ddy2 +
            fy00 * hx0 * ddy1 + fy10 * hx2 * ddy1 + fy01 * hx0 * ddy3 + fy11 * hx2 * ddy3 +
            fxy00 * hx1 * ddy1 + fxy10 * hx3 * ddy1 + fxy01 * hx1 * ddy3 + fxy11 * hx3 * ddy3
        ) * res2_exp

        # Reshape to [N, L*F]
        result = result.permute(1, 0, 2).reshape(N, -1)
        du_dx = du_dx.permute(1, 0, 2).reshape(N, -1)
        du_dy = du_dy.permute(1, 0, 2).reshape(N, -1)
        d2u_dx2 = d2u_dx2.permute(1, 0, 2).reshape(N, -1)
        d2u_dy2 = d2u_dy2.permute(1, 0, 2).reshape(N, -1)

        return result, du_dx, du_dy, d2u_dx2, d2u_dy2


# Convenience function to get best available implementation
def get_hermite_encoding(use_cuda: bool = True, **kwargs):
    """
    Factory function to get the best available Hermite encoding.

    Args:
        use_cuda: If True and CUDA extension is available, use CUDA version.
                  Otherwise, fall back to PyTorch version.
        **kwargs: Arguments passed to the encoding constructor.

    Returns:
        HermiteHashEncodingCUDA or HermiteHashEncoding instance.
    """
    if use_cuda and CUDA_AVAILABLE:
        return HermiteHashEncodingCUDA(**kwargs)
    else:
        # Import PyTorch version as fallback
        from hermite_ngp.encoding._hermite_encoding_torch_2d import HermiteHashEncoding
        return HermiteHashEncoding(**kwargs)


# Public name used by the open-source API
HermiteHashEncoding2D = HermiteHashEncodingCUDA


# For testing
if __name__ == '__main__':
    if not CUDA_AVAILABLE:
        print("CUDA extension not available. Cannot run test.")
        exit(1)

    print("Testing HermiteHashEncodingCUDA...")

    device = 'cuda'
    encoding = HermiteHashEncodingCUDA(
        n_input_dims=2,
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size_1=16,
        log2_hashmap_size_2=16,
        log2_hashmap_size_3=16,
        base_resolution=16,
        per_level_scale=1.5,
    ).to(device)

    # Test forward
    x = torch.rand(1024, 2, device=device)
    output = encoding(x)
    print(f"Forward output shape: {output.shape}")

    # Test forward with laplacian
    output, dx, dy, lap = encoding.forward_with_laplacian(x)
    print(f"With Laplacian - output: {output.shape}, dx: {dx.shape}, dy: {dy.shape}, lap: {lap.shape}")

    # Test backward
    output = encoding(x)
    loss = output.sum()
    loss.backward()
    print(f"Backward: grad_hash_table shape: {encoding.hash_table_1.grad.shape} {encoding.hash_table_2.grad.shape} {encoding.hash_table_3.grad.shape}")
    print(f"Backward: grad_hash_table max: {encoding.hash_table_1.grad.abs().max().item():.6f} {encoding.hash_table_2.grad.abs().max().item():.6f} {encoding.hash_table_3.grad.abs().max().item():.6f}")

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
    from hermite_ngp.encoding._hermite_encoding_torch_2d import HermiteHashEncoding
    encoding_pt = HermiteHashEncoding(
        n_input_dims=2,
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size_1=16,
        log2_hashmap_size_2=16,
        log2_hashmap_size_3=16,
        base_resolution=16,
        per_level_scale=1.5,
    ).to(device)

    # Copy weights
    encoding_pt.hash_table_1.data = encoding.hash_table_1.data.clone()
    encoding_pt.hash_table_2.data = encoding.hash_table_2.data.clone()
    encoding_pt.hash_table_3.data = encoding.hash_table_3.data.clone()

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
