"""
Hermite Hash Encoding for C3 continuous representation.

Key advantage: Can directly compute derivatives using Hermite basis derivatives,
no autograd needed! This is much faster and more accurate.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class HermiteHashEncoding(nn.Module):
    """
    Multi-resolution hash encoding with Hermite interpolation.
    Can directly output value AND derivatives without autograd.
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
        # Initialize: f values random, fx/fy/fxy start at zero for stable training
        F = n_features_per_level
        init_table = torch.zeros(n_levels, self.hashmap_size_1, F)
        init_table[..., :F] = torch.randn(n_levels, self.hashmap_size_1, F) * 0.01
        # fx, fy, fxy start at zero (columns F:4F already zero)
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

        self.register_buffer('primes', torch.tensor([1, 2654435761], dtype=torch.long))
        self.output_dim = n_levels * n_features_per_level

    def _get_grid_data(self, x: torch.Tensor):
        """Get grid coordinates, local coords, and corner features."""
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

        # Grid spacing
        h = 1.0 / self.resolutions  # [L]

        # Hash corners
        c00 = floor_coords
        c10 = torch.stack([ceil_coords[..., 0], floor_coords[..., 1]], dim=-1)
        c01 = torch.stack([floor_coords[..., 0], ceil_coords[..., 1]], dim=-1)
        c11 = ceil_coords

        def hash_coords_1(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1]) % self.hashmap_size_1

        def hash_coords_2(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1]) % self.hashmap_size_2

        def hash_coords_3(coords):
            return (coords[..., 0] * primes[0] ^ coords[..., 1] * primes[1]) % self.hashmap_size_3

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


        return {
            'tx': tx, 'ty': ty, 'h': h, 'N': N, 'L': L, 'F': F,
            'f': (f00, f10, f01, f11),
            'fx': (fx00, fx10, fx01, fx11),
            'fy': (fy00, fy10, fy01, fy11),
            'fxy': (fxy00, fxy10, fxy01, fxy11),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: just returns encoded features."""
        return self.forward_with_derivatives(x)[0]

    def forward_with_derivatives(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning value AND first derivatives.
        Uses Hermite basis derivatives directly - no autograd needed!

        Returns:
            features: [N, L*F] encoded features
            df_dx: [N, L*F] derivative w.r.t. x
            df_dy: [N, L*F] derivative w.r.t. y
        """
        data = self._get_grid_data(x)
        tx, ty, h = data['tx'], data['ty'], data['h']
        N, L, F = data['N'], data['L'], data['F']

        f00, f10, f01, f11 = data['f']
        fx00, fx10, fx01, fx11 = data['fx']
        fy00, fy10, fy01, fy11 = data['fy']
        fxy00, fxy10, fxy01, fxy11 = data['fxy']

        # Hermite basis functions
        tx2, tx3 = tx * tx, tx * tx * tx
        ty2, ty3 = ty * ty, ty * ty * ty

        # Value basis: H(t)
        hx0 = 2*tx3 - 3*tx2 + 1
        hx1 = tx3 - 2*tx2 + tx
        hx2 = -2*tx3 + 3*tx2
        hx3 = tx3 - tx2

        hy0 = 2*ty3 - 3*ty2 + 1
        hy1 = ty3 - 2*ty2 + ty
        hy2 = -2*ty3 + 3*ty2
        hy3 = ty3 - ty2

        # Derivative basis: H'(t)
        dhx0 = 6*tx2 - 6*tx
        dhx1 = 3*tx2 - 4*tx + 1
        dhx2 = -6*tx2 + 6*tx
        dhx3 = 3*tx2 - 2*tx

        dhy0 = 6*ty2 - 6*ty
        dhy1 = 3*ty2 - 4*ty + 1
        dhy2 = -6*ty2 + 6*ty
        dhy3 = 3*ty2 - 2*ty

        # Reshape for broadcasting: [L, N, 1]
        hx0, hx1, hx2, hx3 = [v.unsqueeze(-1) for v in [hx0, hx1, hx2, hx3]]
        hy0, hy1, hy2, hy3 = [v.unsqueeze(-1) for v in [hy0, hy1, hy2, hy3]]
        dhx0, dhx1, dhx2, dhx3 = [v.unsqueeze(-1) for v in [dhx0, dhx1, dhx2, dhx3]]
        dhy0, dhy1, dhy2, dhy3 = [v.unsqueeze(-1) for v in [dhy0, dhy1, dhy2, dhy3]]

        # Note: Don't scale by h here. The hash table learns "scaled derivatives"
        # directly (i.e., h*f' rather than f'). This avoids numerical issues with
        # very small h at fine grid levels.

        # ============ VALUE ============
        # u = sum over corners of H_i(tx) * H_j(ty) * coefficients
        result = (
            f00 * hx0 * hy0 + f10 * hx2 * hy0 + f01 * hx0 * hy2 + f11 * hx2 * hy2 +
            fx00 * hx1 * hy0 + fx10 * hx3 * hy0 + fx01 * hx1 * hy2 + fx11 * hx3 * hy2 +
            fy00 * hx0 * hy1 + fy10 * hx2 * hy1 + fy01 * hx0 * hy3 + fy11 * hx2 * hy3 +
            fxy00 * hx1 * hy1 + fxy10 * hx3 * hy1 + fxy01 * hx1 * hy3 + fxy11 * hx3 * hy3
        )

        # ============ du/dx ============
        # Use H'(tx) * H(ty), then multiply by resolution (chain rule: d/dx = d/dt * dt/dx = d/dt * resolution)
        res_exp = self.resolutions.view(-1, 1, 1)  # [L, 1, 1]

        du_dx = (
            f00 * dhx0 * hy0 + f10 * dhx2 * hy0 + f01 * dhx0 * hy2 + f11 * dhx2 * hy2 +
            fx00 * dhx1 * hy0 + fx10 * dhx3 * hy0 + fx01 * dhx1 * hy2 + fx11 * dhx3 * hy2 +
            fy00 * dhx0 * hy1 + fy10 * dhx2 * hy1 + fy01 * dhx0 * hy3 + fy11 * dhx2 * hy3 +
            fxy00 * dhx1 * hy1 + fxy10 * dhx3 * hy1 + fxy01 * dhx1 * hy3 + fxy11 * dhx3 * hy3
        ) * res_exp

        # ============ du/dy ============
        du_dy = (
            f00 * hx0 * dhy0 + f10 * hx2 * dhy0 + f01 * hx0 * dhy2 + f11 * hx2 * dhy2 +
            fx00 * hx1 * dhy0 + fx10 * hx3 * dhy0 + fx01 * hx1 * dhy2 + fx11 * hx3 * dhy2 +
            fy00 * hx0 * dhy1 + fy10 * hx2 * dhy1 + fy01 * hx0 * dhy3 + fy11 * hx2 * dhy3 +
            fxy00 * hx1 * dhy1 + fxy10 * hx3 * dhy1 + fxy01 * hx1 * dhy3 + fxy11 * hx3 * dhy3
        ) * res_exp

        # Reshape to [N, L*F]
        result = result.permute(1, 0, 2).reshape(N, -1)
        du_dx = du_dx.permute(1, 0, 2).reshape(N, -1)
        du_dy = du_dy.permute(1, 0, 2).reshape(N, -1)

        return result, du_dx, du_dy

    def forward_with_laplacian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning value, first derivatives, AND Laplacian.
        All computed analytically using Hermite basis - no autograd!

        Returns:
            features: [N, L*F]
            df_dx: [N, L*F]
            df_dy: [N, L*F]
            laplacian: [N, L*F] (d²f/dx² + d²f/dy²)
        """
        data = self._get_grid_data(x)
        tx, ty, h = data['tx'], data['ty'], data['h']
        N, L, F = data['N'], data['L'], data['F']

        f00, f10, f01, f11 = data['f']
        fx00, fx10, fx01, fx11 = data['fx']
        fy00, fy10, fy01, fy11 = data['fy']
        fxy00, fxy10, fxy01, fxy11 = data['fxy']

        tx2, tx3 = tx * tx, tx * tx * tx
        ty2, ty3 = ty * ty, ty * ty * ty

        # Value basis H(t)
        hx0 = 2*tx3 - 3*tx2 + 1
        hx1 = tx3 - 2*tx2 + tx
        hx2 = -2*tx3 + 3*tx2
        hx3 = tx3 - tx2

        hy0 = 2*ty3 - 3*ty2 + 1
        hy1 = ty3 - 2*ty2 + ty
        hy2 = -2*ty3 + 3*ty2
        hy3 = ty3 - ty2

        # First derivative basis H'(t)
        dhx0 = 6*tx2 - 6*tx
        dhx1 = 3*tx2 - 4*tx + 1
        dhx2 = -6*tx2 + 6*tx
        dhx3 = 3*tx2 - 2*tx

        dhy0 = 6*ty2 - 6*ty
        dhy1 = 3*ty2 - 4*ty + 1
        dhy2 = -6*ty2 + 6*ty
        dhy3 = 3*ty2 - 2*ty

        # Second derivative basis H''(t)
        ddx0 = 12*tx - 6
        ddx1 = 6*tx - 4
        ddx2 = -12*tx + 6
        ddx3 = 6*tx - 2

        ddy0 = 12*ty - 6
        ddy1 = 6*ty - 4
        ddy2 = -12*ty + 6
        ddy3 = 6*ty - 2

        # Reshape for broadcasting
        hx0, hx1, hx2, hx3 = [v.unsqueeze(-1) for v in [hx0, hx1, hx2, hx3]]
        hy0, hy1, hy2, hy3 = [v.unsqueeze(-1) for v in [hy0, hy1, hy2, hy3]]
        dhx0, dhx1, dhx2, dhx3 = [v.unsqueeze(-1) for v in [dhx0, dhx1, dhx2, dhx3]]
        dhy0, dhy1, dhy2, dhy3 = [v.unsqueeze(-1) for v in [dhy0, dhy1, dhy2, dhy3]]
        ddx0, ddx1, ddx2, ddx3 = [v.unsqueeze(-1) for v in [ddx0, ddx1, ddx2, ddx3]]
        ddy0, ddy1, ddy2, ddy3 = [v.unsqueeze(-1) for v in [ddy0, ddy1, ddy2, ddy3]]

        res_exp = self.resolutions.view(-1, 1, 1)
        res2_exp = res_exp * res_exp

        # Note: Don't scale by h. Hash table learns scaled derivatives directly.

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

        laplacian = d2u_dx2 + d2u_dy2

        # Reshape to [N, L*F]
        result = result.permute(1, 0, 2).reshape(N, -1)
        du_dx = du_dx.permute(1, 0, 2).reshape(N, -1)
        du_dy = du_dy.permute(1, 0, 2).reshape(N, -1)
        laplacian = laplacian.permute(1, 0, 2).reshape(N, -1)

        return result, du_dx, du_dy, laplacian

    def forward_with_second_derivatives(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning value, first derivatives, and second derivatives separately.

        Returns:
            features: [N, L*F]
            df_dx: [N, L*F]
            df_dy: [N, L*F]
            d2f_dx2: [N, L*F]
            d2f_dy2: [N, L*F]
        """
        data = self._get_grid_data(x)
        tx, ty, h = data['tx'], data['ty'], data['h']
        N, L, F = data['N'], data['L'], data['F']

        f00, f10, f01, f11 = data['f']
        fx00, fx10, fx01, fx11 = data['fx']
        fy00, fy10, fy01, fy11 = data['fy']
        fxy00, fxy10, fxy01, fxy11 = data['fxy']

        tx2, tx3 = tx * tx, tx * tx * tx
        ty2, ty3 = ty * ty, ty * ty * ty

        # Value basis H(t)
        hx0 = 2*tx3 - 3*tx2 + 1
        hx1 = tx3 - 2*tx2 + tx
        hx2 = -2*tx3 + 3*tx2
        hx3 = tx3 - tx2

        hy0 = 2*ty3 - 3*ty2 + 1
        hy1 = ty3 - 2*ty2 + ty
        hy2 = -2*ty3 + 3*ty2
        hy3 = ty3 - ty2

        # First derivative basis H'(t)
        dhx0 = 6*tx2 - 6*tx
        dhx1 = 3*tx2 - 4*tx + 1
        dhx2 = -6*tx2 + 6*tx
        dhx3 = 3*tx2 - 2*tx

        dhy0 = 6*ty2 - 6*ty
        dhy1 = 3*ty2 - 4*ty + 1
        dhy2 = -6*ty2 + 6*ty
        dhy3 = 3*ty2 - 2*ty

        # Second derivative basis H''(t)
        ddx0 = 12*tx - 6
        ddx1 = 6*tx - 4
        ddx2 = -12*tx + 6
        ddx3 = 6*tx - 2

        ddy0 = 12*ty - 6
        ddy1 = 6*ty - 4
        ddy2 = -12*ty + 6
        ddy3 = 6*ty - 2

        # Reshape for broadcasting
        hx0, hx1, hx2, hx3 = [v.unsqueeze(-1) for v in [hx0, hx1, hx2, hx3]]
        hy0, hy1, hy2, hy3 = [v.unsqueeze(-1) for v in [hy0, hy1, hy2, hy3]]
        dhx0, dhx1, dhx2, dhx3 = [v.unsqueeze(-1) for v in [dhx0, dhx1, dhx2, dhx3]]
        dhy0, dhy1, dhy2, dhy3 = [v.unsqueeze(-1) for v in [dhy0, dhy1, dhy2, dhy3]]
        ddx0, ddx1, ddx2, ddx3 = [v.unsqueeze(-1) for v in [ddx0, ddx1, ddx2, ddx3]]
        ddy0, ddy1, ddy2, ddy3 = [v.unsqueeze(-1) for v in [ddy0, ddy1, ddy2, ddy3]]

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
