import logging
import math
from typing import Literal

import torch
from torch import Tensor, nn

from .rope import RotaryPositionalEmbeddingsBase


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_grid_indices(grid_size: int) -> Tensor:
    """Create 2D grid indices.

    Args:
        grid_size: grid height and width, assuming square grid.

    Returns:
        grid: [2, grid_size, grid_size] Tensor of grid indices.
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    return torch.stack(grid, axis=0)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> Tensor:
    """Create 2D positional embedding with sine and cosine functions.

    Args:
        embed_dim: embedding dimension
        grid_size: grid height and width
        cls_token:  whether to include cls token
        extra_tokens: number of extra tokens to add before the positional embedding

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or
            w/o cls_token) Tensor of positional embeddings.
    """
    grid = get_2d_grid_indices(grid_size)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Tensor) -> Tensor:
    """Create 2D positional embedding with sine and cosine functions.

    Args:
        embed_dim: embedding dimension
        grid: a 4D grid of positions to be encoded: size (2, 1, H, W)

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] Tensor of positional embeddings.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return torch.cat([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Tensor) -> Tensor:
    """Create 1D positional embedding with sine and cosine functions for grid (2D) inputs.

    Args:
        embed_dim: embedding dimension (D)
        pos: a 3D grid of positions to be encoded: size (1, H, W), H*W = M

    Returns:
        out: (M, D) Tensor of positional embeddings.
    """
    assert embed_dim % 2 == 0
    H = pos.size(1)
    # omega = 1.0 / 10000**omega  # (D/2,) fixed sinusoidal embeddings
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)  # (D/2,)
    omega *= 2.0 * math.pi / H
    # Create periodic position functions such that sin(omega*x) and cos(omega*x) are periodic
    # sin(omega*x) = sin(omega*(x+H)) for all x, H, H = W
    # cos(omega*x) = cos(omega*(x+H)) for all x, H, H = W
    # Assume H = W, omega*H = n*2*pi, n is an integer

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    return torch.cat([emb_sin, emb_cos], axis=1)  # (M, D)


def get_1d_sincos_pos_embed(embed_dim: int, max_len: int, max_period: int) -> Tensor:
    """Create 1D positional embeddings with sine and cosine functions for 1D inputs.

    Args:
        embed_dim: D, the dimension of the positional encoding.
        max_len: L, the length of the input sequence.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        embedding: (L, D) Tensor of positional embeddings.
    """
    half = embed_dim // 2
    position = torch.arange(max_len)  # (L,)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )  # (D/2,)
    args = position[:, None].float() * freqs[None]  # (L, D/2)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (L, D)


class PositionalEncoding(nn.Module):
    """Positional encoding module."""

    def __init__(
        self,
        dim: int,
        max_len: int = 5000,
        max_period: int = 10000,
        physical_dim: Literal[1, 2] = 1,
    ) -> None:
        """Initialize the positional encoding.

        Args:
            dim: D, the dimension of the positional encoding.
            max_len: L, the length of the input sequence.
            max_period: controls the minimum frequency of the embeddings.
            physical_dim: the dimensionality of the physical space, e.g. 1D for text, 2D for images
        """
        super().__init__()
        if physical_dim == 1:
            embedding = get_1d_sincos_pos_embed(dim, max_len, max_period)
        elif physical_dim == 2:
            logging.info("max_len: %s", max_len)
            assert math.isqrt(max_len) ** 2 == max_len, "max_len must be a perfect square"
            embedding = get_2d_sincos_pos_embed(dim, int(math.sqrt(max_len)))
        self.register_buffer("pe", embedding)

    def forward(self) -> Tensor:
        """Forward pass."""
        return self.pe


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding module with support for lattice systems.

    This implementation supports:
    - 1D: sequential data or 1D lattices
    - 2D: images or 2D lattices
    - 3D: 3D crystal structures and lattices

    For lattice systems (2D/3D), positions should be provided in fractional coordinates
    (i.e., scaled by lattice constants to be in [0, 1) range).
    """

    def __init__(
        self,
        head_dim: int,
        max_src_len: int = 4096,
        physical_dim: Literal[1, 2, 3] = 1,
        grid_shape: tuple[int, ...] | None = None,
        fixed_positions: Tensor | None = None,
        periodicity: float | tuple[float, ...] = 1.0,
    ) -> None:
        """Initialize the positional encoding.

        Args:
            head_dim: the dimension of each head.
            max_src_len: Maximum length of the input sequence. For regular grids (no positions input),
                this determines the grid size. For 2D: sqrt(max_src_len), 3D: cbrt(max_src_len).
                Ignored if grid_shape or fixed_positions is provided.
            physical_dim: the dimensionality of the physical space, e.g. 1D for text, 2D for images,
                3D for crystal structures.
            grid_shape: Optional explicit grid shape for non-cubic lattices. Examples:
                - 2D: (8, 8) for square, (6, 10) for rectangular
                - 3D: (3, 3, 3) for cubic, (2, 2, 4) for non-cubic FCC
                If provided, overrides max_src_len for grid construction.
                Ignored if fixed_positions is provided.
            fixed_positions: Optional tensor of fixed fractional coordinates. Shape: (num_atoms, D)
                where D is physical_dim. For materials with fixed atomic positions (e.g., CuAu alloy).
                Positions should be in fractional coordinates [0, 1).
                If provided, RoPE cache is built for these EXACT continuous positions (no discretization).
                This mode is ideal for: structures with fixed atoms, continuous positions, periodic systems.
            periodicity: Period for each dimension. Can be scalar (same for all) or tuple (per dimension).
                Default 1.0 assumes fractional coords in [0, 1). For periodic RoPE with custom cell size,
                set to (Lx, Ly, Lz) where Li is the period in dimension i.
        """
        super().__init__()
        self.physical_dim = physical_dim
        self.head_dim = head_dim
        self.max_src_len = max_src_len
        self.grid_shape = grid_shape
        self.fixed_positions = fixed_positions

        # Handle periodicity
        if isinstance(periodicity, (int, float)):
            self.periodicity = torch.tensor([periodicity] * physical_dim)
        else:
            self.periodicity = torch.tensor(periodicity)
            assert len(self.periodicity) == physical_dim

        # MODE 1: Fixed continuous positions (for materials with fixed atomic structures)
        if fixed_positions is not None:
            self._init_fixed_positions_mode(fixed_positions)
            logging.info("Using fixed positions mode for RoPE with physical_dim=%d", physical_dim)
            return

        # MODE 2: Grid-based positions (for discrete lattices or when positions vary)
        # Create RoPE instances for each spatial dimension
        if self.physical_dim == 1:
            # 1D: full head_dim for single dimension
            self.pe = RotaryPositionalEmbeddingsBase(head_dim, max_seq_len=max_src_len)
            # Default positions for regular grids (can be overridden in forward)
            input_pos = torch.arange(max_src_len).long()  # (L,)
            self.register_buffer("default_input_pos", input_pos)
            self.register_buffer("grid_dims", torch.tensor([max_src_len]))

        elif self.physical_dim == 2:
            # 2D: split head_dim across 2 dimensions (as evenly as possible)
            # Both splits must be even for RoPE (which pairs features)
            if head_dim % 4 == 0:
                # Evenly divisible by 4: split exactly in half
                dim1, dim2 = head_dim // 2, head_dim // 2
            elif head_dim % 2 == 0:
                # Even but not divisible by 4: make one split larger
                dim1 = (head_dim // 2 + 1) & ~1  # Round up to even
                dim2 = head_dim - dim1
            else:
                # Odd head_dim: give one extra to first, both should be even/odd
                # But RoPE needs even, so this shouldn't happen with proper usage
                dim1 = (head_dim + 1) // 2
                dim2 = head_dim - dim1
                # Make both even by adjusting
                if dim1 % 2 == 1:
                    dim1 += 1
                    dim2 -= 1

            if grid_shape is not None:
                assert (
                    len(grid_shape) == 2
                ), f"grid_shape for 2D must have 2 elements, got {grid_shape}"
                h, w = grid_shape
                assert (
                    h * w == max_src_len or max_src_len == 4096
                ), f"grid_shape product {h * w} must match max_src_len {max_src_len}"
                max_src_len = h * w  # Update if using grid_shape
            else:
                h = int(math.sqrt(max_src_len))
                assert (
                    h * h == max_src_len
                ), "max_src_len must be a perfect square for 2D (or provide grid_shape)"
                w = h

            # Use max dimension for RoPE base to handle non-square grids
            max_dim = max(h, w)
            # Create separate RoPE instances for potentially different dimensions
            self.pe_x = RotaryPositionalEmbeddingsBase(dim1, max_seq_len=max_dim)
            self.pe_y = RotaryPositionalEmbeddingsBase(dim2, max_seq_len=max_dim)

            # Default positions for 2D grid (possibly rectangular)
            coords_h = torch.arange(h, dtype=torch.float32)
            coords_w = torch.arange(w, dtype=torch.float32)
            grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=0)
            input_pos = grid.reshape(2, -1).long()  # (2, H*W)
            self.register_buffer("default_input_pos", input_pos)
            self.register_buffer("grid_dims", torch.tensor([h, w]))
            self.register_buffer("head_splits", torch.tensor([dim1, dim2]))

        elif self.physical_dim == 3:
            # 3D: split head_dim across 3 dimensions (as evenly as possible)
            # All splits must be EVEN for RoPE (which pairs features for rotation)
            base_dim = head_dim // 3
            remainder = head_dim % 3

            # First try: distribute remainder
            if remainder == 0:
                dim_x, dim_y, dim_z = base_dim, base_dim, base_dim
            elif remainder == 1:
                dim_x, dim_y, dim_z = base_dim + 1, base_dim, base_dim
            else:  # remainder == 2
                dim_x, dim_y, dim_z = base_dim + 1, base_dim + 1, base_dim

            # Ensure all splits are even (required by RoPE base class)
            # If any split is odd, adjust by redistributing
            if dim_x % 2 == 1 or dim_y % 2 == 1 or dim_z % 2 == 1:
                # Recalculate to ensure all even
                # Strategy: round each to nearest even, then adjust the largest
                base_even = (base_dim // 2) * 2  # Round down to even
                remaining = head_dim - 3 * base_even

                # Distribute remaining (ensuring all splits are even)
                if remaining >= 6:
                    dim_x = base_even + 2
                    dim_y = base_even + 2
                    dim_z = head_dim - dim_x - dim_y
                elif remaining >= 4:
                    dim_x = base_even + 2
                    dim_y = base_even + 2
                    dim_z = base_even
                elif remaining >= 2:
                    dim_x = base_even + 2
                    dim_y = base_even
                    dim_z = base_even
                else:
                    dim_x, dim_y, dim_z = base_even, base_even, base_even

            if grid_shape is not None:
                assert (
                    len(grid_shape) == 3
                ), f"grid_shape for 3D must have 3 elements, got {grid_shape}"
                nx, ny, nz = grid_shape
                assert (
                    nx * ny * nz == max_src_len or max_src_len == 4096
                ), f"grid_shape product {nx * ny * nz} must match max_src_len {max_src_len}"
                max_src_len = nx * ny * nz  # Update if using grid_shape
            else:
                # Assume cubic if no grid_shape provided
                h = round(max_src_len ** (1 / 3))
                assert (
                    abs(h**3 - max_src_len) < 1e-6
                ), "max_src_len must be a perfect cube for 3D (or provide grid_shape)"
                nx, ny, nz = h, h, h

            # Use max dimension for RoPE base to handle non-cubic grids
            max_dim = max(nx, ny, nz)
            # Create separate RoPE instances for potentially different dimensions
            self.pe_x = RotaryPositionalEmbeddingsBase(dim_x, max_seq_len=max_dim)
            self.pe_y = RotaryPositionalEmbeddingsBase(dim_y, max_seq_len=max_dim)
            self.pe_z = RotaryPositionalEmbeddingsBase(dim_z, max_seq_len=max_dim)

            # Default positions for 3D grid (possibly non-cubic)
            coords_x = torch.arange(nx, dtype=torch.float32)
            coords_y = torch.arange(ny, dtype=torch.float32)
            coords_z = torch.arange(nz, dtype=torch.float32)
            grid = torch.stack(
                torch.meshgrid(coords_x, coords_y, coords_z, indexing="ij"), dim=0
            )  # (3, nx, ny, nz)
            input_pos = grid.reshape(3, -1).long()  # (3, nx*ny*nz)
            self.register_buffer("default_input_pos", input_pos)
            self.register_buffer("grid_dims", torch.tensor([nx, ny, nz]))
            self.register_buffer("head_splits", torch.tensor([dim_x, dim_y, dim_z]))

        self.max_src_len = max_src_len  # Update in case it was changed

    def _init_fixed_positions_mode(self, fixed_positions: Tensor) -> None:
        """Initialize RoPE for fixed continuous positions (no discretization).

        This mode is for structures where atomic positions are fixed (e.g., CuAu alloy).
        Builds RoPE cache for EXACT fractional coordinates, preserving periodic + rotational properties.

        Args:
            fixed_positions: Tensor of shape (num_atoms, D) with fractional coordinates in [0, 1)
        """
        assert fixed_positions.ndim == 2, f"Expected (num_atoms, D), got {fixed_positions.shape}"
        num_atoms, dim = fixed_positions.shape
        assert dim == self.physical_dim, f"Position dim {dim} != physical_dim {self.physical_dim}"

        self.max_src_len = num_atoms
        self.use_fixed_mode = True

        # Split head_dim across spatial dimensions
        if self.physical_dim == 1:
            self.pe = RotaryPositionalEmbeddingsBase(self.head_dim, max_seq_len=num_atoms)
            # Build cache for exact continuous positions scaled by periodicity
            self._build_continuous_cache(self.pe, fixed_positions[:, 0], self.periodicity[0])

        elif self.physical_dim == 2:
            # Split head dimension
            dim1 = (self.head_dim + 1) // 2
            dim2 = self.head_dim - dim1
            if self.head_dim % 4 == 0:
                dim1, dim2 = self.head_dim // 2, self.head_dim // 2
            elif self.head_dim % 2 == 0:
                dim1 = (self.head_dim // 2 + 1) & ~1
                dim2 = self.head_dim - dim1

            self.pe_x = RotaryPositionalEmbeddingsBase(dim1, max_seq_len=num_atoms)
            self.pe_y = RotaryPositionalEmbeddingsBase(dim2, max_seq_len=num_atoms)

            # Build caches for exact continuous positions
            self._build_continuous_cache(self.pe_x, fixed_positions[:, 0], self.periodicity[0])
            self._build_continuous_cache(self.pe_y, fixed_positions[:, 1], self.periodicity[1])

            self.register_buffer("head_splits", torch.tensor([dim1, dim2]))

        elif self.physical_dim == 3:
            # Split head dimension across 3 dims (ensuring even splits)
            base_dim = self.head_dim // 3
            remainder = self.head_dim % 3

            if remainder == 0:
                dim_x, dim_y, dim_z = base_dim, base_dim, base_dim
            elif remainder == 1:
                dim_x, dim_y, dim_z = base_dim + 1, base_dim, base_dim
            else:
                dim_x, dim_y, dim_z = base_dim + 1, base_dim + 1, base_dim

            # Ensure all even
            if dim_x % 2 == 1 or dim_y % 2 == 1 or dim_z % 2 == 1:
                base_even = (base_dim // 2) * 2
                remaining = self.head_dim - 3 * base_even

                if remaining >= 6:
                    dim_x = base_even + 2
                    dim_y = base_even + 2
                    dim_z = self.head_dim - dim_x - dim_y
                elif remaining >= 4:
                    dim_x = base_even + 2
                    dim_y = base_even + 2
                    dim_z = base_even
                elif remaining >= 2:
                    dim_x = base_even + 2
                    dim_y = base_even
                    dim_z = base_even
                else:
                    dim_x, dim_y, dim_z = base_even, base_even, base_even

            self.pe_x = RotaryPositionalEmbeddingsBase(dim_x, max_seq_len=num_atoms)
            self.pe_y = RotaryPositionalEmbeddingsBase(dim_y, max_seq_len=num_atoms)
            self.pe_z = RotaryPositionalEmbeddingsBase(dim_z, max_seq_len=num_atoms)

            # Build caches for exact continuous positions
            self._build_continuous_cache(self.pe_x, fixed_positions[:, 0], self.periodicity[0])
            self._build_continuous_cache(self.pe_y, fixed_positions[:, 1], self.periodicity[1])
            self._build_continuous_cache(self.pe_z, fixed_positions[:, 2], self.periodicity[2])
            self.register_buffer("head_splits", torch.tensor([dim_x, dim_y, dim_z]))

    def _build_continuous_cache(
        self, rope_module: RotaryPositionalEmbeddingsBase, positions_1d: Tensor, period: float
    ) -> None:
        """Build RoPE cache for continuous position values.

        Args:
            rope_module: The RoPE module to build cache for
            positions_1d: 1D tensor of fractional positions [0, 1) for one spatial dimension
            period: Period for this dimension (for periodic boundary conditions)
        """
        # Base RoPE has: theta = [0,1,2,...] * 2π / max_seq_len
        # We want: theta_effective = [0,1,2,...] * 2π / period
        # Solution: scale positions by (max_seq_len / period)
        #
        # Then: idx_theta = (positions * max_seq_len/period) ⊗ theta_base
        #                 = (positions * max_seq_len/period) ⊗ ([0,1,2,...] * 2π/max_seq_len)
        #                 = positions ⊗ ([0,1,2,...] * 2π/period)  ✓

        num_atoms = rope_module.max_seq_len
        # Ensure positions_1d is on the same device as rope_module.theta
        positions_1d = positions_1d.to(rope_module.theta.device)
        scaled_positions = positions_1d * (num_atoms / period)

        # Build RoPE cache: idx_theta = scaled_positions ⊗ theta
        idx_theta = torch.einsum("i, j -> ij", scaled_positions, rope_module.theta).float()

        # Cache with cos and sin
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # Replace the default cache with our continuous position cache
        rope_module.register_buffer("cache", cache, persistent=False)

    def _forward_fixed_mode(self, x: Tensor) -> Tensor:
        """Forward pass for fixed positions mode (continuous positions pre-cached).

        Args:
            x: input tensor of shape (B, L, nh, hs)

        Returns:
            x: output tensor with RoPE applied using pre-computed continuous position cache
        """
        batch_size, seq_len = x.size(0), x.size(1)
        assert (
            seq_len == self.max_src_len
        ), f"Fixed mode requires seq_len={self.max_src_len}, got {seq_len}"

        # Use sequential indices - the continuous position info is in the cache
        pos_indices = torch.arange(seq_len, device=x.device)

        if self.physical_dim == 1:
            # Simple case - just apply RoPE with sequential indices
            x = self.pe(x, input_pos=pos_indices)

        elif self.physical_dim == 2:
            # Split and apply to each dimension
            dim_x, dim_y = self.head_splits[0].item(), self.head_splits[1].item()

            x_part_x = x[:, :, :, :dim_x]
            embed_x = self.pe_x(x_part_x, input_pos=pos_indices)

            x_part_y = x[:, :, :, dim_x:]
            embed_y = self.pe_y(x_part_y, input_pos=pos_indices)

            x = torch.cat([embed_x, embed_y], dim=-1)

        elif self.physical_dim == 3:
            # Split and apply to each dimension
            dim_x, dim_y, dim_z = (
                self.head_splits[0].item(),
                self.head_splits[1].item(),
                self.head_splits[2].item(),
            )

            x_part_x = x[:, :, :, :dim_x]
            embed_x = self.pe_x(x_part_x, input_pos=pos_indices)

            x_part_y = x[:, :, :, dim_x : dim_x + dim_y]
            embed_y = self.pe_y(x_part_y, input_pos=pos_indices)

            x_part_z = x[:, :, :, dim_x + dim_y :]
            embed_z = self.pe_z(x_part_z, input_pos=pos_indices)

            x = torch.cat([embed_x, embed_y, embed_z], dim=-1)

        return x

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """Forward pass with optional position input.

        Args:
            x: input tensor of shape (B, L, nh, hs), where:
                - B: batch size
                - L: sequence length
                - nh: number of heads
                - hs: head size (should equal head_dim)
            positions: Optional position tensor. Ignored if using fixed_positions mode.
                - For 1D: shape (B, L) with integer positions
                - For 2D: shape (B, L, 2) with (x, y) fractional coordinates in [0, 1)
                - For 3D: shape (B, L, 3) with (x, y, z) fractional coordinates in [0, 1)

                For lattice systems, positions should be fractional coordinates scaled to [0, max_grid_size).

        Returns:
            x: output tensor of shape (B, L, nh, hs), with RoPE applied.
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # MODE 1: Fixed positions - cache is pre-built for exact continuous positions
        if hasattr(self, "use_fixed_mode") and self.use_fixed_mode:
            return self._forward_fixed_mode(x)

        # MODE 2: Grid-based positions
        # Get positions to use
        if positions is None:
            # Use default regular grid positions
            if self.physical_dim == 1:
                pos = self.default_input_pos[:seq_len].unsqueeze(0).expand(batch_size, -1)  # (B, L)
            else:
                pos = (
                    self.default_input_pos[:, :seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                )  # (B, D, L)
        else:
            # Use provided positions
            if self.physical_dim == 1:
                # positions: (B, L) -> integer indices
                pos = positions.long()
            elif self.physical_dim == 2:
                # positions: (B, L, 2) -> scale to grid indices
                assert (
                    positions.size(-1) == 2
                ), f"Expected 2D positions, got shape {positions.shape}"
                # Scale each dimension by its grid size (handles non-square grids)
                grid_dims = self.grid_dims.to(positions.device)  # (2,) [h, w]
                # Scale fractional coordinates [0, 1) to grid indices
                pos = (positions * grid_dims).long()  # (B, L, 2)
                pos = pos.clamp(min=0)  # Ensure non-negative
                # Clamp each dimension separately
                pos[:, :, 0] = pos[:, :, 0].clamp(max=grid_dims[0] - 1)
                pos[:, :, 1] = pos[:, :, 1].clamp(max=grid_dims[1] - 1)
                # Transpose to (B, 2, L) for consistency with default format
                pos = pos.transpose(1, 2)  # (B, 2, L)
            elif self.physical_dim == 3:
                # positions: (B, L, 3) -> scale to grid indices
                assert (
                    positions.size(-1) == 3
                ), f"Expected 3D positions, got shape {positions.shape}"
                # Scale each dimension by its grid size (handles non-cubic grids like 2x2x4)
                grid_dims = self.grid_dims.to(positions.device)  # (3,) [nx, ny, nz]
                # Scale fractional coordinates [0, 1) to grid indices
                pos = (positions * grid_dims).long()  # (B, L, 3)
                pos = pos.clamp(min=0)  # Ensure non-negative
                # Clamp each dimension separately to its max
                pos[:, :, 0] = pos[:, :, 0].clamp(max=grid_dims[0] - 1)
                pos[:, :, 1] = pos[:, :, 1].clamp(max=grid_dims[1] - 1)
                pos[:, :, 2] = pos[:, :, 2].clamp(max=grid_dims[2] - 1)
                # Transpose to (B, 3, L) for consistency with default format
                pos = pos.transpose(1, 2)  # (B, 3, L)

        # Apply RoPE based on dimensionality
        # Note: Base RoPE class expects input shape [b, s, n_h, h_d]
        num_heads = x.size(2)

        if self.physical_dim == 1:
            # pos: (B, L)
            # Apply RoPE to full head dimension
            x = self.pe(x, input_pos=pos)  # (B, L, nh, hs)

        elif self.physical_dim == 2:
            # pos: (B, 2, L) - [x, y] positions
            # Split head dimension according to head_splits (allows uneven splits)
            dim_x, dim_y = self.head_splits[0].item(), self.head_splits[1].item()

            # Process x-coordinates: use first dim_x features
            x_part_x = x[:, :, :, :dim_x]  # (B, L, nh, dim_x)
            embed_x = self.pe_x(x_part_x, input_pos=pos[:, 0, :])  # (B, L, nh, dim_x)

            # Process y-coordinates: use remaining dim_y features
            x_part_y = x[:, :, :, dim_x:]  # (B, L, nh, dim_y)
            embed_y = self.pe_y(x_part_y, input_pos=pos[:, 1, :])  # (B, L, nh, dim_y)

            # Concatenate along head dimension
            x = torch.cat([embed_x, embed_y], dim=-1)  # (B, L, nh, hs)

        elif self.physical_dim == 3:
            # pos: (B, 3, L) - [x, y, z] positions
            # Split head dimension according to head_splits (allows uneven splits)
            dim_x, dim_y, dim_z = (
                self.head_splits[0].item(),
                self.head_splits[1].item(),
                self.head_splits[2].item(),
            )

            # Process x-coordinates: use first dim_x features
            x_part_x = x[:, :, :, :dim_x]  # (B, L, nh, dim_x)
            embed_x = self.pe_x(x_part_x, input_pos=pos[:, 0, :])  # (B, L, nh, dim_x)

            # Process y-coordinates: use next dim_y features
            x_part_y = x[:, :, :, dim_x : dim_x + dim_y]  # (B, L, nh, dim_y)
            embed_y = self.pe_y(x_part_y, input_pos=pos[:, 1, :])  # (B, L, nh, dim_y)

            # Process z-coordinates: use final dim_z features
            x_part_z = x[:, :, :, dim_x + dim_y :]  # (B, L, nh, dim_z)
            embed_z = self.pe_z(x_part_z, input_pos=pos[:, 2, :])  # (B, L, nh, dim_z)

            # Concatenate along head dimension
            x = torch.cat([embed_x, embed_y, embed_z], dim=-1)  # (B, L, nh, hs)

        return x


# Modified from yang-song/score_sde_pytorch
class GaussianFourierBasis(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, num_basis: int):
        super().__init__()
        assert num_basis % 2 == 0
        self.num_basis = num_basis
        freqs = torch.randn(num_basis // 2) * 2 * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        return emb


# From TimestepEmbedder class in https://github.com/facebookresearch/DiT/blob/main/models.py
class ThermodynamicEmbedder(nn.Module):
    """Embeds scalar thermodynamic quantities such as temperatures and chemical potentials into
    vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        use_mlp: bool = False,
        num_scalars: int = 1,
    ) -> None:
        """Initialize the embedder.

        Args:
            hidden_size: the size of the hidden layer.
            frequency_embedding_size: the size of the frequency embedding.
            use_mlp: whether to use an MLP.
            num_scalars: number of scalar inputs to embed (for combined embedding).
        """
        super().__init__()
        self.use_mlp = use_mlp
        self.num_scalars = num_scalars

        if self.use_mlp:
            # For multiple scalars, the input size is multiplied
            input_size = (
                frequency_embedding_size * num_scalars
                if num_scalars > 1
                else frequency_embedding_size
            )
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.frequency_embedding_size = frequency_embedding_size
        else:
            # For multiple scalars, ensure output size is correct
            if num_scalars > 1:
                self.frequency_embedding_size = hidden_size // num_scalars
                assert (
                    hidden_size % num_scalars == 0
                ), "hidden_size must be divisible by num_scalars"
            else:
                self.frequency_embedding_size = hidden_size

    @staticmethod
    def embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal embeddings.

        Args:
            t: a 1-D Tensor of L indices, one per batch element. These may be fractional.
            dim: D, the dimension of the output embeddings.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            (L, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]  # (L, D/2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (L, D)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t: Input scalars [B] or [B, num_scalars] for combined embedding

        Returns:
            Embeddings [B, hidden_size]
        """
        if self.num_scalars > 1:
            # Handle multiple scalars (e.g., temperature and field)
            if t.dim() == 1:
                raise ValueError(
                    f"Expected input shape [B, {self.num_scalars}] for num_scalars={self.num_scalars}"
                )

            # Embed each scalar separately
            embeddings = [
                self.embedding(t[:, i], self.frequency_embedding_size)
                for i in range(self.num_scalars)
            ]
            t_freq = torch.cat(embeddings, dim=-1)  # [B, frequency_embedding_size * num_scalars]
        else:
            # Single scalar
            t_freq = self.embedding(t, self.frequency_embedding_size)

        return self.mlp(t_freq) if self.use_mlp else t_freq


class BesselBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float):
        super().__init__()
        self.num_basis = num_basis
        freqs = torch.arange(1, num_basis + 1, dtype=torch.float) * math.pi / r_max
        prefactor = torch.tensor(math.sqrt(2.0 / r_max), dtype=torch.float)
        self.register_buffer("freqs", freqs)
        self.register_buffer("prefactor", prefactor)

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        rbf = self.prefactor * torch.sin(args) / x[..., None]
        return rbf


class CosineCutoff(nn.Module):
    def __init__(self, r_max: float):
        super().__init__()
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float))

    def forward(self, x: torch.Tensor):
        x_cut = 0.5 * (1.0 + torch.cos(x * math.pi / self.r_max))
        x_cut = x_cut * (x < self.r_max).float()
        return x_cut
