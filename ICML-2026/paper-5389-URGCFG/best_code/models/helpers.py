"""Auxiliary modules for intercept parameterizations used in FiniteModel."""

import torch
from torch import nn


class ZeroMean(nn.Module):
    """
    Parameterization layer enforcing per-dimension zero-mean.

    Learns θ ∈ ℝ^{N×D}, exposes b ∈ ℝ^{(N+1)×D} such that:

        b[:-1] = θ
        b[-1]  = -θ.sum(0)

    This ensures:
        b.sum(0) = 0
        b.mean(0) = 0
    """

    def __init__(self, nrows: int, ndims: int, init_std: float = 0.1):
        """
        Args:
            nrows: number of learnable rows (θ has shape nrows x ndims)
            ndims: number of dimensions (columns)
            init_std: std for random initialization
        """
        super().__init__()
        theta = torch.randn(nrows, ndims) * init_std
        self.theta = nn.Parameter(theta)

    def forward(self) -> torch.Tensor:
        """
        Construct b from θ.
        Returns:
            b of shape (nrows+1, ndims)
        """
        last = -self.theta.sum(dim=0, keepdim=True)
        return torch.cat([self.theta, last], dim=0)

    @property
    def value(self) -> torch.Tensor:
        """Convenience alias."""
        return self.forward()

    def project_from_b(self, b: torch.Tensor):
        """
        Take any b of shape (nrows+1, ndims), center it,
        and update θ to match the first nrows rows.

        This is needed for refresh/initialization steps.

        Args:
            b: tensor of shape (nrows+1, ndims)
        """
        assert b.dim() == 2, "b must be 2D"
        nrows_plus, ndims = b.shape
        assert nrows_plus == self.theta.shape[0] + 1
        assert ndims == self.theta.shape[1]

        # Center b per dimension
        b_centered = b - b.mean(dim=0, keepdim=True)

        # Write into θ
        with torch.no_grad():
            self.theta.copy_(b_centered[:-1])


class FixedFirstIntercept(nn.Module):
    """
    Intercept parameterization with a fixed first row:

        b ∈ ℝ^{ny×D},  with gauge  b[0, d] = 0  for all d.

    Internally stores only the free rows (i ≥ 1) in θ ∈ ℝ^{(ny−1)×D} and
    exposes the full intercepts via:

        value = cat([0_row, θ], dim=0)

    This removes the additive-constant ambiguity in the potential by
    anchoring each column so that the first entry is always zero.
    """

    def __init__(self, ny: int, dim: int, init_std: float = 0.1):
        """
        Args:
            ny:  total number of intercept rows (including the fixed first row)
            dim: number of dimensions (columns)
            init_std: standard deviation for random initialization of θ
        """
        super().__init__()
        if ny < 1:
            raise ValueError("ny must be at least 1.")
        self.ny = ny
        self.dim = dim
        theta = torch.randn(ny - 1, dim) * init_std
        self.theta = nn.Parameter(theta)  # stores b[1:], free rows

    def forward(self) -> torch.Tensor:
        """Return the full intercept matrix b with b[0,:] = 0."""
        zero_row = torch.zeros(1, self.dim, device=self.theta.device, dtype=self.theta.dtype)
        return torch.cat([zero_row, self.theta], dim=0)

    @property
    def value(self) -> torch.Tensor:
        """Alias for forward(), for symmetry with ZeroMean."""
        return self.forward()

    def project_from_b(self, b: torch.Tensor):
        """
        Project a full intercept matrix b (ny, dim) back into θ, enforcing the gauge.

        The projection simply discards the first row and stores b[1:,:] in θ,
        while the reconstructed b will always have b[0,:] = 0.
        """
        assert b.dim() == 2, "b must be 2D"
        assert b.shape == (self.ny, self.dim), f"Expected shape {(self.ny, self.dim)}, got {tuple(b.shape)}"
        with torch.no_grad():
            self.theta.copy_(b[1:, :])

    # Backwards-compatible alias (if older code used the trailing underscore).
    def project_from_b_(self, b: torch.Tensor):
        self.project_from_b(b)

    def set_column_from_raw_(self, dim: int, raw_column: torch.Tensor):
        """
        Update a single column of θ from a raw column of full b values.

        Args:
            dim: column index in [0, D)
            raw_column: tensor of shape (ny,) giving desired b[:, dim].

        The first entry b[0, dim] is ignored (gauge-fixed to 0); the remaining
        entries are copied into θ[ :, dim ].
        """
        if not (0 <= dim < self.dim):
            raise ValueError(f"dim={dim} out of bounds for dim={self.dim}")
        if raw_column.dim() != 1 or raw_column.shape[0] != self.ny:
            raise ValueError(f"raw_column must have shape ({self.ny},), got {tuple(raw_column.shape)}")
        with torch.no_grad():
            self.theta[:, dim].copy_(raw_column[1:])
