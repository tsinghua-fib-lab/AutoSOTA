#!/usr/bin/env python3
"""
Description: Implementation of discretized spherical harmonics for positional
encoding. Implementation by the paper "GEOGRAPHIC LOCATION ENCODING
WITH SPHERICAL SHARMONICS AND SINUSOIDAL REPRESENTATION NETWORKS"
"""

import torch
from torch import nn

from .spherical_harmonics_ylm import SH


def SH_(args: tuple) -> torch.Tensor:
    """
    Wrapper function for the spherical harmonics function.

    Parameters:
    ----------
    args : tuple
        Tuple with the arguments for the spherical harmonics function.

    Returns:
    -------
    torch.Tensor
        Spherical harmonics function.
    """
    return SH(*args)


class DiscretizedSphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10) -> None:
        """
        Discretized spherical harmonics for positional encoding.
        legendre_polys determines the number of legendre polynomials.
        More polynomials lead more fine-grained resolutions.
        embedding_dims determines the dimension of the embedding.

        Parameters:
        ----------
        legendre_polys : int
            Determines the number of legendre polynomials.
            More polynomials lead more fine-grained resolutions.
        """

        super().__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        lon = torch.tensor(torch.linspace(-180, 180, 360))
        lat = torch.tensor(torch.linspace(-90, 90, 180))
        lons, lats = torch.meshgrid(lon, lat)

        # ij indexing to xy indexing
        lons, lats = lons.T, lats.T

        phi = torch.deg2rad(lons + 180)
        theta = torch.deg2rad(lats + 90)

        Ys = []
        for l_index in range(self.L):
            for m in range(-l_index, l_index + 1):
                Ys.append(SH(m, l_index, phi, theta) * torch.ones_like(phi))

        self.Ys = torch.stack(Ys)
        self.Ys = self.Ys.permute(0, 2, 1)

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discretized spherical harmonics positional
        encoding.

        Parameters:
        ----------
        lonlat : torch.Tensor
            Coordinates in the format [lon, lat].

        Returns:
        -------
        torch.Tensor
            Encoded coordinates in the format [lon, lat].
        """

        lonlat = lonlat + torch.tensor([180, 90], device=lonlat.device)

        Ys = interpolate_pixel_values(self.Ys.to(lonlat.device), lonlat).T
        return Ys

    def get_coeffs(self, l_index: int, m_index: int) -> torch.Tensor:
        """
        Convenience function to store two triangle matrices in one where m can
        be negative.

        Parameters:
        ----------
        l_index : int
            Legendre polynomial index.
        m_index : int
            Frequency component index.
        """
        if m_index == 0:
            return self.weight[l_index, 0]
        if m_index > 0:  # on diagnoal and right of it
            return self.weight[l_index, m_index]
        if m_index < 0:  # left of diagonal
            return self.weight[-l_index, m_index]

    def get_weight_matrix(self) -> torch.Tensor:
        """
        A convenience function to restructure the weight matrix (L x M x E)
        into a double triangle matrix (L x 2 * L + 1 x E) where with legrende
        polynomials are on the rows and frequency components -m ... m on the
        columns.

        Returns:
        -------
        torch.Tensor
            Weight matrix in the format [l, m, e].
        """
        unfolded_coeffs = torch.zeros(
            self.L, self.L * 2 + 1, self.E, device=self.weight.device
        )
        for l_index in range(0, self.L):
            for m_index in range(-l_index, l_index + 1):
                unfolded_coeffs[l_index, m_index + self.L] = self.get_coeffs(
                    l_index, m_index
                )
        return unfolded_coeffs


def interpolate_pixel_values(
    image: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """
    Interpolates pixel values from an image at sub-pixel coordinates.

    Parameters:
    ----------
    image : torch.Tensor
        Image with shape [C, H, W].
    points : torch.Tensor
        Sub-pixel coordinates with shape [N, 2].

    Returns:
    -------
    torch.Tensor
        Interpolated pixel values with shape [C, N].
    """
    num_points = len(points)
    rows, cols = image.size()[1], image.size()[2]

    # Convert sub-pixel coordinates to integer indices
    floor_coords = torch.floor(points).long()
    ceil_coords = torch.ceil(points).long()

    # Compute fractional parts for interpolation weights
    frac_coords = points - floor_coords.float()

    # Clamp the indices to ensure they are within image boundaries
    floor_coords[:, 0] = torch.clamp(floor_coords[:, 0], 0, rows - 1)
    floor_coords[:, 1] = torch.clamp(floor_coords[:, 1], 0, cols - 1)
    ceil_coords[:, 0] = torch.clamp(ceil_coords[:, 0], 0, rows - 1)
    ceil_coords[:, 1] = torch.clamp(ceil_coords[:, 1], 0, cols - 1)

    # Extract pixel values from the image
    floor_pixels = image[:, floor_coords[:, 0], floor_coords[:, 1]]
    ceil_pixels = image[:, ceil_coords[:, 0], ceil_coords[:, 1]]

    # Compute interpolation weights
    weights_floor = (1 - frac_coords[:, 0]) * (1 - frac_coords[:, 1])
    weights_ceil = frac_coords[:, 0] * (1 - frac_coords[:, 1])
    weights = torch.stack([weights_floor, weights_ceil], dim=1)

    # Interpolate pixel values
    interpolated_pixels = torch.sum(
        torch.stack([floor_pixels, ceil_pixels], dim=2)
        * weights.view(1, num_points, 2),
        dim=2,
    )

    return interpolated_pixels
