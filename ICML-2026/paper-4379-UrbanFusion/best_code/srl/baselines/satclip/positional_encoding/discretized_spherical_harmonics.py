"""
Original code from: https://github.com/microsoft/satclip

Original source:
Klemmer, Konstantin; Rolf, Esther; Robinson, Caleb; Mackey, Lester; Rußwurm, Marc.
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."
arXiv preprint published November 28, 2023. Later published as conference paper at AAAI 2025.
"""

import torch
from torch import nn

from .spherical_harmonics_ylm import SH


def SH_(args):
    return SH(*args)


"""
Discretized Spherical Harmonics
"""


class DiscretizedSphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        embedding_dims: determines the dimension of the embedding.
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
        for l in range(self.L):
            for m in range(-l, l + 1):
                Ys.append(SH(m, l, phi, theta) * torch.ones_like(phi))

        self.Ys = torch.stack(Ys)
        self.Ys = self.Ys.permute(0, 2, 1)

    def forward(self, lonlat):

        lonlat = lonlat + torch.tensor([180, 90], device=lonlat.device)

        Ys = interpolate_pixel_values(self.Ys.to(lonlat.device), lonlat).T
        return Ys

    def get_coeffs(self, l, m):
        """
        convenience function to store two triangle matrices in one where m can be negative
        """
        if m == 0:
            return self.weight[l, 0]
        if m > 0:  # on diagnoal and right of it
            return self.weight[l, m]
        if m < 0:  # left of diagonal
            return self.weight[-l, m]

    def get_weight_matrix(self):
        """
        a convenience function to restructure the weight matrix (L x M x E) into
        a double triangle matrix (L x 2 * L + 1 x E) where with legrende polynomials
        are on the rows and frequency components -m ... m on the columns.
        """
        unfolded_coeffs = torch.zeros(
            self.L, self.L * 2 + 1, self.E, device=self.weight.device
        )
        for l in range(0, self.L):
            for m in range(-l, l + 1):
                unfolded_coeffs[l, m + self.L] = self.get_coeffs(l, m)
        return unfolded_coeffs


def interpolate_pixel_values(image, points):
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
