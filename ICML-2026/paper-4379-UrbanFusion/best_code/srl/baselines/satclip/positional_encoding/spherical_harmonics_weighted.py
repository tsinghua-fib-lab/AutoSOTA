"""
Original code from: https://github.com/microsoft/satclip

Original source:
Klemmer, Konstantin; Rolf, Esther; Robinson, Caleb; Mackey, Lester; Rußwurm, Marc.
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."
arXiv preprint published November 28, 2023. Later published as conference paper at AAAI 2025.
"""

from datetime import datetime

import torch
from torch import nn

from .spherical_harmonics_ylm import SH


def SH_(args):
    return SH(*args)


class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10, embedding_dim: int = 16):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        embedding_dims: determines the dimension of the embedding.
        """
        super().__init__()
        self.L, self.M, self.E = (
            int(legendre_polys),
            int(legendre_polys),
            int(embedding_dim),
        )
        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(self.L, self.M, self.E)
        )
        self.embedding_dim = embedding_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.33)

    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = torch.zeros_like(phi)
        for l in range(self.L):
            for m in range(-l, l + 1):
                Y = Y + SH(m, l, phi, theta) * self.get_coeffs(l, m).unsqueeze(
                    1
                )

        return Y.T

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
