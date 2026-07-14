#!/usr/bin/env python3
"""
Description: Implementation by the paper "GEOGRAPHIC
LOCATION ENCODING WITH SPHERICAL SHARMONICS AND SINUSOIDAL REPRESENTATION
NETWORKS". This function prints the source code for
spherical_harmonics.py to console. spherical_harmonics pre-computes the
analytical solutions to each real spherical harmonic with sympy
the script contains different functions for different degrees l and orders m.
"""

import torch
from torch import nn

from .spherical_harmonics_closed_form import SH as SH_closed_form
from .spherical_harmonics_ylm import SH as SH_analytic


class SphericalHarmonics(nn.Module):
    def __init__(
        self, legendre_polys: int = 10, harmonics_calculation: str = "analytic"
    ) -> None:
        """
        Implementation of the Spherical Harmonics Encoder

        Parameters:
        -----------
        legendre_polys: int
            number of legendre polynomials to calculate.
            More polynomials lead to more fine-grained resolutions.
        harmonics_calculation: str
            method to calculate spherical harmonics. Options are:
                - closed-form: uses one equation to calculate the spherical
                harmonics. This is exact, but computationally slower (
                especially for high degrees).
                - analytic: uses pre-computed equations to calculate the
                spherical harmonics up to degree 100.
        """
        super().__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        if harmonics_calculation == "closed-form":
            self.SH = SH_closed_form
        elif harmonics_calculation == "analytic":
            self.SH = SH_analytic

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spherical harmonics encoder.

        Parameters:
        -----------
        lonlat: torch.Tensor
            tensor of shape (batch_size, 2) containing the longitude and
            latitude values in degrees.

        Returns:
        --------
        torch.Tensor
            tensor of shape (batch_size, embedding_dim) containing the
            spherical harmonics representation of the input coordinates.
        """
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l_index in range(self.L):
            for m in range(-l_index, l_index + 1):
                y = self.SH(m, l_index, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)

        return torch.stack(Y, dim=-1)
