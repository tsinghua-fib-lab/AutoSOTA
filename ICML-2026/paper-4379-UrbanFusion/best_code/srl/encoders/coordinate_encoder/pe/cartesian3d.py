#!/usr/bin/env python3
"""
Description: Implementation of the coordinate encoder 3D Cartesian from
MacDodha et al. Implementation by the paper "GEOGRAPHIC LOCATION ENCODING
WITH SPHERICAL SHARMONICS AND SINUSOIDAL REPRESENTATION NETWORKS"
"""

import torch
from torch import nn


class Cartesian3D(nn.Module):
    def __init__(self):
        super().__init__()
        """
        The 3D Cartesian coordinate encoder is a simple encoder that takes
        longitude and latitude coordinates and transforms them into a 3D
        Cartesian coordinate system. The encoder is defined as follows:
        x = cos(lon) * cos(lat)
        y = sin(lon) * cos(lat)
        z = sin(lat)
        """

        # Adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 3

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 3D Cartesian coordinate encoder

        Parameters:
        ----------
        coords : torch.Tensor
            A tensor of shape (batch_size, 2) containing the longitude
            and latitude coordinates in degrees.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (batch_size, 3) containing the 3D Cartesian
            coordinates.
        """
        # Place lon lat coordinates in a -pi, pi range
        coords = torch.deg2rad(coords)

        cos_lon = torch.cos(coords[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(coords[:, 0]).unsqueeze(-1)
        cos_lat = torch.cos(coords[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(coords[:, 1]).unsqueeze(-1)

        return torch.cat((cos_lon * cos_lat, sin_lon * cos_lat, sin_lat), 1)
