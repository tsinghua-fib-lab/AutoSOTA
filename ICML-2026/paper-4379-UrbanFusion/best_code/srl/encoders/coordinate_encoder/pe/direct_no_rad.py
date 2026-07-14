#!/usr/bin/env python3
"""
Description: Implementation of direct positional encoding for coordinates.
"""

import torch
from torch import nn


class DirectNoRad(nn.Module):
    def __init__(self) -> None:
        """
        Direct positional encoding for coordinates.
        Just transforms longitude and latitude to radians and shifts them to
        the range of -pi to pi.
        """
        super().__init__()

        # Adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the direct positional encoding.

        Parameters:
        ----------
        coords : torch.Tensor
            Coordinates in the format [lon, lat].

        Returns:
        -------
        coords : torch.Tensor
            Encoded coordinates in the format [lon, lat].
        """
        # Place lon lat coordinates in a -pi, pi range
        return coords
