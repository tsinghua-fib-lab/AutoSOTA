#!/usr/bin/env python3
"""
Description: Implementation of a coordinate encoder that applies a positional
encoding to coordinates and passes the result through a FCNet network.
Implementation by the paper "GEOGRAPHIC LOCATION ENCODING WITH SPHERICAL
HARMONICS AND SINUSOIDAL REPRESENTATION NETWORKS"
"""

import torch
from torch import nn


class ResLayer(nn.Module):
    def __init__(self, linear_size: int) -> None:
        """
        Residual layer for FCNet

        Parameters
        ----------
        linear_size : int
            Size of the linear layer
        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        dim_hidden: int,
    ) -> None:
        """ "
        Fully connected network for the coordinate encoder.

        Parameters
        ----------
        num_inputs : int
            Number of input features.
        num_classes : int
            Number of classes.
        dim_hidden : int
            Dimension of the hidden layer.
        """
        super().__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)

        self.feats = nn.Sequential(
            nn.Linear(num_inputs, dim_hidden),
            nn.ReLU(inplace=True),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        loc_emb = self.feats(x)
        class_pred = self.class_emb(loc_emb)
        return class_pred
