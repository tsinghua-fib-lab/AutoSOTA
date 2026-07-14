#!/usr/bin/env python3
"""
Description: Implementation of a coordinate encoder that applies a positional
encoding to coordinates and passes the result through a SIREN network.
Implementation by the paper "GEOGRAPHIC LOCATION ENCODING WITH SPHERICAL
HARMONICS AND SINUSOIDAL REPRESENTATION NETWORKS"
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def exists(val):
    """
    Check if a value is not None.

    Parameters
    ----------
    val : any
        Value to check.

    Returns
    -------
    bool
        True if val is not None, False otherwise.
    """
    return val is not None


def cast_tuple(val, repeat=1):
    """
    Cast a value to a tuple of specified length.

    Parameters
    ----------
    val : any
        Value to cast. If already a tuple, returned unchanged.
    repeat : int, default=1
        Length of the resulting tuple if val is not a tuple.

    Returns
    -------
    tuple
        Tuple of length `repeat` containing `val`.
    """
    return val if isinstance(val, tuple) else (val,) * repeat


class Sine(nn.Module):
    """
    Sinusoidal activation function.

    Parameters
    ----------
    w0 : float, default=1.0
        Frequency multiplier for input.
    """

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sine activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sine of scaled input.
        """
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """
    Single Siren layer with sine activation.

    Parameters
    ----------
    dim_in : int
        Input feature dimension.
    dim_out : int
        Output feature dimension.
    w0 : float, default=1.0
        Frequency for sine activation.
    c : float, default=6.0
        Scaling constant for weight initialization.
    is_first : bool, default=False
        Flag for first layer initialization.
    use_bias : bool, default=True
        Whether to include bias term.
    activation : nn.Module, optional
        Custom activation module. Defaults to sine.
    dropout : bool, default=False
        Whether to apply dropout.
    dropout_rate : float, default=0.1
        Dropout rate for layers.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn.Module = None,
        dropout: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:

        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(
        self, weight: torch.Tensor, bias: torch.Tensor, c: float, w0: float
    ) -> None:
        """
        Initialize weights and bias uniformly.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to initialize.
        bias : torch.Tensor
            Bias tensor to initialize.
        c : float
            Scaling constant for weight initialization.
        w0 : float
            Frequency for sine activation.S
        """
        dim = self.dim_in
        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear transformation, optional dropout, and
        activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after linear transformation, dropout, and activation.
        """
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training, p=self.dropout_rate)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    """
    Multi-layer Siren network.

    Parameters
    ----------
    dim_in : int
        Input feature dimension.
    dim_hidden : int
        Hidden feature dimension.
    dim_out : int
        Output feature dimension.
    num_layers : int
        Number of hidden layers.
    w0 : float, default=1.0
        Frequency for sine activation on non-first layers.
    w0_initial : float, default=30.0
        Frequency for sine activation on first layer.
    use_bias : bool, default=True
        Whether to include bias in layers.
    final_activation : nn.Module, optional
        Activation for final layer. Defaults to identity.
    degreeinput : bool, default=False
        Whether to convert degree input to radians.
    dropout : bool, default=False
        Whether to apply dropout in layers.
    dropout_rate : float, default=0.1
        Dropout rate for layers.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: nn.Module = None,
        degreeinput: bool = False,
        dropout: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=dropout,
                    dropout_rate=dropout_rate,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=False,
        )

    def forward(self, x: torch.Tensor, mods=None) -> torch.Tensor:
        """
        Forward pass with optional modulatory inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mods : tuple, optional
            Tuple of modulatory inputs for each layer. Defaults to None.

        Returns
        -------
        torch.Tensor
            Output tensor after forward
        """

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi
        mods = cast_tuple(mods, self.num_layers)
        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if exists(mod):
                x *= rearrange(mod, "d -> () d")
        return self.last_layer(x)
