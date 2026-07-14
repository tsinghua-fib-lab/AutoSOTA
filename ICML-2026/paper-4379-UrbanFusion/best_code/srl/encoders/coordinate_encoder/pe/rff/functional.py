#!/usr/bin/env python3
"""
Description: Implementation of a coordinate encoder that applies a
hierarchical random fourier features. Based on GeoCLIP implementation.
"""
import numpy as np
import torch
from torch import Tensor


def sample_b(sigma: float, size: tuple) -> Tensor:
    """
    Generate a tensor sampled from a zero-mean normal distribution with
    standard deviation ``sigma``.

    Parameters
    ----------
    sigma : float
        Standard deviation of the normal distribution.
    size : tuple
        Shape of the output tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``size``.
    """
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
    """
    Compute the Gaussian random feature encoding of the input tensor.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of shape ``(N, *, input_size)``.
    b : torch.Tensor
        Projection matrix of shape ``(encoded_layer_size, input_size)``.

    Returns
    -------
    torch.Tensor
        Encoded tensor of shape ``(N, *, 2 * encoded_layer_size)``, where
        the last dimension contains concatenated cosine and sine components.
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def basic_encoding(v: Tensor) -> Tensor:
    """
    Compute the basic sinusoidal encoding of the input tensor.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of shape ``(N, *, input_size)``.

    Returns
    -------
    torch.Tensor
        Encoded tensor of shape ``(N, *, 2 * input_size)``, where the last
        dimension contains concatenated cosine and sine components.
    """
    vp = 2 * np.pi * v
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def positional_encoding(v: Tensor, sigma: float, m: int) -> Tensor:
    """
    Compute a multi-scale positional encoding of the input tensor.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of shape ``(N, *, input_size)``.
    sigma : float
        Base scaling factor, typically chosen based on the domain of ``v``.
    m : int
        Number of frequency scales to use in the encoding.

    Returns
    -------
    torch.Tensor
        Encoded tensor of shape ``(N, *, 2 * m * input_size)``, where the
        last dimension contains concatenated cosine and sine components
        for each frequency scale.
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)
