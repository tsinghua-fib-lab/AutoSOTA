#!/usr/bin/env python3
"""
Description: Implementation of a coordinate encoder that applies a
hierarchical random fourier features. Based on GeoCLIP implementation.
"""
from typing import Optional

import torch.nn as nn
from torch import Tensor

from . import functional


class GaussianEncoding(nn.Module):
    """
    Layer for mapping coordinates using random Fourier features
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[float] = None,
        encoded_size: Optional[float] = None,
        b: Optional[Tensor] = None,
    ) -> None:
        """
        Map coordinates to a higher-dimensional space using random Fourier
        features.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation for sampling the projection matrix ``b``.
            Required if ``b`` is not provided.
        input_size : float, optional
            Number of input dimensions. Required if ``b`` is not provided.
        encoded_size : float, optional
            Number of output dimensions for the projection matrix ``b``.
            Required if ``b`` is not provided.
        b : torch.Tensor, optional
            Pre-specified projection matrix of shape ``(encoded_size,
            input_size)``.
            If provided, ``sigma``, ``input_size``, and ``encoded_size`` must
            be None.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and '
                    '"encoded_size" are required.'
                )

            b = functional.sample_b(sigma, (encoded_size, input_size))

        elif (
            sigma is not None
            or input_size is not None
            or encoded_size is not None
        ):
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        """
        Apply Gaussian random feature mapping to the input tensor.

        Parameters
        ----------
        v : torch.Tensor
            Input tensor of shape ``(N, *, input_size)``.

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape ``(N, *, 2 * encoded_size)``,
            containing concatenated cosine and sine components.
        """
        return functional.gaussian_encoding(v, self.b)


class BasicEncoding(nn.Module):
    """
    Layer for mapping coordinates using the basic encoding
    """

    def forward(self, v: Tensor) -> Tensor:
        """
        Apply basic sinusoidal encoding to the input tensor.

        Parameters
        ----------
        v : torch.Tensor
            Input tensor of shape ``(N, *, input_size)``.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``(N, *, 2 * input_size)``, containing
            concatenated cosine and sine components.
        """
        return functional.basic_encoding(v)


class PositionalEncoding(nn.Module):
    """
    Layer for mapping coordinates using the positional encoding
    """

    def __init__(self, sigma: float, m: int):
        """
        Initialize the positional encoding module.

        Parameters
        ----------
        sigma : float
            Frequency scaling constant for the encoding.
        m : int
            Number of frequency scales to use in the mapping.
        """
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        """
        Apply multi-scale positional encoding to the input tensor.

        Parameters
        ----------
        v : torch.Tensor
            Input tensor of shape ``(N, *, input_size)``.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``(N, *, 2 * m * input_size)``, where the
            last dimension contains concatenated cosine and sine components
            for each frequency scale.
        """
        return functional.positional_encoding(v, self.sigma, self.m)
