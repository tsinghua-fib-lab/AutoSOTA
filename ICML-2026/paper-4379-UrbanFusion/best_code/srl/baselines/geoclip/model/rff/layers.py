"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from . import functional


class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[float] = None,
        encoded_size: Optional[float] = None,
        b: Optional[Tensor] = None,
    ):
        r"""
        Args:
            sigma (Optional[float]): standard deviation
            input_size (Optional[float]): the number of input dimensions
            encoded_size (Optional[float]): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.'
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
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(N, *, 2 \cdot \text{encoded_size})`
        """
        return functional.gaussian_encoding(v, self.b)


class BasicEncoding(nn.Module):
    """Layer for mapping coordinates using the basic encoding"""

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{v}} , \sin{2 \pi \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{input_size})`
        """
        return functional.basic_encoding(v)


class PositionalEncoding(nn.Module):
    """Layer for mapping coordinates using the positional encoding"""

    def __init__(self, sigma: float, m: int):
        r"""
        Args:
            sigma (float): frequency constant
            m (int): number of frequencies to map to
        """
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`
        """
        return functional.positional_encoding(v, self.sigma, self.m)
