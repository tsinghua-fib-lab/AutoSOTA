"""
Components for use in NN architectures
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Distribution

from typing import Sequence, Union

from distributions.distributions import GaussianMixtureModel


class MLP(nn.Sequential):

    def __init__(self, layer_sizes: Sequence[int],
                 activation: Union[str, nn.Module] = nn.GELU()):
        """
        Simple MLP Block

        Args:
            layer_sizes: Sequence of layer sizes, including input and output sizes.
            activation: Activation function. Must be one of "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.

        """
        super().__init__()

        if not isinstance(activation, nn.Module):
            match activation:
                case "relu":
                    activation = nn.ReLU()
                case "gelu":
                    activation = nn.GELU()
                case _:
                    print(activation)
                    raise ValueError('activation must be one of "relu", "gelu" or a nn.Module subclass')

        super().__init__(*sum([[activation, nn.Linear(size_in, size_out)]
                               for size_in, size_out in zip(layer_sizes[1:-1], layer_sizes[2:])],
                              start=[nn.Linear(layer_sizes[0], layer_sizes[1])]))
