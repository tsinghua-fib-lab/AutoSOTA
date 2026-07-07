from abc import abstractmethod

import torch
from torch import nn


class PDE(nn.Module):
    def __init__(
            self,
            linear=bool,
            gamma=torch.Tensor or None
    ):
        super(PDE, self).__init__()

        self.linear = linear
        if gamma is not None:
            self.register_buffer('gamma', gamma)

        self.first = True
        self.derivative_values = None
        self.source_values = None

    @abstractmethod
    def zeroth(self, x, u):
        pass

    @abstractmethod
    def derivative(self, x, u):
        pass

    @abstractmethod
    def source(self, x):
        pass

    def reset(self):
        self.first = True
        self.derivative_values = None
        self.source_values = None

    def forward(self, x, u):
        if self.linear:
            if self.first:
                self.derivative_values = self.derivative(x, u)
                self.source_values = self.source(x)
            n_u = self.zeroth(x, u) + self.derivative_values - self.source_values

        else:
            n_u = self.zeroth(x, u) + self.derivative(x, u) - self.source(x)
        self.first = False
        return n_u
