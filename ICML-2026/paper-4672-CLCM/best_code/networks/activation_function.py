import torch
import torch.nn as nn


class ActivationFunction(nn.Module):
    """Base class for activation functions with derivatives."""

    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return torch.sigmoid(x)

    def derivative(self, x):
        sig = torch.sigmoid(x)
        return sig * (1 - sig)


class ReLU(ActivationFunction):
    def forward(self, x):
        return torch.relu(x)

    def derivative(self, x):
        grad = torch.ones_like(x)
        grad[x < 0] = 0
        return grad


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return torch.ones_like(x)


class Tanh(ActivationFunction):
    def forward(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2


class Softplus(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.modulation = torch.tensor(1)
        self.beta = 1
        self.softplus = nn.Softplus(beta=self.beta)
        self.sigmoid = nn.Sigmoid()

    def set_modulation(self, modulation):
        self.modulation = modulation

    def reset_modulation(self):
        self.modulation = torch.tensor(1)

    def forward(self, x):
        return self.softplus(x)

    def derivative(self, x):
        return self.sigmoid(self.beta * x) * self.modulation
