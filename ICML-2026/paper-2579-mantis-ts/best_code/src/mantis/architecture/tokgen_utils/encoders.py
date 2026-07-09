import torch

from torch import nn


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=output_dim, eps=1e-15)

    def forward(self, x):
        return self.layer_norm(self.linear(x))


class ScalarEncoder(nn.Module):
    def __init__(self, k, hidden_dim):
        super(ScalarEncoder, self).__init__()
        self.w = torch.nn.Parameter(torch.rand(
            (1, hidden_dim), dtype=torch.float, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(
            (1, hidden_dim), dtype=torch.float, requires_grad=True))
        self.k = k
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=hidden_dim, eps=1e-15)

    def forward(self, x):
        z = x * self.w + self.k * self.b
        y = self.layer_norm(z)
        return y


class MultiScaledScalarEncoder(nn.Module):
    def __init__(self, scales, hidden_dim, epsilon):
        """
        A multi-scaled encoding of a scalar variable:
        https://arxiv.org/pdf/2310.07402.pdf

        Parameters
        ----------
        scales: list, default=None
            List of scales. By default, initialized as [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4].
        hidden_dim: int, default=32
            Hidden dimension of a scalar encoder.
        epsilon: float, default=1.1
            A constant term used to tolerate the computational error in computation of scale weights.
        """
        super(MultiScaledScalarEncoder, self).__init__()
        self.register_buffer('scales', torch.tensor(scales))
        self.epsilon = epsilon
        self.encoders = nn.ModuleList(
            [ScalarEncoder(k, hidden_dim) for k in scales])

    def forward(self, x):
        alpha = abs(1 / torch.log(torch.matmul(abs(x), 1 /
                    self.scales.reshape(1, -1)) + self.epsilon))
        alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        alpha = torch.unsqueeze(alpha, dim=-1)
        y = [encoder(x) for encoder in self.encoders]
        y = torch.stack(y, dim=-2)
        y = torch.sum(y * alpha, dim=-2)
        return y
