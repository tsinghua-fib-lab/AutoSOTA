import torch
import torch.nn as nn
import torch.nn.functional as F

class _BaseBayesianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tol = 1e-18

class LinearBayesian(nn.Linear, _BaseBayesianNet):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        if self.bias is None:
            self.bias = nn.Parameter(
                torch.tensor([0], dtype=self.weight.dtype), requires_grad=False
            )

    def forward(self, input, alpha):
        mean = F.linear(input, self.weight, self.bias)
        var = alpha * F.linear(input.pow(2), self.weight.pow(2))
        if self.training:
            return mean + torch.sqrt(var + self.tol) * torch.randn_like(mean)
        return mean