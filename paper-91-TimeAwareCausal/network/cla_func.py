import torch.nn as nn
from engine.configs import Classifers

import torch
from typing import Any, Optional, Tuple
from torch.autograd import Function

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

@Classifers.register('nonlinear_cla')
class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLayerClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.linear2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.linear3 = nn.Linear(input_dim // 4, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x.view(x.size(0), -1))
        out = self.relu(out)
        out = self.relu(self.linear2(out))
        out = self.linear3(out)
        return out


@Classifers.register('linear_cla')
class SingleLayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.grl_layer1 = GRL_Layer()
        self.grl_layer2 = GRL_Layer()

    def forward(self, x):
        out = self.linear(x.view(x.size(0), -1))
        return out

    def grl_forward(self, x):
        x_ = self.grl_layer1(x)
        out = self.linear(x_.view(x_.size(0), -1))
        out = self.grl_layer2(out)

        return out