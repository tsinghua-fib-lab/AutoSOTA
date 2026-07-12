import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from .dmol import DMoL_Network

class FeedbackAlignment(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, B):
        ctx.save_for_backward(input, weight, bias, B)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, B = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_B = None

        grad_input = grad_output.mm(B) 
        grad_weight = grad_output.t().mm(input)
        if bias is not None:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias, grad_B

class FALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.register_buffer('B', torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.B, a=np.sqrt(5))

    def forward(self, input):
        return FeedbackAlignment.apply(input, self.weight, self.bias, self.B)

class FA_ProcessingModule(nn.Module):
    def __init__(self, num_classes, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            FALinear(feature_dim + num_classes, 256), nn.ReLU(),
            FALinear(256, num_classes)
        )
    def forward(self, p_prev, shared_features):
        combined_input = torch.cat((p_prev, shared_features), dim=1)
        return self.mlp(combined_input)

class FA_Network(DMoL_Network): 
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__(num_modules, num_classes, in_channels, feature_dim)
        self.processing_modules = nn.ModuleList([FA_ProcessingModule(num_classes, feature_dim) for _ in range(num_modules)])
