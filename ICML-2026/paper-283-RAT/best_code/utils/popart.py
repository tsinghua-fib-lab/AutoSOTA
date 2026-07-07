# https://github.com/zoeyuchao/mappo/blob/main/onpolicy/algorithms/utils/popart.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopArt(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-6):
        super().__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.randn(output_shape, input_shape))
        self.bias = nn.Parameter(torch.randn(output_shape))
        
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    def update(self, input_vector):
        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)
        
        self.weight[:] = self.weight * old_stddev / new_stddev
        self.bias[:] = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=self.epsilon)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def unnormalize(self, input_vector):
        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out
