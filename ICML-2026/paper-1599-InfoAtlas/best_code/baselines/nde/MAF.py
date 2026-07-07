import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
import optimizer
from copy import deepcopy


class MAF(nn.Sequential):
    """ 
        Mask autoregressive flow
    """
    def __init__(self, n_blocks, n_inputs, n_hidden, n_cond_inputs):
        module = []
        self.n_blocks = n_blocks
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_cond_inputs = n_cond_inputs
        for _ in range(n_blocks): module += [MADE(n_inputs, n_hidden, n_cond_inputs), Reverse(n_inputs)]
        super().__init__(*module)
        self.max_iteration = 50 
    
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        self.num_inputs = inputs.size(-1)
        sum_logdet = torch.zeros(inputs.size(0), 1, device=inputs.device)
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        return inputs, sum_logdet
    
    def sample(self, num_samples=1, cond_inputs=None):
        with torch.no_grad():
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
            device = next(self.parameters()).device
            noise = noise.to(device)
            if cond_inputs is not None:
                cond_inputs = cond_inputs.to(device)
            samples, _ = self.forward(noise, cond_inputs, mode='inverse')
        return samples
        
    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self.forward(inputs, cond_inputs)
        log_base_prob = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(dim=1, keepdim=True)
        return (log_base_prob + log_jacob).sum(dim=1)
    
    def objective_func(self, x, y):
        return self.log_probs(x, y).mean() if self.cond else self.log_probs(x).mean() 

    def learn(self, inputs, cond_inputs=None):
        if cond_inputs is not None:
            self.cond = True
            return optimizer.NNOptimizer.learn(self, x=inputs, y=cond_inputs)
        else:
            self.cond = False
            return optimizer.NNOptimizer.learn(self, x=inputs, y=inputs)
    

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None):
        super(MADE, self).__init__()
        input_mask = self.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = self.get_mask(num_hidden, num_hidden, num_inputs, mask_type='hidden')
        output_mask = self.get_mask(num_hidden, num_inputs, num_inputs, mask_type='output')
        self.join = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.hiddens = nn.Sequential(nn.Tanh(),
                            MaskedLinear(num_hidden, num_hidden, hidden_mask), 
                            nn.Tanh())
        self.mu = MaskedLinear(num_hidden, num_inputs, output_mask)
        self.alpha = MaskedLinear(num_hidden, num_inputs, output_mask)
                            
    def get_mask(self, n_in, n_out, d, mask_type):
        if mask_type == 'input':
            in_degrees = torch.arange(n_in) 
            out_degrees = torch.arange(n_out) % (d-1)
            mask = (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float() 
        if mask_type == 'output':
            in_degrees = torch.arange(n_in) % (d-1)
            out_degrees = torch.arange(n_out)
            mask = (out_degrees.unsqueeze(-1) > in_degrees.unsqueeze(0)).float()
        if mask_type == 'hidden':
            in_degrees = torch.arange(n_in) % (d-1)
            out_degrees = torch.arange(n_out) % (d-1)
            mask = (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float() 
        return mask
        
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        # x -> u, J
        if mode == 'direct':
            h = self.join(inputs, cond_inputs)
            h = self.hiddens(h)
            m, a = self.mu(h), self.alpha(h)
            u = (inputs - m)*torch.exp(-a)
            return u, -a.sum(dim=1, keepdim=True)
        else:
        # u -> x, J
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.join(x, cond_inputs)
                h = self.hiddens(h)
                m, a = self.mu(h), self.alpha(h)
                x[:, i_col] = inputs[:, i_col]*torch.exp(a[:, i_col])+m[:, i_col]
            return x, -a.sum(dim=1, keepdim=True)

        
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features)
            # self.cond_linear = nn.Sequential(
            #     nn.Linear(cond_in_features, out_features),
            #     nn.LeakyReLU(0.2),
            #     nn.Linear(out_features, out_features),
            # )
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        out = F.linear(inputs, self.linear.weight*self.mask, self.linear.bias)
        if cond_inputs is not None:
            out += self.cond_linear(cond_inputs)
        return out
    
    
class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)