import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
import optimizer
from copy import deepcopy


class MDN(nn.Module):
    """ 
        Mixture density network (for multi-dimensional data)
    """
    def __init__(self, n_in, n_hidden, n_out, K=1):
        super().__init__()
        # self.main = nn.Sequential(
        #     nn.Linear(n_in, n_hidden),
        #     #nn.Softplus(),
        #     nn.Tanh(),
        #     nn.Linear(n_hidden, n_hidden),
        #     nn.Tanh(),
        #     #nn.Softplus(),
        # )
        self.K = K
        self.dim = n_out
        self.n_hidden = n_hidden
        self.coeff_layer = CoeffLayer(n_hidden, K)
        self.mean_layer = nn.ModuleList([MeanLayer(n_hidden, n_out) for i in range(K)])
        self.cov_layer = nn.ModuleList([CovLayer(n_hidden, n_out) for i in range(K)])
        
        
    def main(self, x):
        return torch.zeros(len(x), self.n_hidden).to(x.device)

    def forward(self, cond_inputs):
        # nn(x) = {coeff}, {mu}, {cov}
        h = self.main(cond_inputs)
        mu_array, C_array, log_det_array = [], [], []
        for k in range(self.K):
            # > mu
            mu = self.mean_layer[k](h)
            mu_array.append(mu)
            # > cov
            C, log_det = self.cov_layer[k](h)
            C_array.append(C)
            log_det_array.append(log_det)
        coeff = self.coeff_layer(h)
        return coeff, mu_array, C_array, log_det_array
    
    def sample(self, cond_inputs, n=1):
        device = cond_inputs.device
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        categorical = distribution.Categorical(coeff)
        samples = []
        for i in range(n):
            k = categorical.sample()    # pick a component
            mu, C = mu_array[k][0], C_array[k][0].inverse()
            V = C.mm(C.t())
            normal = distribution.MultivariateNormal(mu, V)
            x = normal.sample()        
            samples.append(x)
        return torch.cat(samples, dim=0)
            
    def log_probs(self, inputs, cond_inputs, eps=1e-40):   
        # pdf = \sum coeff[k] * N(x; mu[k], cov[k])
        n, d = inputs.size()
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        prob = torch.zeros(len(inputs)).to(inputs.device)
        normal = distribution.Normal(torch.tensor([0.0]).to(inputs.device), torch.tensor([1.0]).to(inputs.device))
        # for k in range(self.K):   # <- pdf for each Gaussian component
        #     mu, C, log_det = mu_array[k], C_array[k], log_det_array[k]
        #     z = (inputs - mu).unsqueeze(dim=1)
        #     C_T = C.transpose(dim0=1, dim1=2)
        #     z = z.bmm(C_T)
        #     z = z.squeeze(dim=1)
        #     log_base_prob = normal.log_prob(z).sum(dim=1)
        #     log_prob = log_base_prob + log_det
        #     prob += coeff[:,k] * log_prob.exp() 
        # return (prob+eps).log()
        log_probs = []
        for k in range(self.K):   # <- pdf for each Gaussian component
            mu, C, log_det = mu_array[k], C_array[k], log_det_array[k]
            z = (inputs - mu).unsqueeze(dim=1)
            C_T = C.transpose(dim0=1, dim1=2)
            z = z.bmm(C_T)
            z = z.squeeze(dim=1)
            log_base_prob = normal.log_prob(z).sum(dim=1)
            log_prob = log_base_prob + log_det
            log_prob_with_weight = log_prob + coeff[:, k].log()
            log_probs.append(log_prob_with_weight.view(n, 1))
        log_probs = torch.cat(log_probs, dim=1)
        return log_probs.logsumexp(dim=1).view(-1)

    def log_probs_marginal(self, inputs, cond_inputs, marginals, eps=1e-40):  
        n, d = inputs.size()
        cond_inputs = cond_inputs[0:1, :]
        assert len(cond_inputs) == 1                                         # <-- only support the case where cond_inputs is 1*D   
        assert len(inputs.size()) == 2                                       # <-- inputs must be N*D
        # pdf = \sum coeff[k] * N(x; mu[k], cov[k])
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        prob = torch.zeros(len(inputs)).to(inputs.device)
        normal = distribution.Normal(torch.tensor([0.0]).to(inputs.device), torch.tensor([1.0]).to(inputs.device))
        # for k in range(self.K):   # <- pdf for each Gaussian component
        #     mu, C = mu_array[k][0], C_array[k][0].inverse()
        #     mu, C = mu.view(-1), C.view(self.dim, self.dim)
        #     V = C.mm(C.t())
        #     mu2 = mu[marginals]
        #     V2 = V[marginals, :][:, marginals]
        #     inputs2 = inputs[:, marginals]
        #     normal = distribution.MultivariateNormal(mu2, V2)
        #     log_prob = normal.log_prob(inputs2)
        #     prob += coeff[:,k] * log_prob.exp() 
        # return (prob + eps).log()
        log_probs = []
        for k in range(self.K):   # <- pdf for each Gaussian component
            mu, C = mu_array[k][0], C_array[k][0].inverse()
            mu, C = mu.view(-1), C.view(self.dim, self.dim)
            V = C.mm(C.t())
            mu2 = mu[marginals]
            V2 = V[marginals, :][:, marginals]
            inputs2 = inputs[:, marginals]
            normal = distribution.MultivariateNormal(mu2, V2)
            log_prob = normal.log_prob(inputs2)
            log_prob_with_weight = log_prob + coeff[:, k].log()
            log_probs.append(log_prob_with_weight.view(n, 1))
        log_probs = torch.cat(log_probs, dim=1)
        return log_probs.logsumexp(dim=1).view(-1)
    
    def objective_func(self, x, y):
        log_probs = self.log_probs(x, y)
        is_nan_or_inf = torch.isinf(log_probs) + torch.isnan(log_probs)
        idx_non_nan_inf = (is_nan_or_inf.int() <= 0).int().nonzero(as_tuple=True)[0]
        return log_probs[idx_non_nan_inf].mean()
 
    def learn(self, inputs, cond_inputs, shuffle=False):
        return optimizer.NNOptimizer.learn(self, x=inputs, y=cond_inputs, shuffle=shuffle)
        
    def params(self, cond_inputs, k=0):
        device = cond_inputs.device
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        categorical = distribution.Categorical(coeff)
        mu, C = mu_array[k][0], C_array[k][0].inverse()
        V = C.mm(C.t())
        return mu, V
        
    
    
        
class CoeffLayer(nn.Module):
    def __init__(self, n_in, K):
        super(CoeffLayer, self).__init__()
        self.n_in = n_in
        self.K = K
        self.linear = nn.Linear(n_in, K)
        
    def forward(self, h):
        m, d = h.size()
        out = self.linear(h)
        s = out.exp()
        coeff = s/s.sum(dim=1, keepdim=True) 
        return coeff
    
        
class MeanLayer(nn.Module): 
    def __init__(self, n_in, n_out):
        super(MeanLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        
    def forward(self, h):
        m, d = h.size()
        out = self.linear(h)
        mean = out.view(m, self.n_out)
        return mean
        

class CovLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(CovLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out*n_out)
   
    def mask(self, h):
        n = len(h)
        ones = 1 + torch.zeros(self.n_out, self.n_out)
        ltri_mask = torch.tril(ones, diagonal=-1).expand(n, self.n_out, self.n_out)
        diag_mask = torch.eye(self.n_out).expand(n, self.n_out, self.n_out)
        return ltri_mask.to(h.device), diag_mask.to(h.device)
        
    def forward(self, h):
        n, d = h.size()
        out = self.linear(h)
        out = out.view(n, self.n_out, self.n_out)
        ltri_mask, diag_mask = self.mask(h)
        ltri, diag = out*ltri_mask, (out.exp()*diag_mask)
        C = ltri + diag                    
        log_det = (out*diag_mask).sum(dim=2).sum(dim=1)
        return C, log_det   # x = C^{-1}z, z = Cx, det|C| = -det|dx/dz|  C^{-1}C^{-T} = Sigma
    
    
        
        
        
        
