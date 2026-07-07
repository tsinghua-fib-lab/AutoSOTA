import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
import optimizer
from . import NAF, MAF, MDN
from copy import deepcopy


class VGC(nn.Sequential):
    """ 
        Vector Gaussian copula
    """
    def __init__(self, n_blocks, n_inputs, n_hidden, n_cond_inputs=2, K=1):
        super().__init__()
        self.maf1 = NAF(n_blocks, n_inputs, n_hidden, n_cond_inputs)
        self.maf2 = NAF(n_blocks, n_inputs, n_hidden, n_cond_inputs)
        self.base = MDN(n_in=2, n_hidden=10, n_out=2*n_inputs, K=K)
        self.max_iteration = 200
        self.lr = 1e-3
        self.bs = 250
        self.normal = None
        self.freeze_marginal = False
        self.phrase = 'joint'

    def forward(self, x, y):
        xx, _ = self.maf1.forward(x)
        yy, _ = self.maf2.forward(y)
        return xx, yy
    
    def sample(self, size):
        with torch.no_grad():
            z = self.normal.rsample(size)
            n, d = z.size()
            z_x, z_y = z[:, 0:d//2], z[:, d//2:]
            x, _ = self.maf1.forward(inputs=z_x, mode='inverse')
            y, _ = self.maf2.forward(inputs=z_y, mode='inverse')
            return torch.cat([x.clone().detach(), y.clone().detach()], dim=1)
    
    def log_prob(self, xy):
        n, d = xy.size()
        x, y = xy[:, 0:d//2], xy[:, d//2:]
        xx, log_jacob_xx = self.maf1.forward(x)
        yy, log_jacob_yy = self.maf2.forward(y)
        if self.freeze_marginal:
            xx, yy = xx.clone().detach(), yy.clone().detach()
            log_jacob_xx, log_jacob_yy = log_jacob_xx.clone().detach(), log_jacob_yy.clone().detach()
        xxyy = torch.cat([xx, yy], dim=1)
        log_jacob = (log_jacob_xx + log_jacob_yy).view(n, -1)
        t = torch.ones(n, 2).to(xy.device)
        if self.normal is None:
            log_base_prob = self.base.log_probs(inputs=xxyy, cond_inputs=t).view(n, -1)          
        else:
            log_base_prob = self.normal.log_prob(xxyy).view(n, -1)
        return (log_base_prob + log_jacob).view(-1)
    
    def log_prob_marginal(self, xy):
        n, d = xy.size()
        x, y = xy[:, 0:d//2], xy[:, d//2:]
        t = torch.ones(n, 2).to(xy.device)
        xx, log_jacob_xx = self.maf1.forward(x)
        yy, log_jacob_yy = self.maf2.forward(y)
        log_base_prob_xx = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=[i for i in range(d//2)]).view(n, -1) 
        log_base_prob_yy = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=[i+d//2 for i in range(d//2)]).view(n, -1) 
        return log_jacob_xx + log_base_prob_xx + log_base_prob_yy + log_jacob_yy
        
    
    def objective_func(self, x, y):
        xy = torch.cat([x, y], dim=1)
        if self.phrase == 'marginal':
            log_probs = self.maf1.log_prob(x).mean() + self.maf2.log_prob(y).mean() 
        else:
            log_probs= self.log_prob(xy).mean()
        return log_probs
    
    def learn(self, x, y):
        n, d = x.size()
        if self.maf1.max_iteration > 0:
            self.maf1.learn(x)
        if self.maf2.max_iteration > 0:
            self.maf2.learn(y)
        if self.max_iteration > 0:
            optimizer.NNOptimizer.learn(self, x=x, y=y)
        return 

    def learn_marginal_only(self, x, y):
        self.phrase = 'marginal'
        optimizer.NNOptimizer.learn(self, x=x, y=y)

    def learn_jointly(self, x, y):
        self.phrase = 'joint'
        optimizer.NNOptimizer.learn(self, x=x, y=y)
    
    def empirical_params(self, z):
        n, d = z.size()
        mu = z.mean(dim=0, keepdim=True)
        V = (z-mu).t() @ (z-mu)/(n+1)
        return mu.view(-1), V
    
    
    # copula transformation
    def copula_transformation(self, z):
        data = z
        # calculate empirical CDF
        sorted_data, idx = torch.sort(data, dim=0)
        _, idx2 = torch.sort(idx, dim=0)
        u = (idx2.float()+1)/(len(data)+1)    
        zeros, ones = torch.zeros(data.size()).to(data.device), torch.ones(data.size()).to(data.device)
        normal = distribution.Normal(zeros, ones)
        # calculate the latent Z
        z = normal.icdf(u)
        n, d = z.size()
        return z
    
    # compute MI
    def MI(self, x, y, inner=True):
        if inner is False:
            x, y = self.forward(x, y)
        xy = torch.cat([x, y], dim=1)
        n, dx = x.size()
        t = torch.ones(n, 2).to(xy.device)
        log_copula_density_xy = self.base.log_probs(inputs=xy, cond_inputs=t)
        log_copula_density_x = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=[i for i in range(dx)])
        log_copula_density_y = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=[dx+i for i in range(dx)])       
        mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
        return mi.mean().item()
        
#     # used in separate learning mode
#     def MI(self, x, y, inner=True):
#         if inner is False:
#             x, y = self.forward(x, y)
#         xy = torch.cat([x, y], dim=1)
#         xy = self.copula_transformation(xy)
#         log_copula_density_xy = self.normal.log_prob(xy)
#         log_copula_density_x = self.normal_x.log_prob(x)
#         log_copula_density_y = self.normal_y.log_prob(y)       
#         mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
#         return mi.mean().item()
    
#     # used in joint learning mode
#     def MI2(self, x, y, inner=True):
#         if inner is False:
#             x, y = self.forward(x, y)
#         xy = torch.cat([x, y], dim=1)
#         n, dx = x.size()
#         t = torch.ones(n, 2).to(xy.device)
#         log_copula_density_xy = self.base.log_probs(inputs=xy, cond_inputs=t)
#         log_copula_density_x = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=[i for i in range(dx)])
#         log_copula_density_y = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=[dx+i for i in range(dx)])       
#         mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
#         return mi.mean().item()
    
    def print(self):
        print('mu=', self.mu)
        print('V=',  (self.V*100).int()/100.0)


    #optimizer.NNOptimizer.learn(self, x=x, y=y)