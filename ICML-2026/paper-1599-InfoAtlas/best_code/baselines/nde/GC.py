import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
from scipy import stats
import scipy.special as special
from scipy.stats import binom
from scipy.stats import norm
from .MDN import MDN


        
            

class GC(nn.Module):
    """ 
        Gaussian copula (non-parametric)
    """
    def __init__(self):
        super().__init__()
        self.dummy_params = nn.Linear(1, 1)
        
    def learn(self, x, y):
        xy = torch.cat([x, y], dim=1)
        n, d = xy.size()
        # calculate the latent Z
        xx, yy = self.forward(x, y)
        z = torch.cat([xx, yy], dim=1)
        V = torch.matmul(z.t(), z)/(len(z)+1)
        A = torch.cholesky(V, upper=False)
        A_t_inv = torch.inverse(A.t())
        # calculate e in z = Ae
        eps = torch.matmul(z, A_t_inv)
        # assign values
        self.V = V
        self.V2 = torch.eye(d).to(x.device)
        self.V2[0:d//2, 0:d//2] = self.V[0:d//2, 0:d//2]
        self.V2[d//2:, d//2:] = self.V[d//2:, d//2:]                  # V2's x-block-matrix and y-block-matrix are the same as V
        self.Vx = self.V[0:d//2, 0:d//2]
        self.Vy = self.V[d//2:, d//2:]
        self.V_inv, self.Vx_inv, self.Vy_inv = torch.inverse(self.V), torch.inverse(self.Vx), torch.inverse(self.Vy)
        self.normal = distribution.multivariate_normal.MultivariateNormal(torch.zeros(d).to(x.device), self.V)
        self.normal2 = distribution.multivariate_normal.MultivariateNormal(torch.zeros(d).to(x.device), self.V2)
    
    def forward(self, x, y):
        data = torch.cat([x, y], dim=1)
        # calculate empirical CDF
        sorted_data, idx = torch.sort(data, dim=0)
        _, idx2 = torch.sort(idx, dim=0)
        u = (idx2.float()+1)/(len(data)+1)    
        zeros, ones = torch.zeros(data.size()).to(data.device), torch.ones(data.size()).to(data.device)
        normal = distribution.Normal(zeros, ones)
        # calculate the latent Z
        z = normal.icdf(u)
        n, d = z.size()
        return z[:, 0:d//2], z[:, d//2:]
            
    def sample(self, n=10000, inner=True):
        # # some preparation
        # sorted_xy = self.sorted_xy
        # N, D = sorted_xy.size()
        #sample z ~ N(0, V)
        z = self.normal.rsample([n])
        # return z
        if inner==True:
            return z[0:n, :]
        else:
            return None
        
        # # convert z to u
        # normal = distribution.Normal(torch.zeros(N, D).to(sorted_xy.device), torch.ones(N, D).to(sorted_xy.device))
        # u = normal.cdf(z).clamp(0.00001, 0.99999)
        # # convert u to idx
        # idx = (N*u).long()
        # # idx to x
        # x = torch.zeros(N, D).to(sorted_xy.device)
        # for d in range(D):
        #     idx_d = idx[:, d]
        #     sorted_x_d = sorted_xy[:, d]
        #     x_d = sorted_x_d[idx_d]
        #     x[:, d] = x_d
        # return x[0:n, :]
    
    @staticmethod
    def log_copula_density(z, V):
        # log_det_V = torch.logdet(V)
        # d, d = V.size()
        # device = z.device
        # V_inv = torch.inverse(V)
        # inside_exp = torch.diag(z@V_inv@z.t())
        # return -0.5*inside_exp - 0.5*log_det_V
        d, d = V.size()
        mu = torch.zeros(d).to(V.device)
        normal = distribution.MultivariateNormal(mu, V)
        return normal.log_prob(z)
    
    def fake_critic(self, x, y):
        #x, y = self.forward(x, y)
        xy = torch.cat([x, y], dim=1)
        log_copula_density_xy = GC.log_copula_density(xy, self.V)
        log_copula_density_x = GC.log_copula_density(x, self.Vx)
        log_copula_density_y = GC.log_copula_density(y, self.Vy)
        mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
        return mi
        
    def KL_joint_marginal(self, x, y, forwarding=False):                                     # E[log q(x ,y)/q(x)q(y)]
        if forwarding:
            x, y = self.forward(x, y)
        xy = torch.cat([x, y], dim=1)
        log_copula_density_xy = GC.log_copula_density(xy, self.V)
        log_copula_density_x = GC.log_copula_density(x, self.Vx)
        log_copula_density_y = GC.log_copula_density(y, self.Vy)
        mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
        return mi.mean().item()
    
    def dv(self, x, y):
        x, y = self.forward(x, y)
        return dv_representation(x, y, self.fake_critic)
    
    def params(self):
        print(self.V.cpu().numpy())
        return
    
    
    # def MI(self):
    #     xy = self.normal.sample([10000])
    #     n, D = xy.size()
    #     x, y = xy[:, 0:D//2], xy[:, D//2:]
    #     log_copula_density_xy = GC.log_copula_density(xy, self.V)
    #     log_copula_density_x = GC.log_copula_density(x, self.Vx)
    #     log_copula_density_y = GC.log_copula_density(y, self.Vy)
    #     mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
    #     return mi.mean().item()
    
    
    
    
    
    
    
class MGC(nn.Module):
    """ 
        Mixture of Gaussian copula
    """
    def __init__(self, d, K, bs):
        super().__init__()
        #self.mog = MDN(n_in=2*d, n_hidden=25, n_out=2*d, K=K)
        self.mog = MDN(n_in=2*d, n_hidden=1, n_out=2*d, K=K)
        self.mog.bs = bs
        self.mog.lr = 5e-2
        self.mog.wd = 0e-5
        self.mog.max_iteration = 2000
        self.forwarding = False
        
    def learn(self, x, y):
        xy = torch.cat([x, y], dim=1)
        n, d = xy.size()
        # compute ranks
        x, y = self.forward(x, y)
        z = torch.cat([x, y], dim=1)
        # fit mog on Z
        print('bs=', self.mog.bs)
        self.mog.learn(inputs=z, cond_inputs=torch.ones_like(z), shuffle=True)
        print('K components=', self.mog.K, 'finished')
        
    # def learn2(self, x, y):
    #     xy = torch.cat([x, y], dim=1)
    #     n, d = xy.size()
    #     x, y = self.forward(x, y)
    #     z = torch.cat([x, y], dim=1)
    #     # fit mog on Z
    #     self.mog.learn2(inputs=z, cond_inputs=torch.ones_like(z))
        
    def forward(self, x, y):
        if self.forwarding:
            data = torch.cat([x, y], dim=1)
            # calculate empirical CDF
            sorted_data, idx = torch.sort(data, dim=0)
            _, idx2 = torch.sort(idx, dim=0)
            u = (idx2.float()+1)/(len(data)+1)    
            zeros, ones = torch.zeros(data.size()).to(data.device), torch.ones(data.size()).to(data.device)
            normal = distribution.Normal(zeros, ones)
            # calculate the latent Z
            z = normal.icdf(u)
            n, d = z.size()
            return z[:, 0:d//2], z[:, d//2:]
        else:
            return x, y
            
    def KL_joint_marginal(self, x, y):                                         # E[log q(x ,y)/q(x)q(y)]
        x, y = self.forward(x, y)
        return self.fake_critic(x, y).mean().item()
    
    
    def fake_critic(self, x, y, eps=0e-50, filter_nan=False):
        with torch.no_grad():
            n, d = x.size()
            marginals_x = [i for i in range(d)]
            marginals_y = [i+d for i in range(d)]     
            xy = torch.cat([x, y], dim=1)
            cond_inputs=torch.ones_like(xy)
            log_copula_density_xy = self.mog.log_probs(xy, cond_inputs, eps=eps)
            log_copula_density_x = self.mog.log_probs_marginal(xy, cond_inputs, marginals_x, eps=eps)
            log_copula_density_y = self.mog.log_probs_marginal(xy, cond_inputs, marginals_y, eps=eps)
            mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
            return mi
            # if filter_nan:
            #     is_nan_or_inf = torch.isinf(mi) + torch.isnan(mi)
            #     idx_non_nan_inf = (is_nan_or_inf.int() <= 0).int().nonzero(as_tuple=True)[0]
            #     print('ratio kept:', len(idx_non_nan_inf)/n)
            #     return mi[idx_non_nan_inf].view(-1)
            # else:
            #     return mi.view(-1)
    
    def dv(self, x, y):
        x, y = self.forward(x, y)
        return dv_representation(x, y, self.fake_critic)
        
    def params(self, x, y):
        xy = torch.cat([x, y], dim=1)
        cond_inputs = torch.ones_like(xy)
        mu, V = self.mog.params(cond_inputs, 0)
        print('mu=', mu.detach().cpu().numpy()[0:10])
        print('V=', (V.detach().cpu().numpy()[0:10, 0:10]*100).astype(int)/100.0)
        return
    
    
    
    
    

              
              
              
              
              
              
              
def dv_representation(x, y, critic_func, clip_value=100, neg_sampling_times=100):
    m, d = x.size()
    # compute f-pos and f-neg
    f_pos, f_neg = [], []
    for i in range(neg_sampling_times): 
        idx_pos = np.linspace(0, m-1, m).tolist()
        idx_neg = torch.randperm(m).cpu().numpy().tolist()
        f_pos.append(critic_func(x[idx_pos], y[idx_pos]))
        f_neg.append(critic_func(x[idx_pos], y[idx_neg]))
    f_pos, f_neg = torch.cat(f_pos, dim=0), torch.cat(f_neg, dim=0)
    # cliping values
    f_pos, f_neg = f_pos.clamp(-clip_value, clip_value), f_neg.clamp(-clip_value, clip_value)
    # applying the DV representation
    mi = f_pos.mean() - (f_neg.exp().mean()+1e-40).log()
    return mi.item()