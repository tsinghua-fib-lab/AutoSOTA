import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import torch.distributions as distribution
import numpy as np
import scipy
import math
import time
import optimizer



class CopulaLayer(nn.Module):
    '''
        copula layer: converting data to be approximately N(0, I)
    '''
    def __init__(self):
        super().__init__()

                    
    def forward(self, x):
        data = x
        # calculate empirical CDF
        sorted_data, idx = torch.sort(data, dim=0)
        _, idx2 = torch.sort(idx, dim=0)
        u = (idx2.float()+1)/(len(data)+1)    
        zeros, ones = torch.zeros(data.size()).to(data.device), torch.ones(data.size()).to(data.device)
        normal = distribution.Normal(zeros, ones)
        # calculate the latent Z
        z = normal.icdf(u)
        n, d = z.size()
        # decorrelation
        V = torch.matmul(z.t(), z)/(len(z)+1)
        A = torch.cholesky(V, upper=False)
        A_t_inv = torch.inverse(A.t())
        # calculate e in z = Ae
        eps = torch.matmul(z, A_t_inv)
        return eps
    
    
    
