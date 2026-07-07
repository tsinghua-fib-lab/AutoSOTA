import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import scipy.stats as stats
from scipy.linalg import block_diag
from copy import deepcopy




class QuadraticCriticLayer(nn.Module): 
    '''
        r(x, y, k) = h^T(Q_k)h, where h=[x, y]
    '''
    def __init__(self, architecture, K, hyperparams=None):                            # for NCE, K=1
        super().__init__()       
        in_dims = architecture[0]
        self.K = K
        dim_hidden, dim_before_critic = architecture[1], architecture[-2]
        self.input = nn.Sequential(
            nn.Linear(in_dims, dim_hidden),
        )
        self.main = nn.Sequential(*(nn.Linear(architecture[i+1], architecture[i+2], bias=True) for i in range(len(architecture)-3)))        
        Q_shape = (K, dim_before_critic, dim_before_critic)
        self.Q_array = nn.Parameter(data=torch.zeros(size=Q_shape), requires_grad=True)                     
        print('self.Q_array', self.Q_array.size())

    def reparam_Q(self, unconstrained_Q):
        Q = pt_enforce_symmetric_and_pos_diag(unconstrained_Q, shift=5.0)
        return Q
    
    def sum_Q(self):
        Q_sum = self.reparam_Q(self.Q_array[0].squeeze())  
        for k in range(self.K-1):
            Q_sum += self.reparam_Q(self.Q_array[k+1].squeeze())  
        self.Q_sum = Q_sum
        return Q_sum
            
    def forward(self, xy, k=0):
        # compute representation for xy
        h = self.input(xy) 
        for layer in self.main: h = layer(F.leaky_relu(h))                             
        # result for the k-th head
        Q = self.reparam_Q(self.Q_array[k].squeeze())    
        V = torch.matmul(torch.matmul(h, Q), h.t())         # h^T Q h
        out = torch.diag(V)
        return out.view(len(h), -1)
    
    
    
    
    
    
    
    



def pt_enforce_lower_diag_and_pos_diag(A, shift=0.0):
    A1 = torch.tril(A, diagonal=-1)  # lower diagonal without diagonal
    A2 = torch.exp(A - shift) * torch.eye(A.shape[-1], device=A.device)
    return A1 + A2


def pt_enforce_symmetric_and_pos_diag(A, shift=0.0):
    B = torch.tril(A, diagonal=-1)
    if len(A.shape) == 3:
        symB = B + torch.transpose(B, 1, 2)
    else:
        symB = B + torch.t(B)
    return symB + torch.exp(A - shift) * torch.eye(A.shape[-1], device=A.device)


def pt_batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.
    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.
    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)
