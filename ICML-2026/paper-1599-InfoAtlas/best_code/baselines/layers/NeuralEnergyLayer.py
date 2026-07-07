import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import optimizer
from copy import deepcopy




class NeuralEnergyLayer(nn.Module): 
    '''
        energy(x)
    '''
    def __init__(self, architecture, K, hyperparams=None):                  
        super().__init__()       
        self.V = 30 if not hasattr(hyperparams, 'V') else hyperparams.V 
        in_dims = architecture[0]                                        
        dim_hidden = architecture[1]
        self.input = nn.Sequential(
            nn.Linear(in_dims, dim_hidden),
        )
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.bn2 = nn.BatchNorm1d(in_dims + dim_hidden)
        self.main = nn.Sequential(*[nn.Linear(dim_hidden, dim_hidden) for _ in range(len(architecture)-3)])      
        self.out = nn.Linear(in_dims + dim_hidden, self.V)

    # def forward(self, xy):
    #     h = F.softplus(self.input(xy)) 
    #     for layer in self.main: h = F.softplus(layer(h))                             
    #     out = self.out(h)   
    #     out = torch.sin(out).sum(dim=1)        
    #     return out.view(len(h), -1)
        
    def forward(self, xy):
        h = self.input(xy) 
        for i, layer in enumerate(self.main): 
            h = layer(F.softplus(h))    
        h = F.softplus(h)
        h = torch.cat([h, xy], dim=1)                     # dense net arch very important!   
        out = self.out(h)
        return torch.sin(out).sum(dim=1)                  # so output will be within -30, 30