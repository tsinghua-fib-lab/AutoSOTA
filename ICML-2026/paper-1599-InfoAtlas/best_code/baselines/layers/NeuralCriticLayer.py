import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import scipy.stats as stats




class NeuralCriticLayer(nn.Module): 
    '''
        a small NLP that output the critic value
    '''
    def __init__(self, architecture, K, hyperparams=None):                  
        super().__init__()       
        in_dims = architecture[0]                                                                           
        dim_hidden = architecture[1]
        self.input = nn.Sequential(
            nn.Linear(in_dims, dim_hidden),
        )
        self.BN = False                                   # for critic used in InfoNCE & MINE, bn does not work well
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden) 
        self.main = nn.Sequential(*[nn.Linear(dim_hidden, dim_hidden) for _ in range(len(architecture)-3)])   
        self.out = nn.Linear(in_dims + dim_hidden, 1)
        self.dropout = nn.Dropout(0.25)
 
    def forward(self, xy):
        h = self.input(xy) 
        h = self.bn1(h) if self.BN else h
        for i, layer in enumerate(self.main): 
            h = layer(F.leaky_relu(h, 0.2))
        h = self.bn2(h) if self.BN else h
        h = F.leaky_relu(h, 0.2)
        h = torch.cat([h, xy], dim=1)                     # dense net arch very important!  
        out = self.out(h)
        return out