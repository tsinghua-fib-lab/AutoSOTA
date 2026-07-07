import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import optimizer



class EncodeLayer(nn.Module):
    '''
        encoder for x (and y): z = f(x)
    '''
    def __init__(self, architecture, hyperparams=None):
        super().__init__()
        self.dropout = False if not hasattr(hyperparams, 'dropout') or hyperparams is None else hyperparams.dropout 
        self.head = nn.Linear(architecture[0], architecture[1], bias=True)
        self.main = nn.Sequential( 
           *(nn.Linear(architecture[i+1], architecture[i+2], bias=True) for i in range(len(architecture)-3)),
        )  
        self.drop = nn.Dropout(p=0.20)
        self.out = nn.Sequential(nn.Linear(architecture[-2], architecture[-1], bias=True))
                    
    def forward(self, x):
        x = self.head(x)
        for layer in self.main: x = F.relu(layer(x))
        x = self.drop(x) if self.dropout else x
        x = self.out(x)
        return x
    
