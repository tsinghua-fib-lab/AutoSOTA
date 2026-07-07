import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import optimizer
import estimators.layers as layers
from copy import deepcopy

from nde.FM import FMVGC
from nde.GC import MGC
from nde.VGC import VGC
from nde.GC import GC


class FastMI(nn.Module):
    """ 
        Fast Copula-based MI estimation
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()

        # default hyperparameters 
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 4 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.K_components = 4 if not hasattr(hyperparams, 'K_components') else hyperparams.K_components
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        
        # layers
        d = architecture_critic[0]
        self.gc = None
        self.mog = MGC(d=d//2, K=self.K_components)
        self.mog.forwarding = True                                              # <-- this is copula-based
        print('K components=', self.K_components)
        

    def MI(self, x, y, mode='mc'):
        self.eval()
        with torch.no_grad(): 
            if mode == 'mc':
                return self.mog.KL_joint_marginal(x, y)
            else:
                return self.mog.dv(x, y)
           
    def learn(self, x, y):
        self.mog.learn(x, y)