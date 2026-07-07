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


class VCE(nn.Module):
    """ 
        Vector Copula MI estimation
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()

        # default hyperparameters 
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 4 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.nde_type = 'FM' if not hasattr(hyperparams, 'nde_type') else hyperparams.nde_type
        self.K_components = 5 if not hasattr(hyperparams, 'K_components') else hyperparams.K_components
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        
        # layers
        d = architecture_critic[0]
        self.gc = None
        self.mog = MGC(d=d//2, K=self.K_components, bs=self.bs)
        self.mog.forwarding = True
        print('K components=', self.K_components, 'copula transform=', self.mog.forwarding)
        

    def MI(self, x, y, mode='mc'):
        self.eval()
        with torch.no_grad(): 
            if self.nde_type is not None:
                v, w = self.gc.forward(x, y)
            else:
                v, w = x, y
            if mode == 'mc':
                return self.mog.KL_joint_marginal(v, w)
            else:
                return self.mog.dv(v, w)
           
    def learn(self, x, y):
        self.mog_init_state_dict = self.mog.state_dict()
        # A. learn flow
        if self.nde_type is not None:
            gc = self.learn_flow(x, y)
            self.gc = gc
            self.gc_state_dict = deepcopy(gc.state_dict())    
            with torch.no_grad():
                v, w = gc.forward(x, y)
                v, w = v.clone().detach(), w.clone().detach()
        else:
            v, w = x, y
        # B. learn copula
        self.mog.load_state_dict(self.mog_init_state_dict)
        self.mog.learn(v, w)
        
    def learn_mog(self, x, y, K=5, forwarding=False):
        # only for debug
        n, d = x.size()
        self.mog = MGC(d=d, K=K, bs=self.bs).to(x.device)
        self.mog.forwarding = forwarding
        with torch.no_grad():
            v, w = self.gc.forward(x, y)
            v, w = v.clone().detach(), w.clone().detach()
        self.mog.learn(v, w)
        

    def learn_flow(self, x, y):
        n, d = x.size()
        if self.gc is not None:
            return self.gc
        print('nde type:', self.nde_type)
        if self.nde_type == 'VGC':
            gc = VGC(n_blocks=2, n_inputs=d, n_hidden=500, n_cond_inputs=2)
            gc.to(x.device)
            gc.maf1.max_iteration = 1000
            gc.maf2.max_iteration = 1000
            gc.max_iteration = 0
            gc.bs = 200
        if self.nde_type == 'FM':
            gc = FMVGC(n_inputs=d, bs=self.bs)
            gc.maf1.max_iteration = 2500
            gc.maf2.max_iteration = 2500
            gc.max_iteration = 0
            gc.bs = 200
        gc.to(x.device)
        gc.learn(x, y)
        return gc
    