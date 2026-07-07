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
from nde.VGC import VGC
from nde.GC import GC


class MIENF(nn.Module):
    """ 
        Neural Mutual Information Estimate via Normalizing Flows
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()

        # default hyperparameters 
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.encode_x = False if not hasattr(hyperparams, 'encode_x') else hyperparams.encode_x
        self.encode_y = False if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.K_components = 1 if not hasattr(hyperparams, 'K_components') else hyperparams.K_components
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.joint_learning = True if not hasattr(hyperparams, 'joint_learning') else hyperparams.joint_learning
        
        # layers
        self.encode_layer = None
        self.encode2_layer = None
        print('K components', self.K_components, 'joint learning', self.joint_learning, '\n')
  
    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x) if self.encode_x else x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode_layer(y) if self.encode_y else y
           
    def MI(self, x, y, mode='mc'):
        self.eval()
        with torch.no_grad(): 
            v, w = self.gc.forward(x, y)
            return self.gc.MI(v, w)

    def learn(self, x, y):
        gc = self.learn_nde(x, y)
        self.gc = gc
        self.gc_state_dict = deepcopy(gc.state_dict())
        
    def _relearn_copula(self, x, y, K, existing_gc):
        n, d = x.size()
        print('relearn copula K=', K)
        gc = VGC(n_blocks=2, n_inputs=d, n_hidden=250, n_cond_inputs=2, K=K) 
        gc.maf1.load_state_dict(deepcopy(existing_gc.maf1.state_dict()))
        gc.maf2.load_state_dict(deepcopy(existing_gc.maf2.state_dict()))
        gc.freeze_marginal = True
        gc.maf1.max_iteration = 0
        gc.maf2.max_iteration = 0
        gc.max_iteration = 2000
        gc.bs = 2500
        gc.to(x.device)
        gc.learn(x, y)
        self.gc = gc
        return
        
    def learn_nde(self, x, y):
        n, d = x.size()
    
        # Neural density estimate
        gc = VGC(n_blocks=2, n_inputs=d, n_hidden=250, n_cond_inputs=2, K=self.K_components)      
        gc.to(x.device)
        gc.bs = 250
        if self.joint_learning:
            gc.maf1.max_iteration = 0
            gc.maf2.max_iteration = 0
            gc.max_iteration = 2000
            gc.learn(x, y)
        else:
            gc.maf1.max_iteration = 2000
            gc.maf2.max_iteration = 2000
            gc.max_iteration = 0
            gc.learn(x, y)
            gc.freeze_marginal = True
            gc.maf1.max_iteration = 0
            gc.maf2.max_iteration = 0
            gc.max_iteration = 1000
            gc.bs = 2500
            gc.learn(x, y)
        return gc

    