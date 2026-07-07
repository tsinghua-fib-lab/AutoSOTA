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
from nde import MAF, MDN, NAF



class DoE(nn.Module):
    """ 
        Difference of Entropy
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()
        
        # hyperparameters
        self.estimator = 'DoE' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 250 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.max_iteration = 300
        
        # nde
        d = architecture_critic[0]
        n_hidden = 200
        self.flow = MAF(n_blocks=3, n_inputs=d//2, n_hidden=n_hidden, n_cond_inputs=d//2)
        self.flow_x = MAF(n_blocks=3, n_inputs=d//2, n_hidden=n_hidden, n_cond_inputs=2)
            
    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.log_ratio(x, y).mean().item()

    def log_ratio(self, x, y):
        n, d = x.size()
        ll = self.flow.log_probs(x, y)                                  # p(x|y)
        ll_x = self.flow_x.log_probs(x)                                 # p(x)
        return ll - ll_x
    
    def objective_func(self, x, y):
        n, d = x.size()
        ll = self.flow.log_probs(x, y)                                  # p(x|y)
        ll_x = self.flow_x.log_probs(x)                                 # p(x)
        return ll.mean() + ll_x.mean()

    def learn(self, x, y):
        self.flow.max_iteration = self.max_iteration
        self.flow_x.max_iteration = self.max_iteration
        self.flow.learn(x, y) 
        self.flow_x.learn(x)