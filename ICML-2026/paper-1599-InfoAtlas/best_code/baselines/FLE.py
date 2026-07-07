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
from nde import MAF, TAF
from copy import deepcopy


class FLE(nn.Module):
    """ 
        Flow-based MI estimator (plug-in estimation)
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()
        
        # hyperparameters
        self.estimator = 'FLE' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 250 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.nde = 'MAF' if not hasattr(hyperparams, 'nde') else hyperparams.nde
        self.max_iteration = 2000
        
        # layers
        d = architecture_critic[0]
        n_hidden = 200
        print('nde type', self.nde)
        if self.nde == 'MAF':
            self.flow_joint = MAF(n_blocks=3, n_inputs=d, n_hidden=n_hidden, n_cond_inputs=2)
            self.flow_marginal = MAF(n_blocks=3, n_inputs=d, n_hidden=n_hidden, n_cond_inputs=2)
        else:
            self.flow_joint = TAF(n_blocks=3, n_inputs=d, n_hidden=n_hidden)
            self.flow_marginal = TAF(n_blocks=3, n_inputs=d, n_hidden=n_hidden)
            
    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.log_ratio(x, y).mean().item()

    def log_ratio(self, x, y):
        n, d = x.size()
        data = torch.cat([x, y], dim=1)
        ll_joint = self.flow_joint.log_probs(data)                                        # p(x, y)
        ll_marginal = self.flow_marginal.log_probs(data)                                  # p(x)p(y)
        return ll_joint - ll_marginal
    
    def objective_func(self, x, y):
        n, d = x.size()
        # prepare data
        idx = torch.randperm(n)
        data_joint = torch.cat([x, y], dim=1).clone().detach()
        data_marginal = torch.cat([x, y[idx]], dim=1).clone().detach()
        # likelihood computation
        ll_joint = self.flow_joint.log_probs(data_joint)                                  # p(x, y)
        ll_marginal = self.flow_marginal.log_probs(data_marginal)                         # p(x)p(y)
        #return ll_joint.mean() if self.stage == 1 else ll_marginal.mean()
        return ll_joint.mean() + ll_marginal.mean()
        
    def learn(self, x, y):
        # self.stage = 1
        # optimizer.NNOptimizer.learn(self, x, y)
        # joint_state_dic = deepcopy(self.flow_joint.state_dict())
        # self.stage = 2
        # optimizer.NNOptimizer.learn(self, x, y)
        # self.flow_joint.load_state_dict(joint_state_dic)
        # return
        return optimizer.NNOptimizer.learn(self, x, y)