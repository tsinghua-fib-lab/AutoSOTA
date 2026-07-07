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
from nde.MAF import MAF


class InfoNCE(nn.Module):
    """ 
        InfoNCE estimator to Mutual Information
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()
        
        # hyperparameters
        self.bs = 250 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 4 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.encode_x = False if not hasattr(hyperparams, 'encode_x') else hyperparams.encode_x
        self.encode_y = False if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.critic = 'neural' if not hasattr(hyperparams, 'critic') else hyperparams.critic
        self.max_iteration = 1500
        
        CriticLayer = layers.NeuralCriticLayer if hyperparams.critic == 'neural' else layers.QuadraticCriticLayer
        
        # layers
        self.encode_layer = None
        self.encode2_layer = None
        self.critic_layer = CriticLayer(architecture_critic, 1, hyperparams)
            
    def encode(self, x):
        # s = s(x), get the representation of x
        return self.encode_layer(x) if self.encode_x else x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode2_layer(y) if self.encode_y else y
    
    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.objective_func(x, y).item()

    def log_ratio(self, x, y):
        z, y = self.encode(x), self.encode2(y)
        zy =  torch.cat([z, y], dim=1)
        t = self.critic_layer(zy)
        return t.view(-1)
    
    def objective_func(self, x, y):
        # InfoNCE (InfoNCE, NIPS'18) 
        m, d = x.size()
        z, y = self.encode(x), self.encode2(y)
        idx_pos = []
        idx_neg = []
        n_neg = self.n_neg if self.training else min(m, 50)
        for i in range(m):
            idx = torch.tensor(np.linspace(0, m-1, m))
            idx_not_i = idx               #idx_not_i = idx[idx.ne(i).nonzero().view(-1)].numpy()
            subset = torch.randperm(m-1)[0:n_neg-1].cpu().numpy()
            idx_pos = idx_pos + (np.zeros(n_neg)+i).tolist()
            idx_neg = idx_neg + (idx_not_i[subset].tolist() + [i]) 
        zy_pos = torch.cat([z, y], dim=1)
        zy_neg = torch.cat([z[idx_pos], y[idx_neg]], dim=1)
        f_pos = self.critic_layer(zy_pos)
        f_neg = self.critic_layer(zy_neg).view(m, n_neg)
        nll = f_pos - f_neg.logsumexp(dim=1).mean()
        mi = nll + np.log(n_neg)
        return mi.mean()
 
    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)