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



class SMILE(nn.Module):
    """ 
        Smoothed Mutual Information ``Lower Bound'' Estimator
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()
        
        # hyperparameters
        self.estimator = 'NCE' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 250 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 4 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.encode_x = False if not hasattr(hyperparams, 'encode_x') else hyperparams.encode_x
        self.encode_y = False if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.clip_value = 8 if not hasattr(hyperparams, 'clip_value') else hyperparams.clip_value 
        self.max_iteration = 500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        
        CriticLayer = layers.NeuralCriticLayer if hyperparams.critic == 'neural' else layers.QuadraticCriticLayer
        
        # layers
        self.encode_layer = None
        self.encode2_layer = None
        self.critic_layer = CriticLayer(architecture_critic, 1, hyperparams)

        print('clip value=', self.clip_value)
            
    def encode(self, x):
        # s = s(x), get the representation of x
        return self.encode_layer(x) if self.encode_x else x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode2_layer(y) if self.encode_y else y
    
    def critic(self, xy):
        b = self.clip_value
        return self.critic_layer(xy).clamp(-b, b)

    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.objective_func(x, y).item()

    def log_ratio(self, x, y):
        z, y = self.encode(x), self.encode2(y)
        zy =  torch.cat([z, y], dim=1)
        t = self.ctitic(zy)
        return t.view(-1)
    
    def objective_func(self, x, y):
        # classifier to classifier (x,y) ~ p(x,y) and (x,y) ~ p(x)p(y)
        m, d = x.size()
        z, y = self.encode(x), self.encode2(y)
        idx_pos = []
        idx_neg = []
        n_neg = self.n_neg if self.training else min(m, 50)
        for i in range(n_neg): 
            idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
            idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            
        zy_pos = torch.cat([z[idx_pos], y[idx_pos]], dim=1)
        zy_neg = torch.cat([z[idx_pos], y[idx_neg]], dim=1)
        f_pos = self.critic(zy_pos)
        f_neg = self.critic(zy_neg)
        mi = f_pos.mean() - (f_neg.exp().mean()+1e-30).log()
        return mi
 
    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)