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


class MRE(nn.Module):
    """ 
        Multin-class density-ratio estimate
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()

        # default hyperparameters 
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 4 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.encode_x = False if not hasattr(hyperparams, 'encode_x') else hyperparams.encode_x
        self.encode_y = False if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.K = 5 if not hasattr(hyperparams, 'n_bridges') else hyperparams.n_bridges
        self.bridge_mode = 'intrapolation' if not hasattr(hyperparams, 'bridge_mode') else hyperparams.bridge_mode
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        
        # layers
        self.encode_layer = None
        self.encode2_layer = None
        self.critic_layer = layers.SoftmaxLayer(architecture_critic, self.K+2, hyperparams)

        print('self.K', self.K)
        
    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x) if self.encode_x else x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode_layer(y) if self.encode_y else y
    
    def _generate_bridge_samples(self, samples_P, samples_Q, k):   
        # generate samples ~ P', Q', where KL(P', Q') < KL(P, Q)
        a1 = 1.0/(self.K+1)*k                                     # when k=0, P'=P; when k=K+1, P'=Q 
        n_P, n_Q = len(samples_P), len(samples_Q)
        if self.bridge_mode == 'mixture':
            idx_P1, idx_Q1 = torch.randperm(n_P)[:int(n_P*(1-a1))], torch.randperm(n_Q)[:int(n_Q*a1)]
            samples_new_P = torch.cat([samples_P[idx_P1]] + [samples_Q[idx_Q1]], dim=0)    
        if self.bridge_mode == 'intrapolation':
            samples_new_P = samples_P*(1-a1) + samples_Q*a1
        return samples_new_P
    
    def MI(self, x, y, mode='mc'):
        self.eval()
        with torch.no_grad(): 
            if mode=='mc':
                return self.log_ratio(x, y).mean().item() 
            else:
                return self.dv(x, y, self.log_ratio).item()
            
    def KL_intermediate(self, z, y, k):
        self.eval()
        with torch.no_grad(): 
            # construct samples P = p(x,y), Q = p(x)p(y)
            m, d = z.size()
            idx_pos = []
            idx_neg1 = []
            idx_neg2 = []
            for i in range(self.n_neg): 
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg1 = idx_neg1 + torch.randperm(m).cpu().numpy().tolist()
                idx_neg2 = idx_neg2 + torch.randperm(m).cpu().numpy().tolist()
            zy_P = torch.cat([z[idx_pos], y[idx_pos]], dim=1)
            zy_Q = torch.cat([z[idx_neg1], y[idx_neg2]], dim=1)
            
            # construct samples from P'(x, y)
            zy_new = self._generate_bridge_samples(zy_P, zy_Q, k)
            
            # intermediate KL
            KL = self.log_ratio(zy_new[:, 0:d], zy_new[:, d:], k, k+1).mean().item()
            return KL
            
    def log_ratio(self, x, y, idx1=0, idx2=-1):
        z, y = self.encode(x), self.encode2(y)
        zy = torch.cat([z, y], dim=1)
        softmax_score = self.critic_layer(zy)
        lr = softmax_score[:, idx1] - softmax_score[:, idx2]
        return lr
    
    def objective_func(self, x, y):
        # compute representation of z, y
        z, y = self.encode(x), self.encode2(y)
        m, d = x.size()
        
        # construct samples P = p(x,y), Q = p(x)p(y)
        idx_pos = []
        idx_neg1 = []
        idx_neg2 = []
        for i in range(self.n_neg): 
            idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
            idx_neg1 = idx_neg1 + torch.randperm(m).cpu().numpy().tolist()
            idx_neg2 = idx_neg2 + torch.randperm(m).cpu().numpy().tolist()
        zy_P = torch.cat([z[idx_pos], y[idx_pos]], dim=1)
        zy_Q = torch.cat([z[idx_neg1], y[idx_neg2]], dim=1)

        # generate bridge + normal samples
        xy_array, label_array = [], []
        with torch.no_grad():
            for k in range(self.K+2):
                xy_k = self._generate_bridge_samples(zy_P, zy_Q, k)            # class 0 is P, class K+1 is Q
                label_k = torch.zeros(len(zy_P)).to(zy_P.device).long() + k
                xy_array.append(xy_k)
                label_array.append(label_k)
        xy_array, label_array = torch.cat(xy_array, dim=0), torch.cat(label_array, dim=0)

        # optimize softmax score
        softmax_score = self.critic_layer(xy_array)
        return -F.cross_entropy(softmax_score, label_array)
    
    def dv(self, x, y, critic):
        # classifier to classifier (x,y) ~ p(x,y) and (x,y) ~ p(x)p(y)
        m, d = x.size()
        z, y = self.encode(x), self.encode2(y)
        idx_pos = []
        idx_neg = []
        n_neg = self.n_neg if self.training else min(m, 50)
        for i in range(n_neg): 
            idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
            idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            
        f_pos = critic(x=z[idx_pos], y=y[idx_pos])
        f_neg = critic(x=z[idx_pos], y=y[idx_neg])
        mi = f_pos.mean() - (f_neg.exp().mean()+1e-30).log()
        return mi
            
    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)
