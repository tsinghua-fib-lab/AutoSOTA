'''
Taken from (Zhang et al. 2024): https://github.com/rsinghlab/scNODE
Description:
    Loss functions.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import geomloss


# ===========================================

def MSELoss(true_obs, est_obs):
    '''
    Mean squared error (MSE).
    :param true_obs (torch.FloatTensor): True expression data.
    :param est_obs (torch.FloatTensor): Predicted expression data.
    :return: (float) MSE value.
    '''
    loss_func = nn.MSELoss(reduction="mean")
    return loss_func(est_obs, true_obs)


def SinkhornLoss(true_obs, est_obs, blur=0.05, scaling=0.5, batch_size=None):
    '''
    Wasserstein distance computed by Sinkhorn algorithm.
    :param true_obs (torch.FloatTensor): True expression data.
    :param est_obs (torch.FloatTensor): Predicted expression data.
    :param blur (float): Sinkhorn algorithm hyperparameter. Default as 0.05.
    :param scaling (float): Sinkhorn algorithm hyperparameter. Default as 0.5.
    :param batch_size (None or int): Either None indicates using all true cell in computation, or an integer indicates
                                     using only a batch of true cells to save computational costs.
    :return: (float) Wasserstein distance average over all timepoints.
    '''
    n_tps = len(true_obs)
    loss = 0.0
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    for t in range(n_tps):
        t_est = est_obs[:, t, :]
        t_true = true_obs[t]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(t_true.shape[0]), size = batch_size, replace = (t_true.shape[0] < batch_size))
            t_true = t_true[cell_idx, :]
        t_loss = ot_solver(t_true, t_est)
        loss += t_loss
    loss = loss / n_tps
    return loss