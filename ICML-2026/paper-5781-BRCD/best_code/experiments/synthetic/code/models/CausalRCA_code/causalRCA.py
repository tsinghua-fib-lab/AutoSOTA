import argparse
import datetime
import json
import math
import os
import pickle as pkl
import requests
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.notebook import tqdm, trange

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
from sknetwork.ranking import PageRank

from models.CausalRCA_code.utils import *
from models.CausalRCA_code.modules import *
from models.CausalRCA_code.config import CONFIG

prox_plus = torch.nn.Threshold(0.,0.)
# from utils import get_triu_offdiag_indices, get_tril_offdiag_indices, matrix_poly, nll_gaussian, kl_gaussian_sem, A_connect_loss, A_positive_loss, stau
def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def causalRCA(df_a, metric_names=None):
    """
    Fully functional version of CausalRCA using DAG-GNN 
    
    Args:
        df_a (pd.DataFrame): Time-series metric data (rows = time, cols = metrics)
        metric_names (list): Optional list of metric names (for labeling output)
    
    Returns:
        dict: {'root_cause': ranked list of column names}
    """
    CONFIG.cuda = torch.cuda.is_available()
    CONFIG.factor = not CONFIG.no_factor

    # --------------------
    # Step 1: Normalize
    # --------------------
    scaler = StandardScaler()
    data = scaler.fit_transform(df_a.values)
    data = pd.DataFrame(data, columns=df_a.columns)

    data_sample_size = data.shape[0]
    data_variable_size = data.shape[1]
    train_data = data

    # --------------------
    # Step 2: Initialize DAG-GNN
    # --------------------
    adj_A = np.zeros((data_variable_size, data_variable_size))

    encoder = MLPEncoder(data_variable_size * CONFIG.x_dims, CONFIG.x_dims, CONFIG.encoder_hidden,
                         int(CONFIG.z_dims), adj_A,
                         batch_size=CONFIG.batch_size,
                         do_prob=CONFIG.encoder_dropout, factor=CONFIG.factor).double()
    
    decoder = MLPDecoder(data_variable_size * CONFIG.x_dims, CONFIG.z_dims, CONFIG.x_dims, encoder,
                         data_variable_size=data_variable_size,
                         batch_size=CONFIG.batch_size,
                         n_hid=CONFIG.decoder_hidden,
                         do_prob=CONFIG.decoder_dropout).double()

    if CONFIG.cuda:
        encoder.cuda()
        decoder.cuda()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr)

    triu_indices = get_triu_offdiag_indices(data_variable_size)
    tril_indices = get_tril_offdiag_indices(data_variable_size)
    if CONFIG.cuda:
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    # Training settings
    lambda_A = CONFIG.lambda_A
    c_A = CONFIG.c_A
    h_tol = CONFIG.h_tol
    gamma = CONFIG.gamma
    eta = CONFIG.eta
    k_max_iter = int(CONFIG.k_max_iter)
    h_A_old = np.inf

    best_ELBO_loss = np.inf

    for step_k in range(k_max_iter):
        while c_A < 1e+20:
            for epoch in range(CONFIG.epochs):
                encoder.train()
                decoder.train()
                batch = torch.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1)).double()
                if CONFIG.cuda:
                    batch = batch.cuda()
                batch = Variable(batch)

                optimizer.zero_grad()

                enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(batch)
                dec_x, output, adj_A_tilt_decoder = decoder(batch, logits, data_variable_size * CONFIG.x_dims, origin_A, adj_A_tilt_encoder, Wa)

                loss_nll = nll_gaussian(output, batch, 0.)
                loss_kl = kl_gaussian_sem(logits)
                loss = loss_nll + loss_kl

                one_adj_A = origin_A
                sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

                if CONFIG.use_A_connect_loss:
                    connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
                    loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap**2

                if CONFIG.use_A_positiver_loss:
                    positive_gap = A_positive_loss(one_adj_A, z_positive)
                    loss += 0.1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap**2)

                h_A = torch.trace(matrix_poly(one_adj_A @ one_adj_A, data_variable_size)) - data_variable_size
                loss += lambda_A * h_A + 0.5 * c_A * h_A**2 + 100. * torch.trace(one_adj_A @ one_adj_A) + sparse_loss

                loss.backward()
                optimizer.step()

                myA.data = stau(myA.data, CONFIG.tau_A * CONFIG.lr)

                if loss.item() < best_ELBO_loss:
                    best_ELBO_loss = loss.item()
                    best_graph = one_adj_A.data.clone().cpu().numpy()

            h_A_new = torch.trace(matrix_poly(one_adj_A @ one_adj_A, data_variable_size)) - data_variable_size
            if h_A_new.item() > gamma * h_A_old:
                c_A *= eta
            else:
                break
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()
        if h_A_new.item() <= h_tol:
            break

    # --------------------
    # Step 3: Inference via PageRank
    # --------------------
    adj = best_graph
    adj[np.abs(adj) < CONFIG.graph_threshold] = 0
    pagerank = PageRank()
    pagerank.fit(np.abs(adj.T)) 
    scores = pagerank.scores_

    if metric_names is None:
        metric_names = list(df_a.columns)

    score_dict = {metric_names[i]: score for i, score in enumerate(scores)}
    ranked_metrics = sorted(score_dict, key=score_dict.get, reverse=True)

    return {"root_cause": ranked_metrics}
    
