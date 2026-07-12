"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch
from torch_scatter import scatter_add
from sklearn.metrics import f1_score
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np
import os
import sys
import random
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_missing_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """ 
    Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing.
    If `type`='uniform', then each feature of each node is missing uniformly at random with probability `rate`.
    Instead, if `type`='structural', either we observe all features for a node, or we observe none. For each node
    there is a probability of `rate` of not observing any feature. 
    """
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def get_row_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1} \mathbf{\hat{A}}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DA = deg_inv_sqrt[row] * edge_weight

    return edge_index, DA


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def performance(output, labels, pre=None, evaluator=None):
    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output
    
    if evaluator:
        acc = evaluator.eval({"y_true": labels, "y_pred": preds.unsqueeze(1)})["acc"]
        acc = acc * 100
    else:
        correct = preds.eq(labels).double()
        acc = correct.sum() * 100 / len(labels)
    
    macro_F = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')*100

    return acc, macro_F

def setup_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")

    return logger

def set_filename(args):
    if 'GOODIE' in args.embedder:
        logs_path = f'./logs/{args.dataset}/ours'
    else:
        logs_path = f'./logs/{args.dataset}/baseline'
        os.makedirs(logs_path, exist_ok=True)

    logs_path += f'/{args.mask_type}'
    os.makedirs(logs_path, exist_ok=True)
    
    filename = args.embedder
    if args.embedder == 'GNN':
        filename = args.gnn + f'_{args.filling_method}'

    if 'GOODIE' in args.embedder:
        file = f'{logs_path}/{filename}_lp_alpha_{args.lp_alpha}_lambda_{args.lamb}'
        file += '.txt'
    else:
        file = f'{logs_path}/{filename}.txt'
                
    return file