import sys
from typing import List

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import random
import os

import torch
from sklearn.preprocessing import MinMaxScaler


from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
import pyciphod.causal_discovery.federated.regret_based.ges as ges

from sklearn.metrics import f1_score


# from pyciphod.causal_discovery.federated.regret_based.iperi.graphs import *
from pyciphod.causal_discovery.federated.regret_based.ges.scores.gauss_obs_l0_pen import GaussObsL0Pen


""" function retriving scoring function """
def get_scoring_class(scoring_function: str, lmbda: float = None, ebic_gamma: float = 1.0, bic_coeff: float = 0.5):
    if scoring_function == 'aic':
        return lambda data: GaussObsL0Pen(data, lmbda=1, method='raw')
    elif scoring_function == 'bic':
        return lambda data: GaussObsL0Pen(data, lmbda=bic_coeff * np.log(data.shape[0]), method='raw')
    elif scoring_function == 'ebic':
        return lambda data: GaussObsL0Pen(data,
            lmbda=0.5 * np.log(data.shape[0]) + ebic_gamma * np.log(data.shape[1]**2),
            method='raw')
    elif scoring_function == 'bic_pen':
        if lmbda is None:
            raise ValueError("labda must be provided for bic_pen scoring function")
        return lambda data: GaussObsL0Pen(data, lmbda=lmbda, method='raw')
    else:
        raise ValueError(f"Unknown scoring function: {scoring_function}")
    

""" function retriving cd function """
def get_cd_function(cd_function: str, linear: bool = True, alpha: float = 0.1):
    if cd_function == 'pc':
       return lambda data: pc_wrapper(data, linear, alpha=alpha)
    elif cd_function == 'lingam':
       return lambda data: lingam_wrapper(data)
    elif cd_function == 'ges':
       return lambda data : ges.fit_bic(data, phases=['forward', 'backward'])[0]
    else:
       raise ValueError(f"Unknown cd function: {cd_function}")
    

""" LiNGAM wrapper """
def lingam_wrapper(data: np.ndarray):
   model = lingam.DirectLiNGAM()
   model.fit(data)
   graph = model.adjacency_matrix_.T
   graph = np.where(graph != 0, 1, 0)  # Binarize the adjacency matrix
   return graph

    
""" PC wrapper """
def pc_wrapper(data: np.ndarray, linear: bool = True, bk: bool = False, alpha: float = 0.1):
   indep_test = 'fisherz' if linear else 'chisq'
   graph = pc(data, alpha=alpha, indep_test=indep_test, verbose=False)
   graph.to_nx_graph()
   graph = nx.to_numpy_array(graph.nx_graph)
   return graph


""" CI-based edge pruning using Fisher's z-test on observational data """
def prune_by_ci(adj: np.ndarray, obs_data: np.ndarray, ci_alpha: float = 0.05) -> np.ndarray:
    """
    Prune edges from a DAG adjacency using conditional independence tests.
    For each directed edge i->j, test if X_i is conditionally independent of X_j
    given other parents of j. If so, remove the edge.

    Parameters
    ----------
    adj : np.ndarray
        Binary adjacency matrix (n_nodes, n_nodes)
    obs_data : np.ndarray
        Observational data matrix (n_samples, n_nodes)
    ci_alpha : float
        Significance level for Fisher's z-test

    Returns
    -------
    pruned_adj : np.ndarray
        Pruned adjacency matrix
    """
    from scipy.stats import norm
    n_samples, n_nodes = obs_data.shape
    pruned = adj.copy()

    for j in range(n_nodes):
        parents = list(np.where(pruned[:, j] == 1)[0])
        if len(parents) <= 1:
            continue
        for i in parents:
            other_parents = [p for p in parents if p != i]
            # Compute partial correlation: corr(X_i, X_j | X_other_parents)
            # Use the inverse of the covariance matrix (precision matrix)
            idx = [i, j] + other_parents
            cov = np.cov(obs_data[:, idx].T)
            try:
                prec = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                continue
            # Partial correlation = -prec[0,1] / sqrt(prec[0,0] * prec[1,1])
            partial_corr = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
            # Fisher's z-transform
            z_stat = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr + 1e-12))
            z_stat *= np.sqrt(n_samples - len(other_parents) - 3)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            if p_value > ci_alpha:
                pruned[i, j] = 0
    return pruned


def shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance (SHD) between two adjacency matrices.
    Both true_adj and pred_adj are binary adjacency matrices (0/1), shape (n, n).
    """
    # Ignore self-loops
    np.fill_diagonal(true_adj, 0)
    np.fill_diagonal(pred_adj, 0)
    
    # Convert to undirected adjacency for comparison of structure (ignoring orientation)
    true_undirected = ((true_adj + true_adj.T) > 0).astype(int)
    pred_undirected = ((pred_adj + pred_adj.T) > 0).astype(int)
    
    # Structural difference (ignoring orientation)
    undirected_diff = np.sum(np.abs(true_undirected - pred_undirected)) // 2
    
    # Orientation errors (edges present in both undirected, but wrong direction)
    common_edges = np.logical_and(true_undirected, pred_undirected)
    orientation_errors = np.sum(np.abs(true_adj[common_edges==1] - pred_adj[common_edges==1]))
    
    return int(undirected_diff + orientation_errors)


def shd_skeleton(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance (SHD) between the skeletons of two adjacency matrices.
    Both true_adj and pred_adj are binary adjacency matrices (0/1), shape (n, n).
    """
    # Ignore self-loops
    np.fill_diagonal(true_adj, 0)
    np.fill_diagonal(pred_adj, 0)
    
    # Convert to undirected adjacency for comparison of structure (ignoring orientation)
    true_undirected = ((true_adj + true_adj.T) > 0).astype(int)
    pred_undirected = ((pred_adj + pred_adj.T) > 0).astype(int)
    
    # Structural difference (ignoring orientation)
    undirected_diff = np.sum(np.abs(true_undirected - pred_undirected)) // 2
    
    return int(undirected_diff)


def f1_skeleton(true_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Compute skeleton F1 score between two DAG adjacency matrices.
    Only considers edges that exist in either true or predicted graph.
    """
    # Convert to undirected adjacency (ignore direction)
    true_undirected = ((true_adj + true_adj.T) > 0).astype(int)
    pred_undirected = ((pred_adj + pred_adj.T) > 0).astype(int)
    
    # Flatten (ignore diagonal)
    mask = ~np.eye(true_undirected.shape[0], dtype=bool)
    y_true = true_undirected[mask].astype(int)
    y_pred = pred_undirected[mask].astype(int)
    
    return f1_score(y_true, y_pred)


def f1_orientation(true_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Compute orientation F1 score between two DAG adjacency matrices.
    Only considers edges that exist in either true or predicted graph.
    """
    # Flatten (ignore diagonal)
    mask = ~np.eye(true_adj.shape[0], dtype=bool)
    y_true = true_adj[mask].astype(int)
    y_pred = pred_adj[mask].astype(int)
    
    return f1_score(y_true, y_pred)


def set_determine(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


def identity(A: np.ndarray) -> np.ndarray:
    return A


def cpdag_to_ucpdag(dag: np.ndarray, interventions: List[int]) -> np.ndarray:
    """ Convert a CPDAG to a UCPDAG by orienting edges that create unshielded colliders. """
    ucpdag = ges.utils.dag_to_cpdag(dag)
    for intervention in interventions:
        A = dag.copy()
        # remove all parents of intervention node
        parents = np.where(A[:, intervention] == 1)[0]
        for parent in parents:                
            A[parent, intervention] = 0
        cpdag_intervened = ges.utils.dag_to_cpdag(A)
        for i in range(ucpdag.shape[0]):
            for j in range(ucpdag.shape[1]):
                if (ucpdag[i, j] == 1 and ucpdag[j, i] == 1 and 
                    cpdag_intervened[i, j] != cpdag_intervened[j, i]):
                    ucpdag[i, j] = cpdag_intervened[i, j]
                    ucpdag[j, i] = cpdag_intervened[j, i]
    return ucpdag

def union_graph(dag: np.ndarray, graphs: List[np.ndarray]) -> np.ndarray:
    """ Compute the union of multiple graphs represented as adjacency matrices. """
    union_graph = np.zeros_like(dag)
    for graph in graphs:
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if (union_graph[i, j] == 1 and union_graph[j, i] == 1 and 
                    graph[i, j] != graph[j, i]):
                    union_graph[i, j] = graph[i, j]
                    union_graph[j, i] = graph[j, i]
                else:
                    union_graph[i, j] = max(union_graph[i, j], graph[i, j])
                    # union_graph[j, i] = max(union_graph[j, i], graph[j, i])
    return union_graph

def intervened_graph(dag: np.ndarray, interventions: List[int]) -> np.ndarray:
    """ Remove all incoming edges to intervention nodes in the DAG. """
    intervened_graph = dag.copy()
    for intervention in interventions:
        parents = np.where(intervened_graph[:, intervention] == 1)[0]
        for parent in parents:                
            intervened_graph[parent, intervention] = 0
    return intervened_graph


""" function to convert string to boolean for argparse """
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes'):  return True
    if v.lower() in ('0', 'false', 'no'):  return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got "{v}"')

