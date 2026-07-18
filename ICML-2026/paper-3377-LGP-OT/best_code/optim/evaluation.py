'''
Taken from (Zhang et al. 2024): https://github.com/rsinghlab/scNODE
Description:
    Model evaluations.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import torch
from scipy.spatial.distance import cdist
from geomloss import SamplesLoss
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
import scipy
import ot as pot

# =====================================
#                UTILS
# =====================================
def basicStats(data, axis="cell"):
    '''
    Compute mean, variance, and fraction of zeros for each cell or gene.
    '''
    data  = np.asarray(data)
    n_cells, n_genes = data.shape
    if axis == "cell":
        # cell average and var
        expression_avg = np.mean(data, axis=1)
        expression_var = np.var(data, axis=1)
        # fraction of zero
        zero_fraction = np.array([len(np.where(cell==0)[0])/n_genes for cell in data])
    elif axis == "gene":
        # gene average and var
        expression_avg = np.mean(data, axis=0)
        expression_var = np.var(data, axis=0)
        # fraction of zero
        zero_fraction = np.array([len(np.where(data[:,i]==0)[0])/n_cells for i in range(n_genes)])
    else:
        raise ValueError("Undefined axis {}! Should be \"cell\" or \"gene\".".format(axis))
    return expression_avg, expression_var, zero_fraction


# =====================================
#     GLOBAL EVALUATION
# =====================================

def _unbalancedDist(true_data, pred_data):
    '''
    Compute pair-wise distance.

    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    :param true_data (numpy.ndarray): True expression data.
    :param pred_data (numpy.ndarray): Predicted expression data.
    :return:
        (float) Pair-wise L2 distance.
        (float) Pair-wise cosine distance.
        (float) Pair-wise correlation distance.
    '''
    l2_dist = cdist(true_data, pred_data, metric="euclidean")
    cos_dist = cdist(true_data, pred_data, metric="cosine")
    corr_dist = cdist(true_data, pred_data, metric="correlation")
    avg_l2 = l2_dist.sum() / np.prod(l2_dist.shape)
    avg_cos = cos_dist.sum() / np.prod(cos_dist.shape)
    avg_corr = corr_dist.sum() / np.prod(corr_dist.shape)
    return avg_l2, avg_cos, avg_corr


def _ot(true_data, pred_data):
    '''
    Compute Wasserstein distance with Sinkhorn algorithm.
    :param true_data (numpy.ndarray): True expression data.
    :param pred_data (numpy.ndarray): Predicted expression data.
    :return: (float) Wasserstein distance.
    '''
    ot_solver = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
    # ot_solver = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=1.0, debias=True, backend="tensorized")
    if isinstance(true_data, np.ndarray):
        true_data = torch.DoubleTensor(true_data)
    if isinstance(true_data, torch.FloatTensor):
        true_data = torch.DoubleTensor(true_data.detach().numpy())
    if isinstance(pred_data, np.ndarray):
        pred_data = torch.DoubleTensor(pred_data)
    if isinstance(pred_data, torch.FloatTensor):
        pred_data = torch.DoubleTensor(pred_data.detach().numpy())
    ot_loss = ot_solver(true_data, pred_data).item()
    return ot_loss


def globalEvaluation(true_data, pred_data):
    '''Evaluate the difference between true and reconstructed data at a single time point.'''
    assert true_data.shape[1] == pred_data.shape[1]
    l2_dist, cos_dist, corr_dist = _unbalancedDist(true_data, pred_data)
    ot_loss = _ot(true_data, pred_data)
    emd_loss = earth_mover_distance(true_data, pred_data)
    return {
        "l2": l2_dist, "cos": cos_dist, "corr": corr_dist, "ot": ot_loss, "emd": emd_loss
    }


def earth_mover_distance(
    p,
    q,
    eigenvals=None,
    weights1=None,
    weights2=None,
    return_matrix=False,
    metric="sqeuclidean",
):
    """
    Returns the earth mover's distance between two point clouds
    Parameters
    ----------
    cloud1 : 2-D array
        First point cloud
    cloud2 : 2-D array
        Second point cloud
    Returns
    -------
    distance : float
        The distance between the two point clouds
    """
    p = p.toarray() if scipy.sparse.isspmatrix(p) else p
    q = q.toarray() if scipy.sparse.isspmatrix(q) else q
    if eigenvals is not None:
        p = p.dot(eigenvals)
        q = q.dot(eigenvals)
    if weights1 is None:
        p_weights = np.ones(len(p)) / len(p)
    else:
        weights1 = weights1.astype("float64")
        p_weights = weights1 / weights1.sum()

    if weights2 is None:
        q_weights = np.ones(len(q)) / len(q)
    else:
        weights2 = weights2.astype("float64")
        q_weights = weights2 / weights2.sum()

    pairwise_dist = np.ascontiguousarray(
        pairwise_distances(p, Y=q, metric=metric, n_jobs=-1)
    )

    result = pot.emd2(
        p_weights, q_weights, pairwise_dist, numItermax=1e7, return_matrix=return_matrix
    )
    if return_matrix:
        square_emd, log_dict = result
        if metric == "sqeuclidean":
            return np.sqrt(square_emd), log_dict
        elif metric == "euclidean":
            return square_emd, log_dict
    else:
        if metric == "sqeuclidean":
            return np.sqrt(result)
        elif metric == "euclidean":
            return result