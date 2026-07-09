import numpy as np
import networkx as nx
import scipy as sp
from tqdm import trange


### Euclidean Projection for Sum Fair PR
def g_sum_fair(x_S_k, phi_k, lam_k):
    
    return np.sum(np.maximum(0,x_S_k-lam_k)) - phi_k

def bisection_sum_fair(x_S_k, phi_k, tol, lam_min=-10, lam_max=10):

    while (lam_max - lam_min) / 2 > tol:
        lam_mid = (lam_min+lam_max)/2
        g_mid = g_sum_fair(x_S_k, phi_k, lam_mid)
        if g_mid == 0:
            return lam_mid
        elif g_mid > 0:
            lam_min = lam_mid
        else:
            lam_max = lam_mid

    return (lam_min + lam_max) / 2, g_mid

def projection_sum_fair_simplex(x, S_0, S_1, S_2, S_3, phi):

    phi_0 = phi[0]
    phi_1 = phi[1]
    phi_2 = phi[2]
    phi_3 = phi[3]

    x_S_0 = x[S_0==1]
    x_S_1 = x[S_1==1]
    x_S_2 = x[S_2==1]
    x_S_3 = x[S_3==1]

    lam_star_0, g_mid_0 = bisection_sum_fair(x_S_0, phi[0], tol=1e-6)
    lam_star_1, g_mid_1 = bisection_sum_fair(x_S_1, phi[1], tol=1e-6)
    lam_star_2, g_mid_2 = bisection_sum_fair(x_S_2, phi[2], tol=1e-6)
    lam_star_3, g_mid_3 = bisection_sum_fair(x_S_3, phi[3], tol=1e-6)
    
    x[S_0==1] = np.maximum(0, x[S_0==1]-lam_star_0)
    x[S_1==1] = np.maximum(0, x[S_1==1]-lam_star_1)
    x[S_2==1] = np.maximum(0, x[S_2==1]-lam_star_2)
    x[S_3==1] = np.maximum(0, x[S_3==1]-lam_star_3)

    return x
###
    
def sum_fair_FairRARI(G, S_0, S_1, S_2, S_3, phi, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]
    
    # Iteratively update the PageRank scores
    with trange(0, max_iter) as tqdm:
        for _ in tqdm:
            xlast = x
            x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p

            x = projection_sum_fair_simplex(x, S_0, S_1, S_2, S_3, phi)

            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            err_.append(err)
            # if err < N * tol:
            #     return dict(zip(nodelist, map(float, x))), err_
    # raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(nodelist, map(float, x))), err_, loss


def sum_fair_post_processing(G, S_0, S_1, S_2, S_3, phi, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]
    
    # Iteratively update the PageRank scores
    with trange(0, max_iter) as tqdm:
        for _ in tqdm:
            xlast = x
            x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p

            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            err_.append(err)
            # if err < N * tol:
            #     return dict(zip(nodelist, map(float, x))), err_
    # raise nx.PowerIterationFailedConvergence(max_iter)
    x = projection_sum_fair_simplex(x, S_0, S_1, S_2, S_3, phi)
    return dict(zip(nodelist, map(float, x))), err_, loss