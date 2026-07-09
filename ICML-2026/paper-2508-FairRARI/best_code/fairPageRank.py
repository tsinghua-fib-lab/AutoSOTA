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
            return lam_mid, g_mid
        elif g_mid > 0:
            lam_min = lam_mid
        else:
            lam_max = lam_mid

    return (lam_min + lam_max) / 2, g_mid

def projection_sum_fair_simplex(x, S_p, S_up, phi):

    phi_p = phi
    phi_up = 1-phi

    x_S_p = x[S_p==1]
    x_S_up = x[S_up==1]

    lam_star_p, g_mid_p = bisection_sum_fair(x_S_p, phi_p, tol=1e-6)
    lam_star_up, g_mid_up = bisection_sum_fair(x_S_up, phi_up, tol=1e-6)
    
    x[S_p==1] = np.maximum(0, x[S_p==1]-lam_star_p)
    x[S_up==1] = np.maximum(0, x[S_up==1]-lam_star_up)

    return x
###

### Euclidean Projection for Min Fair PR
def g_min_fair(x_A, x_notA, A, notA, alph_, lam):
    
    return np.sum(np.maximum(alph_,x_A-lam)) + np.sum(np.maximum(0,x_notA-lam)) - 1

def bisection_min_fair(x_A, x_notA, A, notA, alph_, tol, lam_min=-1, lam_max=1):

    while (lam_max - lam_min) / 2 > tol:
        lam_mid = (lam_min+lam_max)/2
        g_mid = g_min_fair(x_A, x_notA, A, notA, alph_, lam_mid)
        if g_mid == 0:
            return lam_mid, g_mid
        elif g_mid > 0:
            lam_min = lam_mid
        else:
            lam_max = lam_mid

    return (lam_min + lam_max) / 2

def projection_min_fair_simplex(x, A, notA, alph_):

    x_A = x[A==1]
    x_notA = x[notA==1]

    lam_star = bisection_min_fair(x_A, x_notA, A, notA, alph_, tol=1e-6)
    
    x[A==1] = np.maximum(alph_, x[A==1]-lam_star)
    x[notA==1] = np.maximum(0, x[notA==1]-lam_star)

    return x
###

### Euclidean Projection for Sum + Min Fair PR
def g_sum_min_fair(x_S_k_A, x_S_k_notA, phi_k, alpha, lam_k):
    
    return np.sum(np.maximum(alpha,x_S_k_A-lam_k)) + np.sum(np.maximum(0,x_S_k_notA-lam_k)) - phi_k

def bisection_sum_min_fair(x_S_k_A, x_S_k_notA, phi_k, alpha, tol, lam_min=-10, lam_max=10):

    while (lam_max - lam_min) / 2 > tol:
        lam_mid = (lam_min+lam_max)/2
        g_mid = g_sum_min_fair(x_S_k_A, x_S_k_notA, phi_k, alpha, lam_mid)
        if g_mid == 0:
            return lam_mid, g_mid
        elif g_mid > 0:
            lam_min = lam_mid
        else:
            lam_max = lam_mid

    return (lam_min + lam_max) / 2, g_mid

def projection_sum_min_fair_simplex(x, S_p, S_up, A, notA, phi, alpha, tol_bis):

    S_p_A = (S_p == 1) & (A == 1)
    S_p_notA = (S_p == 1) & (notA == 1)
    S_up_A = (S_up == 1) & (A == 1)
    S_up_notA = (S_up == 1) & (notA == 1)

    phi_p = phi
    phi_up = 1-phi

    x_S_p_A = x[S_p_A]
    x_S_p_notA = x[S_p_notA]
    x_S_up_A = x[S_up_A]
    x_S_up_notA = x[S_up_notA]

    lam_star_p, g_mid_p = bisection_sum_min_fair(x_S_p_A, x_S_p_notA, phi_p, alpha, tol=tol_bis)
    lam_star_up, g_mid_up = bisection_sum_min_fair(x_S_up_A, x_S_up_notA, phi_up, alpha, tol=tol_bis)
    
    x[S_p_A] = np.maximum(alpha, x[S_p_A]-lam_star_p)
    x[S_p_notA] = np.maximum(0, x[S_p_notA]-lam_star_p)
    x[S_up_A] = np.maximum(alpha, x[S_up_A]-lam_star_up)
    x[S_up_notA] = np.maximum(0, x[S_up_notA]-lam_star_up)

    return x
###

### FairRARI ###
    
def sum_fair_FairRARI(G, S_p, S_up, phi, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None, edge_reweight_factor=None):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Edge reweighting by group membership (ALGO-04)
    if edge_reweight_factor is not None and edge_reweight_factor > 1.0:
        w = edge_reweight_factor
        A = A.tolil()
        S_p_np = S_p.numpy()
        for u, v in G.edges():
            ui = nodelist.index(u)
            vi = nodelist.index(v)
            if S_p_np[ui] == S_p_np[vi]:  # same group -> up-weight
                A[ui, vi] *= w
                if not G.is_directed():
                    A[vi, ui] *= w
            else:  # cross-group -> down-weight
                A[ui, vi] /= w
                if not G.is_directed():
                    A[vi, ui] /= w
        A = A.tocsr()

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

            x = projection_sum_fair_simplex(x, S_p, S_up, phi)

            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            err_.append(err)
            if err < N * tol:
                return dict(zip(nodelist, map(float, x))), err_, loss
    # raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(nodelist, map(float, x))), err_, loss


def min_fair_FairRARI(G, A_cal, notA_cal, alph_, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Edge reweighting by group membership (ALGO-04)
    if edge_reweight_factor is not None and edge_reweight_factor > 1.0:
        w = edge_reweight_factor
        A = A.tolil()
        S_p_np = S_p.numpy()
        for u, v in G.edges():
            ui = nodelist.index(u)
            vi = nodelist.index(v)
            if S_p_np[ui] == S_p_np[vi]:  # same group -> up-weight
                A[ui, vi] *= w
                if not G.is_directed():
                    A[vi, ui] *= w
            else:  # cross-group -> down-weight
                A[ui, vi] /= w
                if not G.is_directed():
                    A[vi, ui] /= w
        A = A.tocsr()

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

            x = projection_min_fair_simplex(x, A_cal, notA_cal, alph_)

            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            err_.append(err)
            if err < N * tol:
                return dict(zip(nodelist, map(float, x))), err_, loss
    # raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(nodelist, map(float, x))), err_, loss

def sum_min_fair_FairRARI(G, S_p, S_up, A_cal, notA_cal, phi, alph_, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None, tol_bis=1e-6):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Edge reweighting by group membership (ALGO-04)
    if edge_reweight_factor is not None and edge_reweight_factor > 1.0:
        w = edge_reweight_factor
        A = A.tolil()
        S_p_np = S_p.numpy()
        for u, v in G.edges():
            ui = nodelist.index(u)
            vi = nodelist.index(v)
            if S_p_np[ui] == S_p_np[vi]:  # same group -> up-weight
                A[ui, vi] *= w
                if not G.is_directed():
                    A[vi, ui] *= w
            else:  # cross-group -> down-weight
                A[ui, vi] /= w
                if not G.is_directed():
                    A[vi, ui] /= w
        A = A.tocsr()

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

            x = projection_sum_min_fair_simplex(x, S_p, S_up, A_cal, notA_cal, phi, alph_, tol_bis)

            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            err_.append(err)
            if err < N * tol:
                return dict(zip(nodelist, map(float, x))), err_, loss
    # raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(nodelist, map(float, x))), err_, loss


### Post-Processing ###


def min_fair_post_processing(G, A_cal, notA_cal, alph_, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Edge reweighting by group membership (ALGO-04)
    if edge_reweight_factor is not None and edge_reweight_factor > 1.0:
        w = edge_reweight_factor
        A = A.tolil()
        S_p_np = S_p.numpy()
        for u, v in G.edges():
            ui = nodelist.index(u)
            vi = nodelist.index(v)
            if S_p_np[ui] == S_p_np[vi]:  # same group -> up-weight
                A[ui, vi] *= w
                if not G.is_directed():
                    A[vi, ui] *= w
            else:  # cross-group -> down-weight
                A[ui, vi] /= w
                if not G.is_directed():
                    A[vi, ui] /= w
        A = A.tocsr()

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
            if err < N * tol:
                return dict(zip(nodelist, map(float, x))), err_, loss
    # raise nx.PowerIterationFailedConvergence(max_iter)
    
    x = projection_min_fair_simplex(x, A_cal, notA_cal, alph_)

    return dict(zip(nodelist, map(float, x))), err_, loss


def sum_fair_post_processing(G, S_p, S_up, phi, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Edge reweighting by group membership (ALGO-04)
    if edge_reweight_factor is not None and edge_reweight_factor > 1.0:
        w = edge_reweight_factor
        A = A.tolil()
        S_p_np = S_p.numpy()
        for u, v in G.edges():
            ui = nodelist.index(u)
            vi = nodelist.index(v)
            if S_p_np[ui] == S_p_np[vi]:  # same group -> up-weight
                A[ui, vi] *= w
                if not G.is_directed():
                    A[vi, ui] *= w
            else:  # cross-group -> down-weight
                A[ui, vi] /= w
                if not G.is_directed():
                    A[vi, ui] /= w
        A = A.tocsr()

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
            if err < N * tol:
                return dict(zip(nodelist, map(float, x))), err_, loss
    # raise nx.PowerIterationFailedConvergence(max_iter)
    x = projection_sum_fair_simplex(x, S_p, S_up, phi)
    return dict(zip(nodelist, map(float, x))), err_, loss

def sum_min_fair_post_processing(G, S_p, S_up, A_cal, notA_cal, phi, alph_, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight", dangling=None, tol_bis=1e-6):
    loss = []
    err_ = []
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Edge reweighting by group membership (ALGO-04)
    if edge_reweight_factor is not None and edge_reweight_factor > 1.0:
        w = edge_reweight_factor
        A = A.tolil()
        S_p_np = S_p.numpy()
        for u, v in G.edges():
            ui = nodelist.index(u)
            vi = nodelist.index(v)
            if S_p_np[ui] == S_p_np[vi]:  # same group -> up-weight
                A[ui, vi] *= w
                if not G.is_directed():
                    A[vi, ui] *= w
            else:  # cross-group -> down-weight
                A[ui, vi] /= w
                if not G.is_directed():
                    A[vi, ui] /= w
        A = A.tocsr()

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
            if err < N * tol:
                return dict(zip(nodelist, map(float, x))), err_, loss
    # raise nx.PowerIterationFailedConvergence(max_iter)
    x = projection_sum_min_fair_simplex(x, S_p, S_up, A_cal, notA_cal, phi, alph_, tol_bis)
    return dict(zip(nodelist, map(float, x))), err_, loss