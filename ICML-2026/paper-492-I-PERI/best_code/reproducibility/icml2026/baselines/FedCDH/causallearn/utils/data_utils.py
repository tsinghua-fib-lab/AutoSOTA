"""
Code modified from:
https://github.com/xunzheng/notears/blob/master/notears/utils.py
"""
import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=d)#m=d, p=0.3 p=0.5
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=d, directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    # print(S)
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        # print(S == i)
        W += B * (S == i) * U
    return W

# heterogeneous: gaussian mixture model / general cases
def my_simulate_general_hetero(W, K, n, sem_type):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] unweighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """   
    def f(x):
        det = np.random.randint(4) 
        if det == 0:
            y = np.sin(x)
        elif det == 1:
            y = x**2
        elif det == 2:
            y = np.tanh(x)
        elif det == 3:
            y = x
        return y 

    d = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])

    choice = np.random.choice(d, 2, replace=False) 
    # choice = [1]
    print(f"In general cases. The changing variable is: {choice}. The order is: {ordered_vertices}.")

    c_indx = np.asarray(list(range(K)))
    c_indx = np.repeat(c_indx, int(n/K)) 
    # c_indx = np.reshape(c_indx, (n,1)) 

    b = np.random.uniform(low=0.5, high=2.5, size=(d,d))
    b = b * W
    # print("I am here test 101.")
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if j in choice: # changes
            sigma = np.random.uniform(low=1, high=3, size=K)
            sigma = np.repeat(sigma, int(n/K))

            # b_t = np.random.uniform(low=0.5, high=2, size=K)
            # b_t = np.repeat(b_t, int(n/K))
            # print(f"sigma shape: {sigma.shape}. bt shape: {b_t.shape}.")

            assert len(sigma) == n
            if np.random.rand() > 0.5:
                eps = np.random.uniform(low=-0.5, high=0.5, size=n)
            else:
                eps = np.random.normal(0, 1, size=n)

            # # eps1 = np.random.uniform(-0.5, 0.5, size=int(n/2))
            # eps1 = np.random.normal(-1, 1, size=int(n/2))
            # eps2 = np.random.normal(1, 1, size=int(n/2))
            # eps = np.concatenate((eps1,eps2))
            # random.shuffle(eps)

            sample = int(n/K)
            for k in range(K):
                b_ = np.random.uniform(low=0.5, high=2.5, size=(d,d))
                b_ = b_ * W
                X[k*sample:(k+1)*sample,j] = f(X[k*sample:(k+1)*sample,parents]) @ b_[parents,j]
            X[:,j] = X[:,j] + sigma * eps 

        else: # not change
            if np.random.rand() > 0.5:
                eps = np.random.uniform(low=-0.5, high=0.5, size=n)
            else:
                eps = np.random.normal(0, 1, size=n)

            # # eps1 = np.random.uniform(-0.5, 0.5, size=int(n/2))
            # eps1 = np.random.normal(-1, 1, size=int(n/2))
            # eps2 = np.random.normal(1, 1, size=int(n/2))
            # eps = np.concatenate((eps1,eps2))
            # random.shuffle(eps)
            X[:,j] = f(X[:,parents]) @ b[parents,j] + eps
        
    return X, choice 


# heterogeneous: linear Gaussian model
def my_simulate_linear_gaussian(W, K, n, sem_type):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] unweighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """   

    d = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])

    choice = np.random.choice(d, 2, replace=False) 
    # choice = [1]
    print(f"This is Linear Gaussian Model. The changing variable is: {choice}. The order is: {ordered_vertices}.")

    # c_indx = np.asarray(list(range(K)))
    # c_indx = np.repeat(c_indx, int(n/K)) 
    # c_indx = np.reshape(c_indx, (n,1)) 

    b = np.random.uniform(low=0.5, high=2.5, size=(d,d))
    b = b * W
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if j in choice: # changes
            sigma = np.random.uniform(low=1, high=3, size=K)
            sigma = np.repeat(sigma, int(n/K))

            assert len(sigma) == n
            # eps1 = np.random.normal(-1, 1, size=int(n/2))
            # eps2 = np.random.normal(1, 1, size=int(n/2))
            # eps = np.concatenate((eps1,eps2))
            # random.shuffle(eps)

            # var = 0.1 * np.random.randint(5,10)
            var = np.random.uniform(low=1.0, high=2.0)
            eps = np.random.normal(0, var, size=n)

            sample = int(n/K)
            for k in range(K):
                b_ = np.random.uniform(low=0.5, high=2.5, size=(d,d))
                b_ = b_ * W
                X[k*sample:(k+1)*sample,j] = X[k*sample:(k+1)*sample,parents] @ b_[parents,j]
            X[:,j] = X[:,j] + sigma * eps 


        else: # not change
            # eps1 = np.random.normal(-1, 1, size=int(n/2))
            # eps2 = np.random.normal(1, 1, size=int(n/2))
            # eps = np.concatenate((eps1,eps2))
            # random.shuffle(eps)

            # var = 0.1 * np.random.randint(5,10)
            var = np.random.uniform(low=1.0, high=2.0)
            eps = np.random.normal(0, var, size=n)
        
            X[:,j] = X[:,parents] @ b[parents,j] + eps
        
    return X, choice 


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n) # loc=mean, scale=std.
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        # print(f"Parent: {parents+np.ones(len(parents))}. child:{j+1}.")
    return X

# non-stationary
def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        # if np.random.rand() > 0.5:
        #     z = np.random.uniform(low=-0.5, high=0.5, size=n)
        # else:
        #     z = np.random.normal(scale=1, size=n)

        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    choice = np.random.choice(d, int(d*0.2), replace=False) 
    
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T)) # B_est == binary
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T)) # B_true == calculate. 
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

def transfer_to_bin_skeleton(g, B_true):
    # a,b = g.shape
    c,d = B_true.shape 
    # a,b = a+1, b+1
    bin_g = np.zeros((c,d))
    for i in range(c):
        for j in range(d):
            if g[j,i]==1 and g[i,j]==-1:
                bin_g[i,j] = 1
            elif g[j,i]==-1 and g[i,j]==1:
                bin_g[j,i] = 1
            elif g[j,i]==-1 and g[i,j]==-1:
                bin_g[j,i] = -1
            elif g[j,i]==1 and g[i,j]==1:
                bin_g[j,i] = -1   
    return bin_g

def tranfer_to_bin_orientation(g):
    a,b = g.shape
    bin_g = np.zeros((a-1,b-1))
    for i in range(a-1):
        for j in range(b-1):
            if g[j,i]==1 and g[i,j]==-1:
                bin_g[i,j] = 1
            elif g[j,i]==-1 and g[i,j]==1:
                bin_g[j,i] = 1
            # elif g[j,i]==-1 and g[i,j]==-1:
            #     bin_g[j,i] = 1
            # elif g[j,i]==1 and g[i,j]==1:
            #     bin_g[j,i] = 1   
    return bin_g

def my_count_accuracy(graph, B_true):
    B_est = transfer_to_bin_skeleton(graph, B_true)
    # SHD: structural hamming distance. For skeleton. 
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T)) # tril: lower triangle. 
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T)) 
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True) # value in arr1, not in arr2. 
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower)  
    # print(extra_lower, missing_lower)
    # print(B_true)
    
    # for orientation-arrow accuracy.
    # B_est = tranfer_to_bin_orientation(graph)
    # TPR: true positive rate
    pred_1 = np.flatnonzero(B_est==1)
    true = np.flatnonzero(B_true)
    true_pos = np.intersect1d(pred_1, true, assume_unique=True)
    tpr = float(len(true_pos)) / max(len(true), 1)
    
    cond_reversed = np.flatnonzero(B_true.T)
    reverse = np.intersect1d(true_pos, cond_reversed, assume_unique=True)

    
    pred_2 = np.flatnonzero(B_est==-1)
    pred = np.concatenate([pred_1, pred_2])

    # FDR: false discovery rate 
    false_pos = np.setdiff1d(pred, true, assume_unique=True) # false positve includes no directions, wrong directions or reverse directions.
    fdr = float(len(false_pos)) / max(len(pred), 1)

    # FPR: false positive rate 
    d = B_true.shape[0]
    cond_neg_size = 0.5 * d * (d - 1) - len(true)
    gt_missing = d*d - len(true)
    false_pos = np.setdiff1d(pred, true, assume_unique=True)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)
    
    fpr_2 = float(len(false_pos)) / max(gt_missing, 1)

    return {'shd': shd, 'tpr': tpr, 'fdr':fdr, 'fpr':fpr, 'fpr2':fpr_2}, [shd,tpr,fdr,fpr,fpr_2]












# -------------------------- New Metrics ------------------------------

"""
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.
    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
"""
def get_skeleton(B):
    B_bin = (B != 0).astype(int)
    return ((B_bin + B_bin.T) != 0).astype(int)

def count_skeleton_accuracy(B_bin_true, B_bin_est):
    skeleton_true = get_skeleton(B_bin_true) # b_bin_true[i,j]=1  <==> skeleton[i,j]=skeleton[j,i]=1
    skeleton_est = get_skeleton(B_bin_est)   # b_bin_est[i,j]=-1 & b_bin_est[j,i]=1  <==>  skeleton[i,j]=skeleton[j,i]=1

    # print(3, skeleton_true)
    # print(4, skeleton_est) 

    d = len(skeleton_true)
    skeleton_triu_true = skeleton_true[np.triu_indices(d, k=1)]
    skeleton_triu_est = skeleton_est[np.triu_indices(d, k=1)]
    pred = np.flatnonzero(skeleton_triu_est)  # estimated graph
    cond = np.flatnonzero(skeleton_triu_true) # true graph 

    # true pos: an edge estimated with correct direction.
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos: an edge that is in estimated graph but not in the true graph.
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # false neg: an edge that is not in estimated graph but in the true graph.
    false_neg = np.setdiff1d(cond, pred, assume_unique=True) # This is also OK: np.setdiff1d(cond, true_pos, assume_unique=True)
    # true negative: an edge that is neither in estimated graph nor in true graph.
    # true negative: normally equals 0.

    # compute ratio
    nnz = len(pred)
    cond_neg_size = len(skeleton_triu_true) - len(cond)
    fdr = float(len(false_pos)) / max(nnz, 1)  # fdr = (FP) / (TP + FP) = FP / |pred_graph|
    tpr = float(len(true_pos)) / max(len(cond), 1)  # tpr: TP / (TP + FN) = TP / |true_graph|
    fpr = float(len(false_pos)) / max(cond_neg_size, 1) # fpr: (FP) / (TN + FP) = FP / ||
    try:
        f1 = len(true_pos) / (len(true_pos) + 0.5 * (len(false_pos) + len(false_neg)))
    except:
        f1 = None

    # structural hamming distance
    extra_lower = np.setdiff1d(pred, cond, assume_unique=True)
    missing_lower = np.setdiff1d(cond, pred, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower)
    return {'f1_skeleton': f1, 'precision_skeleton': 1 - fdr, 'recall_skeleton': tpr, 'shd_skeleton': shd}
    # return {'f1_skeleton': f1, 'precision_skeleton': 1 - fdr, 'recall_skeleton': tpr,
            # 'shd_skeleton': shd, 'TPR_skeleton': tpr, 'FDR_skeleton': fdr, "number_edge_pred":len(pred), "number_edge_true":len(cond)}


def get_cpdag_from_cdnod(g):
    a,b = g.shape
    cpdag = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            if g[j,i]==1 and g[i,j]==-1:
                cpdag[i,j] = 1
            elif g[j,i]==-1 and g[i,j]==1:
                cpdag[j,i] = 1
            elif g[j,i]==-1 and g[i,j]==-1:
                cpdag[j,i] = 1
                cpdag[i,j] = 1
            # elif g[j,i]==1 and g[i,j]==1:
            #     bin_g[j,i] = 1   
    return cpdag

def get_dag_from_cdnod(g):
    a,b = g.shape
    cpdag = np.zeros((a,b))
    for i in range(a):
        for j in range(i,b):
            if g[j,i]==1 and g[i,j]==-1:
                cpdag[i,j] = 1
            elif g[j,i]==-1 and g[i,j]==1:
                cpdag[j,i] = 1
            elif g[j,i]==-1 and g[i,j]==-1:
                nb = np.random.rand()
                # print(f"the number is: {nb}")
                if nb<=0.5:
                    cpdag[j,i] = 1
                else:
                    cpdag[i,j] = 1
            # elif g[j,i]==1 and g[i,j]==1:
            #     bin_g[j,i] = 1   
    return cpdag

import causaldag as cd
def get_dag_from_pdag(B_bin_pdag):
    # There is bug for G.to_dag().to_amat() from cd package
    # i.e., the shape of B is not preserved
    # So we need to manually preserve the shape
    B_bin_dag = np.zeros_like(B_bin_pdag)
    if np.all(B_bin_pdag == 0):
        # All entries in B_pdag are zeros
        return B_bin_dag
    else:
        G = cd.PDAG.from_amat(B_bin_pdag)  # return a PDAG with arcs/edges.
        # print(G.to_amat()[0])
        B_bin_sub_dag, nodes = G.to_dag().to_amat() # The key is: to_dag() - converting a PDAG to a DAG using some rules. 
        # print("G:", G.to_dag().to_amat())
        B_bin_dag[np.ix_(nodes, nodes)] = B_bin_sub_dag
        return B_bin_dag



def count_arrows_accuracy(B_bin_true, B_bin_est):
    dag_est = cd.DAG.from_amat(B_bin_est)
    dag_true = cd.DAG.from_amat(B_bin_true)

    # print(1, dag_est)
    # print(2, dag_true)
    # return 0

    cm_cpdag = dag_est.confusion_matrix(dag_true)
    tp_arrows = len(cm_cpdag['true_positive_arcs'])
    fp_arrows = len(cm_cpdag['false_positive_arcs'])
    fn_arrows = len(cm_cpdag['false_negative_arcs'])

    # print(tp_arrows)

    precision_arrows, recall_arows, f1_arrows \
        = count_precision_recall_f1(tp_arrows, fp_arrows, fn_arrows)
    return {'f1_arrows': f1_arrows, 'precision_arrows': precision_arrows, 'recall_arows': recall_arows}


def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


import networkx as nx
def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def get_cpdag(B):
    assert is_dag(B)
    return cd.DAG.from_amat(B).cpdag().to_amat()[0]

def compute_shd_cpdag(B_bin_true, B_bin_est):
    assert is_dag(B_bin_true)
    assert is_dag(B_bin_est)
    cpdag_true = get_cpdag(B_bin_true)
    cpdag_est = get_cpdag(B_bin_est)
    return cd.PDAG.from_amat(cpdag_true).shd(cd.PDAG.from_amat(cpdag_est))


def count_dag_accuracy(B_bin_true, B_bin_est):
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    # return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
    #         'precision': precision, 'recall': recall, 'f1': f1}
    return {'f1': f1,  'precision': precision, 'recall': recall, 'shd': shd}


def count_dag_accuracy_2(B_bin_true, B_bin_est):
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    # return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
    #         'precision': precision, 'recall': recall, 'f1': f1}
    return {'f1_rd': f1,  'precision_rd': precision, 'recall_rd': recall, 'shd_rd': shd}
