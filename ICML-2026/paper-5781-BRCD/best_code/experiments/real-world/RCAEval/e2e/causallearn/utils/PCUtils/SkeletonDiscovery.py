from itertools import combinations

import numpy as np
from tqdm.auto import tqdm
from typing import Optional

from RCAEval.e2e.causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import chisq, gsq
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import chisq, gsq, CIT
from RCAEval.e2e.rcg_core.para_kpc.cit import ArrayCIT


# --- HSIC / RCoT-style CI test (nonparametric) ---
from sklearn.ensemble import RandomForestRegressor

def _rbf_kernel(x: np.ndarray, sigma: float = None) -> np.ndarray:
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    sq = np.sum(x**2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * x @ x.T
    if sigma is None:
        v = np.sqrt(np.maximum(d2[np.triu_indices_from(d2, 1)], 1e-12))
        med = np.median(v) if v.size else 1.0
        sigma = med if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * sigma**2))

def _center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _hsic_stat(x: np.ndarray, y: np.ndarray) -> float:
    K = _center_kernel(_rbf_kernel(x))
    L = _center_kernel(_rbf_kernel(y))
    n = K.shape[0]
    return float(np.sum(K * L) / (n * n))   # V-statistic (fine with permutation)

def _residualize_rf(u: np.ndarray, Z: Optional[np.ndarray]) -> np.ndarray:
    if Z is None or Z.size == 0:
        return u - np.mean(u)
    Z2 = Z if Z.ndim == 2 else Z.reshape(-1, 1)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               random_state=0, n_jobs=-1)
    rf.fit(Z2, u)
    return u - rf.predict(Z2)

def hsic_rcot_pvalue(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray],
                     n_perms: int = 200, random_state: int = 0) -> float:
    """Residualize X,Y on Z; test HSIC(resX,resY) with permutation p-value."""
    rng = np.random.default_rng(random_state)
    x = x.ravel(); y = y.ravel()
    Z = None if (z is None or z.size == 0) else z

    rx = _residualize_rf(x, Z)
    ry = _residualize_rf(y, Z)
    t_obs = _hsic_stat(rx, ry)

    # permutation null
    cnt = 1
    for _ in range(n_perms):
        t_perm = _hsic_stat(rx, rng.permutation(ry))
        if t_perm >= t_obs:
            cnt += 1
    return float(cnt / (n_perms + 1))

# --- Adapter: make any (x,y,z)->p callable look like a CIT object on your data ---


def _is_discrete_method(indep_test):
    # treat both string names and callables
    return (indep_test == 'chisq' or indep_test == chisq or
            indep_test == 'gsq'   or indep_test == gsq)

def _to_cit(data: np.ndarray, indep_test):
    # If user passed a CausalLearn method string (e.g., 'fisherz','kci','chisq',...)
    if isinstance(indep_test, str):
        return CIT(data, method=indep_test)
    # If user passed a known library function that CIT understands
    if indep_test in (chisq, gsq):
        return CIT(data, method=indep_test)
    # Otherwise assume a raw (x,y,z)->p callable (e.g., hsic_rcot_pvalue)
    return ArrayCIT(data, indep_test)

def skeleton_discovery(data, alpha, indep_test, stable=True, background_knowledge=None,
                       labels={}, verbose=False, show_progress=True):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : the function of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    #### UNCOMMENT THIS IF YOU WANT TO FALL BACK TO OLD CODE ####
    # cg.set_ind_test(indep_test)
    # cg.data_hash_key = hash(str(data))
    # if indep_test == chisq or indep_test == gsq:
    #     # if dealing with discrete data, data is numpy.ndarray with n rows m columns,
    #     # for each column, translate the discrete values to int indexs starting from 0,
    #     #   e.g. [45, 45, 6, 7, 6, 7] -> [2, 2, 0, 1, 0, 1]
    #     #        ['apple', 'apple', 'pear', 'peach', 'pear'] -> [0, 0, 2, 1, 2]
    #     # in old code, its presumed that discrete `data` is already indexed,
    #     # but here we make sure it's in indexed form, so allow more user input e.g. 'apple' ..
    #     def _unique(column):
    #         return np.unique(column, return_inverse=True)[1]

    #     cg.is_discrete = True
    #     cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
    #     cg.cardinalities = np.max(cg.data, axis=0) + 1
    # else:
    #     cg.data = data
    ############ New code for hsic_rcot_pvalue ############
    # Prepare data encoding for discrete-only tests
    if _is_discrete_method(indep_test):
        def _unique(col):
            return np.unique(col, return_inverse=True)[1]
        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    # Uniformly set a CIT-like callable regardless of what indep_test was
    cit_obj = _to_cit(cg.data, indep_test)
    cg.set_ind_test(cit_obj)
    cg.data_hash_key = hash(str(cg.data))
    #########################################################


    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress: pbar.reset()
        for x in range(no_of_var):
            if show_progress: pbar.update()
            if show_progress: pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        append_value(cg.p_values, x, y, p)
                        if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress: pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress: pbar.close()

    return cg

def local_skeleton_discovery(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False):
    assert type(data) == np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    #cg.set_ind_test(indep_test)
    #### UNCOMMENT THIS IF YOU WANT TO FALL BACK TO OLD CODE ####
    # cg.data_hash_key = hash(str(data))
    # if indep_test == chisq or indep_test == gsq:
    #     def _unique(column):
    #         return np.unique(column, return_inverse=True)[1]

    #     cg.is_discrete = True
    #     cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
    #     cg.cardinalities = np.max(cg.data, axis=0) + 1
    # else:
    #     cg.data = data

    # indep_test = CIT(data, method=indep_test)
    # cg.set_ind_test(indep_test)
    #########################################################
    #### NEW CODE FOR hsic_rcot_pvalue ####
    cg.data_hash_key = hash(str(data))

    if _is_discrete_method(indep_test):
        def _unique(col): return np.unique(col, return_inverse=True)[1]
        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    cit_obj = _to_cit(cg.data, indep_test)   # works for strings or callables
    cg.set_ind_test(cit_obj)
    #########################################################
    depth = -1
    x = local_node
    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    while cg.max_degree() - 1 > depth:
        depth += 1

        local_neigh = np.random.permutation(cg.neighbors(x))
        # local_neigh = cg.neighbors(x)
        for y in local_neigh:
            Neigh_y = cg.neighbors(y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))
            Neigh_y_f = []
            if depth > 0:
                Neigh_y_f = [s for s in Neigh_y if x in cg.neighbors(s)]
                # Neigh_y_f += mi

            for S in combinations(Neigh_y_f, depth):
                p = cg.ci_test(x, y, S)
                if p > alpha:
                    if verbose: print(f'{cg.labels[x]} ind {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print(f'{cg.labels[x]} dep {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')

    return cg