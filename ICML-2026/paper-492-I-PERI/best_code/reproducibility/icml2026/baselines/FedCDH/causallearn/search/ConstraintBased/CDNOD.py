import time
from itertools import permutations, combinations
from typing import Dict, List, Optional

import networkx as nx
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.utils.cit import *
from causallearn.search.ConstraintBased.PC import get_parent_missingness_pairs, skeleton_correction

from copy import deepcopy
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

def my_cov(X, Y:np.ndarray=None):
    if Y is not None:  
        X = X - X.mean(axis=0) #(n,h)
        Y = Y - Y.mean(axis=0) #(n,h)
        cov = X.T @ Y  
    else:
        X = X - X.mean(axis=0) #(n,h)
        cov = X.T @ X
    factor = len(X)
    return cov / factor 

from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import RBFSampler
from mlxtend.preprocessing import standardize

def get_hsic_score(X, Y, C):
    h = 5
    n, _ = X.shape # (n,1)

    feature_map_nystroem = Nystroem(gamma=0.2, n_components=h, random_state=1)  
    X_f = feature_map_nystroem.fit_transform(X) #(n,h)
    Y_f = feature_map_nystroem.fit_transform(Y)
    C_f = feature_map_nystroem.fit_transform(C)
    XY = np.concatenate((X,Y), axis=1) #(n,2)
    XY_f = feature_map_nystroem.fit_transform(XY) #(n,h)

    # rbf_feature = RBFSampler(gamma=1.0, n_components=100, random_state=1)
    # X_f = rbf_feature.fit_transform(X) #(n,h)
    # Y_f = rbf_feature.fit_transform(Y)
    # C_f = rbf_feature.fit_transform(C)
    # XY = np.concatenate((X,Y), axis=1) #(n,2)
    # XY_f = rbf_feature.fit_transform(XY) #(n,h)
    
    # H = np.eye(n) - 1.0/n
    # X_f = standardize(X_f) #(n,h)
    # Y_f = standardize(Y_f)
    # C_f = standardize(C_f)
    # XY_f = standardize(XY_f) 

    Cxyc = my_cov(XY_f, C_f) #(n,h)X(n,h) -> (h,h)
    Cxc = my_cov(X_f, C_f) 
    Cyc = my_cov(Y_f, C_f)
    Ccc = my_cov(C_f, C_f)

    iCcc = np.linalg.inv(Ccc + np.eye(h)*1e-10)
    Mu_xy = Cxyc @ iCcc @ C_f.T #(h,n)
    Mu_x = Cxc @ iCcc @ C_f.T
    Mu_y = Cyc @ iCcc @ C_f.T

    hsic_x_y = np.sum( my_cov(Mu_x, Mu_xy)**2 ) / np.trace( my_cov(Mu_x) )
    hsic_y_x = np.sum( my_cov(Mu_y, Mu_xy)**2 ) / np.trace( my_cov(Mu_y) )
    # print(hsic_x_y, hsic_y_x)

    flag = 0
    if hsic_x_y < hsic_y_x:
        flag = 1
        # print("The direction is: x->y.")
    else:
        flag = 2
        # print("The direction is: y->x.")
    return flag


    
def cdnod(data: ndarray, c_indx: ndarray, K: int=1, alpha: float=0.05, indep_test: str=fisherz, stable: bool=True,
          uc_rule: int=0, uc_priority: int=2, mvcdnod: bool=False, correction_name: str='MV_Crtn_Fisher_Z',
          background_knowledge: Optional[BackgroundKnowledge]=None, verbose: bool=False,
          show_progress: bool = True, **kwargs) -> CausalGraph:
    """
    Causal discovery from nonstationary/heterogeneous data
    phase 1: learning causal skeleton,
    phase 2: identifying causal directions with generalization of invariance, V-structure. Meek rule
    phase 3: identifying directions with independent change principle, and (TODO: under development)
    phase 4: recovering the nonstationarity driving force (TODO: under development)

    Parameters
    ----------
     c_indx: time index or domain index that captures the unobserved changing factors

    Returns
    -------
    cg : a CausalGraph object over the augmented dataset that includes c_indx
    """
    # augment the variable set by involving c_indx to capture the distribution shift
    # data_aug = np.concatenate((data, c_indx), axis=1)
    # data_aug = data 
    if mvcdnod:
        return mvcdnod_alg(data=data, alpha=alpha, indep_test=indep_test, correction_name=correction_name,
                           stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, verbose=verbose,
                           show_progress=show_progress, **kwargs)
    else:
        return cdnod_alg(data=data, c_indx=c_indx, alpha=alpha, K=K, indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                         uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                         show_progress=show_progress, **kwargs)


def cdnod_alg(data: ndarray, c_indx: ndarray, alpha: float, K: int, indep_test: str, stable: bool, uc_rule: int, uc_priority: int,
              background_knowledge: Optional[BackgroundKnowledge] = None, verbose: bool = False,
              show_progress: bool = True, **kwargs) -> CausalGraph:
    """
    Perform Peter-Clark algorithm for causal discovery on the augmented data set that captures the unobserved changing factors

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    alpha : desired significance level (float) in (0, 1)
    indep_test : name of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
           - "MV_Fisher_Z": Missing-value Fishers'Z conditional independence test
           - "kci": kernel-based conditional independence test (If C is time index, KCI test is recommended)
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()

    data_aug = np.concatenate((data, c_indx), axis=1)
    indep_test_all = CIT(data_aug, 'kci', **kwargs)
    
    s_a, d = data_aug.shape 
    fed_data = data_aug.reshape(K, int(s_a/K), d) 
    cg_list = []  
    for i in range(K):
        fed_dt = fed_data[i]
        fed_cg = CausalGraph(no_of_var = data_aug.shape[1], node_names=None)
        fed_indep_test = CIT(fed_dt, 'kci')
        fed_cg.set_ind_test(fed_indep_test)
        cg_list.append(fed_cg)

    """
    Just skeleton learning, without surrogate variable.
    Flag:
        -1: use random feature;
        0: original cdnod; 
        1: print all fed p-values; 
        2: fed cdnod voting-based; 
    """
    flag = -1
    print(f"#######  Skeleton Discovery Stage 1: No surrogate. Flag={flag}")
    cg_0 = SkeletonDiscovery.skeleton_discovery(flag, cg_list, data, K, alpha, indep_test_all, stable)
    # print("\n")
    
    # from causallearn.utils.GraphUtils import GraphUtils
    # pyd = GraphUtils.to_pydot(cg_0.G)
    # pyd.write_png('result/fedcdn2/Lktest_cg0.png')
    # t_1 = time.time()
    # print(f"Time for skeleton learning: {t_1-start}s.")

    """
    Changing causal module detection with surrogate variable in two steps:
        - add new surrogate variable C;
        - add new edges between C and X_i;
    Flag:
        -1: use random feature;
        0: original cdnod; 
        1: print all fed p-values; 
        2: fed cdnod voting-based; 
        3: fed cdnod linearGaussian;
        4: fed cdnod GMM-based. 
    """ 
    flag = -1
    print(f"#######  Skeleton Discovery Stage 2: with surrogate. Flag={flag}")
    cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(flag, cg_0, cg_list, data_aug, K, alpha, indep_test_all, stable)
    # print("\n")
    # cg_1 = cg_0 

    # t_2 = time.time()
    # print(f"Time for causal detection: {t_2-t_1}s.")

    # orient the direction from c_indx to X, if there is an edge between c_indx and X
    c_indx_id = data_aug.shape[1] - 1
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    # By default: uc_rule=0, uc_priority=-1.
    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            # print(f"101: In CDNOD: {uc_rule},{uc_priority}")
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge,cg_list=cg_list, K=K)
        
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")



    """
    Stage 3: Independent change. Use HSIC
    This code is devoloped based on CDNOD long-version paper and their MATLAB code.
    """
    print("#######  Direction ditermination: using HSIC.")
    # 1. Find all undirected edges. d=obs+sur
    # 1.1 Get undirected edges from all surrogate variables.
    vh = [] # variables in heteregeneity.
    for i in range(d-1):
        if (cg.G.graph[i,d-1]==1) and (cg.G.graph[d-1,i]==-1):
            vh.append(i)
    if len(vh)>=2:
        vh = combinations(vh,2)
    else:
        vh = []
    # print(f"The list 1 is: {vh}. All combinations: {vhs}.")

    # 1.2 Get undirected edges from all undirected variables.
    # vh = []
    # for i in range(d-1):
    #     for j in range(i,d-1):
    #         if (cg.G.graph[i,j]==-1) and (cg.G.graph[j,i]==-1):
    #             vh.append((i,j))
    # print(f"the list 2 is: {vh}.")

    # 2. Calculate HSIC.
    for v in vh:
        i,j = v
        # print(f"i={i}, j={j}.")
        """
        Score_i_j: 
            1: i -> j;
            2: j -> i.
        """
        score_i_j = get_hsic_score(data[:,i].reshape(-1,1), data[:,j].reshape(-1,1), c_indx)
        if score_i_j==1:
            cg.G.add_edge(Edge(cg.G.nodes[i], cg.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
        else:
            cg.G.add_edge(Edge(cg.G.nodes[j], cg.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))
        

    end = time.time()
    cg.PC_elapsed = end - start

    # from causallearn.utils.GraphUtils import GraphUtils
    # pyd = GraphUtils.to_pydot(cg.G)
    # pyd.write_png('result/fedcdn2/Lktest_cg6.png')
    
    # pyd = GraphUtils.to_pydot(cg_2.G)
    # pyd.write_png('result/Lktest_exp3_test2.png')

    # pyd = GraphUtils.to_pydot(cg.G)
    # pyd.write_png('result/Lktest_exp3_test3.png')

    return cg


def cdnod_alg_origianal(data: ndarray, c_indx: ndarray, alpha: float, K: int, indep_test: str, stable: bool, uc_rule: int, uc_priority: int,
              background_knowledge: Optional[BackgroundKnowledge] = None, verbose: bool = False,
              show_progress: bool = True, **kwargs) -> CausalGraph:
    """
    Perform Peter-Clark algorithm for causal discovery on the augmented data set that captures the unobserved changing factors

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    alpha : desired significance level (float) in (0, 1)
    indep_test : name of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
           - "MV_Fisher_Z": Missing-value Fishers'Z conditional independence test
           - "kci": kernel-based conditional independence test (If C is time index, KCI test is recommended)
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()

    data_aug = np.concatenate((data, c_indx), axis=1)
    indep_test_all = CIT(data_aug, indep_test, **kwargs)
    
    s_a, s_b = data_aug.shape 
    fed_data = data_aug.reshape(K, int(s_a/K), s_b) 
    cg_list = [] # list 
    for i in range(K):
        fed_dt = fed_data[i]
        fed_cg = CausalGraph(no_of_var = data_aug.shape[1], node_names=None)
        fed_indep_test = CIT(fed_dt, indep_test)
        fed_cg.set_ind_test(fed_indep_test)
        cg_list.append(fed_cg)
    
    
    # Flag: 0-original CD-NOD; 1-print all the fed p-values into csv; 2-fed CD-NOD.
    flag = 0
    print(f"Stage 1: In CDNOD-Skeleton: flag={flag}")
    cg_1 = SkeletonDiscovery.skeleton_discovery(flag, cg_list, data_aug, K, alpha, indep_test_all, stable)

    # from causallearn.utils.GraphUtils import GraphUtils
    # pyd = GraphUtils.to_pydot(cg_0.G)
    # pyd.write_png('result/fedcdn2/Lktest_cg0.png')

    """
        If run CD-NOD:
            add new sorrogate variable C;
            add new edges pointing from C to X_i;
    """ 
    # flag = 0
    # print(f"Stage 2: In CDNOD-Skeleton: flag={flag}")
    # cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate(flag, cg_0, cg_list, data_aug, K, alpha, indep_test, stable)
    # from causallearn.utils.GraphUtils import GraphUtils
    # pyd = GraphUtils.to_pydot(cg_1.G)
    # pyd.write_png('result/fedcdn2/Lktest_cg1.png')

    # return 
    # orient the direction from c_indx to X, if there is an edge between c_indx and X
    c_indx_id = data_aug.shape[1] - 1
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            # print(f"101: In CDNOD: {uc_rule},{uc_priority}")
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge,cg_list=cg_list, K=K)
        
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    # from causallearn.utils.GraphUtils import GraphUtils
    # pyd = GraphUtils.to_pydot(cg.G)
    # pyd.write_png('result/fedcdn2/Lktest_cg6.png')
    
    # pyd = GraphUtils.to_pydot(cg_2.G)
    # pyd.write_png('result/Lktest_exp3_test2.png')

    # pyd = GraphUtils.to_pydot(cg.G)
    # pyd.write_png('result/Lktest_exp3_test3.png')

    return cg


def mvcdnod_alg(data: ndarray, alpha: float, indep_test: str, correction_name: str, stable: bool, uc_rule: int,
                uc_priority: int, verbose: bool, show_progress: bool, **kwargs) -> CausalGraph:
    """
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level (float) in (0, 1)
    :param indep_test: name of the test-wise deletion independence test being used
           - "MV_Fisher_Z": Fisher's Z conditional independence test
           - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    : param correction_name: name of the missingness correction
            - "MV_Crtn_Fisher_Z": Permutation based correction method
            - "MV_Crtn_G_sq": G-squared conditional independence test (TODO: under development)
            - "MV_DRW_Fisher_Z": density ratio weighting based correction method (TODO: under development)
            - "MV_DRW_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    :param uc_rule: how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    :param uc_priority: rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    :return:
    cg: a CausalGraph object
    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test, stable)
    # print('Finish detecting the parents of missingness indicators.  ')

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg_pre = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable, verbose=verbose,
                                                  show_progress=show_progress)
    cg_pre.to_nx_skeleton()
    # print('Finish skeleton search with test-wise deletion.')

    ## b) Correction of the extra edges
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)
    # print('Finish missingness correction.')

    ## Step 3: Orient the edges
    # orient the direction from c_indx to X, if there is an edge between c_indx and X
    c_indx_id = data.shape[1] - 1
    for i in cg_corr.G.get_adjacent_nodes(cg_corr.G.nodes[c_indx_id]):
        cg_corr.G.add_directed_edge(i, cg_corr.G.nodes[c_indx_id])

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_corr, uc_priority)
        else:
            cg_2 = UCSepset.uc_sepset(cg_corr)
        cg = Meek.meek(cg_2)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_corr, uc_priority)
        else:
            cg_2 = UCSepset.maxp(cg_corr)
        cg = Meek.meek(cg_2)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha, uc_priority)
        else:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha)
        cg_before = Meek.definite_meek(cg_2)
        cg = Meek.meek(cg_before)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg
