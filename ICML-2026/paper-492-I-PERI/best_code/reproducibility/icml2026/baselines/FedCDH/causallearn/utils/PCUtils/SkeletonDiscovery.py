from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy import ndarray
from typing import List
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import CIT

from causallearn.utils.cit import kci 
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns


def skeleton_discovery(
    flag: int, 
    cg_list: List[CausalGraph], 
    data: ndarray, 
    K: int,
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
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
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """
    
    # print(f"This is K: {K}")
    # print(data.shape)
    # print(fed_data.shape, fed_data[0].shape)

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, None)
    cg.set_ind_test(indep_test) # indep_test = CIT(data, indep_test, **kwargs)


    if flag==1:
        f = open('result/fedcd/k6-n8-pvalue_fed_2.csv', 'a+')
        f.write("{x},{y},{S},{p},{p>alpha}\n")

    dep_list = np.zeros(10)
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
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
                    if flag==0:
                        p = cg.ci_test(x, y, S)   

                    if flag==-1:
                        p = cg.ci_test(x, y, S, gmm=3)

                    # print all the fed p-values.
                    if flag==1: 
                        f.write(f"{x},{y},{S},{p},{p>alpha}")
                        for i in range(K):
                            p_ = cg_list[i].ci_test(x, y, S) 
                            f.write(f",{p_}")
                        f.write("\n")
                    
                    # fed voting scheme.
                    if flag==2:
                        ############################################
                        # 1-Max; 2-Avg; 3/4-voting; 
                        # 5-enhanced voting (sapling-based);
                        # 6-bilevel voting (seperate C and other nodes.)
                        style = 4                  
                        ratio = 0.6

                        if style == 1:                           
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            p = max(p_list)    
                        elif style == 2:
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            p = sum(p_list)/len(p_list)
                        elif style == 3:    
                            count = 0
                            for i in range(K):
                                p_ = cg_list[i].ci_test(x, y, S)
                                if p_ <= alpha:
                                    count += 1
                            if count >= ratio * K: # ratio
                                p = 0
                            else:
                                p = 1 
                        elif style == 4:
                            # p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                            # count = np.sum(p_list)
                            count = np.sum([cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)])
                            p = (count<ratio*K)
                            # print("Linear Gaussian. Using Voting method")
                            # if count >= ratio * K: # ratio
                            #     p = 0
                            # else:
                            #     p = 1 
                        elif style == 5:    
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            big_K = 100
                            p_list_new = np.random.choice(p_list, big_K)
                            p_list_new = [i<=alpha for i in p_list_new] 
                            count = np.sum(p_list_new)
                            if count >= ratio * big_K: # ratio
                                p = 0
                            else:
                                p = 1
                        elif style == 6:
                            if x == no_of_var-1 or y == no_of_var-1 or (no_of_var-1) in S:
                                p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                                count = np.sum(p_list)
                                if count >= 0.2 * K: # ratio
                                    p = 0
                                else:
                                    p = 1
                            else:
                                p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                                count = np.sum(p_list)
                                if count >= 0.1 * K: # ratio
                                    p = 0
                                else:
                                    p = 1

                        elif style == 7:
                            p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                            count = np.sum(p_list)
                            if count >= ratio * K: # ratio
                                p = 0
                            else:
                                p = 1
                            
                            p_cen = cg.ci_test(x, y, S)
                            if (p_cen > alpha and p == 0) or (p_cen <= alpha and p == 1):
                                p_fed = [np.round(cg_list[i].ci_test(x, y, S),4) for i in range(K)]
                                print(f"{x, y, S} | P-cen: {np.round(p_cen,4)},{p_cen>alpha} | P-fed: {p_fed}.")

                        
                    
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
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
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if flag==1:
        f.close()

    if show_progress:
        pbar.close()

    # print(f"This is the depth list: {dep_list}.")
    # print(f"average: {np.sum(dep_list), np.sum(dep_list)/np.sum(dep_list>0)}")


    return cg



def skeleton_discovery_with_surrogate(
    flag: int, 
    cg: CausalGraph, 
    cg_list: List[CausalGraph], 
    data: ndarray, 
    K: int,
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
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
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """
    
    # print(f"This is K: {K}")
    # print(data.shape)
    # print(fed_data.shape, fed_data[0].shape)

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    # ----------------------------------------------
    # no_of_var = data.shape[1] 
    # cg = CausalGraph(no_of_var, None)
    # cg.set_ind_test(indep_test) # indep_test = CIT(data, indep_test, **kwargs)
    """
        Introduce new surrogate variables.
        1. add new variable;
        2. add new edges between C and X_i;
    """
    # print(f"Before surrogate: number of nodes: {cg.G.num_vars}.")

    no_of_var = data.shape[1] 
    surrogate_node = GraphNode("X%d" % (no_of_var))
    cg.G.add_node(surrogate_node) # return True
    # print(f"a bool: {a_bool}. ", "X%d" % (no_of_var), )
    # print(f"2 number of nodes: {cg.G.num_vars}.")
    # print(f"After surrogate: number of nodes: {cg.G.num_vars}.")

    c_indx_id = no_of_var - 1
    for i in range(no_of_var-1):
        # cg.G.add_directed_edge(cg.G.nodes[c_indx_id], cg.G.nodes[i])
        cg.G.add_edge(Edge(cg.G.nodes[c_indx_id], cg.G.nodes[i], Endpoint.TAIL, Endpoint.TAIL))
    # ----------------------------------------------

    if flag==1:
        f = open('result/fedcd/k6-n8-pvalue_fed_2.csv', 'a+')
        f.write("{x},{y},{S},{p},{p>alpha}\n")

    dep_list = np.zeros(10)
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        
        x_list = [c_indx_id]
        for x in x_list: #range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            # print(f"With surrogate variable, the neighbors are {Neigh_x}")
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
                    if flag==0:
                        p = cg.ci_test(x, y, S)    ################################### 

                    # print all the fed p-values.
                    if flag==1: 
                        f.write(f"{x},{y},{S},{p},{p>alpha}")
                        for i in range(K):
                            p_ = cg_list[i].ci_test(x, y, S) 
                            f.write(f",{p_}")
                        f.write("\n")
                    
                    # fed voting scheme.
                    if flag==2:
                        ############################################
                        # 1-Max; 2-Avg; 3/4-voting; 
                        # 5-enhanced voting (sapling-based);
                        # 6-bilevel voting (seperate C and other nodes.)
                        style = 4                  
                        ratio = 0.1

                        if style == 1:                           
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            p = max(p_list)    
                        elif style == 2:
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            p = sum(p_list)/len(p_list)
                        elif style == 3:    
                            count = 0
                            for i in range(K):
                                p_ = cg_list[i].ci_test(x, y, S)
                                if p_ <= alpha:
                                    count += 1
                            if count >= ratio * K: # ratio
                                p = 0
                            else:
                                p = 1 
                        elif style == 4:
                            # p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                            # count = np.sum(p_list)
                            count = np.sum([cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)])
                            p = (count<ratio*K)
                        elif style == 5:    
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            big_K = 100
                            p_list_new = np.random.choice(p_list, big_K)
                            p_list_new = [i<=alpha for i in p_list_new] 
                            count = np.sum(p_list_new)
                            if count >= ratio * big_K: # ratio
                                p = 0
                            else:
                                p = 1
                        elif style == 6:
                            if x == no_of_var-1 or y == no_of_var-1 or (no_of_var-1) in S:
                                p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                                count = np.sum(p_list)
                                if count >= 0.2 * K: # ratio
                                    p = 0
                                else:
                                    p = 1
                            else:
                                p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                                count = np.sum(p_list)
                                if count >= 0.1 * K: # ratio
                                    p = 0
                                else:
                                    p = 1

                        elif style == 7:
                            p_list = [cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)]
                            count = np.sum(p_list)
                            if count >= ratio * K: # ratio
                                p = 0
                            else:
                                p = 1
                            
                            p_cen = cg.ci_test(x, y, S)
                            if (p_cen > alpha and p == 0) or (p_cen <= alpha and p == 1):
                                p_fed = [np.round(cg_list[i].ci_test(x, y, S),4) for i in range(K)]
                                print(f"{x, y, S} | P-cen: {np.round(p_cen,4)},{p_cen>alpha} | P-fed: {p_fed}.")

                        
                    
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
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
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                # print("test fine here.")
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if flag==1:
        f.close()

    if show_progress:
        pbar.close()

    # print(f"This is the depth list: {dep_list}.")
    # print(f"average: {np.sum(dep_list), np.sum(dep_list)/np.sum(dep_list>0)}")


    return cg



def skeleton_discovery_with_surrogate_GMM(
    flag: int, 
    cg: CausalGraph, 
    cg_list: List[CausalGraph], 
    data: ndarray, 
    K: int,
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
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
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=-1 and cg.G.graph[i,j]=1 indicates  i -> j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """
    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    """
        Introduce new surrogate variables.
        1. add new variable;
        2. add new edges between C and X_i;
    """
    # print(f"Before surrogate: number of nodes: {cg.G.num_vars}.")

    no_of_var = data.shape[1] 
    surrogate_node = GraphNode("X%d" % (no_of_var))
    cg.G.add_node(surrogate_node) 
    # print(f"After surrogate: number of nodes: {cg.G.num_vars}.")

    c_indx_id = no_of_var - 1
    for i in range(no_of_var-1):
        cg.G.add_edge(Edge(cg.G.nodes[c_indx_id], cg.G.nodes[i], Endpoint.TAIL, Endpoint.TAIL))
    # ----------------------------------------------

    if flag==1:
        f = open('result/fedcd/k6-n8-pvalue_fed_2.csv', 'a+')
        f.write("{x},{y},{S},{p},{p>alpha}\n")

    # dep_list = np.zeros(10)
    depth = -1
    # pbar = tqdm(total=no_of_var) if show_progress else None
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        
        x_list = [c_indx_id]
        for x in x_list: #range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            # print(f"With surrogate variable, the neighbors are {Neigh_x}")
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
                # print(f"Neighbor test: Neigh_x_noy={Neigh_x_noy}.")
                for S in combinations(Neigh_x_noy, depth):
                    if flag==0:
                        # print("KCI test. for CI test: ", x,y,S)
                        p = cg.ci_test(x, y, S)    

                    if flag==-1:
                        p = cg.ci_test(x, y, S, gmm=3)

                    # Gaussian Mixture Model 
                    if flag==4:
                        if len(S)==0:
                            print("GMM - Unconditional test: ", x,y,S)
                        else:
                            print("GMM - Conditional test: ", x,y,S)                       
                        p = cg.ci_test(x, y, S, gmm=1, K=K)
                        # print(f"The groundtruth pvalue for CIT is: {cg.ci_test(x, y, S)}.")
                    
                    # Linear Gaussian Model
                    if flag==3:
                        if len(S)==0:
                            print("Linear Gaussian - Unconditional test: ", x,y,S)
                        else:
                            print("Linear Gaussian - Conditional test: ", x,y,S)  
                        p = cg.ci_test(x, y, S, gmm=2, K=K)    


                    # print all the fed p-values.
                    if flag==1: 
                        f.write(f"{x},{y},{S},{p},{p>alpha}")
                        for i in range(K):
                            p_ = cg_list[i].ci_test(x, y, S) 
                            f.write(f",{p_}")
                        f.write("\n")
                    
                    # Fed voting scheme.
                    if flag==2:
                        # style: 1-Max; 2-Avg; 3/4-voting; 
                        style = 5                 
                        ratio = 0.1
                        # print("Linear Gaussian. Using Voting method")
                        if style == 1:                           
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            p = max(p_list)    
                        elif style == 2:
                            p_list = [cg_list[i].ci_test(x, y, S) for i in range(K)]
                            p = sum(p_list)/len(p_list)
                        elif style == 3:    
                            count = 0
                            for i in range(K):
                                p_ = cg_list[i].ci_test(x, y, S)
                                if p_ <= alpha:
                                    count += 1
                            if count >= ratio * K: # ratio
                                p = 0
                            else:
                                p = 1 
                        elif style == 4:
                            count = np.sum([cg_list[i].ci_test(x, y, S)<=alpha for i in range(K)])
                            p = (count<ratio*K)

                        elif style==5:
                            p = 1
                                       
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
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
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)
    if flag==1:
        f.close()
    if show_progress:
        pbar.close()
    return cg