import pickle
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import KBinsDiscretizer

from causallearn.utils.cit import CIT
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.graph.GraphClass import CausalGraph

START_ALPHA = 0.001
ALPHA_STEP = 0.1
ALPHA_LIMIT = 1

VERBOSE = False
F_NODE = 'F-node'

ANOMALOUS_NODE = 'a_node'
GRAPH_GEN_INFO = 'gen_graph_info.pkl'
GROUND_TRUTH_BN_GRAPH = 'g_bn_graph.pkl'

_get_labels = lambda data: {i: name for i, name in enumerate(data.columns)}


def _cg_remove_edge(cg, x, y):
    edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
    if edge1 is not None:
        cg.G.remove_edge(edge1)
    edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
    if edge2 is not None:
        cg.G.remove_edge(edge2)

def _local_skeleton_discovery(data, local_node, alpha, indep_test,
                              mi=[], labels={}, verbose=False):
    assert type(data) == np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names=labels)
    cg.set_ind_test(indep_test)

    new_mi = []
    tests = 0
    p_values = np.empty((no_of_var, no_of_var), object)

    depth = -1
    x = local_node
    # Remove edges between nodes in MI and F-node
    for i in mi:
        _cg_remove_edge(cg, x, i)

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
                tests += 1
                if p > alpha:
                    if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                    _cg_remove_edge(cg, x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        new_mi.append(y)
                    break
                else:
                    append_value(p_values, x, y, p)
                    if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))

    return cg, new_mi, p_values, tests

def load_datasets(normal, anomalous, verbose=VERBOSE):
    if verbose:
        print('Loading the dataset ...')
    normal_df = pd.read_csv(normal)
    anomalous_df = pd.read_csv(anomalous)
    return (normal_df, anomalous_df)

def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def add_fnode_to_graph(path):
    nx_graph: nx.DiGraph = load_graph(f'{path}/{GROUND_TRUTH_BN_GRAPH}')
    graph_info = load_graph(f'{path}/{GRAPH_GEN_INFO}')
    nx_graph.add_node(F_NODE)
    nx_graph.add_edge(F_NODE, graph_info[ANOMALOUS_NODE])
    return nx_graph

def add_fnode(normal_df, anomalous_df):
    normal_df[F_NODE] = 0
    anomalous_df[F_NODE] = 1
    return pd.concat([normal_df, anomalous_df])

# Run PC on the given dataset.
# The last column of the data must be the F-node
def run_pc(data, alpha, localized=True, labels=None, mi=[],
           max_depth=np.inf, cg_opts=None, ci_test=None, verbose=VERBOSE):
    if labels is None: labels = _get_labels(data)

    np_data = data.to_numpy()
    indep_test = CIT(np_data, ci_test)
    if localized:
        f_node = np_data.shape[1] - 1
        result = _local_skeleton_discovery(np_data,f_node, alpha,
                                           indep_test=indep_test, mi=mi,
                                           labels=list(labels.values()),
                                           verbose=verbose)
    else:
        raise Exception('Not updated with new version of causal-learn')
    return result

def top_k_rc(normal_df, anomalous_df, bins=None, mi=[],
             localized=True, start_alpha=None, min_nodes=-1,
             max_depth=np.inf, cg_opts=dict(), ci_test=None, verbose=VERBOSE):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)

    if min_nodes == -1:
        # Order all nodes (if possible) except F-node
        min_nodes = len(data.columns) - 1
    assert(min_nodes < len(data.columns))

    G = None
    no_ci = 0
    i_to_labels = {i: name for i, name in enumerate(data.columns)}
    labels_to_i = {name: i for i, name in enumerate(data.columns)}
    cg_opts['i_to_labels'] = i_to_labels

    _preprocess_mi = lambda l: [labels_to_i.get(i) for i in l]
    _postprocess_mi = lambda l: [i_to_labels.get(i) for i in list(filter(None, l))]
    processed_mi = _preprocess_mi(mi)
    _run_pc = lambda alpha: run_pc(data, alpha, localized=localized, mi=processed_mi,
                                   labels=i_to_labels, cg_opts=cg_opts, max_depth=max_depth,
                                   ci_test=ci_test, verbose=verbose)

    rc = []
    _alpha = START_ALPHA if start_alpha is None else start_alpha
    for i in np.arange(_alpha, ALPHA_LIMIT, ALPHA_STEP):
        cg, new_mi, p_values, no_tests = _run_pc(i)
        no_ci += no_tests
        f_neigh = [x.name for x in cg.G.get_adjacent_nodes(cg.G.get_node(F_NODE))]
        new_neigh = [x for x in f_neigh if x not in rc]
        if cg_opts.get('oracle'):
            rc = f_neigh
            break

        if len(new_neigh) == 0: continue
        else:
            f_p_values = p_values[-1][[labels_to_i.get(key) for key in new_neigh]]
            rc += _order_neighbors(new_neigh, f_p_values)

        if len(rc) == min_nodes: break

    return (rc, G, _postprocess_mi(new_mi), no_ci)

def _order_neighbors(neigh, p_values):
    _neigh = neigh.copy()
    _p_values = p_values.copy()
    stack = []

    while len(_neigh) != 0:
        i = np.argmax(_p_values)
        node = _neigh[i]
        stack = [node] + stack
        _neigh.remove(node)
        _p_values = np.delete(_p_values, i)
    return stack

def _preprocess_for_fnode(normal_df, anomalous_df, bins):
    df = add_fnode(normal_df, anomalous_df)
    if df is None: return None
    return df
