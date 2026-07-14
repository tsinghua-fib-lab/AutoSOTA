import pickle
import warnings

import numpy as np
import pandas as pd
import networkx as nx
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import KBinsDiscretizer

from causallearn.utils.cit import CIT
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

START_ALPHA = 0.001
ALPHA_STEP = 0.1
ALPHA_LIMIT = 1

VERBOSE = False
F_NODE = 'F-node'

ANOMALOUS_NODE = 'a_node'
GRAPH_GEN_INFO = 'gen_graph_info.pkl'
GROUND_TRUTH_BN_GRAPH = 'g_bn_graph.pkl'

_get_labels = lambda data: {i: name for i, name in enumerate(data.columns)}

def get_node_name(node):
    return f"X{node}"

def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]

def preprocess(n_df, a_df, bins):
    # _process = lambda df: _select_lat(_rm_time(df), per)
    # _process = lambda df: _select_lat(_scale_down_mem(_rm_time(df)), per)

    # n_df = _process(n_df)
    # a_df = _process(a_df)

    # n_df = drop_constant(n_df)
    # a_df = drop_constant(a_df)

    # n_df, a_df = _match_columns(n_df, a_df)

    # # Enable for sock-shop
    # n_df = _select_useful_cols(n_df)
    # a_df = _select_useful_cols(a_df)

    # df = _discretize(df, bins)
    # n_df = df[df[F_NODE] == 0].drop(columns=[F_NODE])
    # a_df = df[df[F_NODE] == 1].drop(columns=[F_NODE])

    df = add_fnode(n_df, a_df)
    df = _select_useful_cols(df, keep_cols=[F_NODE])
    df = _discretize(df, bins)
    n_df = df[df[F_NODE] == 0].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == 1].drop(columns=[F_NODE])
    return n_df, a_df

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
def run_pc(data, alpha, localized=False, labels=None, mi=[],
           max_depth=np.inf, cg_opts=None, ci_test=None, verbose=VERBOSE):
    if labels is None: labels = _get_labels(data)

    np_data = data.to_numpy()
    indep_test = CIT(np_data, ci_test)
    # localized=False
    if localized:
        f_node = np_data.shape[1] - 1
        cg = SkeletonDiscovery.local_skeleton_discovery(np_data, f_node, alpha,
                                                        indep_test=indep_test, mi=mi,
                                                        labels=list(labels.values()),
                                                        max_depth=max_depth,
                                                        cg_opts=cg_opts, verbose=verbose)
    else:
        cg = SkeletonDiscovery.skeleton_discovery(np_data, alpha, indep_test=indep_test,
                                                  background_knowledge=None,
                                                  stable=False, verbose=verbose,
                                                  labels=list(labels.values()),
                                                  max_depth=max_depth,
                                                  cg_opts=cg_opts, show_progress=False)
    return cg

def save_graph(graph, file):
    nx.draw_networkx(graph)
    plt.savefig(file)

# def pc_with_fnode(normal_df, anomalous_df, alpha, bins=None,
#                   localized=False, verbose=VERBOSE):
#     data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
#     cg = run_pc(data, alpha, localized=localized, verbose=verbose)
#     return cg.nx_graph

def ges_with_fnode(normal_df, anomalous_df, bins=None, labels=None):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
    if labels is None: labels = _get_labels(data)

    ges_r = ges(data.to_numpy(), score_func='local_score_BDeu', labels=labels)
    G = ges_r['G']
    f_neigh = G.get_adjacent_nodes(G.get_node(F_NODE))
    return [x.get_name() for x in f_neigh]

def top_k_rc(normal_df, anomalous_df, bins=None, mi=[],
             localized=False, start_alpha=None, min_nodes=-1,
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
        cg = _run_pc(i)
        no_ci += cg.no_ci_tests
        f_neigh = [x.name for x in cg.G.get_adjacent_nodes(cg.G.get_node(F_NODE))]
        new_neigh = [x for x in f_neigh if x not in rc]
        if cg_opts.get('oracle'):
            rc = f_neigh
            break

        if len(new_neigh) == 0: continue
        else:
            f_p_values = cg.p_values[-1][[labels_to_i.get(key) for key in new_neigh]]
            rc += _order_neighbors(new_neigh, f_p_values)

        if len(rc) == min_nodes: break

    return (rc, G, _postprocess_mi(cg.mi), no_ci)

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

# ==================== Private methods =============================

_rm_time = lambda df: df.loc[:, ~df.columns.isin(['time'])]
_list_intersection = lambda l1, l2: [x for x in l1 if x in l2]

def _preprocess_for_fnode(normal_df, anomalous_df, bins):
    df = add_fnode(normal_df, anomalous_df)
    if df is None: return None

    return df

def _select_useful_cols(df, keep_cols=[]):
    names = ['front-end', 'user', 'catalogue', 'orders', 'carts', 'payment', 'shipping']
    metrics = ['cpu', 'mem', 'lod', 'lat_50']

    l_names = [] + keep_cols
    for i in names:
        for j in metrics:
            _name = f'{i}_{j}'
            if _name in df.columns:
                l_names.append(_name)
    df = df[l_names]

    # Drop constants
    df = df.loc[:, (df != df.iloc[0]).any()]

    # df = _select_lat(_rm_time(df), SOCKSHOP_LATENCY_PREC)
    return _discretize(df, 5)
    # return df

    # i = df.loc[:, df.columns != F_NODE].std() > 1
    # cols = i[i].index.tolist()
    # cols.append(F_NODE)
    # if len(cols) == 1:
    #     return None
    # elif len(cols) == len(df.columns):
    #     return df

    # # print(f"Out of {df.columns.tolist()}, selecting only {cols} columns")
    # return df[cols]

def _match_columns(n_df, a_df):
    cols = _list_intersection(n_df.columns, a_df.columns)
    return (n_df[cols], a_df[cols])

# Convert all memeory columns to MBs
def _scale_down_mem(df):
    def update_mem(x):
        if not x.name.endswith('_mem'):
            return x
        x /= 1e6
        x = x.astype(int)
        return x

    return df.apply(update_mem)

# Select all the non-latency columns and only select latecy columns
# with given percentaile
def _select_lat(df, per):
    return df.filter(regex=(".*(?<!lat_\d{2})$|_lat_" + str(per) + "$"))

# NOTE: THIS FUNCTION THROWS WARNGINGS THAT ARE SILENCED!
def _discretize(data, bins=5):
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discretizer.fit(data)
    disc_d = discretizer.transform(data)
    disc_d = pd.DataFrame(disc_d, columns=data.columns)
    disc_d = disc_d.astype(int)
    return disc_d