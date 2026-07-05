# Disable warnings from sklearn
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import KBinsDiscretizer

from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import SkeletonDiscovery

# Note: Some of the functions defined here are only used for data
# from sock-shop or real-world application.


START_ALPHA = 0.001
ALPHA_STEP = 0.1
ALPHA_LIMIT = 1

VERBOSE = False
F_NODE = 'F-node'

def get_node_name(node):
    return f"X{node}"

def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]

# Only used for sock-shop and real outage datasets
def preprocess(n_df, a_df, per):
    _process = lambda df: _select_lat(_scale_down_mem(_rm_time(df)), per)

    n_df = _process(n_df)
    a_df = _process(a_df)

    n_df = drop_constant(n_df)
    a_df = drop_constant(a_df)

    n_df, a_df = _match_columns(n_df, a_df)

    df = _select_useful_cols(add_fnode(n_df, a_df))
    n_df = df[df[F_NODE] == '0'].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == '1'].drop(columns=[F_NODE])

    return (n_df, a_df)

def load_datasets(normal, anomalous, verbose=VERBOSE):
    if verbose:
        print('Loading the dataset ...')
    normal_df = pd.read_csv(normal)
    anomalous_df = pd.read_csv(anomalous)
    return (normal_df, anomalous_df)

def add_fnode(normal_df, anomalous_df):
    normal_df[F_NODE] = '0'
    anomalous_df[F_NODE] = '1'
    return pd.concat([normal_df, anomalous_df])

# Run PC (only the skeleton phase) on the given dataset.
# The last column of the data *must* be the F-node
def run_pc(data, alpha, localized=False, labels={}, mi=[], verbose=VERBOSE):
    if labels == {}:
        labels = {i: name for i, name in enumerate(data.columns)}

    np_data = data.to_numpy()
    if localized:
        f_node = np_data.shape[1] - 1
        # Localized PC
        cg = SkeletonDiscovery.local_skeleton_discovery(np_data, f_node, alpha,
                                                        indep_test=CIT(np_data, "chisq"), mi=mi,
                                                        verbose=verbose)
    else:
        cg = SkeletonDiscovery.skeleton_discovery(np_data, alpha, indep_test=CIT(np_data, "chisq"),
                                                  background_knowledge=None,
                                                  stable=False, verbose=verbose,
                                                  show_progress=False)

    cg.to_nx_graph()
    return cg

def get_fnode_child(G):
    if F_NODE in G:
        return [*G.successors(F_NODE)]

    # causal-learn récent garde parfois les indices numériques.
    # Le F-node est toujours la dernière colonne.
    f_node = max(G.nodes)
    return [*G.successors(f_node)]

def save_graph(graph, file):
    nx.draw_networkx(graph)
    plt.savefig(file)

def pc_with_fnode(normal_df, anomalous_df, alpha, bins=None,
                  localized=False, verbose=VERBOSE):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
    cg = run_pc(data, alpha, localized=localized, verbose=verbose)
    return cg.nx_graph

# Equivelant to \Psi-PC from the main paper
def top_k_rc(normal_df, anomalous_df, bins=None, mi=[],
             localized=False, start_alpha=None, min_nodes=-1,
             verbose=VERBOSE):
    if 0 in [len(normal_df.columns), len(anomalous_df.columns)]:
        return ([], None, [], 0)
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)

    if min_nodes == -1:
        # Order all nodes (if possible) except F-node
        min_nodes = len(data.columns) - 1
    assert(min_nodes < len(data))

    G = None
    no_ci = 0
    i_to_labels = {i: name for i, name in enumerate(data.columns)}
    labels_to_i = {name: i for i, name in enumerate(data.columns)}

    _preprocess_mi = lambda l: [labels_to_i.get(i) for i in l]
    _postprocess_mi = lambda l: [i_to_labels.get(i) for i in list(filter(None, l))]
    processed_mi = _preprocess_mi(mi)
    _run_pc = lambda alpha: run_pc(data, alpha, localized=localized, mi=processed_mi,
                                   labels=i_to_labels, verbose=verbose)

    rc = []
    _alpha = START_ALPHA if start_alpha is None else start_alpha
    for i in np.arange(_alpha, ALPHA_LIMIT, ALPHA_STEP):
        cg = _run_pc(i)
        G = cg.nx_graph
        no_ci += getattr(cg, "no_ci_tests", 0)

        if G is None: continue

        f_neigh = get_fnode_child(G)
        f_neigh = [i_to_labels.get(x, x) for x in f_neigh]
        new_neigh = [x for x in f_neigh if x not in rc]
        if len(new_neigh) == 0: continue
        else:
            if hasattr(cg, "p_values"):
                f_p_values = cg.p_values[-1][[labels_to_i.get(key) for key in new_neigh]]
                rc += _order_neighbors(new_neigh, f_p_values)
            else:
                rc += sorted(new_neigh, key=str)

        if len(rc) == min_nodes: break

    return (rc, G, _postprocess_mi(getattr(cg, "mi", [])), no_ci)

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

    return _discretize(df, bins) if bins is not None else df

def _select_useful_cols(df):
    i = df.loc[:, df.columns != F_NODE].std() > 1
    cols = i[i].index.tolist()
    cols.append(F_NODE)
    if len(cols) == 1:
        return None
    elif len(cols) == len(df.columns):
        return df

    return df[cols]

# Only select the metrics that are in both datasets
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
def _discretize(data, bins):
    d = data.iloc[:, :-1]
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    discretizer.fit(d)
    disc_d = discretizer.transform(d)
    disc_d = pd.DataFrame(disc_d, columns=d.columns.values.tolist())
    disc_d[F_NODE] = data[F_NODE].tolist()

    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)

    return disc_d
