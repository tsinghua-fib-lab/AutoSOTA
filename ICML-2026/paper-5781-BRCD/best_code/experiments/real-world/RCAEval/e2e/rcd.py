import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.utils.cit import chisq
from RCAEval.e2e.causallearn.utils.PCUtils import SkeletonDiscovery
from sklearn.preprocessing import KBinsDiscretizer

from RCAEval.io.time_series import drop_extra

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")


# =========== UTILS.py ====================
# Note: Some of the functions defined here are only used for data
# from sock-shop or real-world application.
CI_TEST = chisq

START_ALPHA = 0.001
ALPHA_STEP = 0.1
ALPHA_LIMIT = 1

VERBOSE = False
F_NODE = "F-node"


def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]


# Only used for sock-shop and real outage datasets
def preprocess_sock_shop(n_df, a_df, per, dk_select_useful=False):
    _process = lambda df: _select_lat(_scale_down_mem(_rm_time(df)), per)

    n_df = _process(n_df)
    a_df = _process(a_df)

    n_df = drop_constant(n_df)
    a_df = drop_constant(a_df)

    n_df, a_df = _match_columns(n_df, a_df)

    df = add_fnode_and_concat(n_df, a_df)

    if dk_select_useful is True:
        df = _select_useful_cols(df)

    n_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    return (n_df, a_df)


def load_datasets(normal, anomalous):
    normal_df = pd.read_csv(normal)
    anomalous_df = pd.read_csv(anomalous)
    return (normal_df, anomalous_df)


def add_fnode_and_concat(normal_df, anomalous_df):
    normal_df[F_NODE] = "0"
    anomalous_df[F_NODE] = "1"
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
        cg = SkeletonDiscovery.local_skeleton_discovery(
            np_data,
            f_node,
            alpha,
            indep_test=CI_TEST,
            mi=mi,
            labels=labels,
            verbose=verbose,
        )
    else:
        cg = SkeletonDiscovery.skeleton_discovery(
            np_data,
            alpha,
            indep_test=CI_TEST,
            background_knowledge=None,
            stable=False,
            verbose=verbose,
            labels=labels,
            show_progress=False,
        )

    cg.to_nx_graph()
    return cg


def get_fnode_child(G):
    return [*G.successors(F_NODE)]


def save_graph(graph, file):
    nx.draw_networkx(graph)
    plt.savefig(file)


def pc_with_fnode(normal_df, anomalous_df, alpha, bins=None, localized=False, verbose=VERBOSE):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
    cg = run_pc(data, alpha, localized=localized, verbose=verbose)
    return cg.nx_graph


# Equivelant to \Psi-PC from the main paper
def run_psi_pc(
    normal_df,
    anomalous_df,
    bins=None,
    mi=None,  # TODO: this is just wrong, refactor it
    localized=False,
    start_alpha=None,
    min_nodes=-1,
    verbose=VERBOSE,
):
    """
    Run \Psi-PC on the given dataset.
    The last column of the data *must* be the F-node

    Parameters
    ----------
    normal_df: pd.DataFrame
        Normal data
    anomalous_df: pd.DataFrame
        Anomalous data
    bins: int
        Number of bins to use for discretization
    mi: list
        List of tuples of mutual information
    localized: bool
        Whether to use localized PC
    start_alpha: float
        Starting alpha value
    min_nodes: int
        Minimum number of nodes to order
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    # TODO: refactor this
    """

    if mi is None:
        mi = []
    if 0 in [len(normal_df.columns), len(anomalous_df.columns)]:
        return ([], None, [], 0)
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)

    if min_nodes == -1:
        # Order all nodes (if possible) except F-node
        min_nodes = len(data.columns) - 1
    assert min_nodes < len(data)

    G = None
    no_ci = 0
    i_to_labels = {i: name for i, name in enumerate(data.columns)}
    labels_to_i = {name: i for i, name in enumerate(data.columns)}

    _preprocess_mi = lambda l: [labels_to_i.get(i) for i in l]  # noqa
    _postprocess_mi = lambda l: [i_to_labels.get(i) for i in list(filter(None, l))]  # noqa
    processed_mi = _preprocess_mi(mi)
    _run_pc = lambda alpha: run_pc(
        data,
        alpha,
        localized=localized,
        mi=processed_mi,
        labels=i_to_labels,
        verbose=verbose,
    )

    rc = []
    _alpha = START_ALPHA if start_alpha is None else start_alpha
    for i in np.arange(_alpha, ALPHA_LIMIT, ALPHA_STEP):
        cg = _run_pc(i)
        G = cg.nx_graph
        no_ci += cg.no_ci_tests

        if G is None:
            continue

        f_neigh = get_fnode_child(G)
        new_neigh = [x for x in f_neigh if x not in rc]
        if len(new_neigh) == 0:
            continue
        else:
            f_p_values = cg.p_values[-1][[labels_to_i.get(key) for key in new_neigh]]
            rc += _order_neighbors(new_neigh, f_p_values)

        if len(rc) == min_nodes:
            break

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

_rm_time = lambda df: df.loc[:, ~df.columns.isin(["time"])]
_list_intersection = lambda l1, l2: [x for x in l1 if x in l2]


def _preprocess_for_fnode(normal_df, anomalous_df, bins):
    df = add_fnode_and_concat(normal_df, anomalous_df)
    if df is None:
        return None

    return _discretize(df, bins) if bins is not None else df


def _select_useful_cols(df):
    i = df.loc[:, df.columns != F_NODE].std() > 1
    cols = i[i].index.tolist()
    cols.append(F_NODE)
    if len(cols) == 1:
        return None
    elif len(cols) == len(df.columns):
        return df

    print(f"Drop {len(df.columns) - len(cols)} columns, left with {len(cols)}")

    return df[cols]


# Only select the metrics that are in both datasets
def _match_columns(n_df, a_df):
    cols = _list_intersection(n_df.columns, a_df.columns)
    return (n_df[cols], a_df[cols])


# Convert all memeory columns to MBs
def _scale_down_mem(df):
    def update_mem(x):
        if not x.name.endswith("_mem"):
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
    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
    discretizer.fit(d)
    disc_d = discretizer.transform(d)
    disc_d = pd.DataFrame(disc_d, columns=d.columns.values.tolist())
    disc_d[F_NODE] = data[F_NODE].tolist()

    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)

    return disc_d


# =========== UTILS.py ====================

# np.random.seed(0)

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will produce a sparse graph
# so we might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give dense graph so the size of the separating set will increase and phase-1 will
# take more time.
# We tried a few different values and found that 0.01 gives the best result in our case
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5


# Split the dataset into multiple subsets
def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma : (i * gamma) + gamma])

    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks


def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose):
    """
    Run phase-1 of RCD algorithm

    Parameters
    ----------
    normal_df : pandas.DataFrame
        Normal data
    anomalous_df : pandas.DataFrame
        Anomalous data
    gamma : int
        Number of nodes in each subset
    localized : bool
        Run localized version of PSI-PC
    bins : int
        Number of bins
    verbose : bool
        Verbose mode

    Returns
    -------
    f_child_union : list
        List of nodes in the separating set
    mi_union : list
        List of mutual information values
    ci_tests : int
        Number of conditional independence tests
    """
    ci_tests = 0
    chunks = create_chunks(normal_df, gamma)
    if verbose:
        print(f"Created {len(chunks)} subsets")

    f_child_union = []
    mi_union = []
    f_child = []
    for c in chunks:
        # Try this segment with multiple values of alpha until we find at least one node
        rc, _, mi, ci = run_psi_pc(
            normal_df.loc[:, c],
            anomalous_df.loc[:, c],
            bins=bins,
            localized=localized,
            start_alpha=LOCAL_ALPHA,
            min_nodes=1,
            verbose=verbose,
        )
        f_child_union += rc
        mi_union += mi
        ci_tests += ci
        if verbose:
            f_child.append(rc)

    if verbose:
        print(f"Output of individual chunk {f_child}")
        print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

    return f_child_union, mi_union, ci_tests


def run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose):
    """
    Run RCD algorithm with two phases (phase-1 and phase-2) to find the root causes of the anomaly in the data.

    Parameters
    ----------
    normal_df : pandas.DataFrame
        Normal data
    anomalous_df : pandas.DataFrame
        Anomalous data
    gamma : int
        Number of nodes in each subset
    localized : bool
        Run localized version of PSI-PC
    bins : int
        Number of bins for discretization
    verbose : bool
        Verbose mode


    Returns
    -------
    rc : list
        List of root causes
    """
    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    while True:
        start = time.time()
        f_child_union, mi, ci_tests = run_level(
            normal_df.loc[:, f_child_union],
            anomalous_df.loc[:, f_child_union],
            gamma,
            localized,
            bins,
            verbose,
        )
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        # Phase-1 with only one level
        # break

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        if len_child <= gamma or len_child == prev:
            break
        prev = len(f_child_union)

    # Phase-2
    mi_union = []
    new_nodes = f_child_union
    rc, _, mi, ci = run_psi_pc(
        normal_df.loc[:, new_nodes],
        anomalous_df.loc[:, new_nodes],
        bins=bins,
        mi=mi_union,
        localized=localized,
        verbose=verbose,
    )
    ci_tests += ci

    # return rc, ci_tests
    return rc

def rcd_helper(normal_df,anomal_df,gamma=5,localized=True,bins=5, verbose=False):
    rc = run_multi_phase(normal_df, anomal_df, gamma, localized, bins, verbose)
    # return rc
    return {
        "ranks": rc,
    }

def rcd(
    data,
    inject_time,
    dk_select_useful=False,
    gamma=5,
    localized=True,
    bins=5,
    verbose=False,
    dataset=None,
    seed=None,
    **kwargs,
):
    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]

    if dk_select_useful is True:
        normal_df = drop_extra(normal_df)
        anomal_df = drop_extra(anomal_df)

    # if dataset == real outages:
    if dataset == "sock-shop":
        normal_df, anomal_df = preprocess_sock_shop(normal_df, anomal_df, 90, dk_select_useful)
    elif dataset is not None:
        from RCAEval.io.time_series import convert_mem_mb, drop_constant, drop_time, preprocess

        normal_df = drop_constant(convert_mem_mb(drop_time(normal_df)))
        anomal_df = drop_constant(convert_mem_mb(drop_time(anomal_df)))

        normal_df, anomal_df = _match_columns(normal_df, anomal_df)

        df = add_fnode_and_concat(normal_df, anomal_df)
        if dk_select_useful is True:
            df = _select_useful_cols(df)

        normal_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
        anomal_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    if seed is not None:
        np.random.seed(seed)

    # return rc
    return rcd_helper(normal_df,anomal_df, gamma, localized, bins, verbose)
