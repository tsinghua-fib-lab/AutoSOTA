import os
import time
import pickle
import datetime
import warnings

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from causallearn.graph.GeneralGraph import GeneralGraph


ANOMALOUS_NODE = 'a_node'
GRAPH_GEN_INFO = 'gen_graph_info.pkl'
NORMAL_DATA = 'normal.csv'
ANOMALOUS_DATA = 'anomalous.csv'
GROUND_TRUTH_NX_GRAPH = 'g_bn_graph.pkl'
GROUND_TRUTH_PDF = 'ground-truth.pdf'
NORMAL_BN = 'normal.bif'
ANOMALOUS_BN = 'anomalous.bif'
F_NODE = 'F-node'

BINS = 5
VERBOSE = False

current_time = lambda: time.perf_counter()


def readable_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def get_node_name(node):
    return f'X{node}'

def get_nodes_dir_name(nodes):
    return f'{nodes}-nodes'

def get_prior_graph_name(k, oracle):
    if oracle and k == -1: return 'oracle-cpdag'
    return f"{'oracle' if oracle else 'sample'}-{k}"

def load_datasets(path, verbose=VERBOSE):
    if verbose:
        print(f'Loading the dataset from {path}...')
    normal_df = pd.read_csv(f'{path}/{NORMAL_DATA}')
    anomalous_df = pd.read_csv(f'{path}/{ANOMALOUS_DATA}')
    return (normal_df, anomalous_df)

def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph
load_data = load_graph

# Iterate over all the directories in the given path that have the format 
# {int}-{str}. A directory that does not start with a leading integer will
# be ignored. The yielded results are sorted by the leading integer.
def dir_iterator(path):
    sorted_dirs = list()
    for f in os.listdir(path):
        try:
            _int, _str = f.split('-')
            sorted_dirs.append(f)
        except:
            pass
    sorted_dirs = sorted(sorted_dirs, key=lambda s: int(s.split('-')[0]))
    for f in sorted_dirs:
        p_path = f'{path}/{f}'
        if not os.path.isdir(p_path): continue
        yield(f.split('-')[0], p_path)

def add_fnode(normal_df, anomalous_df):
    normal_df[F_NODE] = 0
    anomalous_df[F_NODE] = 1
    return pd.concat([normal_df, anomalous_df])

def store_causal_learn_graph(G: GeneralGraph, k: int, path: str):
    with open(path, 'wb') as f:
        pickle.dump({'graph': G, 'k':k}, f)

# Used for sock-shop

def _select_cols(df, keep_cols=[]):
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
    return df

def discretize(data, bins=BINS):
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discretizer.fit(data)
    disc_d = discretizer.transform(data)
    disc_d = pd.DataFrame(disc_d, columns=data.columns)
    disc_d = disc_d.astype(int)
    return disc_d
