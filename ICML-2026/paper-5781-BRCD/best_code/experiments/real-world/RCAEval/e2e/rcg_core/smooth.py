import time
import argparse

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2_contingency

from utils import base_utils as bu

current_time = lambda: time.perf_counter() * 1e3


def smooth_traversel(nxGraph, combined_df):
    # traverse each node in the graph
    # for a combined dataframe, compute chi-sq value between node i and F-NODE
    scores = {}
    for node in nxGraph.nodes():
        contingency_table = pd.crosstab(combined_df[node], combined_df[bu.F_NODE])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        scores[node] = chi2

    max_score_tracker = {}
    for node in nxGraph.nodes():
        parents = list(nxGraph.predecessors(node))
        if parents:
            current_max = np.max([(scores[node] - scores[parent]) for parent in parents])
        else:
            current_max = scores[node]
        max_score_tracker[node] = current_max
    return sorted(max_score_tracker, key=lambda x: max_score_tracker[x], reverse=True)

def rank_variables(n_df, a_df, path):
    df = bu.add_fnode(n_df, a_df)
    dag: nx.DiGraph = bu.load_graph(f'{path}/{bu.GROUND_TRUTH_NX_GRAPH}')
    start = current_time()
    result = smooth_traversel(dag, df)
    end = current_time() - start
    return {'time': end, 'root_cause': result, 'tests': n_df.shape[1] - 1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SMOOTH TRAVERSAL on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    path = args.path

    n_df, a_df = bu.load_datasets(path)
    result = rank_variables(n_df, a_df, path=path)

    print(result)
