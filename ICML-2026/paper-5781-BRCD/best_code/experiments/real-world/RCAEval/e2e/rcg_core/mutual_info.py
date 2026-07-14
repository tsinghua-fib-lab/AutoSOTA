import time
import argparse
import pandas as pd

from utils import mutual_info as mt
from utils import base_utils as bu

current_time = lambda: time.perf_counter() * 1e3


def rank_variables(n_df, a_df):
    df = bu.add_fnode(n_df, a_df)
    start = current_time()
    scores = list()
    for c in df.columns[:-1]:
        _s = mt.mutual_information(df, c, bu.F_NODE)
        scores.append((c, _s))
    sorted_nodes = sorted(scores, key=lambda t: t[1], reverse=True)
    sorted_nodes = [x[0] for x in sorted_nodes]
    end = current_time() - start
    return {'time': end, 'root_cause': sorted_nodes, 'tests': n_df.shape[1] - 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run mutual information on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    path = args.path

    n_df, a_df = bu.load_datasets(path)
    result = rank_variables(n_df, a_df)
    print(result)
