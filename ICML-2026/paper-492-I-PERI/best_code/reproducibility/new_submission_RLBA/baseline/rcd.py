#!/usr/bin/env python3

import time
import argparse

import numpy as np

from . import utils_rcd as u

VERBOSE = False

BINS = 5
K = None

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will produce a sparse graph
# so we might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give dense graph so the size of the separating set will increase and phase-1 will
# take more time.
# We tried a few different values and found that 0.01 gives the best result in our case
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5

# SRC_DIR = 'sock-shop-data/carts-mem/1/'
SRC_DIR = 'data/s-2/n-10-d-3-an-1-nor-s-1000-an-s-1000/'

# Split the dataset into multiple subsets
def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma:(i * gamma) + gamma])

    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks

def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose):
    ci_tests = 0
    chunks = create_chunks(normal_df, gamma)
    if verbose:
        print(f"Created {len(chunks)} subsets")

    f_child_union = list()
    mi_union = list()
    f_child = list()
    for c in chunks:
        # Try this segment with multiple values of alpha until we find at least one node
        rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, c],
                                   anomalous_df.loc[:, c],
                                   bins=bins,
                                   localized=localized,
                                   start_alpha=LOCAL_ALPHA,
                                   min_nodes=1,
                                   verbose=verbose)
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
    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    while True:
        start = time.time()
        f_child_union, mi, ci_tests = run_level(normal_df.loc[:, f_child_union],
                                                anomalous_df.loc[:, f_child_union],
                                                gamma, localized, bins, verbose)
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        # Phase-1 with only one level
        # break

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        if len_child <= gamma or len_child == prev: break
        prev = len(f_child_union)

    # Phase-2
    mi_union = []
    new_nodes = f_child_union
    rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, new_nodes],
                               anomalous_df.loc[:, new_nodes],
                               bins=bins,
                               mi=mi_union,
                               localized=localized,
                               verbose=verbose)
    ci_tests += ci

    return rc, ci_tests

def rca_with_rcd(normal_df, anomalous_df, bins,
                 gamma=DEFAULT_GAMMA, localized=False, verbose=VERBOSE):
    start = time.time()
    rc, ci_tests = run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose)
    end = time.time()

    return {'time': end - start, 'root_cause': rc, 'ci_tests': ci_tests}

def top_k_rc(normal_df, anomalous_df, k, bins,
             gamma=DEFAULT_GAMMA, localized=False, verbose=VERBOSE):
    result = rca_with_rcd(normal_df, anomalous_df, bins, gamma, localized, verbose)
    return {**result, 'root_cause': result['root_cause'][:k]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PC on the given dataset')

    parser.add_argument('--path', type=str, default=SRC_DIR,
                        help='Path to the experiment data')
    parser.add_argument('--k', type=int, default=K,
                        help='Top-k root causes')
    parser.add_argument('--local', action='store_true',
                        help='Run localized version to only learn the neighborhood of F-node')

    args = parser.parse_args()
    path = args.path
    k = args.k
    local = args.local
    (normal_df, anomalous_df) = u.load_datasets(path + 'normal.csv',
                                                path + 'anomalous.csv')

    # Enable the following line for sock-shop or real outage dataset
    # normal_df, anomalous_df = u.preprocess(normal_df, anomalous_df, 90)

    result = top_k_rc(normal_df, anomalous_df, k=k, bins=BINS, localized=local)
    print(f"Top {k} took {round(result['time'], 4)} and potential root causes are {result['root_cause']}")
