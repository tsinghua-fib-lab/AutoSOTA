import time
import argparse

import numpy as np

from causallearn.utils.cit import chisq, fisherz

import rcd_utils as u

VERBOSE = False

ORACLE = False
BINS = None # only used for sockshop when the data is continues
K = 1
SEED = 7336

CI_TEST = chisq

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will give you sparse graph
# so you might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give you dense graph so the size of the separating set will increase and phase-1 will
# take more time. I tried with a few values and found that 0.01 gives the best result
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5

current_time = lambda: time.perf_counter() * 1e3


def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma:(i * gamma) + gamma])

    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks

def run_level(normal_df, anomalous_df, gamma, localized, bins, cg_opts, verbose):
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
                                   cg_opts=cg_opts,
                                   ci_test=CI_TEST,
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

def run_multi_phase(normal_df, anomalous_df, path, gamma, localized, bins, oracle, verbose):
    cg_opts = dict()
    if oracle:
        cg_opts = {'oracle': True, 'true_nx_graph': u.add_fnode_to_graph(path)}
    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)
    counter = 1
    ci_tests = 0

    # Phase-1
    while True:
        start = current_time()
        f_child_union, mi, ci = run_level(normal_df.loc[:, f_child_union],
                                                anomalous_df.loc[:, f_child_union],
                                                gamma * counter, localized, bins, cg_opts, verbose)
        end = current_time() - start
        ci_tests += ci
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {end}")
        i += 1
        mi_union += mi

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level didn't remove any node
        if len_child == prev:
            counter += 1
            if counter == 5: break
            continue

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
                               cg_opts=cg_opts,
                               ci_test=CI_TEST,
                               verbose=verbose)
    ci_tests += ci
    return rc, ci_tests

def rca_with_rcd(normal_df, anomalous_df, path, bins,
                 gamma=DEFAULT_GAMMA, localized=True, oracle=False,
                 verbose=VERBOSE):
    start = current_time()
    rc, ci_tests = run_multi_phase(normal_df, anomalous_df, path, gamma, localized, bins, oracle, verbose)
    end = current_time() - start
    return {'time': end, 'root_cause': rc, 'tests': ci_tests}

def top_k_rc(normal_df, anomalous_df, path, k, bins,
             gamma=DEFAULT_GAMMA, localized=True, oracle=False,
             seed=SEED, verbose=VERBOSE):
    np.random.seed(seed)
    result = rca_with_rcd(normal_df, anomalous_df, path, bins, gamma, localized, oracle, verbose)
    return {**result, 'root_cause': result['root_cause'][:k]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PC on the given dataset')

    parser.add_argument('--path', type=str,
                        help='Path to the experiment data')
    parser.add_argument('--k', type=int, default=K,
                        help='Top-k root causes')
    parser.add_argument('--local', action='store_true',
                        help='Run localized version to only learn the neighborhood of F-node')
    args = parser.parse_args()
    path = args.path
    k = args.k
    local = args.local

    (normal_df, anomalous_df) = u.load_datasets(f'{path}/normal.csv', f'{path}/anomalous.csv')
    result = top_k_rc(normal_df, anomalous_df, path, seed=SEED, k=k, bins=BINS, localized=local, oracle=ORACLE)
    print(f"Top {k} took {round(result['time'], 4)} and points to {result['root_cause']} with {result['tests']} tests")
