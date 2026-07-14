import time
import argparse
from collections import deque

import networkx as nx

from causallearn.utils.cit import CIT, chisq
from utils import base_utils as bu

L = 5
ALPHA = 0.05
CI_TEST = chisq
DUMMY_SINK_NODE = 'dummy_sink'

current_time = lambda: time.perf_counter() * 1e3


class CITestWithCache:
    def __init__(self):
        self.test_count = 0

    def is_dependent(self, node):
        raise NotImplemented()


class OracleCITest(CITestWithCache):
    def __init__(self, dag, true_rc):
        super().__init__()
        self.nx_graph = dag
        self.true_rc = true_rc
        self.cache = {DUMMY_SINK_NODE: True} # dummy sink node is always reachable/dependent

    def is_dependent(self, node):
        if node in self.cache:
            result = self.cache[node]
        else:
            self.test_count += 1
            result = self.run_ci_test(node)
            self.cache[node] = result
        return result

    def run_ci_test(self, node):
        return self.true_rc == node or self.true_rc in nx.descendants(self.nx_graph, node)


class DataCITest(CITestWithCache):
    def __init__(self, n_df, a_df, alpha, cache={}):
        super().__init__()
        self.alpha = alpha
        data = bu.add_fnode(n_df, a_df)
        self.cit = CIT(data, CI_TEST)
        self.node_to_i = {col: i for i, col in enumerate(data.columns)}
        self.cache = {DUMMY_SINK_NODE: 0} # dummy sink node is always reachable/dependent
        self.cache.update(cache)
        self.update_list = list()

    def is_dependent(self, node):
        if node in self.cache:
            pval = self.cache[node]
        else:
            self.test_count += 1
            pval = self.run_ci_test(node)
            self.cache[node] = pval
        return pval <= self.alpha

    def run_ci_test(self, node):
        return self.cit(self.node_to_i[node], -1)

    # NOTE: This is a special function and must be used with caution.
    # Iterate over all the items in the cache and picks an item that is
    # closest to self.alpha and changes its value such that now CI test
    # would return the different result.
    # Example: cache = [(X0, 0), (X1, 0.04), (X2, 1), (X3, 0.07), (X5, 0.7)]
    # alpha = 0.05
    # udpate_cache will pick X1 which is the closest and set its value to 0.051
    # Example cache = [(X0, 0), (X1, 0.03), (X2, 1), (X3, 0.06), (X5, 0.7)]
    # udpate_cache will pick X3 which is the closest and set its value to 0.05
    def update_cache(self):
        _cache = [key for key, _ in self.cache.items() if key != DUMMY_SINK_NODE and key not in self.update_list]
        # Don't do anything if all the values in cache have been changed
        if (len(_cache)) == 0:
            return

        x = min(_cache, key=lambda key: abs(self.cache[key] - self.alpha))
        self.update_list.append(x)

        _temp = self.cache[x]
        # self.cache = dict()
        self.cache[x] = self.alpha + 0.001 if _temp <= self.alpha else self.alpha


def _get_subtree_size(dag: nx.DiGraph, root, visited):
    size = 1
    visited |= {root}
    for c in set(dag.successors(root)) - visited:
        if c in visited: continue
        _s = _get_subtree_size(dag, c, visited)
        size += _s
    return size

def _get_heavy_children(dag: nx.DiGraph, root):
    heavy_children = dict({_s: [] for _s in dag.nodes})
    visited = set()
    stack = deque()
    stack.append(root)
    visited |= {root}
    while len(stack) > 0:
        r = stack[-1]
        not_visited_children = set(dag.successors(r)) - visited
        if len(not_visited_children) == 0:
            stack.pop()
            continue

        max_size = -1
        c_with_max_size = None
        for c in not_visited_children:
            size = _get_subtree_size(dag, c, visited.copy())
            if size > max_size:
                c_with_max_size = c
                max_size = size
        visited |= {c_with_max_size}
        stack.append(c_with_max_size)
        heavy_children[r].append(c_with_max_size)
    return heavy_children

def _get_heaviest_path(paths, root):
    path = list([root])
    if len(paths[root]) == 0: return path
    return path + _get_heaviest_path(paths, paths[root][0])

# Performs ci tests on the nodes in the given list using binary search.
# Returns the last node in the list that is dependent. Any node in the list
# after that must be independent.
# Assumes that the first node in the list must be dependent.
def _binary_search_ci(nodes, ci_tester):
    low, high = 0, len(nodes) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if ci_tester.is_dependent(nodes[mid]):
            low = mid
        else:
            high = mid - 1
    return nodes[low]

# Performs ci tests on the nodes in the given list and returns the first node
# that is dependent. None otherwise.
def _sequential_search_ci(nodes, ci_tester):
    if len(nodes) == 0: return None
    for w in nodes:
        if ci_tester.is_dependent(w): return w
    return None

def _run_igs(dag: nx.DiGraph, ci_tester: CITestWithCache):
    source_nodes = [node for node in dag.nodes if dag.in_degree(node) == 0]
    assert len(source_nodes) == 1
    root = source_nodes[0]

    s_time = current_time()
    heavy_children = _get_heavy_children(dag, root)
    candidate = root
    while True:
        pi = _get_heaviest_path(heavy_children, candidate)
        u = _binary_search_ci(pi, ci_tester)
        w = _sequential_search_ci(heavy_children[u], ci_tester)
        if w is None:
            candidate = u
            break
        candidate = w
    e_time = current_time() - s_time

    return {'time': e_time, 'root_cause': [candidate], 'tests': ci_tester.test_count}

def _add_dummy_sink(dag: nx.DiGraph):
    sinks = [node for node in dag.nodes if dag.out_degree(node) == 0]
    if len(sinks) == 1: return dag
    dag.add_node(DUMMY_SINK_NODE)
    for _s in sinks:
        dag.add_edge(_s, DUMMY_SINK_NODE)
    return dag

def _run(n_df, a_df, dag, path=None, perfect_ci=False, max_l=1):
    dag = _add_dummy_sink(dag)
    r_dag = dag.reverse()
    if perfect_ci:
        print('Running with perfect CI')
        true_rc = bu.load_data(f'{path}/{bu.GRAPH_GEN_INFO}')[bu.ANOMALOUS_NODE]
        ci_tester = OracleCITest(r_dag, true_rc)
        return _run_igs(r_dag, ci_tester)

    ci_tester = DataCITest(n_df, a_df, ALPHA)
    result = {'time': 0, 'root_cause': [], 'tests': 0}
    for _ in range(max_l):
        local_r = _run_igs(r_dag, ci_tester)
        ci_tester.update_cache()
        result['root_cause'].append(local_r['root_cause'][0])
        result['tests'] += local_r['tests']
        result['time'] += local_r['time']
    return result

def run_algo(n_df, a_df, path, perfect_ci=False, max_l=1):
    dag: nx.DiGraph = bu.load_graph(f'{path}/{bu.GROUND_TRUTH_NX_GRAPH}')
    return _run(n_df, a_df, dag, path=path, perfect_ci=perfect_ci, max_l=max_l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run modified IGS on the given DAG to find the root cause')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--oracle', action='store_true', help='Use d-sep CI test')
    args = parser.parse_args()
    path = args.path
    oracle = args.oracle

    max_l = L
    if oracle and L > 1:
        print('With oracle, it is not possible to output more than one node; Setting L=1')
        max_l = 1

    n_df, a_df = bu.load_datasets(path)
    result = run_algo(n_df, a_df, path, perfect_ci=oracle, max_l=max_l)
    print(f"IGS found {result['root_cause']} to be the root cause in {result['time']} sec and with {result['tests']} CI tests")
