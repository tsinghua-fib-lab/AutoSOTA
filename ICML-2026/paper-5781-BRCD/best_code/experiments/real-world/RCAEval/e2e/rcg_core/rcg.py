import copy
import time
import argparse
from typing import List

import networkx as nx
from causallearn.graph.Edge import Edge
from causallearn.graph.Graph import Graph
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph

from RCAEval.e2e.rcg_core.utils import mutual_info as mt
from RCAEval.e2e.rcg_core.utils import base_utils as bu

L = 3
current_time = lambda: time.perf_counter() * 1e3


class Scorer:
    def __init__(self, df):
        self.df = df # Assumes that F-node is already added
        self._cache = dict()

    def get_score(self, y, z=set()):
        xz_key = (y, frozenset(z))
        if xz_key in self._cache:
            return self._cache[xz_key]

        # Marginal score
        if len(z) == 0:
            score = mt.mutual_information(self.df, bu.F_NODE, y)
        else:
            score = mt.conditional_mutual_information(self.df, bu.F_NODE, y, z)
        self._cache[xz_key] = score
        return self._cache[xz_key]

    def compute_mi(self):
        scores = []
        for y in self.df.columns[:-1]:
            scores.append((y, self.get_score(y)))
        return scores

    def compute_cmi_on_possible_parents(self, graph: GeneralGraph):
        scores = []
        for y in self.df.columns[:-1]:
            poss_pa = [x.name for x in find_possible_parents(graph, graph.get_node(y))]
            scores.append((y, self.get_score(y, z=poss_pa)))
        return scores


def update_edges(edges: List[Edge], graph: GeneralGraph):
    for old_edge, new_edge in edges:
        graph.remove_edge(old_edge)
        if new_edge:
            graph.add_edge(new_edge)

def is_possible_parent(graph: Graph, potential_parent_node, child_node):
    """
    Test if a node can possibly serve as parent of the given node.
    Make sure that on the connecting edge
        (a) there is no head edge-mark (->) at the tested node and
        (b) there is no tail edge-mark (--) at the given node,
    where variant edge-marks (o) are allowed.
    :param potential_parent_node: the node that is being tested
    :param child_node: the node that serves as the child
    :return:
     """
    if graph.node_map[potential_parent_node] == graph.node_map[child_node]:
        return False
    if not graph.is_adjacent_to(potential_parent_node, child_node):
        return False

    if graph.get_endpoint(child_node, potential_parent_node) == Endpoint.ARROW:
        return False
    else:
        return True

def find_possible_parents(graph: Graph, child_node, en_nodes=None):
    if en_nodes is None:
        nodes = graph.get_nodes()
        en_nodes = [node for node in nodes if graph.node_map[node] != graph.node_map[child_node]]

    possible_parents = [parent_node for parent_node in en_nodes if is_possible_parent(graph, parent_node, child_node)]
    return possible_parents


def _local_run(df, graph: GeneralGraph, l):
    scorer = Scorer(df)
    best_ranking = list()
    alphas = sorted(set([x[1] for x in scorer.compute_mi()]))
    for _alpha in alphas:
        G = copy.deepcopy(graph)

        new_edges = []
        for x in G.get_nodes():
            for y in G.get_adjacent_nodes(x):
                fx = scorer.get_score(x.name)
                fy = scorer.get_score(y.name)
                if fx < _alpha and _alpha <= fy:
                    old_edge = G.get_edge(x, y)
                    new_edge = None
                    if G.is_undirected_from_to(x, y):
                        # Orient X - Y to X -> Y
                        new_edge = Edge(x, y, Endpoint.TAIL, Endpoint.ARROW)
                    new_edges.append((old_edge, new_edge))
        update_edges(new_edges, G)

        cmi = scorer.compute_cmi_on_possible_parents(G)
        sorted_cmi = [x[0] for x in sorted(cmi, key=lambda t: t[1], reverse=True)]

        # Consistency check
        for x in sorted_cmi[:l]:
            if scorer.get_score(x) < _alpha:
                return best_ranking
        best_ranking = sorted_cmi
    return best_ranking

def rank_variables(n_df, a_df, graph: GeneralGraph, l):
    df = bu.add_fnode(n_df, a_df)
    start = current_time()
    result = _local_run(df, graph, l)
    end = current_time() - start
    return {'time': end, 'root_cause': result, 'tests': n_df.shape[1] - 1}

def _nx_to_g_graph(graph: nx.DiGraph) -> GeneralGraph:
    from causallearn.graph.Edge import Edge
    from causallearn.graph.GraphNode import GraphNode

    _nodes = [GraphNode(x) for x in graph.nodes]
    G = GeneralGraph(_nodes)
    for u, v in graph.edges():
        _u = G.get_node(u)
        _v = G.get_node(v)
        G.add_edge(Edge(_u, _v, Endpoint.TAIL, Endpoint.ARROW))
    return G

def run(n_df, a_df, src_dir, l, k=-1, oracle=False, dag=False):
    if dag:
        dag: nx.DiGraph = bu.load_graph(f'{src_dir}/{bu.GROUND_TRUTH_NX_GRAPH}')
        G = _nx_to_g_graph(dag)
    else:
        G = bu.load_graph(f'{src_dir}/{bu.get_prior_graph_name(k, oracle)}')['graph']
    return rank_variables(n_df, a_df, G, l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run mutual information on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    path = args.path

    n_df, a_df = bu.load_datasets(path)
    result = run(n_df, a_df, path, L, oracle=True)
    print(result)
