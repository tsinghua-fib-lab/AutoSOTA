import time
import argparse

import networkx as nx

from utils import base_utils as bu

current_time = lambda: time.perf_counter() * 1e3


def rank_variables(path):
    G: nx.DiGraph = bu.load_graph(f'{path}/{bu.GROUND_TRUTH_NX_GRAPH}')

    start = current_time()
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    personalization = {}
    for node in G.nodes():
        if node in dangling_nodes:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.5
    pagerank = nx.pagerank(G, personalization=personalization)
    sorted_nodes = dict(sorted(pagerank.items(), key=lambda x: x[1], reverse=True))
    sorted_nodes = list(sorted_nodes.keys())
    end = current_time() - start
    return {'time': end, 'root_cause': sorted_nodes, 'tests': G.number_of_nodes()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run mutual information on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    path = args.path

    result = rank_variables(path)
    print(result)
