import time
import argparse

import pickle
import numpy as np
import pandas as pd
import networkx as nx

from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

from models.rcg.para_kpc.kPC_fas import kpc
from models.rcg.utils import base_utils as bu

K = -1 # -1 for CPDAG
ORACLE = True
CI_TEST = 'chisq'
ALPHA = 0.05

CORES = 1

current_time = lambda: time.perf_counter() * 1e3


def _learn_true_cpdag(path):
    nx_graph: nx.DiGraph = bu.load_graph(f'{path}/{bu.GROUND_TRUTH_NX_GRAPH}')
    node_map = {x: GraphNode(x) for x in nx_graph.nodes}
    dag: Dag = Dag(list(node_map.values()))
    for u, v in nx_graph.edges():
        dag.add_edge(Edge(node_map[u], node_map[v], Endpoint.TAIL, Endpoint.ARROW))
    G = dag2cpdag(dag)

    cg_path = f'{path}/{bu.get_prior_graph_name(-1, True)}'
    bu.store_causal_learn_graph(G, -1, cg_path)
    return G

def learn(path, df=None, k=0, oracle=False, store=True):
    if oracle and k == -1:
        return _learn_true_cpdag(path)

    if oracle:
        nx_graph: nx.DiGraph = bu.load_graph(f'{path}/{bu.GROUND_TRUTH_NX_GRAPH}')
        node_names = list(nx_graph.nodes)
        G, _ = kpc(np.array([[]]), independence_test_method='d_separation',
                   true_dag=nx_graph, k=k, n=len(node_names),
                   node_names=node_names, parallel=True, s=None,
                   batch=None, p_cores=CORES)
    else:
        if df is None:
            df = pd.read_csv(f'{path}/{bu.NORMAL_DATA}')
        G, _ = kpc(df.to_numpy(), independence_test_method=CI_TEST,
                   n=len(df.columns), alpha=ALPHA, k=k,
                   node_names=df.columns.tolist(),
                   parallel=True, s=None, batch=None, p_cores=CORES)
    if store:
        cg_path = f'{path}/{bu.get_prior_graph_name(k, oracle)}'
        bu.store_causal_learn_graph(G, k, cg_path)
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates CPDAG from a dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the experiment data')
    args = parser.parse_args()
    path = args.path

    s_time = current_time()
    learn(path, k=K, oracle=ORACLE, store=True)
    e_time = current_time()
    print(f'Learning the essential graph took {e_time - s_time}')
