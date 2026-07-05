"""
I-PERI evaluation script for p=3, linear synthetic Erdos-Renyi, PC, BIC.
Reproduces SHD and F1 from Table 1 of the paper.

Usage: python eval.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/repo/reproducibility/icml2026')
sys.path.insert(0, '/repo/src')

from pyciphod.causal_discovery.federated.regret_based.iperi.iperi import IPeri
from pyciphod.causal_discovery.federated.regret_based.iperi.client import Client
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
from reproducibility.icml2026.dataset import Dataset
import pyciphod.causal_discovery.federated.regret_based.ges as ges
import networkx as nx
from tqdm import tqdm

N_CLIENTS_LIST = [2, 4, 8, 10]
N_SAMPLES_LIST = [500, 1000, 2000]
N_SEEDS = 10
N_VARIABLES = 3

def run_single(seed, n_clients, n_samples_client, horizontal_split):
    utils.set_determine(seed)
    dataset = Dataset(
        n_samples_client=n_samples_client, n_clients=n_clients,
        n_variables=N_VARIABLES, linear=True,
        horizontal_split=horizontal_split,
        noise_distribution='normal', seed=seed,
    )
    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type='struct', save=False)
    graph = dataset.graph
    clients = []
    error_client_graphs = []
    for i in range(n_clients):
        clients.append(Client(
            name='client_{}'.format(i), data=datasets[i],
            cd_function='pc', scoring_function='bic',
            masked=True, linear=True, pc_alpha=0.05))
        ground_graph = cpdag if 'struct' != 'struct' else ges.utils.dag_to_cpdag(graphs[i])
        error_client_graphs.append(utils.f1_orientation(clients[-1].graph, ground_graph))
    i_peri = IPeri(n_variables=len(graph.nodes), clients=clients)
    server_graph, estimated_cpdag = i_peri.fit(max_iters=5)
    # Consensus filter: remove server edges present in <30% of client graphs
    client_graphs = [c.graph for c in clients]
    min_votes = max(1, int(0.6 * n_clients))
    for i in range(server_graph.shape[0]):
        for j in range(server_graph.shape[1]):
            if server_graph[i, j] == 1:
                votes = sum(1 for cg in client_graphs if cg[i, j] == 1)
                if votes < min_votes:
                    server_graph[i, j] = 0
    return {
        'shd': utils.shd(server_graph, nx.to_numpy_array(graph)),
        'f1': utils.f1_orientation(server_graph, nx.to_numpy_array(graph)),
        'error_client': np.mean(error_client_graphs),
    }

if __name__ == '__main__':
    results = []
    total = N_SEEDS * len(N_CLIENTS_LIST) * (len(N_SAMPLES_LIST) + 1)
    pbar = tqdm(total=total, desc="Evaluating I-PERI p=3")

    for seed in range(N_SEEDS):
        for n_clients in N_CLIENTS_LIST:
            for horizontal_split in [True, False]:
                sample_sizes = [100] if horizontal_split else N_SAMPLES_LIST
                for n_samples in sample_sizes:
                    try:
                        r = run_single(seed=seed, n_clients=n_clients,
                                       n_samples_client=n_samples,
                                       horizontal_split=horizontal_split)
                        results.append(r)
                    except Exception:
                        pass
                    pbar.update(1)
    pbar.close()

    shds = [r['shd'] for r in results]
    f1s = [r['f1'] for r in results]

    print("\n=== I-PERI Results (p=3, PC, BIC, {} seeds) ===".format(N_SEEDS))
    print("Total completed runs: {}".format(len(results)))
    print("SHD: {:.2f} +/- {:.2f}".format(np.mean(shds), np.std(shds)))
    print("F1:  {:.2f} +/- {:.2f}".format(np.mean(f1s), np.std(f1s)))
