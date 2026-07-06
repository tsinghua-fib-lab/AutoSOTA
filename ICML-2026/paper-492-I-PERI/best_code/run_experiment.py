import sys
import os
import csv
import numpy as np

# Add repo paths
sys.path.insert(0, '/repo/reproducibility/icml2026')
sys.path.insert(0, '/repo/src')

from pyciphod.causal_discovery.federated.regret_based.iperi.iperi import IPeri
from pyciphod.causal_discovery.federated.regret_based.iperi.client import Client
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
from reproducibility.icml2026.dataset import Dataset
import pyciphod.causal_discovery.federated.regret_based.ges as ges
import networkx as nx
from tqdm import tqdm

def run_experiment(seed, n_clients=4, n_samples_client=1000, n_variables=3, data_type='struct',
                   cd_function='pc', scoring_function='bic', linear=True, masked=True,
                   noise_distribution='normal', horizontal_split=False, max_iters=1):
    utils.set_determine(seed)

    dataset = Dataset(
        n_samples_client=n_samples_client,
        n_clients=n_clients,
        n_variables=n_variables,
        linear=linear,
        horizontal_split=horizontal_split,
        noise_distribution=noise_distribution,
        seed=seed,
    )

    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type=data_type, save=False)
    graph = dataset.graph

    clients = []
    client_graphs = []
    error_client_graphs = []

    for i in range(n_clients):
        # datasets[i] is a pandas DataFrame
        clients.append(Client(
            name='client_{}'.format(i),
            data=datasets[i],  # Pass DataFrame directly, not .values
            cd_function=cd_function,
            scoring_function=scoring_function,
            masked=masked,
            linear=linear
        ))
        client_graphs.append(clients[-1].graph)
        ground_graph = cpdag if data_type != 'struct' else ges.utils.dag_to_cpdag(graphs[i])
        client_graph = clients[-1].graph
        error_client_graphs.append(
            utils.f1_orientation(client_graph, ground_graph)
        )

    i_peri = IPeri(n_variables=len(graph.nodes), clients=clients)
    union_graph = utils.union_graph(nx.to_numpy_array(graph), client_graphs)
    server_graph, estimated_cpdag = i_peri.fit(max_iters=max_iters)

    return {
        'seed': seed,
        'shd': utils.shd(server_graph, nx.to_numpy_array(graph)),
        'f1': utils.f1_orientation(server_graph, nx.to_numpy_array(graph)),
        'shd_cpdag': utils.shd(estimated_cpdag, nx.to_numpy_array(graph)),
        'f1_cpdag': utils.f1_orientation(estimated_cpdag, nx.to_numpy_array(graph)),
        'error_client': np.mean(error_client_graphs),
    }

if __name__ == '__main__':
    # Run 10 seeds with a single configuration
    results = []

    for seed in tqdm(range(10), desc="Running seeds"):
        try:
            r = run_experiment(
                seed=seed,
                n_clients=4,
                n_samples_client=1000,
                n_variables=3,
                data_type='struct',
                cd_function='pc',
                scoring_function='bic',
                linear=True,
                masked=True,
                noise_distribution='normal',
                horizontal_split=False,
            )
            results.append(r)
            print("Seed {}: SHD={}, F1={:.4f}, client_F1={:.4f}".format(
                seed, r['shd'], r['f1'], r['error_client']))
        except Exception as e:
            import traceback
            print("Seed {} FAILED: {}".format(seed, e))
            traceback.print_exc()

    print()
    print("=== Results ===")
    if results:
        shds = [r['shd'] for r in results]
        f1s = [r['f1'] for r in results]
        print("SHD: {:.2f} +/- {:.2f}".format(np.mean(shds), np.std(shds)))
        print("F1:  {:.2f} +/- {:.2f}".format(np.mean(f1s), np.std(f1s)))
        print("Raw SHD: {}".format(shds))
        print("Raw F1:  {}".format([round(x, 4) for x in f1s]))
