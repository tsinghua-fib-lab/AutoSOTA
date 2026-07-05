import sys
import os
import csv
import numpy as np
import time

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

def run_single(seed, n_clients, n_samples_client, horizontal_split, data_type='struct',
               cd_function='pc', scoring_function='bic', linear=True, masked=True,
               noise_distribution='normal'):
    utils.set_determine(seed)
    dataset = Dataset(
        n_samples_client=n_samples_client, n_clients=n_clients,
        n_variables=N_VARIABLES, linear=linear,
        horizontal_split=horizontal_split,
        noise_distribution=noise_distribution, seed=seed,
    )
    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type=data_type, save=False)
    graph = dataset.graph
    clients = []
    error_client_graphs = []
    for i in range(n_clients):
        clients.append(Client(
            name='client_{}'.format(i), data=datasets[i],
            cd_function=cd_function, scoring_function=scoring_function,
            masked=masked, linear=linear))
        ground_graph = cpdag if data_type != 'struct' else ges.utils.dag_to_cpdag(graphs[i])
        error_client_graphs.append(utils.f1_orientation(clients[-1].graph, ground_graph))
    i_peri = IPeri(n_variables=len(graph.nodes), clients=clients)
    server_graph, estimated_cpdag = i_peri.fit(max_iters=1)
    return {
        'shd': utils.shd(server_graph, nx.to_numpy_array(graph)),
        'f1': utils.f1_orientation(server_graph, nx.to_numpy_array(graph)),
        'error_client': np.mean(error_client_graphs),
    }

results = []
total = N_SEEDS * len(N_CLIENTS_LIST) * (len(N_SAMPLES_LIST) + 1)  # +1 for heterogeneous
pbar = tqdm(total=total, desc="Total runs")

for seed in range(N_SEEDS):
    for n_clients in N_CLIENTS_LIST:
        # Heterogeneous: horizontal_split=True, n_samples_client=100 (base)
        for horizontal_split in [True, False]:
            if horizontal_split:
                sample_sizes = [100]  # base value, actual samples randomly chosen from {500,1000,2000}
            else:
                sample_sizes = N_SAMPLES_LIST
            for n_samples in sample_sizes:
                try:
                    r = run_single(seed=seed, n_clients=n_clients,
                                   n_samples_client=n_samples,
                                   horizontal_split=horizontal_split)
                    r['seed'] = seed
                    r['n_clients'] = n_clients
                    r['n_samples_client'] = n_samples
                    r['horizontal_split'] = horizontal_split
                    results.append(r)
                except Exception as e:
                    pass
                pbar.update(1)

pbar.close()

print("\n=== I-PERI Results (p=3, PC, BIC, 10 seeds) ===")
print("Total completed runs: {}".format(len(results)))

all_shd = [r['shd'] for r in results]
all_f1 = [r['f1'] for r in results]
print("Overall SHD: {:.2f} +/- {:.2f}".format(np.mean(all_shd), np.std(all_shd)))
print("Overall F1:  {:.2f} +/- {:.2f}".format(np.mean(all_f1), np.std(all_f1)))

# Per-config breakdown
print("\n--- Per Configuration (avg over seeds) ---")
configs = {}
for r in results:
    key = (r['n_clients'], r['n_samples_client'], r['horizontal_split'])
    if key not in configs:
        configs[key] = {'shd': [], 'f1': []}
    configs[key]['shd'].append(r['shd'])
    configs[key]['f1'].append(r['f1'])

for key in sorted(configs.keys()):
    nc, ns, hs = key
    s = configs[key]
    print("n_clients={}, n_samples={}, hsplit={}: SHD={:.2f}+/-{:.2f}, F1={:.2f}+/-{:.2f}".format(
        nc, ns, hs, np.mean(s['shd']), np.std(s['shd']), np.mean(s['f1']), np.std(s['f1'])))
