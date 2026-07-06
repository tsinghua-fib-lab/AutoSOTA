import sys, time, csv
sys.path.append("")
import argparse

import torch

from reproducibility.icml2026.dataset import Dataset
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
from notears_admm.linear_admm import notears_linear_admm
from notears_admm.nonlinear_admm import notears_nonlinear_admm
from notears_admm.postprocess import postprocess

# Configuration of torch
torch.set_default_dtype(torch.double)

def parse_args():   
    parser = argparse.ArgumentParser(description="HI-Peri causal discovery")
    parser.add_argument('--n_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--n_samples_client', type=int, default=200, help='Number of samples')
    parser.add_argument('--n_variables', type=int, default=10, help='Number of variables')
    parser.add_argument('--data_type', type=str, default='obs')
    parser.add_argument('--linear', type=utils.str2bool, default='true', help='Use linear causal discovery')
    parser.add_argument('--noise_distribution', type=str, default='normal', help='Type of noise_distribution (gaussian, uniform)')
    parser.add_argument('--seed', type=int, default=1846, help='Random seed')
    parser.add_argument('--save', type=utils.str2bool, default='false', help='Save generated data')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    utils.set_determine(args.seed)

    graph_types = 'erdos_renyi'
    if args.data_type == 'sachs':
        graph_types = 'sachs'
        args.n_variables = 11  # Sachs dataset has 11 variables

    if args.data_type == 'causalchambers':
        graph_types = 'causalchambers'
        args.n_variables = 20  # CausalChambers lt dataset has 10 variables
        args.n_samples_client = 7

    dataset = Dataset(
        graph_type=graph_types,
        n_samples_client=args.n_samples_client,
        n_clients=args.n_clients,
        n_variables=args.n_variables,
        horizontal_split=False,
        noise_distribution=args.noise_distribution,
        linear=args.linear,
        seed=args.seed
    )
    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type=args.data_type, save=True)
    if args.data_type == 'sachs':
        args.n_clients = len(datasets)
    graph = nx.to_numpy_array(dataset.graph)
    experiment_name = dataset.folder_name
    datasets = np.array([np.array(ds) for ds in datasets])

    t0 = time.time()
    if args.linear:
        print("Running NOTEARS-ADMM linear...")
        B_est = notears_linear_admm(datasets, verbose=True)
    else:
        print("Running NOTEARS-ADMM nonlinear...")
        B_est = notears_nonlinear_admm(datasets, verbose=True)
    t1 = time.time()

    server_graph = np.array(postprocess(B_est, threshold=0.1))
    server_graph = np.where(server_graph != 0, 1, 0)  # Binarize the matrix

    shd = utils.shd(server_graph, graph)
    f1 = utils.f1_orientation(server_graph, graph)
    print(f"SHD: {shd}, F1: {f1}")
    f1_skeleton = utils.f1_skeleton(server_graph, graph)
    shd_skeleton = utils.shd_skeleton(server_graph, graph)
    print(f"SHD skeleton: {shd_skeleton}, F1 skeleton: {f1_skeleton}")
    print("Learned graph: \n", server_graph)
    print("True graph: \n", graph)

    if args.save:
        result = {
            "n_samples_client": args.n_samples_client,
            "n_clients": args.n_clients,
            "n_variables": args.n_variables,
            "data_type": args.data_type,
            "noise_distribution": args.noise_distribution,   
            "linear": args.linear, 
            "seed": args.seed,
            "shd": shd, 
            "shd_skeleton": shd_skeleton,
            "f1": f1,
            "f1_skeleton": f1_skeleton,
            "time": t1 - t0
        }
        print(result)

        with open(f'baselines/notears-admm/results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)


# python baselines/notears-admm/main.py --n_clients 10 --n_samples_client 200 --n_variables 10 --data_type obs --linear True --seed 1846 --save True