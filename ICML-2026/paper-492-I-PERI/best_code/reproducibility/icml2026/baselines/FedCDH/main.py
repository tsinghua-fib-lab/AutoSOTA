import argparse, time

import numpy as np
import pandas as pd
import networkx as nx

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci, fisherz, chisq
from causallearn.utils.data_utils import get_cpdag_from_cdnod

import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
from reproducibility.icml2026.dataset import Dataset

def parse_args():   
    parser = argparse.ArgumentParser(description="FedCDH causal discovery")
    parser.add_argument('--n_samples_client', type=int, default=100, help='Number of clients')
    parser.add_argument('--n_clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--linear', type=utils.str2bool, default='false', help='Use linear causal discovery')
    parser.add_argument('--n_variables', type=int, default=4, help='Number of variables')
    parser.add_argument('--horizontal_split', type=utils.str2bool, default='false', help='Use uneven sample split among clients')
    parser.add_argument('--data_type', type=str, default='obs')
    parser.add_argument('--noise', type=str, default='normal', help='Type of noise (gaussian, uniform)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_iters', type=int, default=100, help='Maximum number of iterations for HI-Peri')
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
        args.n_clients = 6

    dataset = Dataset(
        graph_type=graph_types,
        n_samples_client=args.n_samples_client,
        n_clients=args.n_clients,
        n_variables=args.n_variables,
        horizontal_split=args.horizontal_split,
        noise_distribution=args.noise,
        linear=args.linear,
        seed=args.seed
    )
    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type=args.data_type, save=True)
    if args.data_type == 'sachs':
        args.n_clients = len(datasets)
    # datasets = np.concatenate(datasets, axis=0)
    # n = int(len(datasets) / args.n_clients)
    # c_indx = np.asarray(list(range(args.n_clients)))
    # c_indx = np.repeat(c_indx, n)
    # c_indx = np.reshape(c_indx, (n*args.n_clients,1))
    # print(c_indx.shape)
    # if args.data_type == 'causalchambers':
    #     datasets = [pd.DataFrame(df) for df in datasets]
    #     blocks = np.array_split(datasets, args.n_clients)  # allows uneven split
    #     min_rows = min(b.shape[0] for b in blocks)
    #     if len(set(b.shape[0] for b in blocks)) != 1:
    #         print(f"[FedCDH] Sachs: rebalancing client blocks to {min_rows} rows each.")
    #     datasets = np.concatenate([b[:min_rows] for b in blocks], axis=0)

    # concatenate dataset 
    n_clients_samples = [len(dataset) for dataset in datasets]
    print(n_clients_samples)
    # import sys 
    # sys.exit()
    datasets = np.concatenate(datasets, axis=0)
    # repeat idx based on dataset sizes
    c_indx = np.array([])
    for i in range(args.n_clients):
        c_indx = np.concatenate((c_indx, np.repeat(i, n_clients_samples[i])))
    # print(c_indx.shape)
    c_indx = np.reshape(c_indx, (len(datasets), 1)) 
    # print(c_indx.shape)

    # Final safety check: CDNOD requires total rows divisible by n_clients
    n_rows = datasets.shape[0]
    # remainder = n_rows % args.n_clients
    # if remainder != 0:
    #     keep = n_rows - remainder
    #     print(f"[FedCDH] Trimming {remainder} row(s): {n_rows} -> {keep} for K={args.n_clients}")
    #     datasets = datasets[:keep]
    #     c_indx = c_indx[:keep]

    # assert datasets.shape[0] == c_indx.shape[0], "datasets and c_indx length mismatch"
    # assert datasets.shape[0] % args.n_clients == 0, "rows must be divisible by n_clients"
    # print(datasets)
    if args.linear:
        indep_test = fisherz
    else:
        indep_test = chisq
    t0 = time.time()
    cg = cdnod(datasets, c_indx, args.n_clients, 0.05, indep_test, True, 0, -1)
    t1 = time.time()

    est_graph = np.zeros((args.n_variables, args.n_variables))
    est_graph = cg.G.graph[0:args.n_variables, 0:args.n_variables]
    est_cpdag = get_cpdag_from_cdnod(est_graph)

    shd = utils.shd(est_cpdag, nx.to_numpy_array(dataset.graph))
    shd_skeleton = utils.shd_skeleton(est_cpdag, nx.to_numpy_array(dataset.graph))
    print(f"SHD: {shd}, SHD Skeleton: {shd_skeleton}")
    f1 = utils.f1_orientation(est_cpdag, nx.to_numpy_array(dataset.graph))
    f1_skeleton = utils.f1_skeleton(est_cpdag, nx.to_numpy_array(dataset.graph))
    print(f"F1: {f1}, F1 Skeleton: {f1_skeleton}")

    result = {
            "method": "FedCDH",
            "n_clients": args.n_clients,
            "n_samples_client": args.n_samples_client,
            "n_variables": args.n_variables,
            "horizontal_split": args.horizontal_split,          
            "data_type": args.data_type,
            "noise_distribution": args.noise,
            "linear": args.linear,
            "seed": args.seed,
            "shd": shd,
            "shd_skeleton": shd_skeleton,
            "f1": f1,
            "f1_skeleton": f1_skeleton,
            "time": t1 - t0
        }
    
    print(result)

    if args.save:
        import csv
        with open(f'baselines/FedCDH/results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)