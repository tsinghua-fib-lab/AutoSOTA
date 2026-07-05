import sys 
import argparse, csv, time

from pyciphod.causal_discovery.federated.regret_based.iperi.iperi import IPeri
from pyciphod.causal_discovery.federated.regret_based.iperi.client import Client
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
from reproducibility.icml2026.dataset import Dataset
import pyciphod.causal_discovery.federated.regret_based.ges as ges

import networkx as nx
import numpy as np

from tqdm import tqdm


def parse_args():   
    parser = argparse.ArgumentParser(description="I-Peri federated causal discovery")
    parser.add_argument('--n_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--n_samples_client', type=int, default=200, help='Number of samples')
    parser.add_argument('--n_variables', type=int, default=3, help='Number of variables')
    parser.add_argument('--horizontal_split', type=utils.str2bool, default='false', help='Use uneven sample split among clients')
    parser.add_argument('--data_type', type=str, default='struct')
    parser.add_argument('--cd_function', type=str, default='pc', help='Causal discovery function for clients (pc, lingam, ges)')
    parser.add_argument('--max_iters', type=int, default=1, help='Maximum number of iterations for HI-Peri')
    parser.add_argument('--scoring_function', type=str, default='bic', help='Scoring function for clients (bic, bdeu)')
    parser.add_argument('--noise_distribution', type=str, default='uniform', help='Type of noise (gaussian, uniform)')
    parser.add_argument('--seed', type=int, default=2000, help='Random seed')
    parser.add_argument('--linear', type=utils.str2bool, default='false', help='Use linear causal discovery')
    parser.add_argument('--masked', type=utils.str2bool, default='false', help='Use masked causal discovery, if false standard PERI')
    parser.add_argument('--save', type=utils.str2bool, default='false', help='Save generated data')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    utils.set_determine(args.seed)

    if args.cd_function == 'lingam':
        args.noise_distribution = 'uniform'
        
    dataset = Dataset(
        n_samples_client=args.n_samples_client,
        n_clients=args.n_clients,
        n_variables=args.n_variables,
        linear=args.linear,
        horizontal_split=args.horizontal_split,
        noise_distribution=args.noise_distribution,
        seed=args.seed,
    )

    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type=args.data_type, save=True)
    graph = dataset.graph 
    experiment_name = dataset.folder_name
    server_variables = graph.nodes 
    
    clients = []
    client_graphs = []
    error_client_graphs = []
    t0 = time.time()
    for i in tqdm(range(args.n_clients)):
        data_path = f'{experiment_name}/client_{i}.csv'
        clients.append(Client(
            name=f'client_{i}',
            data=data_path,
            cd_function=args.cd_function,
            scoring_function=args.scoring_function,
            masked=args.masked,
            linear=args.linear
            )
        )
        client_graphs.append(clients[-1].graph)
        ground_graph = cpdag if args.data_type != 'struct' else ges.utils.dag_to_cpdag(graphs[i])
        client_graph = clients[-1].graph
        error_client_graphs.append(
            utils.f1_orientation(client_graph, ground_graph)
        )  
        
    i_peri = IPeri(
        n_variables=len(graph.nodes),
        clients=clients
    )

    union_graph = utils.union_graph(nx.to_numpy_array(graph), client_graphs)
    server_graph, estimated_cpdag = i_peri.fit(max_iters=args.max_iters)
    t1 = time.time()

    result = {
        "n_samples_client": args.n_samples_client,
        "n_clients": args.n_clients,
        "n_variables": args.n_variables,
        "horizontal_split": args.horizontal_split,
        "data_type": args.data_type,
        "cd_function": args.cd_function,
        "scoring_function": args.scoring_function,
        "linear": args.linear,
        "masked": args.masked,
        "noise_distribution": args.noise_distribution,    
        "seed": args.seed,
        "shd": utils.shd(server_graph, nx.to_numpy_array(graph)),
        "f1": utils.f1_orientation(server_graph, nx.to_numpy_array(graph)),
        "shd_cpdag_est": utils.shd(estimated_cpdag, nx.to_numpy_array(graph)),
        "f1_cpda_est": utils.f1_orientation(estimated_cpdag, nx.to_numpy_array(graph)),
        "shd_union": utils.shd(union_graph, nx.to_numpy_array(graph)),
        "f1_union": utils.f1_orientation(union_graph, nx.to_numpy_array(graph)),
        "shd_ucpdag": utils.shd(server_graph, ucpdag),
        "f1_ucpdag": utils.f1_orientation(server_graph, ucpdag),
        "shd_cpdag": utils.shd(estimated_cpdag, cpdag),
        "f1_cpdag": utils.f1_orientation(estimated_cpdag, cpdag),
        "shd_ucpdag_cpdag": utils.shd(cpdag, ucpdag),
        "f1_ucpdag_cpdag": utils.f1_orientation(cpdag, ucpdag),
        "error_client": np.mean(error_client_graphs),
        "time": t1 - t0
    }

    print(result)

    if args.save:

        with open(f'results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
