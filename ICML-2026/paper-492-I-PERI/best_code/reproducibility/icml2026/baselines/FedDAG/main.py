import argparse, time, csv, sys 
sys.path.append("")

from reproducibility.icml2026.dataset import Dataset
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils
import networkx as nx

from models import GS_FedDAG, AS_FedDAG_linear

import tensorflow as tf 
tf.compat.v1.disable_eager_execution()


def parse_args():   
    parser = argparse.ArgumentParser(description="HI-Peri causal discovery")
    parser.add_argument('--n_clients', type=int, default=2, help='Number of clients')
    parser.add_argument('--n_samples_client', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_variables', type=int, default=3, help='Number of variables')
    parser.add_argument('--horizontal_split', type=utils.str2bool, default='false', help='Use uneven sample split among clients')
    parser.add_argument('--data_type', type=str, default='obs')
    parser.add_argument('--noise_distribution', type=str, default='normal', help='Type of noise (gaussian, uniform)')
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
        args.n_clients = 6

    if args.data_type == 'it_monitoring':
        graph_types = 'it_monitoring'
        args.n_variables = 16  # IT monitoring dataset has 14 variables
        args.n_clients = 2

    dataset = Dataset(
        graph_type=graph_types,
        n_samples_client=args.n_samples_client,
        n_clients=args.n_clients,
        n_variables=args.n_variables,
        horizontal_split=args.horizontal_split,
        noise_distribution=args.noise_distribution,
        seed=args.seed
    )

    datasets, cpdag, ucpdag, graphs = dataset.generate(data_type=args.data_type, save=True)
    if args.data_type == 'sachs':
        args.n_clients = len(datasets)
    graph = nx.to_numpy_array(dataset.graph)
    experiment_name = dataset.folder_name
    print(len(datasets))


    print('Begin running the mothod.....')
    t0 = time.time()
    model = GS_FedDAG(d=args.n_variables,
                    num_client=args.n_clients,
                    use_gpu=False,
                    seed=args.seed)
    
                
    model.learn(datasets)
    t1 = time.time()

    server_graph = model.causal_matrix

    shd = utils.shd(server_graph, graph)
    f1 = utils.f1_orientation(server_graph, graph)
    f1_skeleton = utils.f1_skeleton(server_graph, graph)
    shd_skeleton = utils.shd_skeleton(server_graph, graph)
    print(f"SHD: {shd}, F1: {f1}")
    print(f"SHD skeleton: {shd_skeleton}, F1 skeleton: {f1_skeleton}")
    print("Learned graph: \n", server_graph)
    print("True graph: \n", graph)

    if args.save:
        result = {
            "n_samples_client": args.n_samples_client,
            "n_clients": args.n_clients,
            "n_variables": args.n_variables,
            "horizontal_split": args.horizontal_split,
            "data_type": args.data_type,
            "noise_distribution": args.noise_distribution,   
            # "linear": args.linear, 
            "seed": args.seed,
            "shd": shd, 
            "shd_skeleton": shd_skeleton,
            "f1": f1,
            "f1_skeleton": f1_skeleton,
            "time": t1 - t0
        }
        print(result)

        with open(f'baselines/FedDAG/results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)