from PlanetAlign.datasets import *
from PlanetAlign.algorithms import *
from PlanetAlign.data import BaseData

import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_dense_adj
import time
import argparse

from utils.utils import *
from utils.active_utils import *
from utils.dicts import *


def get_args():
    parser = argparse.ArgumentParser(description="Active OT-based Network Alignment")

    parser.add_argument('--alg', type=str, default='PARROT', help='Algorithm to use')
    parser.add_argument('--dataset', type=str, default='phone-email', help='Dataset name')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--outIter', type=int, default=10, help='Number of outer iterations')
    parser.add_argument('--anchor_selection_seed', type=int, default=0, help='Anchor selection seed')
    parser.add_argument('--init_train_ratio', type=float, default=0.2, help='Initial training ratio')
    parser.add_argument('--query_round', type=int, default=10, help='Number of query rounds')
    parser.add_argument('--query_portion', type=float, default=0.2, help='Portion of queries to make')
    parser.add_argument('--query_mode', type=str, default='offline', choices=['offline'], help='Query mode to use')
    parser.add_argument('--modes', type=str, nargs='+', default=['sq_l2_adjoint_grad'], help='Utility modes to use',
                        choices=['kl_adjoint_grad', 'sq_l2_adjoint_grad', 'consistency_adjoint_grad'])
    parser.add_argument('--weights', type=float, nargs='+', default=None, help='Weights for utility modes')
    parser.add_argument('--use_attr', action='store_true', help='Whether to use node attributes')

    return parser.parse_args()


def main():
    args = get_args()

    alg = args.alg
    dataset = args.dataset
    query_round = args.query_round
    query_portion = args.query_portion
    query_mode = args.query_mode
    outIter = args.outIter
    anchor_selection_seed = args.anchor_selection_seed
    modes = args.modes
    use_attr = args.use_attr
    device = args.device

    print(
        f"\n=== Experiment Settings ===\n"
        f"Algorithm:            {alg}\n"
        f"Dataset:              {dataset}\n"
        f"Initial Train Ratio:  {args.init_train_ratio}\n"
        f"Query Portion:        {query_portion}\n"
        f"Query Round:          {query_round}\n"
        f"Query Mode:           {query_mode}\n"
        f"Outer Iterations:     {outIter}\n"
        f"Anchor Seed:          {anchor_selection_seed}\n"
        f"Modes:                {modes}\n"
        f"Use Attributes:       {use_attr}\n"
        f"===========================\n"
    )

    

    import os
    dataset_root = os.environ.get('AVATAR_DATASETS', '/datasets')
    data = dataset_classes_dict[dataset](root=dataset_root, download=True, train_ratio=args.init_train_ratio, seed=anchor_selection_seed)
    # data.sort()  # Dataset class does not support sort

    data.rwr_anchors = data.train_data.clone()
    total_gnds = data.train_data.shape[0] + data.test_data.shape[0]

    num_query = int(total_gnds * query_portion) if query_portion < 1.0 else query_portion
    assert num_query < total_gnds, "Number of queries must be less than total ground-truth pairs."
    batch_size = (num_query + query_round * (num_query // query_round == 0) - num_query % query_round) // query_round
    print(f"Total number of queries: {num_query}, Batch size per round: {batch_size}")

    visited = set([i for i in data.train_data[:, 0].tolist()])
    print(f"Initial #visited nodes: {len(visited)}")
    gnd_map = {}
    for u, v in data.train_data.tolist():
        gnd_map[u] = v
    for u, v in data.test_data.tolist():
        gnd_map[u] = v

    init_settings, train_settings = read_settings(alg, dataset)
    model = alg_classes_dict[alg](**init_settings, dtype=torch.float64).to(device)
    eps = 0
    for eps_para in eps_dict[alg]:
        eps += init_settings[eps_para]

    # Result settings
    active_settings_dict = {
        "query_portion": query_portion,
        "query_round": query_round,
        "query_mode": query_mode,
        "outIter": outIter,
        "anchor_selection_seed": anchor_selection_seed,
        "eps": eps
    }
    
    print(f"Graph 0 num nodes: {data.pyg_graphs[0].num_nodes}, Graph 1 num nodes: {data.pyg_graphs[1].num_nodes}")

    lap_mat1, lap_mat2 = None, None
    for mode in modes:
        if mode.startswith('consistency'):
            adj1 = to_dense_adj(data.pyg_graphs[0].edge_index, max_num_nodes=data.pyg_graphs[0].num_nodes).squeeze().to(model.dtype).to(device)
            adj2 = to_dense_adj(data.pyg_graphs[1].edge_index, max_num_nodes=data.pyg_graphs[1].num_nodes).squeeze().to(model.dtype).to(device)
            lap_mat1 = get_sym_norm_laplacian(adj1)
            lap_mat2 = get_sym_norm_laplacian(adj2)
            break
    
    # Active learning loop
    for query_idx in range(query_round + 1):
        print(f"Query Round {query_idx + 1}/{query_round}")
        train_settings['outer_iters'] = min(15, 5 + query_idx)
        T, logger, add_dict = model.train(data, gids=(0, 1), use_attr=args.use_attr, **train_settings, save_log=False)
        C = add_dict['cost']

        if query_idx < query_round:
            start_time = time.time()
            queried_anchors, _ = query_anchors(T, C, eps, modes, visited, gnd_map, batch_size, lap_mat1=lap_mat1, lap_mat2=lap_mat2)
            end_time = time.time()
            print(f"Query time for this round: {end_time - start_time:.4f} s")
            queried_anchors = queried_anchors.to(data.train_data.device)
            new_train_data = torch.cat((data.train_data, queried_anchors), dim=0)
            new_test_data = setdiff(data.test_data, queried_anchors)
            data.train_data = new_train_data
            data.test_data = new_test_data
            print(f"#queried anchors: {queried_anchors.shape[0]}")
            print(f"New training size: {data.train_data.shape[0]}, New testing size: {data.test_data.shape[0]}")
            print(f"Total #visited nodes: {len(visited)}")


if __name__ == '__main__':
    main()
