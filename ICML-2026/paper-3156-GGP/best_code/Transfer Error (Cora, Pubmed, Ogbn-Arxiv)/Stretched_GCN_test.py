import torch
import numpy as np
import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
import time
import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from Stretched_models import load_model, get_standard
from Stretched_sampling import (
    build_accumulated_subgraphs,
    sample_random_subgraph_with_features_new,
)


def main():
    # === Configuration via Argparse ===
    parser = argparse.ArgumentParser(description="Evaluate GCN Size Transferability across Sparsity")
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Pubmed', 'ogbn-arxiv'], 
                        help='Dataset to use (default: Cora)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers (default: 2)')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channel size (default: 256)')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of random trials per size (default: 20)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"=== Running Evaluation: {args.dataset} | Layers={args.num_layers}, Hidden={args.hidden_channels} ===")

    # === Intelligent Dataset & Sparsity Scheme Mapping ===
    if args.dataset == 'Cora':
        dataset = Planetoid(root="data", name="Cora")
        out_channels = 7
        schemeIII_density = 0.0014
        c1, c2 = 2.0, 0.2
        acc_group_sizes = [50] * 54
        sample_sizes = list(range(100, 601, 50))  # [100, 150, ..., 600]
        
    elif args.dataset == 'Pubmed':
        dataset = Planetoid(root="data", name="Pubmed")
        out_channels = 3
        schemeIII_density = 0.0014
        c1, c2 = 2.2, 0.2
        acc_group_sizes = [50] * 150
        sample_sizes = list(range(100, 1201, 100)) # [100, 200, ..., 1200]
        
    elif args.dataset == 'ogbn-arxiv':
        # Safely load weights for ogbn-arxiv under PyTorch 2.0+
        torch.serialization.add_safe_globals([Data])
        _orig_torch_load = torch.load
        def torch_load_with_weights_only_false(*load_args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*load_args, **kwargs)
        torch.load = torch_load_with_weights_only_false
        
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data')
        out_channels = 40
        schemeIII_density = 0.004
        c1, c2 = 8.0, 1.0
        acc_group_sizes = [50] * 20 + [100] * 100
        sample_sizes = list(range(100, 2001, 100)) # [100, 200, ..., 2000]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.x = data.x.float()
    data = data.to(device)

    # Dynamic file paths matching training configs
    model_path = f"{args.dataset}_model_{args.num_layers}_{args.hidden_channels}.pt"
    save_path = f"{args.dataset}_Test_{args.num_layers}_{args.hidden_channels}.csv"

    # Define density functions based on mapped parameters
    funcs = [
        lambda n, _: c1 / n,                     # Scheme I
        lambda n, _: c2 * np.log(n) / n,         # Scheme II
        lambda n, _: 0 * n + schemeIII_density   # Scheme III
    ]
    
    # Replicate group sizes for the three schemes
    acc_group_sizes_list = [acc_group_sizes] * 3

    # === Initialize and Load Pretrained Model ===
    model = load_model(args.num_layers, model_path, data.num_features, args.hidden_channels, out_channels, device)

    # === Compute Ground-Truth Standard Vectors on Full Graph ===
    deg = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    for edge in data.edge_index.t():
        deg[edge[0]] += 1
    sorted_nodes = torch.argsort(deg, descending=True)

    standard_all_nodes = get_standard(data, model, device)
    print(f"Standard vectors (all nodes) shape: {standard_all_nodes.shape}")

    # Build accumulated subgraphs based on sorting and group sizes
    Gm_Idx_groups_cpu = build_accumulated_subgraphs(sorted_nodes, acc_group_sizes_list[2])
    Gm_Idx_groups = [idx.to(device) for idx in Gm_Idx_groups_cpu]

    # === Storage for Evaluation Results ===
    all_results = []

    # === Iterate Through Three Sparsity Schemes ===
    for scheme_id, (acc_group_sizes_curr, scheme_func) in enumerate(zip(acc_group_sizes_list, funcs), start=1):
        for sum_nodes in sample_sizes:
            t0 = time.time()
            diffs = []
            
            for _ in range(args.num_trials):
                sub_data, original_nodes = sample_random_subgraph_with_features_new(
                    data, Gm_Idx_groups, n=sum_nodes,
                    target_density_func=scheme_func
                )
                
                with torch.no_grad():
                    # 1. Forward pass on the sampled subgraph
                    out_sub = model(sub_data.x, sub_data.edge_index)
                    # 2. Extract corresponding ground-truth representations from full graph
                    standard_subset = standard_all_nodes[original_nodes]
                    # 3. Compute L2 node-wise discrepancy
                    diffs_per_node = torch.norm(out_sub - standard_subset, p=2, dim=1)
                    # 4. Average difference for the current subgraph sample
                    mean_diff_for_subgraph = diffs_per_node.mean().item()
                    
                diffs.append(mean_diff_for_subgraph)

            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            t1 = time.time()
            
            print(f"[Scheme {scheme_id}] Sample size {sum_nodes}: "
                  f"mean_diff = {mean_diff:.4f}, std_diff = {std_diff:.4f} "
                  f"| Time: {t1-t0:.4f}s")

            all_results.append([scheme_id, sum_nodes, mean_diff, std_diff])

    # === Save Results to CSV ===
    df = pd.DataFrame(all_results, columns=["scheme_id", "sample_size", "mean_diff", "std_diff"])
    df.to_csv(save_path, index=False)
    print(f"\nAll results successfully saved to {save_path}")


if __name__ == "__main__":
    main()