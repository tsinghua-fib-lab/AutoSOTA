import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset

from Stretched_models import load_model, get_standard
from Stretched_sampling import build_accumulated_subgraphs, \
    sample_random_subgraph_with_features_new

def main():
    # === 1. Configuration via Argparse ===
    parser = argparse.ArgumentParser(description='Run Downstream Task Evaluation (Acc, F1, Loss, Transfer Error)')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Pubmed', 'ogbn-arxiv'], 
                        help='Dataset to use (default: Cora)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers (default: 3)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channel size (default: 64)')
    parser.add_argument('--num_trials', type=int, default=50, help='Number of trials per sample size (default: 50)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"=== Running Experiment: {args.dataset} | Layers={args.num_layers} | Hidden={args.hidden_channels} ===")

    # === 2. Intelligent Dataset & Sparsity Scheme Mapping ===
    if args.dataset == 'Cora':
        dataset = Planetoid(root="data", name="Cora")
        out_channels = 7
        schemeIII_density = 0.0014
        c1, c2 = 2.0, 0.2
        acc_group_sizes = [50] * 54
        sample_sizes = list(range(100, 601, 50))  
        
    elif args.dataset == 'Pubmed':
        dataset = Planetoid(root="data", name="Pubmed")
        out_channels = 3
        schemeIII_density = 0.0014
        c1, c2 = 2.2, 0.2
        acc_group_sizes = [50] * 150
        sample_sizes = list(range(100, 1201, 100)) 
        
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
        sample_sizes = list(range(100, 2001, 100)) 
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Prepare data
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.x = data.x.float()
    data = data.to(device)

    # File paths
    model_path = f"{args.dataset}_model_{args.num_layers}_{args.hidden_channels}.pt"
    save_path = f"{args.dataset}_Test_{args.num_layers}_{args.hidden_channels}_metrics.csv"
    
    # Define density functions dynamically
    funcs = [
        lambda n, _: c1 / n,                           # Scheme I
        lambda n, _: c2 * np.log(n) / n,               # Scheme II
        lambda n, _: 0 * n + schemeIII_density         # Scheme III
    ]

    # === 3. Initialize/Load Model and Compute Ground-Truth ===
    model = load_model(args.num_layers, model_path, data.num_features, args.hidden_channels, out_channels, device)

    # Sort nodes by degree descending
    deg = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    for edge in data.edge_index.t():
        deg[edge[0]] += 1
    sorted_nodes = torch.argsort(deg, descending=True)

    # Compute full graph standard output (for direct error)
    standard_all_nodes = get_standard(data, model, device)
    print(f"Standard vectors (all nodes) shape: {standard_all_nodes.shape}")

    # Build accumulated subgraph indices
    Gm_Idx_groups_cpu = build_accumulated_subgraphs(sorted_nodes, acc_group_sizes)
    Gm_Idx_groups = [idx.to(device) for idx in Gm_Idx_groups_cpu]

    # === 4. Testing Loop and Metrics Collection ===
    all_results = []

    for scheme_id, target_density_func in enumerate(funcs, start=1):
        for sum_nodes in sample_sizes:
            t0 = time.time()

            trial_diffs = []
            trial_accs = []
            trial_f1s = []
            trial_losses = []

            for _ in range(args.num_trials):
                # Sample subgraph
                sub_data, original_nodes = sample_random_subgraph_with_features_new(
                    data, Gm_Idx_groups, n=sum_nodes,
                    target_density_func=target_density_func
                )
                
                # Squeeze labels for OGB compatability
                y_true = sub_data.y.squeeze() if sub_data.y.dim() > 1 else sub_data.y
                
                with torch.no_grad():
                    # Forward pass
                    out_sub = model(sub_data.x, sub_data.edge_index)
                    
                    # [Metric 1] Direct Error (L2 Discrepancy)
                    standard_subset = standard_all_nodes[original_nodes]
                    diffs_per_node = torch.norm(out_sub - standard_subset, p=2, dim=1)
                    trial_diffs.append(diffs_per_node.mean().item())
                    
                    # [Metric 2] Cross-Entropy Loss
                    loss = F.cross_entropy(out_sub, y_true).item()
                    trial_losses.append(loss)
                    
                    # [Metric 3] Accuracy
                    pred = out_sub.argmax(dim=1)
                    acc = (pred == y_true).float().mean().item()
                    trial_accs.append(acc)
                    
                    # [Metric 4] F1 Score (Macro)
                    y_true_np = y_true.cpu().numpy()
                    pred_np = pred.cpu().numpy()
                    f1 = f1_score(y_true_np, pred_np, average='macro', zero_division=0)
                    trial_f1s.append(f1)

            # Compute statistics
            mean_diff = np.mean(trial_diffs)
            mean_loss = np.mean(trial_losses)
            mean_acc = np.mean(trial_accs)
            mean_f1 = np.mean(trial_f1s)

            std_diff = np.std(trial_diffs)
            std_loss = np.std(trial_losses)
            std_acc = np.std(trial_accs)
            std_f1 = np.std(trial_f1s)

            t1 = time.time()
            
            print(f"[Scheme {scheme_id}] Size {sum_nodes:4d} | "
                  f"Diff: {mean_diff:.4f}±{std_diff:.4f} | "
                  f"Loss: {mean_loss:.4f}±{std_loss:.4f} | "
                  f"Acc: {mean_acc:.4f}±{std_acc:.4f} | "
                  f"F1: {mean_f1:.4f}±{std_f1:.4f} | "
                  f"Time: {t1-t0:.2f}s")

            all_results.append([
                scheme_id, sum_nodes, 
                mean_diff, std_diff, 
                mean_loss, std_loss, 
                mean_acc, std_acc, 
                mean_f1, std_f1
            ])

    # === 5. Save Results ===
    columns = [
        "scheme_id", "sample_size", 
        "mean_diff", "std_diff", 
        "mean_loss", "std_loss", 
        "mean_acc", "std_acc", 
        "mean_f1", "std_f1"
    ]
    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(save_path, index=False)
    print(f"\nAll metric results successfully saved to {save_path}")

if __name__ == "__main__":
    main()