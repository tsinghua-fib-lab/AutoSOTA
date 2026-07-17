import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from ogb.nodeproppred import PygNodePropPredDataset

# ==========================================
# Core Utility Functions
# ==========================================

def compute_density(data, nodes):
    """Compute the edge density of a given subgraph."""
    edge_index_sub, _ = subgraph(subset=nodes, edge_index=data.edge_index, relabel_nodes=False)
    edge_index_sub = to_undirected(edge_index_sub)
    m_edges = edge_index_sub.size(1) // 2
    n = nodes.numel()
    return 2 * m_edges / (n * (n - 1)) if n > 1 else 0.0

def find_closest_subgraph(data, nodes_seq, n, func, extra_param=None):
    """Find the accumulated subgraph whose density is closest to the target density."""
    target_density = func(n, extra_param)
    best_nodes, best_diff = None, float("inf")
    for nodes in nodes_seq:
        dens = compute_density(data, nodes)
        diff = abs(dens - target_density)
        if diff < best_diff:
            best_diff = diff
            best_nodes = nodes
    return best_nodes

def build_accumulated_subgraphs_of_groups(sorted_nodes, group_sizes_list):
    """Build sequentially expanding subgraphs based on group sizes."""
    all_acc_subgraphs = []
    for group_sizes in group_sizes_list:
        nodes_seq = []
        start = 0
        for size in group_sizes:
            start += size
            nodes_seq.append(sorted_nodes[:start])
        all_acc_subgraphs.append(nodes_seq)
    return all_acc_subgraphs

def sample_random_subgraphs_density(data, acc_subgraphs, sample_sizes, funcs, scheme3_density, num_trials=20):
    """Sample subgraphs and record their empirical densities across different schemes."""
    all_mean_densities = []
    all_std_densities = []
    params = [None, None, scheme3_density]

    for scheme_idx, (nodes_seq, func, param) in enumerate(zip(acc_subgraphs, funcs, params)):
        mean_densities, std_densities = [], []

        for n in sample_sizes:
            Gm_nodes = find_closest_subgraph(data, nodes_seq, n, func, param)
            densities = []
            
            # Ensure sample size does not exceed available nodes
            current_n = min(n, Gm_nodes.numel())

            for _ in range(num_trials):
                # Randomly sample nodes from the accumulated subgraph
                sampled_nodes = Gm_nodes[torch.randperm(Gm_nodes.numel())[:current_n]]

                edge_index_sub, _ = subgraph(subset=sampled_nodes, edge_index=data.edge_index, relabel_nodes=False)
                edge_index_sub = to_undirected(edge_index_sub)
                m_edges = edge_index_sub.size(1) // 2
                
                density = 2 * m_edges / (current_n * (current_n - 1)) if current_n > 1 else 0
                densities.append(density)

            mean_densities.append(torch.tensor(densities).mean().item())
            std_densities.append(torch.tensor(densities).std().item())

        all_mean_densities.append(mean_densities)
        all_std_densities.append(std_densities)

    return all_mean_densities, all_std_densities


# ==========================================
# Plotting Functions (Proportions Unchanged)
# ==========================================

def plot_density_curves_standard(sample_sizes, all_mean_densities, all_std_densities, labels, dataset_name):
    """Original standard plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",      
        "mathtext.rm": "Times New Roman",  
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    def func_fit_I(n, C): return C / n
    def func_fit_II(n, C): return C * np.log(n) / n
    def func_fit_III(n, C): return C * np.ones_like(n)

    fig, ax = plt.subplots(figsize=(6, 4)) 

    x_fit = np.linspace(min(sample_sizes), max(sample_sizes), 500)
    x_data = np.array(sample_sizes)

    for idx, (mean_dens, std_dens, label, color, marker) in \
        enumerate(zip(all_mean_densities, all_std_densities, labels, colors, markers)):
        
        mean_dens = np.array(mean_dens)
        std_dens = np.array(std_dens)

        ax.plot(x_data, mean_dens, marker=marker, linestyle='-', color=color, alpha=0.5, label=f"{label} (Data)")
        ax.fill_between(x_data, mean_dens - std_dens, mean_dens + std_dens, color=color, alpha=0.1)

        popt = None
        fit_label = ""
        y_fit_curve = None

        try:
            if idx == 0:
                popt, _ = curve_fit(func_fit_I, x_data, mean_dens)
                y_fit_curve = func_fit_I(x_fit, *popt)
                fit_label = r"Fit: $\Theta (1 / n)$"
            elif idx == 1:
                popt, _ = curve_fit(func_fit_II, x_data, mean_dens)
                y_fit_curve = func_fit_II(x_fit, *popt)
                fit_label = r"Fit: $\Theta (\log n / n)$"
            elif idx == 2:
                popt, _ = curve_fit(func_fit_III, x_data, mean_dens)
                y_fit_curve = func_fit_III(x_fit, *popt)
                fit_label = r"Fit: $\Theta (1)$"

            if y_fit_curve is not None:
                ax.plot(x_fit, y_fit_curve, linestyle='--', linewidth=2.5, color=color, label=fit_label)
        except Exception as e:
            print(f"Fitting failed for Scheme {idx+1}: {e}")

    ax.set_xlabel(r"Graph Size ($n$)")
    ax.set_ylabel("Edge Density")
    ax.set_xlim(sample_sizes[0]-50, sample_sizes[-1] + 50)
    ax.legend(frameon=False, loc='upper right', ncol=1)
    
    plt.tight_layout()
    plt.grid(False)

    out_file = f"{dataset_name}_density_curves_standard.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to {out_file}")
    plt.show()

def plot_density_curves_icml(sample_sizes, all_mean_densities, all_std_densities, labels, dataset_name):
    """ICML single-column plotting style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",         
        "font.size": 10,                  
        "axes.labelsize": 12,
        "legend.fontsize": 8,             
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,            
        "xtick.direction": "in",          
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    })

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    markers = ["o", "s", "^"]

    def func_fit_I(n, C): return C / n
    def func_fit_II(n, C): return C * np.log(n) / n
    def func_fit_III(n, C): return C * np.ones_like(n)

    fig, ax = plt.subplots(figsize=(3.25, 2.8)) 

    x_fit = np.linspace(min(sample_sizes), max(sample_sizes), 500)
    x_data = np.array(sample_sizes)

    for idx, (mean_dens, std_dens, label, color, marker) in \
        enumerate(zip(all_mean_densities, all_std_densities, labels, colors, markers)):
        
        mean_dens = np.array(mean_dens)
        std_dens = np.array(std_dens)

        ax.plot(x_data, mean_dens, marker=marker, linestyle='', color=color, alpha=0.7, 
                label=f"{label} (Data)", markersize=3.5, markeredgewidth=0.5)
        ax.fill_between(x_data, mean_dens - std_dens, mean_dens + std_dens, color=color, alpha=0.15)

        popt = None
        fit_label = ""
        y_fit_curve = None

        try:
            if idx == 0:
                popt, _ = curve_fit(func_fit_I, x_data, mean_dens)
                y_fit_curve = func_fit_I(x_fit, *popt)
                fit_label = r"Fit: $\Theta(1/n)$"
            elif idx == 1:
                popt, _ = curve_fit(func_fit_II, x_data, mean_dens)
                y_fit_curve = func_fit_II(x_fit, *popt)
                fit_label = r"Fit: $\Theta(\log n/n)$"
            elif idx == 2:
                popt, _ = curve_fit(func_fit_III, x_data, mean_dens)
                y_fit_curve = func_fit_III(x_fit, *popt)
                fit_label = r"Fit: $\Theta(1)$"

            if y_fit_curve is not None:
                ax.plot(x_fit, y_fit_curve, linestyle='--', linewidth=1.5, color=color, label=fit_label)
        except Exception as e:
            print(f"Fitting failed for Scheme {idx+1}: {e}")

    ax.set_xlabel(r"Graph Size ($n$)") 
    ax.set_ylabel("Edge Density")
    
    pad = (max(sample_sizes) - min(sample_sizes)) * 0.05
    ax.set_xlim(min(sample_sizes) - pad, max(sample_sizes) + pad)
    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=False, loc='best', fontsize=8, ncol=1)
    
    plt.tight_layout()
    out_file = f"{dataset_name}_density_curves_icml.pdf"
    plt.savefig(out_file, format='pdf')
    print(f"Plot saved to {out_file}")
    plt.show()

# ==========================================
# Main Execution Pipeline
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Sample and Plot Density Curves across Sparsity Schemes")
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Pubmed', 'ogbn-arxiv'], 
                        help='Dataset to use (default: Cora)')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of random trials per sample size')
    parser.add_argument('--style', type=str, default='standard', choices=['standard', 'icml'], 
                        help='Plotting style to use (default: standard)')
    args = parser.parse_args()

    # === Intelligent Dataset & Sparsity Scheme Mapping ===
    if args.dataset == 'Cora':
        dataset = Planetoid(root="data", name="Cora")
        scheme3_density = 0.0014
        c1, c2 = 2.0, 0.2
        group_sizes_list = [[50] * 54] * 3
        sample_sizes = list(range(100, 601, 50))
        
    elif args.dataset == 'Pubmed':
        dataset = Planetoid(root="data", name="Pubmed")
        scheme3_density = 0.0014
        c1, c2 = 2.2, 0.2
        group_sizes_list = [[50] * 150] * 3
        sample_sizes = list(range(100, 1201, 100))
        
    elif args.dataset == 'ogbn-arxiv':
        torch.serialization.add_safe_globals([Data])
        _orig_torch_load = torch.load
        def torch_load_with_weights_only_false(*load_args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*load_args, **kwargs)
        torch.load = torch_load_with_weights_only_false
        
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data')
        scheme3_density = 0.004
        c1, c2 = 8.0, 1.0
        group_sizes_list = [[50] * 20 + [100] * 100] * 3
        sample_sizes = list(range(100, 2001, 100))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"=== Generating Density Curves for {args.dataset} ===")

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    # Compute full graph density
    n_nodes = data.num_nodes
    n_edges = data.edge_index.size(1) // 2 
    full_density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    print(f"Full Graph: nodes = {n_nodes}, edges = {n_edges}, density = {full_density:.6f}")

    # Sort nodes by degree
    deg = torch.zeros(data.num_nodes, dtype=torch.long)
    for edge in data.edge_index.t():
        deg[edge[0]] += 1
    sorted_nodes = torch.argsort(deg, descending=True)
    
    # Build accumulated subgraphs
    acc_subgraphs = build_accumulated_subgraphs_of_groups(sorted_nodes, group_sizes_list)
    
    # Define density functions dynamically
    funcs = [
        lambda n, _: c1 / n,                           # Scheme I
        lambda n, _: c2 * np.log(n) / n,               # Scheme II
        lambda n, rho: rho                             # Scheme III
    ]

    # Perform sampling
    all_mean_densities, all_std_densities = sample_random_subgraphs_density(
        data, acc_subgraphs, sample_sizes, funcs, scheme3_density, num_trials=args.num_trials
    )

    labels = ["Scheme I", "Scheme II", "Scheme III"]
    
    # Render plot based on specified style
    if args.style == 'standard':
        plot_density_curves_standard(sample_sizes, all_mean_densities, all_std_densities, labels, args.dataset)
    elif args.style == 'icml':
        plot_density_curves_icml(sample_sizes, all_mean_densities, all_std_densities, labels, args.dataset)

if __name__ == "__main__":
    main()