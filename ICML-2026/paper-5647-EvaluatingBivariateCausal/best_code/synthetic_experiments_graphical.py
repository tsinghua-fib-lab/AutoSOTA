"""
Experiment: Statement Graph Error Analysis

Sample random DAGs with fixed n, m, p and compute statement graphs.
Add varying numbers of errors to the statement graph by flipping random entries.
Compute the incompatibility score of the resulting graph and plot results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from graph_functions import (
    generate_random_dag,
    compute_statement_graph,
    incompatibility_score
)


plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18,
    'figure.titlesize': 24,
})

def add_errors_to_graph(directed_adj: np.ndarray, bidirected_adj: np.ndarray, 
                        num_errors: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add errors to a statement graph by flipping random entries.
    
    For directed edges: flip A[i,j] from 0 to 1 or 1 to 0 (excluding diagonal)
    For bidirected edges: flip B[i,j] and B[j,i] together (upper triangular entries only)
    
    Parameters:
        directed_adj: n x n directed adjacency matrix
        bidirected_adj: n x n symmetric bidirected adjacency matrix
        num_errors: Number of entries to flip
        seed: Random seed for reproducibility
    
    Returns:
        corrupted_directed: Directed adjacency with errors
        corrupted_bidirected: Bidirected adjacency with errors
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = directed_adj.shape[0]
    
    corrupted_directed = directed_adj.copy()
    corrupted_bidirected = bidirected_adj.copy()
    
    # Collect all possible positions to flip
    # Directed: all off-diagonal entries (i, j) where i != j
    directed_positions = [(i, j) for i in range(n) for j in range(n) if i != j]
    # Bidirected: upper triangular entries (i, j) where i < j
    bidirected_positions = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Combine all positions with a type indicator
    all_positions = [('directed', pos) for pos in directed_positions] + \
                    [('bidirected', pos) for pos in bidirected_positions]
    
    # Randomly select positions to flip
    if num_errors > len(all_positions):
        num_errors = len(all_positions)
    
    selected_indices = np.random.choice(len(all_positions), size=num_errors, replace=False)
    
    for idx in selected_indices:
        edge_type, (i, j) = all_positions[idx]
        
        if edge_type == 'directed':
            # Flip directed edge
            corrupted_directed[i, j] = 1 - corrupted_directed[i, j]
        else:
            # Flip bidirected edge (both i,j and j,i)
            corrupted_bidirected[i, j] = 1 - corrupted_bidirected[i, j]
            corrupted_bidirected[j, i] = 1 - corrupted_bidirected[j, i]
    
    return corrupted_directed, corrupted_bidirected


def compute_mean_scores(n: int, m: int, p: float, error_range: List[int],
                        num_graphs: int, num_error_samples: int, seed: int) -> np.ndarray:
    """
    Compute mean incompatibility scores for given parameters.
    
    Returns:
        means: Array of mean scores for each error level
    """
    all_scores = {num_errors: [] for num_errors in error_range}
    
    for graph_idx in range(num_graphs):
        graph_seed = seed + graph_idx * 1000
        adj_matrix, perm, hidden = generate_random_dag(n + m, 0, p, seed=graph_seed)
        
        np.random.seed(graph_seed + 1)
        hidden = set(np.random.choice(n + m, size=m, replace=False))
        
        try:
            directed_adj, bidirected_adj, observed = compute_statement_graph(adj_matrix, hidden)
        except Exception:
            continue
        
        for num_errors in error_range:
            for error_idx in range(num_error_samples):
                error_seed = graph_seed + num_errors * 100 + error_idx
                
                corrupted_dir, corrupted_bidir = add_errors_to_graph(
                    directed_adj, bidirected_adj, num_errors, seed=error_seed
                )
                
                try:
                    score, _ = incompatibility_score(corrupted_dir, corrupted_bidir, verbose=False)
                    all_scores[num_errors].append(score)
                except Exception:
                    pass
    
    means = np.array([np.mean(all_scores[e]) if all_scores[e] else np.nan for e in error_range])
    return means


def run_experiment_vary_n(n_values: List[int] = [3, 4, 5, 6], m: int = 2, p: float = 0.5,
                          error_range: List[int] = None, num_graphs: int = 50, 
                          num_error_samples: int = 20, seed: int = 42,
                          save_path: str = None):
    """
    Run experiment varying the number of observed variables (n).
    
    Parameters:
        n_values: List of different n values to test
        m: Number of hidden variables (fixed)
        p: Edge probability (fixed)
        error_range: List of error counts to test
        num_graphs: Number of graphs to sample per configuration
        num_error_samples: Number of error samples per graph
        seed: Base random seed
        save_path: Optional path to save the figure
    """
    if error_range is None:
        error_range = list(range(0, 16, 2))
    
    print(f"Running experiment varying n with m={m}, p={p}")
    print(f"n values: {n_values}")
    print(f"Error range: {error_range}")
    print()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Line styles and markers for B&W incompatibility
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for idx, n in enumerate(n_values):
        print(f"Processing n={n}...")
        means = compute_mean_scores(n, m, p, error_range, num_graphs, num_error_samples, seed + idx * 10000)
        
        ls = line_styles[idx % len(line_styles)]
        mk = markers[idx % len(markers)]
        ax.plot(error_range, means, linestyle=ls, marker=mk, color='black',
                markersize=6, linewidth=1.5, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=1.0, label=f'$n={n}$')
    
    ax.set_xlabel('True number of errors', fontsize=24)
    ax.set_ylabel('Average\nincompatibility score', fontsize=24)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    


def run_experiment_vary_p(n: int = 4, m: int = 2, p_values: List[float] = [0.2, 0.4, 0.6, 0.8],
                          error_range: List[int] = None, num_graphs: int = 50,
                          num_error_samples: int = 20, seed: int = 42,
                          save_path: str = None):
    """
    Run experiment varying the edge probability (p).
    
    Parameters:
        n: Number of observed variables (fixed)
        m: Number of hidden variables (fixed)
        p_values: List of different p values to test
        error_range: List of error counts to test
        num_graphs: Number of graphs to sample per configuration
        num_error_samples: Number of error samples per graph
        seed: Base random seed
        save_path: Optional path to save the figure
    """
    if error_range is None:
        error_range = list(range(0, 16, 2))
    
    print(f"Running experiment varying p with n={n}, m={m}")
    print(f"p values: {p_values}")
    print(f"Error range: {error_range}")
    print()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for idx, p in enumerate(p_values):
        print(f"Processing p={p}...")
        means = compute_mean_scores(n, m, p, error_range, num_graphs, num_error_samples, seed + idx * 10000)
        
        ls = line_styles[idx % len(line_styles)]
        mk = markers[idx % len(markers)]
        ax.plot(error_range, means, linestyle=ls, marker=mk, color='black',
                markersize=6, linewidth=1.5, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=1.0, label=f'$p={p}$')
    
    ax.set_xlabel('True number of errors', fontsize=24)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    


def run_experiment_vary_m(n: int = 4, m_values: List[int] = [0, 1, 2, 3], p: float = 0.5,
                          error_range: List[int] = None, num_graphs: int = 50,
                          num_error_samples: int = 20, seed: int = 42,
                          save_path: str = None):
    """
    Run experiment varying the number of hidden variables (m).
    
    Parameters:
        n: Number of observed variables (fixed)
        m_values: List of different m values to test
        p: Edge probability (fixed)
        error_range: List of error counts to test
        num_graphs: Number of graphs to sample per configuration
        num_error_samples: Number of error samples per graph
        seed: Base random seed
        save_path: Optional path to save the figure
    """
    if error_range is None:
        error_range = list(range(0, 16, 2))
    
    print(f"Running experiment varying m with n={n}, p={p}")
    print(f"m values: {m_values}")
    print(f"Error range: {error_range}")
    print()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for idx, m in enumerate(m_values):
        print(f"Processing m={m}...")
        means = compute_mean_scores(n, m, p, error_range, num_graphs, num_error_samples, seed + idx * 10000)
        
        ls = line_styles[idx % len(line_styles)]
        mk = markers[idx % len(markers)]
        ax.plot(error_range, means, linestyle=ls, marker=mk, color='black',
                markersize=6, linewidth=1.5, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=1.0, label=f'$m={m}$')
    
    ax.set_xlabel('True number of errors', fontsize=24)
    ax.set_ylabel('Average\nincompatibility score', fontsize=24)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    


if __name__ == "__main__":
    # Common parameters
    error_range = list(range(0, 11, 1))
    num_graphs = 20
    num_error_samples = 20
    seed = 42

    # Experiment 1: Vary n (number of observed variables)
    print("=" * 60)
    print("EXPERIMENT 1: Varying n (number of observed variables)")
    print("=" * 60)
    run_experiment_vary_n(
        n_values=[5, 7, 9, 11],
        m=3,
        p=0.3,
        error_range=error_range,
        num_graphs=num_graphs,
        num_error_samples=num_error_samples,
        seed=seed,
       save_path="graphical_synthetic_vary_n.png"
    )
    
    # Experiment 2: Vary p (edge probability)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Varying p (edge probability)")
    print("=" * 60)
    run_experiment_vary_p(
        n=10,
        m=3,
        p_values=[0.05, 0.1, 0.25, 0.5],
        error_range=error_range,
        num_graphs=num_graphs,
        num_error_samples=num_error_samples,
        seed=seed,
        save_path="graphical_synthetic_vary_p.png"
    )
    
    # Experiment 3: Vary m (number of hidden variables)
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Varying m (number of hidden variables)")
    print("=" * 60)
    run_experiment_vary_m(
        n=10,
        m_values=[0, 3, 5, 10],
        p=0.3,
        error_range=error_range,
        num_graphs=num_graphs,
        num_error_samples=num_error_samples,
        seed=seed,
        save_path="graphical_synthetic_vary_m.png" 
    )
    

