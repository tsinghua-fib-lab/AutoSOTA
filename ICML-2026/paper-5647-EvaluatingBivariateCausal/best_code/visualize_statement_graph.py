"""
Visualize Statement Graphs from LLM Graphical Discovery Experiments

This script visualizes the directed and bidirected edges from statement graphs
stored as adjacency matrices in LLM_responses_graphical/.

Usage:
    python visualize_statement_graph.py <model_name> <run_number>
    python visualize_statement_graph.py gpt-oss-120b-1 2
    python visualize_statement_graph.py claude-opus-4-5-20251101-v1 1
    python visualize_statement_graph.py --list  # List all available models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import glob
import re

# Define paths
DATA_DIR = os.path.dirname(__file__)
LLM_RESPONSES_DIR = os.path.join(DATA_DIR, 'LLM_responses_graphical')

# Variable names and short display names
VARIABLE_NAMES = [
    'population_density',
    'literacy_rate',
    'daily_income',
    'sanitation_access',
    'smoking',
    'happiness_score',
    'life_expectancy'
]

SHORT_NAMES = {
    'population_density': 'PopDens',
    'literacy_rate': 'Literacy',
    'daily_income': 'Income',
    'sanitation_access': 'Sanitation',
    'smoking': 'Smoking',
    'happiness_score': 'Happiness',
    'life_expectancy': 'LifeExp'
}


def discover_models():
    """Auto-discover all model names and their runs from CSV files."""
    pattern = os.path.join(LLM_RESPONSES_DIR, 'directed_adj_*_run*.csv')
    files = glob.glob(pattern)
    
    model_runs = {}  # model_name -> list of run numbers
    
    for f in files:
        basename = os.path.basename(f)
        basename = basename.replace('directed_adj_', '').replace('.csv', '')
        
        match = re.match(r'(.+)_run(\d+)$', basename)
        if match:
            model_name = match.group(1)
            run_num = int(match.group(2))
            if model_name not in model_runs:
                model_runs[model_name] = []
            model_runs[model_name].append(run_num)
    
    for model_name in model_runs:
        model_runs[model_name] = sorted(model_runs[model_name])
    
    return model_runs


def load_adjacency_matrices(model_name, run_num):
    """Load directed and bidirected adjacency matrices for a model/run."""
    directed_path = os.path.join(LLM_RESPONSES_DIR, f'directed_adj_{model_name}_run{run_num}.csv')
    bidirected_path = os.path.join(LLM_RESPONSES_DIR, f'bidirected_adj_{model_name}_run{run_num}.csv')
    
    if not os.path.exists(directed_path):
        raise FileNotFoundError(f"Directed adjacency matrix not found: {directed_path}")
    if not os.path.exists(bidirected_path):
        raise FileNotFoundError(f"Bidirected adjacency matrix not found: {bidirected_path}")
    
    directed_df = pd.read_csv(directed_path, index_col=0)
    bidirected_df = pd.read_csv(bidirected_path, index_col=0)
    
    # Reorder to match variable ordering
    directed_df = directed_df.loc[VARIABLE_NAMES, VARIABLE_NAMES]
    bidirected_df = bidirected_df.loc[VARIABLE_NAMES, VARIABLE_NAMES]
    
    return directed_df, bidirected_df


def extract_edges(directed_df, bidirected_df):
    """Extract directed and bidirected edges from adjacency matrices.
    
    Convention: A[i, j] = 1 means j -> i (j causes i)
    """
    variables = list(directed_df.columns)
    
    # Extract directed edges (A[i,j] = 1 means j -> i)
    directed_edges = []
    for i, row_var in enumerate(variables):
        for j, col_var in enumerate(variables):
            if directed_df.iloc[i, j] == 1:
                directed_edges.append((SHORT_NAMES[col_var], SHORT_NAMES[row_var]))
    
    # Extract bidirected edges (only upper triangle since symmetric)
    bidirected_edges = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            if bidirected_df.iloc[i, j] == 1:
                bidirected_edges.append((SHORT_NAMES[variables[i]], SHORT_NAMES[variables[j]]))
    
    return directed_edges, bidirected_edges


def visualize_statement_graph(model_name, run_num, save=True, show=True):
    """
    Visualize the statement graph for a given model and run.
    
    Parameters:
        model_name: Name of the model (e.g., 'gpt-oss-120b-1')
        run_num: Run number (1-5)
        save: Whether to save the figure as PNG
        show: Whether to display the figure
    
    Returns:
        fig: The matplotlib figure object
    """
    # Load adjacency matrices
    directed_df, bidirected_df = load_adjacency_matrices(model_name, run_num)
    
    # Extract edges
    directed_edges, bidirected_edges = extract_edges(directed_df, bidirected_df)
    
    print(f"\nStatement Graph: {model_name} (Run {run_num})")
    print("=" * 50)
    print(f"Directed edges ({len(directed_edges)}):")
    for u, v in directed_edges:
        print(f"  {u} -> {v}")
    print(f"\nBidirected edges ({len(bidirected_edges)}):")
    for u, v in bidirected_edges:
        print(f"  {u} <-> {v}")
    
    # Create graph
    G = nx.DiGraph()
    node_names = [SHORT_NAMES[v] for v in VARIABLE_NAMES]
    G.add_nodes_from(node_names)
    G.add_edges_from(directed_edges)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Position nodes in a circle
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=4500, 
                           edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
    
    # Draw directed edges (black arrows) with visible arrowheads
    if directed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=directed_edges, edge_color='black',
                               arrows=True, arrowsize=30, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.1', width=2.5,
                               node_size=4500, ax=ax, min_source_margin=30, min_target_margin=30)
    
    # Draw bidirected edges (red dashed double arrows)
    for u, v in bidirected_edges:
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5,
                                   connectionstyle='arc3,rad=0.2', 
                                   linestyle='dashed'))
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    if save:
        save_path = os.path.join(LLM_RESPONSES_DIR, f'statement_graph_{model_name}_run{run_num}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def list_available_models():
    """List all available models and their runs."""
    model_runs = discover_models()
    
    print("\nAvailable models and runs:")
    print("=" * 50)
    for model_name in sorted(model_runs.keys()):
        runs = model_runs[model_name]
        print(f"  {model_name}: runs {runs}")
    print()
    print("Usage: python visualize_statement_graph.py <model_name> <run_number>")
    print("Example: python visualize_statement_graph.py gpt-oss-120b-1 2")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_statement_graph.py <model_name> <run_number>")
        print("       python visualize_statement_graph.py --list")
        sys.exit(1)
    
    if sys.argv[1] == '--list' or sys.argv[1] == '-l':
        list_available_models()
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print("Error: Please provide both model name and run number")
        print("Usage: python visualize_statement_graph.py <model_name> <run_number>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    try:
        run_num = int(sys.argv[2])
    except ValueError:
        print(f"Error: Run number must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    # Check if model exists
    model_runs = discover_models()
    if model_name not in model_runs:
        print(f"Error: Model '{model_name}' not found")
        print("\nAvailable models:")
        for m in sorted(model_runs.keys()):
            print(f"  - {m}")
        sys.exit(1)
    
    if run_num not in model_runs[model_name]:
        print(f"Error: Run {run_num} not found for model '{model_name}'")
        print(f"Available runs: {model_runs[model_name]}")
        sys.exit(1)
    
    # Visualize
    visualize_statement_graph(model_name, run_num)


if __name__ == "__main__":
    main()
