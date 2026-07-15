"""
Visualization for LLM Graphical Causal Discovery Experiments

This script processes adjacency matrices from:
- LLM_responses_graphical/ (directed and bidirected adjacency matrices per run)

Computes incompatibility scores using graph_functions.incompatibility_score
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from graph_functions import incompatibility_score

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18,
    'figure.titlesize': 24,
})

# Define paths
DATA_DIR = os.path.dirname(__file__)
LLM_RESPONSES_DIR = os.path.join(DATA_DIR, 'LLM_responses_graphical')

VARIABLE_NAMES = [
    'population_density',
    'literacy_rate',
    'daily_income',
    'sanitation_access',
    'smoking',
    'happiness_score',
    'life_expectancy'
]

# Model name mapping for display in plots
MODEL_DISPLAY_NAMES = {
    'claude-opus-4-6-v1': 'Claude Opus 4.6',
    'claude-opus-4-5-20251101-v1': 'Claude Opus 4.5',
    'claude-opus-4-1-20250805-v1': 'Claude Opus 4.1',
    'kimi-k2-thinking': 'Kimi K2 Thinking',
    'mistral-large-3-675b-instruct': 'Mistral Large 3',
    'magistral-small-2509': 'Magistral Small',
    'gpt-oss-120b-1': 'GPT oss 120B',
    'gpt-oss-20b-1': 'GPT oss 20B',
    'qwen3-next-80b-a3b': 'Qwen3 Next 80B A3B',
    'qwen3-235b-a22b-2507-v1': 'Qwen3 235B A22b',
    'gemma-3-4b-it': 'Gemma 3 4B IT',
    'gemma-3-27b-it': 'Gemma 3 27B IT',
}


def get_display_name(model_name):
    """Get display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def load_adjacency_matrix(filepath):
    """Load an adjacency matrix from CSV."""
    df = pd.read_csv(filepath, index_col=0)
    # Reorder to match variable ordering
    df = df.loc[VARIABLE_NAMES, VARIABLE_NAMES]
    return df.values.astype(int)


def discover_models():
    """Auto-discover all model names and their runs from CSV files."""
    pattern = os.path.join(LLM_RESPONSES_DIR, 'directed_adj_*_run*.csv')
    files = glob.glob(pattern)
    
    model_runs = {}  # model_name -> list of run numbers
    
    for f in files:
        basename = os.path.basename(f)
        # Extract model name and run number
        # Format: directed_adj_{model_name}_run{N}.csv
        basename = basename.replace('directed_adj_', '').replace('.csv', '')
        
        # Find the _run{N} suffix
        match = re.match(r'(.+)_run(\d+)$', basename)
        if match:
            model_name = match.group(1)
            run_num = int(match.group(2))
            if model_name not in model_runs:
                model_runs[model_name] = []
            model_runs[model_name].append(run_num)
    
    # Sort run numbers for each model
    for model_name in model_runs:
        model_runs[model_name] = sorted(model_runs[model_name])
    
    return model_runs


def compute_incompatibility_scores():
    """
    Compute incompatibility scores for all LLM graphical results.
    """
    print(f"\n{'='*60}")
    print("LLM GRAPHICAL INCOMPATIBILITY SCORES")
    print(f"{'='*60}")
    
    # Auto-discover all models and their runs
    model_runs = discover_models()
    print(f"\nFound {len(model_runs)} models:")
    for model, runs in sorted(model_runs.items()):
        print(f"  - {model}: {len(runs)} runs")
    print()
    
    results = {}
    
    # Process each model
    for model_name in sorted(model_runs.keys()):
        runs = model_runs[model_name]
        print(f"-" * 50)
        print(f"Model: {model_name}")
        print("-" * 50)
        
        run_scores = []
        run_details = []
        
        for run_num in runs:
            try:
                # Load directed and bidirected adjacency matrices
                directed_path = os.path.join(LLM_RESPONSES_DIR, f'directed_adj_{model_name}_run{run_num}.csv')
                bidirected_path = os.path.join(LLM_RESPONSES_DIR, f'bidirected_adj_{model_name}_run{run_num}.csv')
                
                directed_adj = load_adjacency_matrix(directed_path)
                bidirected_adj = load_adjacency_matrix(bidirected_path)
                
                # Compute incompatibility score
                score, details = incompatibility_score(directed_adj, bidirected_adj, verbose=False)
                run_scores.append(score)
                run_details.append(details)
                
                print(f"  Run {run_num}: Score = {score} (FAS={details['fas_deletions']}, "
                      f"TE_del={details['te_deletions']}, TE_add={details['te_additions']}, "
                      f"CPC_del={details['cpc_deletions']}, CPC_add={details['cpc_additions']})")
                
            except Exception as e:
                print(f"  Run {run_num}: Error - {e}")
        
        if run_scores:
            results[model_name] = {
                'mean': np.mean(run_scores),
                'std': np.std(run_scores) if len(run_scores) > 1 else 0,
                'min': np.min(run_scores),
                'max': np.max(run_scores),
                'run_scores': run_scores,
                'run_details': run_details,
                'num_runs': len(run_scores)
            }
            print(f"  -> Average: {results[model_name]['mean']:.2f} ± {results[model_name]['std']:.2f} "
                  f"(min={results[model_name]['min']}, max={results[model_name]['max']}, {len(run_scores)} runs)")
        else:
            results[model_name] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'run_scores': [],
                'run_details': [],
                'num_runs': 0
            }
            print(f"  -> No valid runs")
        print()
    
    # Summary comparison
    print("=" * 60)
    print("SUMMARY COMPARISON - GRAPHICAL INCOMPATIBILITY SCORES")
    print("=" * 60)
    print(f"{'Model':<45} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6} {'Runs':>6}")
    print("-" * 80)
    for model, res in sorted(results.items(), 
                              key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else float('inf')):
        if res['mean'] is not None:
            print(f"{model:<45} {res['mean']:>8.2f} {res['std']:>8.2f} {res['min']:>6} {res['max']:>6} {res['num_runs']:>6}")
        else:
            print(f"{model:<45} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>6} {0:>6}")
    
    return results


def compute_incompatibility_scores_filtered(min_edges=20, max_edges=28):
    """
    Compute incompatibility scores only for runs with total edges in the specified range.
    
    Parameters:
        min_edges: Minimum number of edges (inclusive)
        max_edges: Maximum number of edges (inclusive)
    
    Returns:
        results: Dictionary of filtered results per model
    """
    print(f"\n{'='*60}")
    print(f"INCOMPATIBILITY SCORES (FILTERED: {min_edges}-{max_edges} EDGES)")
    print(f"{'='*60}")
    
    model_runs = discover_models()
    print(f"\nFound {len(model_runs)} models")
    print()
    
    results = {}
    
    for model_name in sorted(model_runs.keys()):
        runs = model_runs[model_name]
        print(f"Model: {model_name}")
        
        run_scores = []
        run_details = []
        included_runs = []
        
        for run_num in runs:
            try:
                directed_path = os.path.join(LLM_RESPONSES_DIR, f'directed_adj_{model_name}_run{run_num}.csv')
                bidirected_path = os.path.join(LLM_RESPONSES_DIR, f'bidirected_adj_{model_name}_run{run_num}.csv')
                
                directed_adj = load_adjacency_matrix(directed_path)
                bidirected_adj = load_adjacency_matrix(bidirected_path)
                
                # Count edges
                n_directed = np.sum(directed_adj) - np.trace(directed_adj)
                n_bidirected = np.sum(np.triu(bidirected_adj, k=1))
                n_total = n_directed + n_bidirected
                
                # Only include if in range
                if min_edges <= n_total <= max_edges:
                    score, details = incompatibility_score(directed_adj, bidirected_adj, verbose=False)
                    run_scores.append(score)
                    run_details.append(details)
                    included_runs.append(run_num)
                    print(f"  Run {run_num}: {n_total} edges, Score = {score}")
                
            except Exception as e:
                print(f"  Run {run_num}: Error - {e}")
        
        if run_scores:
            results[model_name] = {
                'mean': np.mean(run_scores),
                'std': np.std(run_scores) if len(run_scores) > 1 else 0,
                'min': np.min(run_scores),
                'max': np.max(run_scores),
                'run_scores': run_scores,
                'run_details': run_details,
                'included_runs': included_runs,
                'num_runs': len(run_scores)
            }
            print(f"  -> {len(run_scores)} runs in range, Mean: {results[model_name]['mean']:.2f} ± {results[model_name]['std']:.2f}")
        else:
            print(f"  -> No runs in edge range")
        print()
    
    # Summary
    print("=" * 60)
    print(f"SUMMARY - FILTERED ({min_edges}-{max_edges} EDGES)")
    print("=" * 60)
    print(f"{'Model':<45} {'Mean':>8} {'Std':>8} {'Runs':>6}")
    print("-" * 70)
    for model, res in sorted(results.items(), 
                              key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else float('inf')):
        print(f"{model:<45} {res['mean']:>8.2f} {res['std']:>8.2f} {res['num_runs']:>6}")
    
    return results


def plot_incompatibility_scores_filtered(results, min_edges=20, max_edges=28, save_path=None, vmin=None, vmax=None):
    """
    Create a bar plot of filtered incompatibility scores.
    
    Parameters:
        vmin, vmax: Optional min/max values for color normalization (for consistent coloring across plots)
    """
    valid_results = {k: v for k, v in results.items() if v.get('mean') is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return None
    
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['mean'])
    model_names = [get_display_name(m[0]) for m in sorted_models]
    means = [m[1]['mean'] for m in sorted_models]
    stds = [m[1]['std'] for m in sorted_models]
    num_runs = [m[1]['num_runs'] for m in sorted_models]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(model_names) * 0.6)))
    
    y_pos = np.arange(len(model_names))
    
    # Color based on score (use provided vmin/vmax for consistent coloring)
    norm_scores = np.array(means)
    score_min = vmin if vmin is not None else norm_scores.min()
    score_max = vmax if vmax is not None else norm_scores.max()
    if score_max > score_min:
        norm_scores = (norm_scores - score_min) / (score_max - score_min)
        norm_scores = np.clip(norm_scores, 0, 1)  # Clip to [0, 1] range
    else:
        norm_scores = np.zeros_like(norm_scores)
    colors = plt.cm.RdYlGn_r(norm_scores)
    
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor='black', 
                   alpha=0.8, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{m} ($n={n}$)" for m, n in zip(model_names, num_runs)])
    ax.set_xlabel('Incompatibility scores (mean \u00b1 std)\n'
                  r'for statement graphs with edge density $\leq 2/3$')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels - position to the right of error bars
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for i, (mean, std) in enumerate(zip(means, stds)):
        label_x = mean + std + x_range * 0.02
        ax.text(label_x, i, f'{mean:.1f}\u00b1{std:.1f}', va='center', fontweight='bold')
    
    # Adjust x-axis to make room for labels
    ax.set_xlim(x_min, x_max + x_range * 0.20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_incompatibility_scores(results, save_path=None, vmin=None, vmax=None):
    """
    Create a bar plot of incompatibility scores with error bars for all models.
    Lower scores are better (fewer modifications needed).
    
    Parameters:
        vmin, vmax: Optional min/max values for color normalization (for consistent coloring across plots)
    """
    # Filter out models with no valid results
    valid_results = {k: v for k, v in results.items() if v['mean'] is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return None
    
    # Sort models by mean score (lower is better)
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['mean'])
    model_names = [get_display_name(m[0]) for m in sorted_models]
    means = [m[1]['mean'] for m in sorted_models]
    stds = [m[1]['std'] for m in sorted_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(model_names) * 0.6)))
    
    y_pos = np.arange(len(model_names))
    
    # Color based on score (green = low/good, red = high/bad)
    # Use provided vmin/vmax for consistent coloring across plots
    norm_scores = np.array(means)
    score_min = vmin if vmin is not None else norm_scores.min()
    score_max = vmax if vmax is not None else norm_scores.max()
    if score_max > score_min:
        norm_scores = (norm_scores - score_min) / (score_max - score_min)
        norm_scores = np.clip(norm_scores, 0, 1)  # Clip to [0, 1] range
    else:
        norm_scores = np.zeros_like(norm_scores)
    colors = plt.cm.RdYlGn_r(norm_scores)  # Red for high, green for low
    
    # Create horizontal bar chart with error bars
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor='black', 
                   alpha=0.8, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Incompatibility Score (Mean \u00b1 Std)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels - position to the right of error bars
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for i, (mean, std) in enumerate(zip(means, stds)):
        label_x = mean + std + x_range * 0.02
        ax.text(label_x, i, f'{mean:.1f}\u00b1{std:.1f}', va='center', fontweight='bold')
    
    # Adjust x-axis to make room for labels
    ax.set_xlim(x_min, x_max + x_range * 0.20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig



def plot_score_breakdown(results, save_path=None):
    """
    Create a stacked bar plot showing the breakdown of incompatibility score components.
    """
    # Filter out models with no valid results
    valid_results = {k: v for k, v in results.items() if v.get('run_details')}
    
    if not valid_results:
        print("No valid results to plot")
        return None
    
    # Sort by mean score (lower is better)
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['mean'])
    model_names = [get_display_name(m[0]) for m in sorted_models]
    
    # Compute average breakdown for each model
    fas_means = []
    te_del_means = []
    te_add_means = []
    cpc_del_means = []
    cpc_add_means = []
    
    for model_name, data in sorted_models:
        details_list = data['run_details']
        fas_means.append(np.mean([d['fas_deletions'] for d in details_list]))
        te_del_means.append(np.mean([d['te_deletions'] for d in details_list]))
        te_add_means.append(np.mean([d['te_additions'] for d in details_list]))
        cpc_del_means.append(np.mean([d['cpc_deletions'] for d in details_list]))
        cpc_add_means.append(np.mean([d['cpc_additions'] for d in details_list]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(model_names) * 0.6)))
    
    y_pos = np.arange(len(model_names))
    
    # Create stacked horizontal bar chart
    bars1 = ax.barh(y_pos, fas_means, label='FAS Deletions (cycles)', color='#e74c3c', alpha=0.8)
    bars2 = ax.barh(y_pos, te_del_means, left=fas_means, label='TE Deletions', color='#f39c12', alpha=0.8)
    
    left2 = np.array(fas_means) + np.array(te_del_means)
    bars3 = ax.barh(y_pos, te_add_means, left=left2, label='TE Additions', color='#9b59b6', alpha=0.8)
    
    left3 = left2 + np.array(te_add_means)
    bars4 = ax.barh(y_pos, cpc_del_means, left=left3, label='CPC Deletions', color='#3498db', alpha=0.8)
    
    left4 = left3 + np.array(cpc_del_means)
    bars5 = ax.barh(y_pos, cpc_add_means, left=left4, label='CPC Additions', color='#2ecc71', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Incompatibility Score Breakdown')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add total labels - position to the right of the stacked bars
    totals = np.array(fas_means) + np.array(te_del_means) + np.array(te_add_means) + np.array(cpc_del_means) + np.array(cpc_add_means)
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for i, total in enumerate(totals):
        label_x = total + x_range * 0.02
        ax.text(label_x, i, f'{total:.1f}', va='center', fontweight='bold')
    
    # Adjust x-axis to make room for labels
    ax.set_xlim(x_min, x_max + x_range * 0.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def compute_edge_counts():
    """
    Compute the total number of edges (directed + bidirected) for each model and run.
    """
    print(f"\n{'='*60}")
    print("EDGE COUNTS PER MODEL AND RUN")
    print(f"{'='*60}")
    
    model_runs = discover_models()
    
    results = {}
    
    for model_name in sorted(model_runs.keys()):
        runs = model_runs[model_name]
        print(f"\nModel: {model_name}")
        
        run_directed = []
        run_bidirected = []
        run_total = []
        
        for run_num in runs:
            try:
                directed_path = os.path.join(LLM_RESPONSES_DIR, f'directed_adj_{model_name}_run{run_num}.csv')
                bidirected_path = os.path.join(LLM_RESPONSES_DIR, f'bidirected_adj_{model_name}_run{run_num}.csv')
                
                directed_adj = load_adjacency_matrix(directed_path)
                bidirected_adj = load_adjacency_matrix(bidirected_path)
                
                # Count directed edges (all non-zero off-diagonal entries)
                n_directed = np.sum(directed_adj) - np.trace(directed_adj)
                
                # Count bidirected edges (upper triangle only since symmetric)
                n_bidirected = np.sum(np.triu(bidirected_adj, k=1))
                
                n_total = n_directed + n_bidirected
                
                run_directed.append(n_directed)
                run_bidirected.append(n_bidirected)
                run_total.append(n_total)
                
                print(f"  Run {run_num}: Directed={n_directed}, Bidirected={n_bidirected}, Total={n_total}")
                
            except Exception as e:
                print(f"  Run {run_num}: Error - {e}")
        
        if run_total:
            results[model_name] = {
                'directed_mean': np.mean(run_directed),
                'directed_std': np.std(run_directed) if len(run_directed) > 1 else 0,
                'bidirected_mean': np.mean(run_bidirected),
                'bidirected_std': np.std(run_bidirected) if len(run_bidirected) > 1 else 0,
                'total_mean': np.mean(run_total),
                'total_std': np.std(run_total) if len(run_total) > 1 else 0,
                'run_directed': run_directed,
                'run_bidirected': run_bidirected,
                'run_total': run_total,
                'num_runs': len(run_total)
            }
            print(f"  -> Total edges: {results[model_name]['total_mean']:.1f} ± {results[model_name]['total_std']:.1f}")
    
    return results


def count_runs_in_edge_range(edge_results, min_edges=20, max_edges=28):
    """
    Count how many runs each model has with total edges in the specified range.
    
    Parameters:
        edge_results: Results from compute_edge_counts()
        min_edges: Minimum number of edges (inclusive)
        max_edges: Maximum number of edges (inclusive)
    
    Returns:
        dict: model_name -> {'in_range': count, 'total': total_runs, 'runs_in_range': [(run_idx, edge_count), ...]}
    """
    print(f"\n{'='*60}")
    print(f"RUNS WITH {min_edges}-{max_edges} EDGES")
    print(f"{'='*60}")
    print(f"{'Model':<50} {'In Range':>10} {'Total':>8}")
    print("-" * 70)
    
    results = {}
    
    for model_name in sorted(edge_results.keys()):
        data = edge_results[model_name]
        run_totals = data.get('run_total', [])
        
        in_range_runs = []
        for i, total in enumerate(run_totals):
            if min_edges <= total <= max_edges:
                in_range_runs.append((i + 1, total))  # 1-indexed run number
        
        results[model_name] = {
            'in_range': len(in_range_runs),
            'total': len(run_totals),
            'runs_in_range': in_range_runs
        }
        
        print(f"{model_name:<50} {len(in_range_runs):>10} {len(run_totals):>8}")
        if in_range_runs:
            for run_num, total in in_range_runs:
                print(f"    Run {run_num}: {total} edges")
    
    return results


def plot_edge_counts(edge_results, save_path=None):
    """
    Create a stacked bar plot showing directed and bidirected edge counts for each model.
    """
    valid_results = {k: v for k, v in edge_results.items() if v.get('run_total')}
    
    if not valid_results:
        print("No valid results to plot")
        return None
    
    # Sort by total edges
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['total_mean'])
    model_names = [get_display_name(m[0]) for m in sorted_models]
    directed_means = [m[1]['directed_mean'] for m in sorted_models]
    bidirected_means = [m[1]['bidirected_mean'] for m in sorted_models]
    total_stds = [m[1]['total_std'] for m in sorted_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(model_names) * 0.6)))
    
    y_pos = np.arange(len(model_names))
    
    # Stacked horizontal bar chart
    bars1 = ax.barh(y_pos, directed_means, label='Directed Edges', color='#3498db', alpha=0.8)
    bars2 = ax.barh(y_pos, bidirected_means, left=directed_means, label='Bidirected Edges', 
                    color='#e74c3c', alpha=0.8)
    
    # Add error bars for total
    totals = np.array(directed_means) + np.array(bidirected_means)
    ax.errorbar(totals, y_pos, xerr=total_stds, fmt='none', color='black', 
                capsize=3, elinewidth=1.5, capthick=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Number of Edges (Mean \u00b1 Std)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add total labels - position to the right of error bars
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for i, (d, b, std) in enumerate(zip(directed_means, bidirected_means, total_stds)):
        total = d + b
        label_x = total + std + x_range * 0.02
        ax.text(label_x, i, f'{total:.0f}\u00b1{std:.1f}', 
                va='center', fontweight='bold')
    
    # Adjust x-axis to make room for labels
    ax.set_xlim(x_min, x_max + x_range * 0.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig



def print_adjacency_matrices(model_name, run_num=1):
    """Print the adjacency matrices for a specific model and run."""
    try:
        directed_path = os.path.join(LLM_RESPONSES_DIR, f'directed_adj_{model_name}_run{run_num}.csv')
        bidirected_path = os.path.join(LLM_RESPONSES_DIR, f'bidirected_adj_{model_name}_run{run_num}.csv')
        
        directed_adj = load_adjacency_matrix(directed_path)
        bidirected_adj = load_adjacency_matrix(bidirected_path)
        
        print(f"\nDirected Adjacency Matrix for {model_name} (Run {run_num}):")
        print(pd.DataFrame(directed_adj, index=VARIABLE_NAMES, columns=VARIABLE_NAMES))
        
        print(f"\nBidirected Adjacency Matrix for {model_name} (Run {run_num}):")
        print(pd.DataFrame(bidirected_adj, index=VARIABLE_NAMES, columns=VARIABLE_NAMES))
        
    except Exception as e:
        print(f"Error loading matrices: {e}")


if __name__ == "__main__":
    import sys
    
    # Check if response directory exists
    if not os.path.exists(LLM_RESPONSES_DIR):
        print(f"Directory not found: {LLM_RESPONSES_DIR}")
        print("Run experiments_llm_graphical.py first to generate results.")
        sys.exit(1)
    
    # Compute incompatibility scores for all models
    results = compute_incompatibility_scores()
    
    # Compute edge counts for all models
    edge_results = compute_edge_counts()
    
    # Count runs in edge range 10-28
    range_results = count_runs_in_edge_range(edge_results, min_edges=10, max_edges=28)
    
    # Compute incompatibility scores filtered to 10-28 edges
    filtered_results = compute_incompatibility_scores_filtered(min_edges=10, max_edges=28)
    
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print("=" * 60)
    
    # Compute global color range for consistent coloring across plots
    all_means = []
    for res in results.values():
        if res['mean'] is not None:
            all_means.append(res['mean'])
    for res in filtered_results.values():
        if res.get('mean') is not None:
            all_means.append(res['mean'])
    global_vmin = min(all_means) if all_means else 0
    global_vmax = max(all_means) if all_means else 1
    
    # Generate and save plots
    scores_path = os.path.join(LLM_RESPONSES_DIR, 'graphical_incompatibility_scores.png')
    plot_incompatibility_scores(results, save_path=scores_path, vmin=global_vmin, vmax=global_vmax)
    
    breakdown_path = os.path.join(LLM_RESPONSES_DIR, 'graphical_score_breakdown.png')
    plot_score_breakdown(results, save_path=breakdown_path)
    
    # Edge count plots
    edge_path = os.path.join(LLM_RESPONSES_DIR, 'graphical_edge_counts.png')
    plot_edge_counts(edge_results, save_path=edge_path)
    
    # Filtered incompatibility scores plot
    if filtered_results:
        filtered_path = os.path.join(LLM_RESPONSES_DIR, 'graphical_incompatibility_scores_filtered.png')
        plot_incompatibility_scores_filtered(filtered_results, min_edges=10, max_edges=28, save_path=filtered_path, vmin=global_vmin, vmax=global_vmax)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE - GRAPHICAL DISCOVERY")
    print("=" * 60)
