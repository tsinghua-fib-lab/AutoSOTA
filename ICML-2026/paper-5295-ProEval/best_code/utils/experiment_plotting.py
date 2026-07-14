#!/usr/bin/env python3

# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plotting utility for experiment results.

This script reads experiment CSV files and generates:
1. Failure rate comparison plot (Active vs Random)
2. Scatter plots showing embedding space distribution

Usage:
    python plot.py --csv experiment_generation_gsm8k_v1.csv --output_prefix gsm8k
    python plot.py --csv experiment_generation_strategyqa_v1.csv --output_prefix strategyqa
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional

# Import plotting utilities from local plotting module
try:
    from .plotting import set_style, set_size, COLOR_LIST, MARKER_LIST, WIDTH
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False
    print("Warning: Could not import plotting utilities")

# OpenRouter Configuration for embeddings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Baseline strategy names for consistent plotting
# HSS-Gen: HSS anchor selection + LLM generation
# HSS: Direct selection from seed pool (no generation)
# Random Topic: Random generation with topic injection
# Pure Random: Random generation without topic injection
STRATEGY_NAMES = {
    'pure_random': 'Pure Random',
    'random_topic': 'Random Topic', 
    'random': 'Random Topic',  # backwards compat
    'hss_gen': 'HSS-Gen',
    'hss': 'HSS',
    'variance': 'Variance',
    'level_set': 'Level Set',
    'thompson': 'Thompson',
}

# Standardized color palette (matching plot_samples_to_failure.py)
# Qualitative palette: Gray #cecece, Purple #a559aa, Teal #59a89c, Gold #f0c571, Red #e02b35, Dark Blue #082a54
METHOD_COLORS = {
    'SS': '#59a89c',          # Teal - Active sampling, no-gen
    'Rand': '#cecece',        # Gray - Random sampling
    'SS-Gen': '#e02b35',      # Red - SS with generation
    'TSS': '#082a54',         # Dark Blue - Topic + SS + generation (best method)
    'Rand-T-Gen': '#f0c571',  # Gold - Random Topic + generation
    'Rand-Gen': '#a559aa',    # Purple - Pure random generation
}

# Preferred method order for legend display
PREFERRED_METHOD_ORDER = ['Rand', 'Rand-Gen', 'Rand-T-Gen', 'SS', 'SS-Gen', 'TSS']

def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for text using OpenRouter embeddings API."""
    import requests
    
    if not OPENROUTER_API_KEY:
        print("Warning: No OpenRouter API key, using random embeddings")
        return np.random.randn(1536)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data or len(data["data"]) == 0:
            raise Exception(f"No embedding returned: {data}")
        
        return np.array(data["data"][0]["embedding"])
        
    except Exception as e:
        print(f"[Error] Embedding failed: {e}")
        return np.random.randn(1536)


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load experiment CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Strategies: {df['Strategy'].unique()}")
    return df


def load_v2_npz_data(npz_path: str) -> dict:
    """
    Load V2 generator results from .npz file.
    
    Returns:
        dict with keys like 'Random_mean', 'HSS (Ours)_mean', 'Random_std', etc.
    """
    data = np.load(npz_path, allow_pickle=True)
    results = {key: data[key] for key in data.files}
    
    print(f"Loaded results from {npz_path}")
    print(f"Keys: {list(results.keys())}")
    if 'n_runs' in results:
        print(f"Number of runs: {results['n_runs']}")
    
    return results


def load_v2_csv_data(csv_path: str) -> pd.DataFrame:
    """Load V2 generator detailed CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Methods: {df['Method'].unique()}")
    print(f"Runs: {df['Run'].unique()}")
    return df


def plot_v2_from_csv(csv_path: str, output_prefix: str = 'gsm8k_v2',
                      filter_methods: list = None):
    """
    Plot comparison from V2 detailed CSV with std shading.
    
    Computes cumulative failures per method and optionally aggregates across runs.
    """
    # Preferred order for legend display
    PREFERRED_METHOD_ORDER = ['Rand', 'Rand-Gen', 'Rand-T-Gen', 'SS', 'SS-Gen', 'TSS']
    
    df = load_v2_csv_data(csv_path)
    
    # Get unique methods
    all_methods = df['Method'].unique().tolist()
    
    if filter_methods:
        methods = [m for m in all_methods if m in filter_methods]
        print(f"Filtering to: {methods}")
    else:
        methods = all_methods
    
    # Sort methods by preferred order
    def method_sort_key(m):
        if m in PREFERRED_METHOD_ORDER:
            return PREFERRED_METHOD_ORDER.index(m)
        return len(PREFERRED_METHOD_ORDER)  # Unknown methods go at end
    
    methods = sorted(methods, key=method_sort_key)
    
    # Get number of runs
    n_runs = df['Run'].nunique()
    n_iterations = df.groupby(['Method', 'Run']).size().max()
    
    print(f"Detected: {n_runs} run(s), ~{n_iterations} iterations per run")
    
    # Compute cumulative failures per method and run
    results_mean = {}
    results_std = {}
    
    for method in methods:
        method_df = df[df['Method'] == method]
        runs_data = []
        
        for run_id in df['Run'].unique():
            run_df = method_df[method_df['Run'] == run_id].sort_values('Iteration')
            if len(run_df) > 0:
                cumulative_failures = (run_df['Score'] == 0).cumsum().values
                runs_data.append(cumulative_failures)
        
        if runs_data:
            # Pad to same length if needed
            max_len = max(len(r) for r in runs_data)
            padded = [np.pad(r, (0, max_len - len(r)), mode='edge') for r in runs_data]
            runs_array = np.array(padded)
            
            results_mean[method] = np.mean(runs_array, axis=0)
            if n_runs > 1:
                results_std[method] = np.std(runs_array, axis=0)
    
    # Plot
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig, ax = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    # Use fallback color for methods not in METHOD_COLORS
    fallback_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, method in enumerate(methods):
        if method not in results_mean:
            continue
        mean_data = results_mean[method]
        x_axis = np.arange(1, len(mean_data) + 1)
        
        # Use standardized color if available, else fallback
        color = METHOD_COLORS.get(method, fallback_colors[i % len(fallback_colors)])
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(x_axis, mean_data, linestyle=linestyle, color=color,
                label=method, linewidth=2)
        
        if method in results_std:
            std_data = results_std[method]
            ax.fill_between(x_axis, mean_data - std_data, mean_data + std_data,
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Failures', fontsize=12, fontweight='bold', rotation=90, va='bottom')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va='center')
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    ax.set_ylim(0, None)  # Start y-axis from 0
    plt.tight_layout()
    
    plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_comparison.pdf', bbox_inches='tight')
    print(f"Comparison plot saved to: {output_prefix}_comparison.png, {output_prefix}_comparison.pdf")
    plt.close()
    
    # Print final results
    print(f"\nFinal Results:")
    for method in methods:
        if method not in results_mean:
            continue
        mean_val = results_mean[method][-1]
        if method in results_std:
            std_val = results_std[method][-1]
            print(f"  {method}: {mean_val:.1f} ± {std_val:.1f} failures")
        else:
            print(f"  {method}: {mean_val:.1f} failures")


def plot_v2_comparison(npz_path: str, output_prefix: str = 'gsm8k_v2', 
                       filter_methods: list = None):
    """
    Plot comparison from V2 generator npz results with std shading.
    
    Args:
        npz_path: Path to npz file
        output_prefix: Prefix for output files
        filter_methods: Optional list of method names to include (e.g., ['Random', 'HSS (Ours)'])
    
    Supports gsm8k_v2_runall_results.npz format with:
    - <method>_mean: mean failure history
    - <method>_std: std deviation (if n_runs > 1)
    """
    data = load_v2_npz_data(npz_path)
    
    # Extract method names (keys ending in _mean)
    all_methods = [k.replace('_mean', '') for k in data.keys() if k.endswith('_mean')]
    
    # Filter if specified
    if filter_methods:
        method_names = [m for m in all_methods if m in filter_methods]
        print(f"Filtering to: {method_names} (from {all_methods})")
    else:
        method_names = all_methods
    
    if not method_names:
        print("No valid method data found in npz file")
        return
    
    print(f"Methods: {method_names}")
    
    # Setup plot
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig, ax = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    # Use fallback color for methods not in METHOD_COLORS
    fallback_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, name in enumerate(method_names):
        mean_key = f'{name}_mean'
        std_key = f'{name}_std'
        
        mean_data = np.array(data[mean_key])
        x_axis = np.arange(1, len(mean_data) + 1)
        
        # Use standardized color if available, else fallback
        color = METHOD_COLORS.get(name, fallback_colors[i % len(fallback_colors)])
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(x_axis, mean_data, linestyle=linestyle, color=color, 
                label=name, linewidth=2)
        
        # Add shading if std available
        if std_key in data:
            std_data = np.array(data[std_key])
            ax.fill_between(x_axis, mean_data - std_data, mean_data + std_data,
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Failures', fontsize=12, fontweight='bold', rotation=90, va='bottom')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va='center')
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=12)
    ax.set_ylim(0, None)  # Start y-axis from 0
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_comparison.pdf', bbox_inches='tight')
    print(f"Comparison plot saved to: {output_prefix}_comparison.png, {output_prefix}_comparison.pdf")
    plt.close()
    
    # Print final results
    print(f"\nFinal Results:")
    for name in method_names:
        mean_val = data[f'{name}_mean'][-1]
        if f'{name}_std' in data:
            std_val = data[f'{name}_std'][-1]
            print(f"  {name}: {mean_val:.1f} ± {std_val:.1f} failures")
        else:
            print(f"  {name}: {mean_val:.1f} failures")


def compute_failure_rates(df: pd.DataFrame) -> tuple:
    """
    Compute cumulative failure rates for each strategy.
    
    Returns:
        (random_failure_pct, active_failure_pct, n_iterations)
    """
    # Separate by strategy
    random_df = df[df['Strategy'] == 'random'].sort_values('Iteration')
    active_df = df[df['Strategy'] == 'active'].sort_values('Iteration')
    
    # Compute cumulative failures
    random_failures = (random_df['Score'] == 0.0).cumsum().values
    active_failures = (active_df['Score'] == 0.0).cumsum().values
    
    n_iterations = len(random_df)
    
    # Convert to percentages
    random_failure_pct = (random_failures / n_iterations) * 100
    active_failure_pct = (active_failures / n_iterations) * 100
    
    return random_failure_pct, active_failure_pct, n_iterations


def plot_failure_rate(df: pd.DataFrame, output_prefix: str):
    """Plot failure rate comparison between Active and Random strategies."""
    print("\nGenerating failure rate plot...")
    
    random_failure_pct, active_failure_pct, n_iterations = compute_failure_rates(df)
    
    # Use LaTeX theme if available
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig, ax = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
        color_random = COLOR_LIST[2]
        color_active = COLOR_LIST[0]
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        color_random = '#E74C3C'
        color_active = '#3498DB'
    
    x_axis = np.arange(1, n_iterations + 1)
    
    # Plot failure percentage
    ax.plot(x_axis, random_failure_pct, linestyle='--', 
            color=color_random, label='Random', linewidth=1.5)
    
    ax.plot(x_axis, active_failure_pct, linestyle='-', 
            color=color_active, label='Active (Ours)', linewidth=2)
    
    ax.set_xlabel('Number of Samples', fontsize=9)
    ax.set_ylabel('Failure Rate (\%)', fontsize=9, rotation=90, va='bottom')
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va='center')
    ax.set_ylim(0, 100)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    
    plt.tight_layout()
    
    # Save in both PNG and PDF formats
    output_path_png = f'{output_prefix}_failures.png'
    output_path_pdf = f'{output_prefix}_failures.pdf'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Failure rate plot saved to: {output_path_png}, {output_path_pdf}")
    
    plt.close()
    
    # Print summary
    print(f"\nFinal Results (after {n_iterations} iterations):")
    print(f"  Random: {int(random_failure_pct[-1] * n_iterations / 100)}/{n_iterations} failures ({random_failure_pct[-1]:.1f}%)")
    print(f"  Active: {int(active_failure_pct[-1] * n_iterations / 100)}/{n_iterations} failures ({active_failure_pct[-1]:.1f}%)")
    diff = active_failure_pct[-1] - random_failure_pct[-1]
    print(f"  Difference: {diff:+.1f}% (higher is better for finding failures)")


def plot_embedding_scatter(df: pd.DataFrame, output_prefix: str, use_cached: bool = True):
    """
    Plot scatter visualization of generated problems in 2D embedding space.
    
    Args:
        df: DataFrame with experiment results
        output_prefix: Prefix for output file names
        use_cached: If True, try to load cached embeddings
    """
    print("\nGenerating scatter plots...")
    
    # Separate by strategy
    random_df = df[df['Strategy'] == 'random'].copy()
    active_df = df[df['Strategy'] == 'active'].copy()
    
    print(f"  Random samples: {len(random_df)}")
    print(f"  Active samples: {len(active_df)}")
    
    if len(random_df) == 0 or len(active_df) == 0:
        print("Not enough data for scatter plot - need both strategies")
        return
    
    # Compute embeddings for questions
    print("Loading/computing embeddings for questions...")
    
    # Check for cache files (plot.py format or generator format)
    cache_file = f'{output_prefix}_embeddings.npz'
    # Also check for generator-created cache (without underscore)
    alt_cache_file = f'{output_prefix.replace("_", "")}_embeddings.npz' if '_' in output_prefix else None
    
    cache_valid = False
    
    # Try primary cache file
    if use_cached and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        random_embeddings = data['random_embeddings']
        active_embeddings = data['active_embeddings']
        random_scores = data['random_scores']
        active_scores = data['active_scores']
        
        # Validate cache size matches current data
        if len(random_embeddings) == len(random_df) and len(active_embeddings) == len(active_df):
            cache_valid = True
            print(f"  Cache valid: {len(random_embeddings)} random, {len(active_embeddings)} active embeddings")
        else:
            print(f"  Cache size mismatch! Cache has {len(random_embeddings)} random, {len(active_embeddings)} active")
            print(f"  Expected {len(random_df)} random, {len(active_df)} active")
    
    # Try alternative cache file (from generator)
    if not cache_valid and use_cached and alt_cache_file and os.path.exists(alt_cache_file):
        print(f"Loading cached embeddings from {alt_cache_file}")
        data = np.load(alt_cache_file)
        random_embeddings = data['random_embeddings']
        active_embeddings = data['active_embeddings']
        random_scores = data['random_scores']
        active_scores = data['active_scores']
        
        if len(random_embeddings) == len(random_df) and len(active_embeddings) == len(active_df):
            cache_valid = True
            print(f"  Cache valid: {len(random_embeddings)} random, {len(active_embeddings)} active embeddings")
        else:
            print(f"  Cache size mismatch! Skipping...")
    
    if not cache_valid:
        print(f"Computing {len(random_df) + len(active_df)} embeddings via API...")
        print("  (Warning: This requires OPENROUTER_API_KEY. If not set, results may be poor)")
        random_embeddings = np.array([get_embedding(q) for q in random_df['Question'].values])
        active_embeddings = np.array([get_embedding(q) for q in active_df['Question'].values])
        random_scores = random_df['Score'].values
        active_scores = active_df['Score'].values
        
        # Cache embeddings
        np.savez(cache_file, 
                 random_embeddings=random_embeddings, 
                 active_embeddings=active_embeddings,
                 random_scores=random_scores,
                 active_scores=active_scores)
        print(f"Cached embeddings to {cache_file}")
    
    print(f"  Total embeddings: {len(random_embeddings) + len(active_embeddings)}")
    
    if len(active_embeddings) == 0 or len(random_embeddings) == 0:
        print("Not enough data for scatter plot")
        return
    
    # Combine for PCA
    all_embeddings = np.vstack([active_embeddings, random_embeddings])
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_embeddings)
    
    # Split back
    active_reduced = all_reduced[:len(active_embeddings)]
    random_reduced = all_reduced[len(active_embeddings):]
    
    # Colors: Blue for success, Red for failure
    if LATEX_AVAILABLE:
        color_success = COLOR_LIST[2]  # Blue
        color_failure = COLOR_LIST[0]  # Red
    else:
        color_success = '#6495ED'  # Blue
        color_failure = '#DC143C'  # Red
    
    # Figure 1: BQ Active Sampling
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig_active, ax_active = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
    else:
        fig_active, ax_active = plt.subplots(figsize=(4, 4))
    
    success_mask = active_scores == 1.0
    failure_mask = active_scores == 0.0
    
    ax_active.scatter(active_reduced[success_mask, 0], active_reduced[success_mask, 1], 
                      c=color_success, alpha=0.7, s=50, label='Success', 
                      edgecolors='white', linewidths=0.5)
    ax_active.scatter(active_reduced[failure_mask, 0], active_reduced[failure_mask, 1], 
                      c=color_failure, alpha=0.7, s=50, label='Failure', 
                      edgecolors='white', linewidths=0.5)
    
    ax_active.set_xlabel('PC1', fontsize=9)
    ax_active.set_ylabel('PC2', fontsize=9, rotation=90, va='bottom')
    ax_active.tick_params(axis='both', which='major', labelsize=8)
    plt.setp(ax_active.yaxis.get_majorticklabels(), rotation=90, va='center')
    ax_active.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    
    plt.tight_layout()
    active_path_png = f'{output_prefix}_scatter_active.png'
    active_path_pdf = f'{output_prefix}_scatter_active.pdf'
    plt.savefig(active_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(active_path_pdf, bbox_inches='tight')
    print(f"Active scatter plot saved to: {active_path_png}, {active_path_pdf}")
    plt.close(fig_active)
    
    # Figure 2: Random Sampling
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig_random, ax_random = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
    else:
        fig_random, ax_random = plt.subplots(figsize=(4, 4))
    
    success_mask = random_scores == 1.0
    failure_mask = random_scores == 0.0
    
    ax_random.scatter(random_reduced[success_mask, 0], random_reduced[success_mask, 1], 
                      c=color_success, alpha=0.5, s=50, label='Success', 
                      edgecolors='white', linewidths=0.5)
    ax_random.scatter(random_reduced[failure_mask, 0], random_reduced[failure_mask, 1], 
                      c=color_failure, alpha=0.5, s=50, label='Failure', 
                      edgecolors='white', linewidths=0.5)
    
    ax_random.set_xlabel('PC1', fontsize=9)
    ax_random.set_ylabel('PC2', fontsize=9, rotation=90, va='bottom')
    ax_random.tick_params(axis='both', which='major', labelsize=8)
    plt.setp(ax_random.yaxis.get_majorticklabels(), rotation=90, va='center')
    ax_random.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    
    plt.tight_layout()
    random_path_png = f'{output_prefix}_scatter_random.png'
    random_path_pdf = f'{output_prefix}_scatter_random.pdf'
    plt.savefig(random_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(random_path_pdf, bbox_inches='tight')
    print(f"Random scatter plot saved to: {random_path_png}, {random_path_pdf}")
    plt.close(fig_random)


def plot_v2_embedding_scatter(csv_path: str, output_prefix: str = 'v2',
                               active_method: str = 'HSS',
                               random_method: str = 'Rand-Gen',
                               run_to_use: int = 1,
                               use_cached: bool = True,
                               failure_only: bool = False):
    """
    Plot PCA scatter visualization for V2 generator results.
    
    Creates a 2x2 grid showing embedding space distributions:
    - Top: GSM8K (Active vs Random)  -- or single dataset
    - Bottom: StrategyQA (Active vs Random) -- if available
    
    For single dataset, creates a 1x2 grid: Active (left) vs Random (right)
    
    Args:
        csv_path: Path to V2 detailed CSV file
        output_prefix: Prefix for output files
        active_method: Which active method to compare with baseline (default: 'HSS')
        random_method: Which method to use as baseline (default: 'Rand-Gen')
        run_to_use: Which run to use for visualization (default: 1)
        use_cached: Whether to use cached embeddings
    """
    from tqdm import tqdm
    
    print(f"\n📊 Generating V2 PCA scatter plots...")
    print(f"  Active method: {active_method}")
    print(f"  Using run: {run_to_use}")
    
    # Load V2 detailed CSV
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} records from {csv_path}")
    
    # Filter to specific run
    df = df[df['Run'] == run_to_use]
    print(f"  Filtered to run {run_to_use}: {len(df)} records")
    
    # Get data for Active method and baseline (random) method
    active_df = df[df['Method'] == active_method].copy()
    random_df = df[df['Method'] == random_method].copy()
    
    print(f"  {active_method}: {len(active_df)} samples")
    print(f"  {random_method}: {len(random_df)} samples")
    
    if len(active_df) == 0 or len(random_df) == 0:
        print("Error: Not enough data for both methods")
        return
    
    # Extract questions and scores
    active_questions = active_df['Question'].tolist()
    active_scores = active_df['Score'].values
    random_questions = random_df['Question'].tolist()
    random_scores = random_df['Score'].values
    
    # Cache file
    cache_file = f'{output_prefix}_v2_embeddings.npz'
    
    # Load or compute embeddings
    cache_valid = False
    if use_cached and os.path.exists(cache_file):
        print(f"  Loading cached embeddings from {cache_file}")
        try:
            data = np.load(cache_file)
            active_embeddings = data['active_embeddings']
            random_embeddings = data['random_embeddings']
            cached_active_scores = data['active_scores']
            cached_random_scores = data['random_scores']
            
            # Validate cache
            if (len(active_embeddings) == len(active_questions) and 
                len(random_embeddings) == len(random_questions)):
                active_scores = cached_active_scores
                random_scores = cached_random_scores
                cache_valid = True
                print(f"  Cache valid: {len(active_embeddings)} active, {len(random_embeddings)} random")
        except Exception as e:
            print(f"  Cache load failed: {e}")
    
    if not cache_valid:
        print(f"  Computing {len(active_questions) + len(random_questions)} embeddings...")
        
        # Compute embeddings
        active_embeddings = []
        for q in tqdm(active_questions, desc=f"  Embedding {active_method}"):
            active_embeddings.append(get_embedding(q))
        active_embeddings = np.array(active_embeddings)
        
        random_embeddings = []
        for q in tqdm(random_questions, desc="  Embedding Random"):
            random_embeddings.append(get_embedding(q))
        random_embeddings = np.array(random_embeddings)
        
        # Save cache
        np.savez(cache_file,
                 active_embeddings=active_embeddings,
                 random_embeddings=random_embeddings,
                 active_scores=active_scores,
                 random_scores=random_scores,
                 active_method=active_method,
                 run=run_to_use)
        print(f"  Cached embeddings to {cache_file}")
    
    # Combine for joint PCA
    all_embeddings = np.vstack([active_embeddings, random_embeddings])
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_embeddings)
    
    # Split back
    active_reduced = all_reduced[:len(active_embeddings)]
    random_reduced = all_reduced[len(active_embeddings):]
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    
    # Get topics for coloring
    active_topics = active_df['Topic'].str.replace(r'^Topic:\s*', '', regex=True).values
    random_topics = random_df['Topic'].str.replace(r'^Topic:\s*', '', regex=True).values
    
    # Get unique topics and assign colors
    all_topics = np.unique(np.concatenate([active_topics, random_topics]))
    n_topics = len(all_topics)
    topic_to_idx = {t: i for i, t in enumerate(all_topics)}
    
    # Topic colors - use a colormap for many topics
    if n_topics <= 10:
        topic_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_topics]
    else:
        topic_colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_topics)))[:n_topics]
    
    # Map topics to color indices
    active_topic_idx = np.array([topic_to_idx[t] for t in active_topics])
    random_topic_idx = np.array([topic_to_idx[t] for t in random_topics])
    # Colors from eval.utils COLOR_LIST for consistency
    color_success = '#6495ED'  # Blue (eval.utils COLOR_LIST[2])
    color_failure = '#DC143C'  # Crimson red (eval.utils COLOR_LIST[0])
    
    # Compute shared axis limits from ALL data (fair comparison)
    all_pc1 = np.concatenate([active_reduced[:, 0], random_reduced[:, 0]])
    all_pc2 = np.concatenate([active_reduced[:, 1], random_reduced[:, 1]])
    
    # Add 10% padding
    pc1_range = all_pc1.max() - all_pc1.min()
    pc2_range = all_pc2.max() - all_pc2.min()
    pc1_lim = (all_pc1.min() - 0.1 * pc1_range, all_pc1.max() + 0.1 * pc1_range)
    pc2_lim = (all_pc2.min() - 0.1 * pc2_range, all_pc2.max() + 0.1 * pc2_range)
    
    # Count failures
    active_failure_count = (active_scores == 0.0).sum()
    random_failure_count = (random_scores == 0.0).sum()
    active_failure_rate = 100 * active_failure_count / len(active_scores)
    random_failure_rate = 100 * random_failure_count / len(random_scores)
    
    # Create legend elements (conditional on failure_only)
    from matplotlib.lines import Line2D
    if failure_only:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_failure,
                   markersize=8, alpha=0.4, markeredgecolor='white', label='Failure')
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_success,
                   alpha=0.4, markersize=8, markeredgecolor='white', label='Success'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_failure,
                   markersize=8, alpha=0.4, markeredgecolor='white', label='Failure')
        ]
    
    # Figure 1: Active method (separate chart)
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig_active, ax = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
    else:
        fig_active, ax = plt.subplots(figsize=(5, 5))
    
    success_mask = active_scores == 1.0
    failure_mask = active_scores == 0.0
    
    # Plot success points (blue, half transparent) - skip if failure_only
    if not failure_only and success_mask.sum() > 0:
        ax.scatter(active_reduced[success_mask, 0], active_reduced[success_mask, 1],
                   c=color_success, marker='o', alpha=0.4, s=50,
                   edgecolors='white', linewidths=0.2)
    
    # Plot failure points (red, more visible)
    if failure_mask.sum() > 0:
        ax.scatter(active_reduced[failure_mask, 0], active_reduced[failure_mask, 1],
                   c=color_failure, marker='o', alpha=0.4, s=50,
                   edgecolors='white', linewidths=0.3)
    
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9, rotation=90, va='bottom')
    ax.set_xlim(pc1_lim)
    ax.set_ylim(pc2_lim)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va='center')
    # ax.set_title(f'Active ({active_method}): {active_failure_count} failures', fontsize=10)
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=False, fontsize=10, numpoints=3, scatterpoints=3)
    
    plt.tight_layout()
    active_png = f'{output_prefix}_scatter_active.png'
    active_pdf = f'{output_prefix}_scatter_active.pdf'
    plt.savefig(active_png, dpi=300, bbox_inches='tight')
    plt.savefig(active_pdf, bbox_inches='tight')
    print(f"\n✓ Active scatter saved to: {active_png}, {active_pdf}")
    plt.close(fig_active)
    
    # Figure 2: Random method (separate chart)
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH * 0.48, fraction=1)
        fig_random, ax = plt.subplots(figsize=(fig_dim[0], fig_dim[0]))
    else:
        fig_random, ax = plt.subplots(figsize=(5, 5))
    
    success_mask = random_scores == 1.0
    failure_mask = random_scores == 0.0
    
    # Plot success points (blue, half transparent) - skip if failure_only
    if not failure_only and success_mask.sum() > 0:
        ax.scatter(random_reduced[success_mask, 0], random_reduced[success_mask, 1],
                   c=color_success, marker='o', alpha=0.4, s=50,
                   edgecolors='white', linewidths=0.2)
    
    # Plot failure points (red, more visible)
    if failure_mask.sum() > 0:
        ax.scatter(random_reduced[failure_mask, 0], random_reduced[failure_mask, 1],
                   c=color_failure, marker='o', alpha=0.4, s=50,
                   edgecolors='white', linewidths=0.3)
    
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9, rotation=90, va='bottom')
    ax.set_xlim(pc1_lim)
    ax.set_ylim(pc2_lim)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va='center')
    # ax.set_title(f'Random: {random_failure_count} failures', fontsize=10)
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=False, fontsize=10, numpoints=3, scatterpoints=3)
    
    plt.tight_layout()
    random_png = f'{output_prefix}_scatter_random.png'
    random_pdf = f'{output_prefix}_scatter_random.pdf'
    plt.savefig(random_png, dpi=300, bbox_inches='tight')
    plt.savefig(random_pdf, bbox_inches='tight')
    print(f"✓ Random scatter saved to: {random_png}, {random_pdf}")
    plt.close(fig_random)
    
    # Print topic color legend
    print(f"\n  Topic Colors (T1-T{n_topics}):")
    for i, topic in enumerate(all_topics):
        print(f"    T{i+1}: {topic}")
    
    # Print statistics
    active_failures = (active_scores == 0.0).sum()
    random_failures = (random_scores == 0.0).sum()
    print(f"\n  Statistics:")
    print(f"    {active_method}: {active_failures}/{len(active_scores)} failures ({100*active_failures/len(active_scores):.1f}%)")
    print(f"    Random: {random_failures}/{len(random_scores)} failures ({100*random_failures/len(random_scores):.1f}%)")
    
    # Print coverage metrics
    active_pc1_range = active_reduced[:, 0].max() - active_reduced[:, 0].min()
    active_pc2_range = active_reduced[:, 1].max() - active_reduced[:, 1].min()
    random_pc1_range = random_reduced[:, 0].max() - random_reduced[:, 0].min()
    random_pc2_range = random_reduced[:, 1].max() - random_reduced[:, 1].min()
    
    print(f"\n  Coverage (embedding space range):")
    print(f"    {active_method}: PC1 range={active_pc1_range:.3f}, PC2 range={active_pc2_range:.3f}")
    print(f"    Random: PC1 range={random_pc1_range:.3f}, PC2 range={random_pc2_range:.3f}")
    print(f"    {active_method} coverage: {active_pc1_range * active_pc2_range:.4f} (PC1 × PC2)")
    print(f"    Random coverage: {random_pc1_range * random_pc2_range:.4f} (PC1 × PC2)")


def plot_v2_topic_distribution(csv_path: str, output_prefix: str = 'v2',
                                methods: list = None):
    """
    Plot horizontal bar chart showing topic distribution across methods.
    
    Y-axis: All topics (sorted by total count)
    X-axis: Count of generated problems
    Bars: One bar per method, grouped by topic
    
    Args:
        csv_path: Path to V2 detailed CSV file
        output_prefix: Prefix for output files
        methods: List of methods to include (default: all)
    """
    print(f"\n📊 Generating topic distribution plot...")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} records from {csv_path}")
    
    # Get methods
    all_methods = df['Method'].unique().tolist()
    if methods:
        methods_to_plot = [m for m in methods if m in all_methods]
    else:
        methods_to_plot = all_methods
    print(f"  Methods: {methods_to_plot}")
    
    # Clean topic names (strip "Topic: " prefix if present)
    df['Topic_Clean'] = df['Topic'].str.replace(r'^Topic:\s*', '', regex=True)
    
    # Get all topics sorted by total count (descending)
    topic_counts = df['Topic_Clean'].value_counts()
    all_topics = topic_counts.index.tolist()
    print(f"  Found {len(all_topics)} topics")
    
    # Create pivot table: rows=topics, columns=methods, values=counts
    pivot = df.groupby(['Topic_Clean', 'Method']).size().unstack(fill_value=0)
    
    # Reorder columns to match methods_to_plot
    pivot = pivot[[m for m in methods_to_plot if m in pivot.columns]]
    
    # Reorder rows by total count (descending)
    pivot = pivot.loc[all_topics]
    
    # Colors for methods
    if LATEX_AVAILABLE:
        colors = COLOR_LIST[:len(methods_to_plot)]
    else:
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C'][:len(methods_to_plot)]
    
    # Create horizontal bar chart
    if LATEX_AVAILABLE:
        set_style()
        fig_dim = set_size(WIDTH, fraction=1)
        fig, ax = plt.subplots(figsize=(fig_dim[0], fig_dim[0] * 0.8))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar positions
    n_topics = len(pivot)
    n_methods = len(pivot.columns)
    bar_height = 0.8 / n_methods
    y_positions = np.arange(n_topics)
    
    # Plot bars for each method
    for i, (method, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - n_methods/2 + 0.5) * bar_height
        bars = ax.barh(y_positions + offset, pivot[method], 
                       height=bar_height, label=method, color=color, alpha=0.85)
    
    # Use short labels T1, T2, T3...
    short_labels = [f'T{i+1}' for i in range(n_topics)]
    topic_mapping = {short_labels[i]: pivot.index[i] for i in range(n_topics)}
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(short_labels, fontsize=9, fontweight='bold')
    ax.set_xlabel('Number of Generated Problems', fontsize=9)
    ax.set_ylabel('Topic', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=7)
    
    # Invert y-axis so most common topic is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    topic_png = f'{output_prefix}_topic_distribution.png'
    topic_pdf = f'{output_prefix}_topic_distribution.pdf'
    plt.savefig(topic_png, dpi=300, bbox_inches='tight')
    plt.savefig(topic_pdf, bbox_inches='tight')
    print(f"\n✓ Topic distribution saved to: {topic_png}, {topic_pdf}")
    plt.close()
    
    # Print topic legend
    print(f"\n  Topic Legend:")
    for short, full in topic_mapping.items():
        print(f"    {short}: {full}")
    
    # Print topic diversity metrics
    print(f"\n  Topic Diversity Metrics:")
    for method in methods_to_plot:
        if method in pivot.columns:
            method_dist = pivot[method].values
            # Entropy as diversity measure
            probs = method_dist / method_dist.sum()
            probs = probs[probs > 0]  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs))
            n_nonzero = (method_dist > 0).sum()
            print(f"    {method}: entropy={entropy:.3f}, topics_covered={n_nonzero}/{n_topics}")


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results from CSV or NPZ')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to experiment CSV file (legacy format)')
    parser.add_argument('--npz', type=str, default=None,
                        help='Path to V2 generator NPZ file (e.g., gsm8k_v2_runall_results.npz)')
    parser.add_argument('--v2csv', type=str, default=None,
                        help='Path to V2 detailed CSV file (e.g., gsm8k_v2_detailed_results.csv)')
    parser.add_argument('--output_prefix', type=str, default='experiment',
                        help='Prefix for output file names')
    parser.add_argument('--scatter', action='store_true',
                        help='Generate scatter plots (requires embeddings)')
    parser.add_argument('--scatter-v2', action='store_true',
                        help='Generate V2 PCA scatter plots for detailed CSV (Active vs Random distribution)')
    parser.add_argument('--topics', action='store_true',
                        help='Generate topic distribution bar chart for V2 detailed CSV')
    parser.add_argument('--active-method', type=str, default='HSS',
                        help='Active method to compare with baseline in scatter plots (default: HSS)')
    parser.add_argument('--random-method', type=str, default='Rand-Gen',
                        help='Baseline/random method to compare with active in scatter plots (default: Rand-Gen)')
    parser.add_argument('--run', type=int, default=1,
                        help='Which run to use for scatter plots (default: 1)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Do not use cached embeddings')
    parser.add_argument('--failure-only', action='store_true',
                        help='Only plot failure points in scatter plots (hide success points)')
    parser.add_argument('--methods', nargs='+', type=str, default=None,
                        help='Filter methods to plot (e.g., --methods Random "HSS (Ours)")')
    
    args = parser.parse_args()
    
    if not args.csv and not args.npz and not args.v2csv:
        print("Error: Must provide --csv, --npz, or --v2csv argument")
        parser.print_help()
        return
    
    # V2 detailed CSV format (new)
    if args.v2csv:
        # Generate failure rate comparison plot
        plot_v2_from_csv(args.v2csv, args.output_prefix, filter_methods=args.methods)
        
        # Generate V2 scatter plots if requested
        if getattr(args, 'scatter_v2', False):
            plot_v2_embedding_scatter(
                args.v2csv, 
                args.output_prefix,
                active_method=args.active_method,
                random_method=getattr(args, 'random_method', 'Rand-Gen'),
                run_to_use=args.run,
                use_cached=not args.no_cache,
                failure_only=getattr(args, 'failure_only', False)
            )
        
        # Generate topic distribution if requested
        if args.topics:
            plot_v2_topic_distribution(
                args.v2csv,
                args.output_prefix,
                methods=args.methods
            )
    
    # V2 NPZ format
    elif args.npz:
        plot_v2_comparison(args.npz, args.output_prefix, filter_methods=args.methods)
    
    # Legacy CSV format
    elif args.csv:
        df = load_csv_data(args.csv)
        
        # Generate failure rate plot
        plot_failure_rate(df, args.output_prefix)
        
        # Generate scatter plots if requested
        if args.scatter:
            plot_embedding_scatter(df, args.output_prefix, use_cached=not args.no_cache)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
