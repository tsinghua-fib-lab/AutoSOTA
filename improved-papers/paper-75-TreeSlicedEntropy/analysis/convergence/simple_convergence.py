#!/usr/bin/env python3
"""
Simple Tree Wasserstein Convergence Analysis
A single file containing all functionality for analyzing convergence with respect to L and k.
Uses only gaussian_raw generation mode and linear function type.
"""

import torch
import numpy as np
import time
import json
import pandas as pd
from tsw import TSW, generate_trees_frames

def run_convergence_analysis(
    X: torch.Tensor,
    Y: torch.Tensor,
    L_values: list = [10, 50, 100, 500, 1000, 5000, 10000, 20000],
    k_values: list = [5, 10, 50, 100],
    n_runs: int = 10,
    ground_truth_L: int = 20000,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Run convergence analysis for Tree Wasserstein distance estimator.
    
    Args:
        X, Y: Input distributions
        L_values: List of number of trees to test
        k_values: List of number of lines per tree to test
        n_runs: Number of independent runs per configuration
        ground_truth_L, ground_truth_k: Parameters for ground truth computation
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results and statistics
    """
    
    X = X.to(device)
    Y = Y.to(device)
    d = X.shape[1]
    
    if verbose:
        print(f"Running convergence analysis for {X.shape[0]} vs {Y.shape[0]} points in {d}D")
        print(f"Testing L={L_values}, k={k_values}, {n_runs} runs each")
        print(f"Using gaussian_raw generation mode")
    
    # Initialize results
    results = {
        'L_values': L_values,
        'k_values': k_values,
        'ground_truths': {},
        'statistics': {}
    }
    
    # Run experiments
    for L in L_values:
        for k in k_values:
            if verbose:
                print(f"Testing L={L}, k={k}...")
            
            # Compute ground truth for this k value
            if k not in results['ground_truths']:
                if verbose:
                    print(f"  Computing ground truth for k={k}...")
                
                theta, intercept = generate_trees_frames(
                    ntrees=ground_truth_L, 
                    nlines=k, 
                    d=d, 
                    gen_mode='gaussian_raw',
                    device=device
                )
                
                tw_obj = TSW(
                    ntrees=ground_truth_L,
                    nlines=k,
                    ftype='linear',
                    mass_division='distance_based',
                    device=device
                )
                
                ground_truth_tw = tw_obj(X, Y, theta, intercept).item()
                results['ground_truths'][k] = ground_truth_tw
                
                if verbose:
                    print(f"  Ground truth for k={k}: {ground_truth_tw:.6f}")
            
            estimates = []
            run_times = []
            
            for run in range(n_runs):
                start_time = time.time()
                
                # Generate projections
                theta, intercept = generate_trees_frames(
                    ntrees=L, 
                    nlines=k, 
                    d=d, 
                    gen_mode='gaussian_raw',
                    device=device
                )
                
                # Create TW object
                tw_obj = TSW(
                    ntrees=L,
                    nlines=k,
                    ftype='linear',
                    mass_division='distance_based',
                    device=device
                )
                
                # Compute distance
                tw_estimate = tw_obj(X, Y, theta, intercept)
                estimates.append(tw_estimate.item())
                run_times.append(time.time() - start_time)
            
            # Compute statistics
            estimates = np.array(estimates)
            run_times = np.array(run_times)
            
            mean_distance = np.mean(estimates)
            std_distance = np.std(estimates)
            mean_time = np.mean(run_times)
            
            ground_truth_tw = results['ground_truths'][k]
            bias = abs(mean_distance - ground_truth_tw)
            relative_bias = (bias / ground_truth_tw) * 100 if ground_truth_tw != 0 else 0
            coefficient_of_variation = std_distance / mean_distance if mean_distance != 0 else 0
            
            # Store results
            key = f"L{L}_k{k}"
            results['statistics'][key] = {
                'L': L,
                'k': k,
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'mean_time': mean_time,
                'bias': bias,
                'relative_bias': relative_bias,
                'coefficient_of_variation': coefficient_of_variation,
                'estimates': estimates.tolist(),
                'run_times': run_times.tolist(),
                'ground_truth': ground_truth_tw
            }
    
    return results

def print_results(results):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print("CONVERGENCE ANALYSIS RESULTS")
    print(f"{'='*80}")
    print("Ground Truth Distances:")
    for k, gt in results['ground_truths'].items():
        print(f"  k={k}: {gt:.6f}")
    print(f"{'='*80}")
    
    # Create table
    data = []
    for key, stats in results['statistics'].items():
        data.append({
            'L (Trees)': stats['L'],
            'k (Lines)': stats['k'],
            'Mean Distance': f"{stats['mean_distance']:.6f}",
            'Std Dev': f"{stats['std_distance']:.6f}",
            'Mean Time (s)': f"{stats['mean_time']:.4f}",
            'Bias': f"{stats['bias']:.6f}",
            'Rel Bias (%)': f"{stats['relative_bias']:.2f}",
            'CV (σ/μ)': f"{stats['coefficient_of_variation']:.4f}"
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['L (Trees)', 'k (Lines)'])
    print(df.to_string(index=False))
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS:")
    print(f"{'='*80}")
    
    best_low_bias = min(results['statistics'].values(), key=lambda x: x['relative_bias'])
    best_stable = min(results['statistics'].values(), key=lambda x: x['coefficient_of_variation'])
    fastest = min(results['statistics'].values(), key=lambda x: x['mean_time'])
    
    print(f"Lowest Bias: L={best_low_bias['L']}, k={best_low_bias['k']} "
          f"(Rel Bias: {best_low_bias['relative_bias']:.2f}%)")
    print(f"Most Stable: L={best_stable['L']}, k={best_stable['k']} "
          f"(CV: {best_stable['coefficient_of_variation']:.4f})")
    print(f"Fastest: L={fastest['L']}, k={fastest['k']} "
          f"(Time: {fastest['mean_time']:.4f}s)")

def print_runtime_analysis(results):
    """Print detailed runtime analysis."""
    print(f"\n{'='*80}")
    print("RUNTIME ANALYSIS")
    print(f"{'='*80}")
    
    # Create runtime data
    runtime_data = []
    for key, stats in results['statistics'].items():
        runtime_data.append({
            'L': stats['L'],
            'k': stats['k'],
            'Mean Time (s)': stats['mean_time'],
            'L×k': stats['L'] * stats['k'],
            'Rel Bias (%)': stats['relative_bias'],
            'CV': stats['coefficient_of_variation']
        })
    
    runtime_df = pd.DataFrame(runtime_data)
    runtime_df = runtime_df.sort_values(['L', 'k'])
    
    print("Runtime scaling analysis:")
    print(runtime_df.to_string(index=False))
    
    # Optimal configurations
    print(f"\n{'='*80}")
    print("OPTIMAL CONFIGURATIONS")
    print(f"{'='*80}")
    
    best_accuracy = min(runtime_data, key=lambda x: x['Rel Bias (%)'])
    best_speed = min(runtime_data, key=lambda x: x['Mean Time (s)'])
    best_stability = min(runtime_data, key=lambda x: x['CV'])
    
    print(f"Best Accuracy: L={best_accuracy['L']}, k={best_accuracy['k']}")
    print(f"  - Relative Bias: {best_accuracy['Rel Bias (%)']:.2f}%")
    print(f"  - Runtime: {best_accuracy['Mean Time (s)']:.4f}s")
    print(f"  - L×k: {best_accuracy['L×k']}")
    
    print(f"\nBest Speed: L={best_speed['L']}, k={best_speed['k']}")
    print(f"  - Runtime: {best_speed['Mean Time (s)']:.4f}s")
    print(f"  - Relative Bias: {best_speed['Rel Bias (%)']:.2f}%")
    print(f"  - L×k: {best_speed['L×k']}")
    
    print(f"\nBest Stability: L={best_stability['L']}, k={best_stability['k']}")
    print(f"  - CV: {best_stability['CV']:.4f}")
    print(f"  - Runtime: {best_stability['Mean Time (s)']:.4f}s")
    print(f"  - Relative Bias: {best_stability['Rel Bias (%)']:.2f}%")

def save_results(results, filename_prefix="convergence_results"):
    """Save results to files."""
    # Save JSON
    with open(f"{filename_prefix}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV table
    data = []
    for key, stats in results['statistics'].items():
        data.append({
            'L (Trees)': stats['L'],
            'k (Lines)': stats['k'],
            'Mean Distance': stats['mean_distance'],
            'Std Dev': stats['std_distance'],
            'Mean Time (s)': stats['mean_time'],
            'Bias': stats['bias'],
            'Rel Bias (%)': stats['relative_bias'],
            'CV (σ/μ)': stats['coefficient_of_variation']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['L (Trees)', 'k (Lines)'])
    df.to_csv(f"{filename_prefix}.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"- {filename_prefix}.json")
    print(f"- {filename_prefix}.csv")

def print_cross_dimensional_analysis(all_results):
    """Print comparison across different dimensions."""
    print(f"\n{'='*80}")
    print("CROSS-DIMENSIONAL CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    
    # Extract dimensions
    dimensions = [int(key.split('_')[1]) for key in all_results.keys()]
    dimensions.sort()
    
    print(f"Comparing dimensions: {dimensions}")
    
    # Compare ground truths across dimensions
    print(f"\nGround Truth Distances by Dimension:")
    for d in dimensions:
        key = f"dim_{d}"
        ground_truths = all_results[key]['ground_truths']
        print(f"  {d}D: {ground_truths}")
    
    # Compare optimal configurations across dimensions
    print(f"\nOptimal Configurations by Dimension:")
    for d in dimensions:
        key = f"dim_{d}"
        stats = all_results[key]['statistics']
        
        # Find best configurations
        best_accuracy = min(stats.values(), key=lambda x: x['relative_bias'])
        best_speed = min(stats.values(), key=lambda x: x['mean_time'])
        best_stability = min(stats.values(), key=lambda x: x['coefficient_of_variation'])
        
        print(f"\n  {d}D:")
        print(f"    Best Accuracy: L={best_accuracy['L']}, k={best_accuracy['k']} "
              f"(Bias: {best_accuracy['relative_bias']:.2f}%, Time: {best_accuracy['mean_time']:.4f}s)")
        print(f"    Best Speed: L={best_speed['L']}, k={best_speed['k']} "
              f"(Time: {best_speed['mean_time']:.4f}s, Bias: {best_speed['relative_bias']:.2f}%)")
        print(f"    Best Stability: L={best_stability['L']}, k={best_stability['k']} "
              f"(CV: {best_stability['coefficient_of_variation']:.4f}, Bias: {best_stability['relative_bias']:.2f}%)")
    
    # Compare convergence rates
    print(f"\nConvergence Rate Analysis:")
    for d in dimensions:
        key = f"dim_{d}"
        stats = all_results[key]['statistics']
        
        # Find CV for L=1000, k=50 across dimensions
        target_key = f"L1000_k50"
        if target_key in stats:
            cv = stats[target_key]['coefficient_of_variation']
            bias = stats[target_key]['relative_bias']
            time = stats[target_key]['mean_time']
            print(f"  {d}D (L=1000, k=50): CV={cv:.4f}, Bias={bias:.2f}%, Time={time:.4f}s")
        else:
            print(f"  {d}D (L=1000, k=50): Not available")

def save_combined_results(all_results, filename_prefix):
    """Save combined results from all dimensions."""
    # Save combined JSON
    with open(f"{filename_prefix}.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create combined CSV
    all_data = []
    for dim_key, results in all_results.items():
        d = int(dim_key.split('_')[1])
        for key, stats in results['statistics'].items():
            all_data.append({
                'Dimension': d,
                'L (Trees)': stats['L'],
                'k (Lines)': stats['k'],
                'Mean Distance': stats['mean_distance'],
                'Std Dev': stats['std_distance'],
                'Mean Time (s)': stats['mean_time'],
                'Bias': stats['bias'],
                'Rel Bias (%)': stats['relative_bias'],
                'CV (σ/μ)': stats['coefficient_of_variation']
            })
    
    df = pd.DataFrame(all_data)
    df = df.sort_values(['Dimension', 'L (Trees)', 'k (Lines)'])
    df.to_csv(f"{filename_prefix}.csv", index=False)
    
    print(f"\nCombined results saved to:")
    print(f"- {filename_prefix}.json")
    print(f"- {filename_prefix}.csv")

def main():
    """Main function to run convergence analysis for multiple dimensions."""
    # Setup
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define dimensions to test
    dimensions = [10, 50, 100]  # Smaller dimensions
    L_values = [10, 50, 100, 500, 1000, 5000, 10000, 20000]
    k_values = [5, 10, 50, 100]  # Removed k=200 to save memory
    n_runs = 10  # Reduced for larger parameters
    
    print(f"Testing dimensions: {dimensions}")
    print(f"Testing L={L_values}")
    print(f"Testing k={k_values}")
    print(f"Running {n_runs} independent trials per configuration")
    
    all_results = {}
    
    for d in dimensions:
        print(f"\n{'='*80}")
        print(f"ANALYZING DIMENSION {d}")
        print(f"{'='*80}")
        
        # Generate sample data for this dimension
        print(f"Generating sample data for {d}D...")
        if d <= 10:
            N, M = 100, 100
        elif d <= 50:
            N, M = 50, 50
        else:  # d == 100
            N, M = 30, 30
        X = torch.randn(N, d, device=device) + torch.tensor([1.0] * d, device=device)
        Y = torch.randn(M, d, device=device) + torch.tensor([-1.0] * d, device=device)
        
        print(f"Data: {X.shape[0]} vs {Y.shape[0]} points in {d}D")
        
        # Adjust ground truth L based on dimension to avoid memory issues
        if d == 10:
            ground_truth_L = 5000
        elif d == 50:
            ground_truth_L = 2000
        else:  # d == 100
            ground_truth_L = 1000
        print(f"Using ground truth L={ground_truth_L} for dimension {d}")
        
        # Run analysis for this dimension
        results = run_convergence_analysis(
            X=X,
            Y=Y,
            L_values=L_values,
            k_values=k_values,
            n_runs=n_runs,
            ground_truth_L=ground_truth_L,
            device=device,
            verbose=True
        )
        
        # Store results
        all_results[f"dim_{d}"] = results
        
        # Print results for this dimension
        print(f"\nRESULTS FOR DIMENSION {d}:")
        print_results(results)
        print_runtime_analysis(results)
        
        # Save results for this dimension
        save_results(results, f"convergence_analysis_dim_{d}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Print cross-dimensional comparison
    print_cross_dimensional_analysis(all_results)
    
    # Save combined results
    save_combined_results(all_results, "convergence_analysis_all_dimensions")

if __name__ == "__main__":
    main() 