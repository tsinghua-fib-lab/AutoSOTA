#!/usr/bin/env python3
"""
Script to process comparison.csv and calculate averages and standard errors
for each method across all seeds for each dataset and baseline combination.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def calculate_statistics(df):
    """
    Calculate mean and standard error for each method/baseline/dataset combination.
    
    Args:
        df: DataFrame with columns: dataset_name, method, baseline, seed, mse, aug_mse, delta_percent, quality_score
    
    Returns:
        DataFrame with aggregated statistics
    """
    # Group by dataset, method, and baseline
    grouped = df.groupby(['dataset_name', 'method', 'baseline'])
    
    # Calculate statistics for each numeric column
    stats = grouped.agg({
        'mse': ['mean', 'std', 'count'],
        'aug_mse': ['mean', 'std', 'count'],
        'delta_percent': ['mean', 'std', 'count'],
        'quality_score': ['mean', 'std', 'count']
    }).round(6)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    
    # Calculate standard error (std / sqrt(n))
    for metric in ['mse', 'aug_mse', 'delta_percent', 'quality_score']:
        std_col = f'{metric}_std'
        count_col = f'{metric}_count'
        se_col = f'{metric}_se'
        
        if std_col in stats.columns and count_col in stats.columns:
            stats[se_col] = stats[std_col] / np.sqrt(stats[count_col])
            stats[se_col] = stats[se_col].round(6)
    
    # Reset index to make grouping columns regular columns
    stats = stats.reset_index()
    
    return stats

def format_results_table(stats_df):
    """
    Create a nicely formatted summary table.
    """
    # Create a summary with key metrics
    summary = stats_df[['dataset_name', 'method', 'baseline', 
                       'mse_mean', 'mse_se', 
                       'aug_mse_mean', 'aug_mse_se',
                       'delta_percent_mean', 'delta_percent_se',
                       'quality_score_mean', 'quality_score_se',
                       'mse_count']].copy()
    
    # Rename columns for clarity
    summary.columns = ['Dataset', 'Method', 'Baseline', 
                      'MSE_Mean', 'MSE_SE', 
                      'Aug_MSE_Mean', 'Aug_MSE_SE',
                      'Delta_Percent_Mean', 'Delta_Percent_SE',
                      'Quality_Score_Mean', 'Quality_Score_SE',
                      'N_Seeds']
    
    return summary

def main():
    # Get the script directory
    script_dir = Path(__file__).parent
    EXP_DIR = "experiments_all_baselines"

    # Look for comparison.csv in the EXP_DIR directory
    comparison_file = script_dir.parent / EXP_DIR / 'comparison.csv'
    
    if not comparison_file.exists():
        print(f"Error: Could not find comparison.csv at {comparison_file}")
        print("Please ensure the file exists in the right directory.")
        sys.exit(1)
    
    print(f"Reading data from: {comparison_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(comparison_file)
        print(f"Loaded {len(df)} rows of data")
        
        # Display basic info about the data
        print(f"\nDatasets: {sorted(df['dataset_name'].unique())}")
        print(f"Methods: {sorted(df['method'].unique())}")
        print(f"Baselines: {sorted(df['baseline'].unique())}")
        print(f"Number of unique seeds: {df['seed'].nunique()}")
        
        # Calculate statistics
        print("\nCalculating statistics...")
        stats = calculate_statistics(df)
        
        # Create formatted summary
        summary = format_results_table(stats)
        
        # Save results
        output_file = script_dir.parent / EXP_DIR / 'aggregated_results.csv'
        stats.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Save summary
        summary_file = script_dir.parent / EXP_DIR / 'results_summary.csv'
        summary.to_csv(summary_file, index=False)
        print(f"Summary results saved to: {summary_file}")
        
        # Display some key results
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Group by dataset and method to show a compact view
        for dataset in sorted(df['dataset_name'].unique()):
            print(f"\n{dataset}:")
            print("-" * len(dataset))
            
            dataset_data = summary[summary['Dataset'] == dataset]
            
            for baseline in sorted(dataset_data['Baseline'].unique()):
                baseline_data = dataset_data[dataset_data['Baseline'] == baseline]
                print(f"\n  {baseline.upper()} Baseline:")
                
                for _, row in baseline_data.iterrows():
                    method = row['Method']
                    delta_mean = row['Delta_Percent_Mean']
                    delta_se = row['Delta_Percent_SE']
                    quality_mean = row['Quality_Score_Mean']
                    quality_se = row['Quality_Score_SE']
                    n_seeds = int(row['N_Seeds'])
                    
                    print(f"    {method:8s}: Δ={delta_mean:8.2f}±{delta_se:5.2f}%, "
                          f"Quality={quality_mean:.4f}±{quality_se:.4f} (n={n_seeds})")
        
        print(f"\nProcessing complete! Check {output_file.name} and {summary_file.name} for detailed results.")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
