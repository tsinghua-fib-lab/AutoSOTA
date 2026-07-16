import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import argparse
def calculate_test_overlap_rate(data_path):
    """
    Calculate the test overlap rate using the formula:
    T_bar = (N * sum(P(Aj)^2)) / (L_bar * (N-1))  - 1/(N-1)
    
    Where:
    - P(Aj) = hj/N where hj is the number of times item j has been administered
    - N = total number of examinees
    - L_bar = mean test length across all examinees
    
    Parameters:
    - data_path: Path to the directory containing selected items CSV files
    
    Returns:
    - test_overlap_rate: The calculated test overlap rate
    - item_exposure_rates: Dictionary mapping item IDs to their exposure rates
    """
    # Get all csv files in the directory
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    # Count number of examinees (N)
    N = len(csv_files)
    
    if N <= 1:
        return 0, {}  # Cannot calculate overlap with only one examinee
    
    # Read each file and calculate mean test length (L_bar)
    total_items = 0
    item_frequency = {}  # To track how many times each item appears
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        total_items += len(df)
        
        # Count item occurrences
        for item_id in df['item_id']:
            if item_id in item_frequency:
                item_frequency[item_id] += 1
            else:
                item_frequency[item_id] = 1
    
    # Calculate L_bar (mean test length)
    L_bar = total_items / N
    
    # Calculate sum of squared probabilities and item exposure rates
    sum_p_squared = 0
    item_exposure_rates = {}
    
    for item_id, frequency in item_frequency.items():
        p_aj = frequency / N  # P(Aj) = hj/N
        item_exposure_rates[item_id] = p_aj  # Item exposure rate
        sum_p_squared += p_aj ** 2
    
    # Calculate test overlap rate
    test_overlap_rate = (N * sum_p_squared ) / (L_bar * (N-1)) - 1/(N-1)
    
    return test_overlap_rate, item_exposure_rates

def calculate_test_overlap_from_frequency_file(frequency_file, N):
    """
    Calculate test overlap rate using a pre-calculated frequency file
    
    Parameters:
    - frequency_file: Path to the frequency file with item frequencies
    - N: Number of examinees
    
    Returns:
    - test_overlap_rate: The calculated test overlap rate
    - item_exposure_rates: Dictionary mapping item IDs to their exposure rates
    """
    if N <= 1:
        return 0, {}  # Cannot calculate overlap with only one examinee
        
    # Read frequency file
    df = pd.read_csv(frequency_file)
    
    # Calculate L_bar (mean test length)
    # We can estimate it from the frequencies - assuming each frequency is the count of examinees
    # who saw that item
    total_selections = df['frequency'].sum()
    L_bar = total_selections / N
    
    # Calculate sum of squared probabilities and item exposure rates
    sum_p_squared = 0
    item_exposure_rates = {}
    
    for _, row in df.iterrows():
        item_id = row['item_id']
        hj = row['frequency']
        p_aj = hj / N  # P(Aj) = hj/N
        item_exposure_rates[item_id] = p_aj  # Item exposure rate
        sum_p_squared += p_aj ** 2
    
    # Calculate test overlap rate
    test_overlap_rate = (N * sum_p_squared ) / (L_bar * (N-1)) - 1/(N-1)
    
    return test_overlap_rate, item_exposure_rates

def analyze_item_exposure_rates(item_exposure_rates, output_prefix):
    """
    Analyze and report item exposure rates
    
    Parameters:
    - item_exposure_rates: Dictionary of item IDs and their exposure rates
    - output_prefix: Prefix for output files
    """
    exposure_rates = list(item_exposure_rates.values())
    
    # Check if there are any exposure rates to analyze
    if len(exposure_rates) == 0:
        print("\nWarning: No items found. Cannot calculate exposure rate statistics.")
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "std": 0
        }
    
    # Calculate summary statistics
    stats = {
        "min": np.min(exposure_rates),
        "max": np.max(exposure_rates),
        "mean": np.mean(exposure_rates),
        "median": np.median(exposure_rates),
        "std": np.std(exposure_rates)
    }
    
    # Create DataFrame for item exposure rates
    exposure_df = pd.DataFrame({
        "item_id": list(item_exposure_rates.keys()),
        "exposure_rate": exposure_rates
    })
    
    # Save to CSV
    exposure_df.to_csv(f"{output_prefix}_item_exposure_rates.csv", index=False)
    
    # Print summary
    print("\nItem Exposure Rate Summary:")
    print(f"  - Minimum: {stats['min']:.4f}")
    print(f"  - Maximum: {stats['max']:.4f}")
    print(f"  - Mean: {stats['mean']:.4f}")
    print(f"  - Median: {stats['median']:.4f}")
    print(f"  - Std Dev: {stats['std']:.4f}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Calculate test overlap rate and item exposure rates")
    parser.add_argument("--data_path", type=str, help="Path to the directory containing selected items CSV files", default="winogrande/atlas_winogrande_random")
    parser.add_argument("--se_theta_stop", type=str, help="Stopping rule for ATLAS", default="0.3")
    parser.add_argument("--benchmark", type=str, help="Benchmark name", default="winogrande")
    args = parser.parse_args()
    
    # Path to data
    selected_items_path = os.path.join(args.data_path, "selected_items_" + args.se_theta_stop)
    frequency_file = os.path.join(args.data_path, "item_selection_frequency_" + args.se_theta_stop + ".csv")
    output_prefix = args.benchmark + "_" + args.se_theta_stop
    
    # Count examinees (number of files in the directory)
    csv_files = glob.glob(os.path.join(selected_items_path, "*.csv"))
    N = len(csv_files)
    
    print(f"Number of examinees (N): {N}")
    
    # Method 1: Calculate from individual files
    overlap_rate1, item_exposure_rates1 = calculate_test_overlap_rate(selected_items_path)
    print(f"Test overlap rate (calculated from individual files): {overlap_rate1:.6f}")
    
    # Method 2: Calculate from frequency file
    overlap_rate2, item_exposure_rates2 = calculate_test_overlap_from_frequency_file(frequency_file, N)
    print(f"Test overlap rate (calculated from frequency file): {overlap_rate2:.6f}")
    
    # Analyze and report item exposure rates
    exposure_stats = analyze_item_exposure_rates(item_exposure_rates1, output_prefix)
    
    # Save results
    results = {
        "N": N,
        "overlap_rate_method1": overlap_rate1,
        "overlap_rate_method2": overlap_rate2,
        "mean_exposure_rate": exposure_stats["mean"],
        "std_exposure_rate": exposure_stats["std"]
    }
    
    pd.DataFrame([results]).to_csv(f"{output_prefix}_test_overlap_results.csv", index=False)
    print(f"\nResults saved to '{output_prefix}_test_overlap_results.csv'")
    print(f"Item exposure rates saved to '{output_prefix}_item_exposure_rates.csv'")
    print(f"Histogram saved to '{output_prefix}_item_exposure_rate_distribution.png'")
    
    return item_exposure_rates1, exposure_stats

if __name__ == "__main__":
    main()