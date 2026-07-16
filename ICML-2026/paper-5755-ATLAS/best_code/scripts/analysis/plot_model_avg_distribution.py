#!/usr/bin/env python3
"""
Script to calculate average scores from csv file and plot their distribution
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

def calculate_average_score(csv_file_path):
    """
    Calculate the average score for each model in the CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing model scores
        
    Returns:
        dict: Dictionary with model names as keys and their average scores as values
    """
    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Calculate average score for each model
    results = {}
    
    # Read the CSV file
    with open(csv_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Read header row
        header = next(csv_reader)
        
        # Process each row (model)
        for row in csv_reader:
            if not row:  # Skip empty rows
                continue
                
            model_name = row[0]  # First column is the model name
            
            # Skip rows that are likely headers or metadata
            if 'Unnamed' in model_name or model_name.lower() in ['model', 'index']:
                print(f"Skipping likely header row: {model_name}")
                continue
                
            # Calculate average by taking mean of all score columns (columns 1 to end)
            try:
                scores = [float(score) for score in row[1:] if score.strip()]  # Convert to floats
                
                if scores:  # Check if there are any scores
                    average_score = sum(scores) / len(scores)
                    # average_score = sum(scores)##⚠️
                    # Store result
                    results[model_name] = average_score
            except ValueError:
                print(f"Skipping row with non-numeric values: {model_name}")
                continue
    
    return results

def write_csv(average_scores, output_file):
    """
    Write average scores to a CSV file.
    
    Args:
        average_scores (dict): Dictionary with model names as keys and their average scores as values
        output_file (str): Path to save the plot image (will be modified to save CSV)
    """
    # Create CSV filename by replacing the image extension with .csv
    csv_output = os.path.splitext(output_file)[0] + '_scores.csv'
    
    print(f"Saving average scores to {csv_output}...")
    
    # Write the data to CSV
    with open(csv_output, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(['Model', 'Score'])
        
        # Write data rows sorted by model name, excluding any rows that look like headers
        for model_name, score in sorted(average_scores.items()):
            # Skip rows that are likely headers or metadata (like 'Unnamed: 0')
            if 'Unnamed' in model_name or model_name.lower() in ['model', 'index']:
                print(f"Skipping likely header row: {model_name}")
                continue
            csv_writer.writerow([model_name, score])
    
    print(f"Average scores saved to {csv_output}")

def plot_score_distribution(scores, output_file=None):
    """Plot histogram of scores and compare with Gaussian."""
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with density=True to get probability density
    n, bins, patches = plt.hist(scores, bins=30, density=True, alpha=0.7, 
                               label='Average Score Distribution')
    
    # Fit a normal distribution to the data
    mu, sigma = stats.norm.fit(scores)
    
    # Plot the PDF (probability density function)
    x = np.linspace(min(scores), max(scores), 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Fitted Gaussian: μ={mu:.2f}, σ={sigma:.2f}')
    
    # Calculate goodness of fit with Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.kstest(scores, 'norm', args=(mu, sigma))
    
    # Add title and labels
    plt.title(f'Distribution of Average Scores for Models\nKS test: statistic={ks_statistic:.4f}, p-value={p_value:.4f}', 
             fontsize=14)
    plt.xlabel('Average Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    # Add text box with statistics
    # Calculate skewness and kurtosis
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)
    
    stats_text = f'Number of models: {len(scores)}\n'
    stats_text += f'Mean: {np.mean(scores):.4f}\n'
    stats_text += f'Median: {np.median(scores):.4f}\n'
    stats_text += f'Std Dev: {np.std(scores):.4f}\n'
    stats_text += f'Min: {np.min(scores):.4f}\n'
    stats_text += f'Max: {np.max(scores):.4f}\n'
    stats_text += f'Skewness: {skewness:.4f}\n'
    stats_text += f'Kurtosis: {kurtosis:.4f}'
    
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add interpretation of Gaussian fit
    if p_value < 0.05:
        fit_text = "The distribution differs significantly from a Gaussian (p<0.05)"
    else:
        fit_text = "The distribution is consistent with a Gaussian (p>=0.05)"
    
    plt.annotate(fit_text, xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=10, horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Draw the plot (average score)')
    parser.add_argument('--csv_path', help='Path to the input CSV file', default="data/clean_response_matrix_gsm8k.csv")
    parser.add_argument('--plotname', '-o', help='Path to save the plot', default="model_scores_distribution.png")
    args = parser.parse_args()
    # Path to the CSV file
    csv_file_path = args.csv_path
    output_file = args.plotname
    
    # Calculate average scores
    print(f"Calculating average scores from {os.path.basename(csv_file_path)}...")
    average_scores = calculate_average_score(csv_file_path)
    write_csv(average_scores, output_file)
    
    # Convert dictionary values to a numpy array for plotting
    scores_array = np.array(list(average_scores.values()))
    
    print(f"Found {len(scores_array)} models with valid scores.")
    print(f"Score range: {np.min(scores_array):.4f} to {np.max(scores_array):.4f}")
    print(f"Mean score: {np.mean(scores_array):.4f}")
    
    # Plot the distribution
    print("Generating plot...")
    plot_score_distribution(scores_array, output_file)
    
    # Additional analysis: skewness and kurtosis
    skewness = stats.skew(scores_array)
    kurtosis = stats.kurtosis(scores_array)
    print(f"Distribution Skewness: {skewness:.4f}")
    print(f"Distribution Kurtosis: {kurtosis:.4f}")
    print("(For a perfect Gaussian: Skewness=0, Kurtosis=0)")

if __name__ == "__main__":
    main()
