#!/usr/bin/env python3
"""
This script compares good and bad model pairs on their performance across difficulty quantiles.
Good models have small rank difference between average score rank and theta rank.
Bad models have large rank difference despite similar average scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict
import csv

def load_item_parameters(benchmark_name):
    """Load item difficulty parameters from CSV file."""
    item_params_file = f'{benchmark_name}/irt_item_parameters_combined.csv'
    try:
        item_df = pd.read_csv(item_params_file)
        # Convert X1, X2, etc. to 1, 2, etc. for easier matching
        item_df['item_id'] = item_df.iloc[:, 0].str.replace('X', '').astype(int)
        return item_df
    except Exception as e:
        print(f"Error loading item parameters: {e}")
        return None

def load_model_responses(benchmark_name, model_name):
    """Load model responses from the matrix format CSV file."""
    # Use different file paths depending on the benchmark
    responses_file = f'data/clean_response_matrix_{benchmark_name}.csv'
    
    try:
        # Read the first few lines to detect the target model
        model_idx = None
        model_responses = []
        
        # Read the CSV using pandas for the header (first row with item IDs)
        response_matrix_df = pd.read_csv(responses_file, nrows=100)  # Read just enough rows to find model
        
        # Get item IDs from the header row
        item_ids = [int(col) for col in response_matrix_df.columns[1:] if col.isdigit()]
        
        # Find the target model row
        for idx, row in response_matrix_df.iterrows():
            model_name_in_file = row[0]
            if model_name in model_name_in_file:
                print(f"Found target model: {model_name_in_file}")
                model_idx = idx
                break
        
        if model_idx is None:
            print(f"Target model {model_name} not found in the first 100 rows")
            # Try to read the whole file to find the model
            with open(responses_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header with item IDs
                for i, row in enumerate(reader):
                    if row and model_name in row[0]:
                        print(f"Found target model at row {i+2}: {row[0]}")
                        # Extract item responses
                        for j, item_response in enumerate(row[1:]):
                            if j < len(item_ids):
                                item_id = item_ids[j]
                                correct = float(item_response) if item_response else float('nan')
                                model_responses.append((item_id, correct))
                        break
        else:
            # We found the model in the first 100 rows
            row_data = response_matrix_df.iloc[model_idx].values
            for j, item_response in enumerate(row_data[1:]):
                if j < len(item_ids):
                    item_id = item_ids[j]
                    correct = float(item_response) if pd.notna(item_response) else float('nan')
                    model_responses.append((item_id, correct))
        
        return model_responses
    except Exception as e:
        print(f"Error loading model responses: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_all_params(item_params, model_responses, n_quantiles=10):
    """
    Analyze model performance across all IRT parameters (difficulty, discrimination, guessing)
    
    Args:
        item_params: DataFrame with item parameters
        model_responses: List of tuples (item_id, correct)
        n_quantiles: Number of quantiles to use for analysis
    """
    # Create mappings from item_id to parameters
    item_to_params = {}
    for _, row in item_params.iterrows():
        if 'item_id' in row and 'd' in row and 'a1' in row and 'g' in row:
            # In MIRT parameterization, higher 'd' values mean easier items
            # So we negate 'd' to convert to traditional difficulty where higher values = harder items
            item_to_params[row['item_id']] = {
                'difficulty': -1 * row['d'] / row['a1'],  # Negate to convert to traditional difficulty
                'discrimination': row['a1'],
                'guessing': row['g']
            }
    
    # Match responses with parameters
    matched_data = []
    for item_id, correct in model_responses:
        if item_id in item_to_params and not np.isnan(correct):
            params = item_to_params[item_id]
            matched_data.append((item_id, params['difficulty'], params['discrimination'], 
                                params['guessing'], correct))
    
    print(f"Analyzing {len(matched_data)} items with both parameters and valid responses")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(matched_data, columns=['item_id', 'difficulty', 'discrimination', 'guessing', 'correct'])
    
    # Compute quantiles for each parameter
    for param in ['difficulty', 'discrimination', 'guessing']:
        # Create quantile labels based on n_quantiles
        if param == 'difficulty':
            # For difficulty (traditional scale), higher values mean harder items
            if n_quantiles == 4:
                labels = ['Q1_Easy', 'Q2', 'Q3', 'Q4_Hard']
            else:
                labels = [f'Q{i+1}_' + ('Easy' if i < n_quantiles/2 else 'Hard' if i >= 3*n_quantiles/4 else 'Medium') 
                         for i in range(n_quantiles)]
        elif param == 'discrimination':
            # For discrimination, higher values mean better discrimination
            if n_quantiles == 4:
                labels = ['Q1_Low', 'Q2', 'Q3', 'Q4_High']
            else:
                labels = [f'Q{i+1}_' + ('Low' if i < n_quantiles/4 else 'High' if i >= 3*n_quantiles/4 else 'Medium') 
                         for i in range(n_quantiles)]
        else:
            # For guessing, we use neutral labels
            labels = [f'Q{i+1}' for i in range(n_quantiles)]
        
        try:    
            df[f'{param}_quantile'] = pd.qcut(df[param], n_quantiles, labels=labels)
        except ValueError:
            # If too many ties for the requested number of quantiles, fall back to quartiles
            print(f"Warning: Too many ties in {param} for {n_quantiles} quantiles. Falling back to quartiles.")
            n_quartiles = min(4, n_quantiles)
            quartile_labels = labels[:n_quartiles] if len(labels) >= n_quartiles else [f'Q{i+1}' for i in range(n_quartiles)]
            df[f'{param}_quantile'] = pd.qcut(df[param], n_quartiles, labels=quartile_labels, duplicates='drop')
    
    # Calculate overall accuracy
    overall_accuracy = df['correct'].mean()
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    # Calculate correlations between parameters and correctness
    correlations = {}
    for param in ['difficulty', 'discrimination', 'guessing']:
        corr = df[param].corr(df['correct'])
        correlations[param] = corr
        print(f"Correlation between {param} and accuracy: {corr:.4f}")
    
    # Analyze accuracy by each parameter's quantile
    quantile_analyses = {}
    for param in ['difficulty', 'discrimination', 'guessing']:
        try:
            quantile_analyses[param] = df.groupby(f'{param}_quantile').agg(
                accuracy=('correct', 'mean'),
                count=('correct', 'count'),
                avg_param=(param, 'mean'),
                min_param=(param, 'min'),
                max_param=(param, 'max')
            )
        except KeyError:
            # If quantile creation failed and the column doesn't exist
            print(f"Skipping quantile analysis for {param} due to previous errors.")
            continue
    
    return quantile_analyses, df, overall_accuracy, correlations

def load_model_pairs():
    """Define pairs of good and bad models for each benchmark."""
    model_pairs = {
        'arc': [
            {
                'good_model': 'MiniMoog/Mergerix-7b-v0.2',
                'bad_model': 'nbeerbower/llama-3-spicy-8B',
                'pair_name': 'Pair 1 - Identical Score (0.637) - Same Rank',
                'good_avg_rank': 1332.0,
                'good_theta_rank': 1333.0,
                'good_rank_diff': -1.0,
                'bad_avg_rank': 1332.0,
                'bad_theta_rank': 3371.0,
                'bad_rank_diff': -2039.0
            },
            {
                'good_model': 'rwitz2/go-bruins-v2.1',
                'bad_model': 'jondurbin/airoboros-70b-3.3',
                'pair_name': 'Pair 2 - Similar Score (0.692 vs 0.667)',
                'good_avg_rank': 621.0,
                'good_theta_rank': 621.0,
                'good_rank_diff': 0.0,
                'bad_avg_rank': 934.0,
                'bad_theta_rank': 2436.0,
                'bad_rank_diff': -1502.0
            },
            {
                'good_model': 'SC56/Mistral-7B-sumz-dpo-3h',
                'bad_model': 'chihoonlee10/T3Q-ko-solar-dpo-v6.0',
                'pair_name': 'Pair 3 - Similar Score (0.711 vs 0.718)',
                'good_avg_rank': 360.0,
                'good_theta_rank': 358.0,
                'good_rank_diff': 2.0,
                'bad_avg_rank': 150.0,
                'bad_theta_rank': 1791.0,
                'bad_rank_diff': -1641.0
            },
            {
                'good_model': 'meraGPT/mera-mix-4x7B',
                'bad_model': 'swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA',
                'pair_name': 'Pair 4 - Similar Score (0.714 vs 0.713)',
                'good_avg_rank': 271.0,
                'good_theta_rank': 270.0,
                'good_rank_diff': 1.0,
                'bad_avg_rank': 309.0,
                'bad_theta_rank': 2612.0,
                'bad_rank_diff': -2303.0
            }
        ],
        'gsm8k': [
            {
                'good_model': 'failspy/Llama-3-8B-Instruct-abliterated',
                'bad_model': 'Locutusque/Hyperion-3.0-Mistral-7B-alpha',
                'pair_name': 'Pair 1 - Identical Score (0.419) - Same Rank',
                'good_avg_rank': 2862.0,
                'good_theta_rank': 2861.0,
                'good_rank_diff': 1.0,
                'bad_avg_rank': 2862.0,
                'bad_theta_rank': 2860.0,
                'bad_rank_diff': 2.0
            },
            {
                'good_model': 'saishf/West-Maid-7B',
                'bad_model': 'Test157t/HerculeanSea-upd-7b-128k',
                'pair_name': 'Pair 2 - Similar Score (0.629 vs 0.616)',
                'good_avg_rank': 1598.0,
                'good_theta_rank': 1596.0,
                'good_rank_diff': 2.0,
                'bad_avg_rank': 1747.0,
                'bad_theta_rank': 1747.0,
                'bad_rank_diff': 0.0
            },
            {
                'good_model': 'aboros98/merlin1.2',
                'bad_model': 'perlthoughts/Chupacabra-7B-v2.04',
                'pair_name': 'Pair 3 - Similar Score (0.516 vs 0.519)',
                'good_avg_rank': 2408.0,
                'good_theta_rank': 2408.0,
                'good_rank_diff': 0.0,
                'bad_avg_rank': 2395.0,
                'bad_theta_rank': 2397.0,
                'bad_rank_diff': -2.0
            },
            {
                'good_model': 'failspy/Llama-3-8B-Instruct-abliterated',
                'bad_model': 'Locutusque/Hyperion-3.0-Mistral-7B-alpha',
                'pair_name': 'Pair 4 - Same Score (0.419)',
                'good_avg_rank': 2862.0,
                'good_theta_rank': 2861.0,
                'good_rank_diff': 1.0,
                'bad_avg_rank': 2862.0,
                'bad_theta_rank': 2860.0,
                'bad_rank_diff': 2.0
            }
        ],
        'truthfulqa': [
            {
                'good_model': 'ValiantLabs/ShiningValiantXS',
                'bad_model': 'kz919/mistral-7b-dpo-open-orca-flan-50k-synthetic-5-models',
                'pair_name': 'Pair 1 - Identical Score (0.200) - Same Avg Rank',
                'good_avg_rank': 2883.0,
                'good_theta_rank': 2883.0,
                'good_rank_diff': 0.0,
                'bad_avg_rank': 2883.0,
                'bad_theta_rank': 4432.0,
                'bad_rank_diff': -1549.0
            },
            {
                'good_model': 'BlueNipples/SnowLotus-v2-10.7B',
                'bad_model': 'occultml/Helios-10.7B',
                'pair_name': 'Pair 2 - Similar Score (0.219 vs 0.213)',
                'good_avg_rank': 2669.0,
                'good_theta_rank': 2667.0,
                'good_rank_diff': 2.0,
                'bad_avg_rank': 2740.5,
                'bad_theta_rank': 4583.0,
                'bad_rank_diff': -1842.5
            },
            {
                'good_model': 'Changgil/K2S3-Mistral-7b-v1.46',
                'bad_model': 'occultml/Helios-10.7B-v2',
                'pair_name': 'Pair 3 - Similar Score (0.227 vs 0.214)',
                'good_avg_rank': 2555.0,
                'good_theta_rank': 2554.0,
                'good_rank_diff': 1.0,
                'bad_avg_rank': 2719.0,
                'bad_theta_rank': 4584.0,
                'bad_rank_diff': -1865.0
            },
            {
                'good_model': 'diffnamehard/Mistral-CatMacaroni-slerp-uncensored',
                'bad_model': 'luqmanxyz/FrankenVillain-7B-v1',
                'pair_name': 'Pair 4 - Similar Score (0.283 vs 0.225)',
                'good_avg_rank': 1840.0,
                'good_theta_rank': 1840.0,
                'good_rank_diff': 0.0,
                'bad_avg_rank': 2584.0,
                'bad_theta_rank': 4578.0,
                'bad_rank_diff': -1994.0
            },
            {
                'good_model': 'ddyuudd/m_b_8_32',
                'bad_model': 'Sao10K/Senko-11B-v1',
                'pair_name': 'Pair 4 - Similar Score (0.238 vs 0.214)',
                'good_avg_rank': 2396.0,
                'good_theta_rank': 2394.0,
                'good_rank_diff': 2.0,
                'bad_avg_rank': 2719.0,
                'bad_theta_rank': 4551.0,
                'bad_rank_diff': -1832.0
            },
            {
                'good_model': 'ValiantLabs/ShiningValiantXS',
                'bad_model': 'rinna/japanese-gpt-neox-3.6b',
                'pair_name': 'Pair 5 - Similar Score (0.200 vs 0.205)',
                'good_avg_rank': 2883.0,
                'good_theta_rank': 2883.0,
                'good_rank_diff': 0.0,
                'bad_avg_rank': 2828.0,
                'bad_theta_rank': 4491.0,
                'bad_rank_diff': -1663.0
            },
            {
                'good_model': 'meta-llama/Meta-Llama-3-70B',
                'bad_model': 'awnr/Mistral-7B-v0.1-signtensors-1-over-4',
                'pair_name': 'Pair 6 - Similar Score (0.160 vs 0.205)',
                'good_avg_rank': 3472.0,
                'good_theta_rank': 3473.0,
                'good_rank_diff': -1.0,
                'bad_avg_rank': 2828.0,
                'bad_theta_rank': 4635.0,
                'bad_rank_diff': -1807.0
            }
        ],
        'winogrande': [

            {
                'good_model': 'alnrg2arg/blockchainlabs_7B_merged_test2_4',
                'bad_model': 'Weyaxi/Stellaris-internlm2-20b-r128',
                'pair_name': 'Pair 2 - Same Score (0.833)',
                'good_avg_rank': 364.0,
                'good_theta_rank': 365.0,
                'good_rank_diff': -1.0,
                'bad_avg_rank': 364.0,
                'bad_theta_rank': 2724.0,
                'bad_rank_diff': -2360.0
            }
        ],
    }
    return model_pairs

def compare_model_pairs(benchmark_name):
    """Compare pairs of good and bad models on difficulty quantiles."""
    print(f"Comparing model pairs for benchmark: {benchmark_name}")
    
    # Number of quantiles to use for analysis
    n_quantiles = 10  # Using 15 quantiles for more detailed analysis
    
    # Load item parameters
    item_params = load_item_parameters(benchmark_name)
    if item_params is None:
        print("Failed to load item parameters. Exiting.")
        return
    
    # Get model pairs for this benchmark
    model_pairs = load_model_pairs()[benchmark_name]
    
    # Create output directory
    output_dir = f'/store01/nchawla/pli9/llmbenchmark/{benchmark_name}/model_pair_comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each pair
    for pair_idx, pair_info in enumerate(model_pairs):
        good_model = pair_info['good_model']
        bad_model = pair_info['bad_model']
        pair_name = pair_info['pair_name']
        
        print(f"\nProcessing pair {pair_idx+1}: {pair_name}")
        print(f"Good model: {good_model}")
        print(f"Bad model: {bad_model}")
        
        # Load model responses
        good_model_responses = load_model_responses(benchmark_name, good_model)
        bad_model_responses = load_model_responses(benchmark_name, bad_model)
        
        if good_model_responses is None or bad_model_responses is None:
            print("Failed to load model responses. Skipping this pair.")
            continue
        
        # Analyze models
        print("\nAnalyzing good model...")
        good_quantile_analyses, good_df, good_overall_acc, good_correlations = analyze_all_params(
            item_params, good_model_responses, n_quantiles=n_quantiles)
        
        print("\nAnalyzing bad model...")
        bad_quantile_analyses, bad_df, bad_overall_acc, bad_correlations = analyze_all_params(
            item_params, bad_model_responses, n_quantiles=n_quantiles)
        
        # Create comparative visualizations for each parameter
        for param_name in ['difficulty', 'discrimination', 'guessing']:
            if param_name not in good_quantile_analyses or param_name not in bad_quantile_analyses:
                print(f"Skipping {param_name} visualization due to missing data.")
                continue
            
            good_analysis = good_quantile_analyses[param_name]
            bad_analysis = bad_quantile_analyses[param_name]
            
            # Create comparative bar chart
            plt.figure(figsize=(12, 7))
            
            # Set width of bars
            bar_width = 0.35
            index = np.arange(len(good_analysis))
            
            # Plot bars with more appealing colors
            plt.bar(index, good_analysis['accuracy'], bar_width, 
                    label=f'Good Model: {good_model.split("/")[-1]}\nAvg Rank: {pair_info["good_avg_rank"]}, Theta Rank: {pair_info["good_theta_rank"]}\nRank Diff: {pair_info["good_rank_diff"]}\nOverall Acc: {good_overall_acc:.4f}, Corr: {good_correlations[param_name]:.4f}', 
                    color='#3498db', alpha=0.8, edgecolor='#2980b9', linewidth=1.5)
            
            plt.bar(index + bar_width, bad_analysis['accuracy'], bar_width, 
                    label=f'Bad Model: {bad_model.split("/")[-1]}\nAvg Rank: {pair_info["bad_avg_rank"]}, Theta Rank: {pair_info["bad_theta_rank"]}\nRank Diff: {pair_info["bad_rank_diff"]}\nOverall Acc: {bad_overall_acc:.4f}, Corr: {bad_correlations[param_name]:.4f}', 
                    color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1.5)
            
            # Add labels and title
            plt.xlabel(f'{param_name.capitalize()} Quantile')
            plt.ylabel('Accuracy')
            plt.title(f'{benchmark_name.upper()} - {pair_name}\nAccuracy vs {param_name.capitalize()} Quantiles')
            plt.xticks(index + bar_width/2, good_analysis.index)
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            plot_file = os.path.join(output_dir, f'comparison_{param_name}_{pair_idx+1}.png')
            plt.savefig(plot_file, dpi=150)
            print(f"Saved comparison plot to {plot_file}")
            
            # Create line plot for trend visualization
            plt.figure(figsize=(12, 7))
            
            plt.plot(index, good_analysis['accuracy'], 'o-', color='#3498db', linewidth=2.5, markersize=9,
                    markeredgecolor='#2980b9', markerfacecolor='#3498db', markeredgewidth=1.5,
                    label=f'Good Model: {good_model.split("/")[-1]}\nAvg Rank: {pair_info["good_avg_rank"]}, Theta Rank: {pair_info["good_theta_rank"]}\nRank Diff: {pair_info["good_rank_diff"]}\nOverall Acc: {good_overall_acc:.4f}, Corr: {good_correlations[param_name]:.4f}')
            
            plt.plot(index, bad_analysis['accuracy'], 's-', color='#e74c3c', linewidth=2.5, markersize=9,
                    markeredgecolor='#c0392b', markerfacecolor='#e74c3c', markeredgewidth=1.5,
                    label=f'Bad Model: {bad_model.split("/")[-1]}\nAvg Rank: {pair_info["bad_avg_rank"]}, Theta Rank: {pair_info["bad_theta_rank"]}\nRank Diff: {pair_info["bad_rank_diff"]}\nOverall Acc: {bad_overall_acc:.4f}, Corr: {bad_correlations[param_name]:.4f}')
            
            # Add parameter value ranges to x-axis
            xlabels = []
            for i, row in good_analysis.iterrows():
                xlabels.append(f"{row['min_param']:.2f}\nto\n{row['max_param']:.2f}")
            
            plt.xticks(index, xlabels, rotation=45 if n_quantiles > 6 else 0)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel(f'{param_name.capitalize()} Range')
            plt.ylabel('Accuracy')
            plt.title(f'{benchmark_name.upper()} - {pair_name}\nAccuracy Trend by {param_name.capitalize()} Range')
            plt.legend()
            plt.tight_layout()
            
            # Save trend plot
            trend_plot_file = os.path.join(output_dir, f'trend_{param_name}_{pair_idx+1}.png')
            plt.savefig(trend_plot_file, dpi=150)
            print(f"Saved trend plot to {trend_plot_file}")
        
        # Save summary to CSV
        summary_data = {
            'Model': [f'Good: {good_model}', f'Bad: {bad_model}'],
            'Overall_Accuracy': [good_overall_acc, bad_overall_acc],
            'Difficulty_Correlation': [good_correlations['difficulty'], bad_correlations['difficulty']],
            'Discrimination_Correlation': [good_correlations['discrimination'], bad_correlations['discrimination']],
            'Guessing_Correlation': [good_correlations['guessing'], bad_correlations['guessing']],
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f'summary_{pair_idx+1}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary data to {summary_file}")

def set_plot_style():
    """Set a custom, visually appealing plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')  # Use seaborn style as base
    
    # Custom style settings
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['axes.facecolor'] = '#f9f9f9'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.family'] = 'sans-serif'

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compare model pairs across difficulty quantiles.')
    parser.add_argument('benchmark', type=str, choices=['truthfulqa', 'winogrande', 'hellaswag', 'gsm8k', 'arc'], 
                        help='Benchmark to analyze')
    
    # Set custom plot style
    set_plot_style()
    
    args = parser.parse_args()
    compare_model_pairs(args.benchmark)

if __name__ == "__main__":
    main()
