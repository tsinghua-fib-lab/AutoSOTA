#!/usr/bin/env python3
"""
Generate a LaTeX table with FID, IS, and runtime statistics across seeds.
"""

import argparse
import os
import re
import csv
import numpy as np
from collections import defaultdict


def parse_fid_results(filepath, model_name, data_per_class, num_splits):
    """Parse FID score results file and extract metrics."""
    results = defaultdict(lambda: {'fid': [], 'is_mean': [], 'is_std': []})
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return results
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line
            # Format: model: biggan-deep-512, data_per_class: 5, num_splits: 10, seed: 1, attention: exact, fid: 32.18, is_mean_fake: 58.37, is_std_fake: 4.21
            match = re.search(r'model:\s*(\S+),\s*data_per_class:\s*(\d+),\s*num_splits:\s*(\d+),\s*seed:\s*(\d+),\s*attention:\s*(\S+)\s*,\s*fid:\s*([\d.]+),\s*is_mean_fake:\s*([\d.]+),\s*is_std_fake:\s*([\d.]+)', line)
            
            if match:
                model = match.group(1)
                dpc = int(match.group(2))
                ns = int(match.group(3))
                seed = int(match.group(4))
                attention = match.group(5).strip()
                fid = float(match.group(6))
                is_mean = float(match.group(7))
                is_std = float(match.group(8))
                
                # Filter by parameters
                if model == model_name and dpc == data_per_class and ns == num_splits:
                    results[attention]['fid'].append(fid)
                    results[attention]['is_mean'].append(is_mean)
                    results[attention]['is_std'].append(is_std)
    
    return results


def parse_runtime_files(times_dir, data_per_class, batch_size=32, dim_bins=1):
    """Parse runtime CSV files and extract attention matrix times.
    
    Note: The 'n' parameter in filenames may refer to number of runs rather than data_per_class.
    """
    results = defaultdict(list)
    
    if not os.path.exists(times_dir):
        print(f"Warning: {times_dir} not found")
        return results
    
    # Pattern: times-n{data_per_class}-{attention}-cuda-bs{batch_size}-bn{dim_bins}.csv
    # Note: We search for any n* pattern since n might be runs not data_per_class
    pattern_suffix = f"-cuda-bs{batch_size}-bn{dim_bins}.csv"
    
    for filename in os.listdir(times_dir):
        if not filename.startswith("times-n") or not filename.endswith(pattern_suffix):
            continue
        
        # Extract attention type from filename
        # Format: times-n10-exact-cuda-bs32-bn1.csv
        parts = filename.replace('.csv', '').split('-')
        # Find the attention type (between n{X} and cuda)
        try:
            n_idx = next(i for i, p in enumerate(parts) if p.startswith('n'))
            cuda_idx = next(i for i, p in enumerate(parts) if p == 'cuda')
            attention = '-'.join(parts[n_idx+1:cuda_idx])
        except:
            print(f"Warning: Could not parse attention type from {filename}")
            continue
        
        # Read CSV file
        filepath = os.path.join(times_dir, filename)
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if 'attention-matrix' in row:
                        results[attention].append(float(row['attention-matrix']))
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
    
    return results


def format_mean_std(values, multiplier=1.0, decimals=2):
    """Format mean ± std for LaTeX."""
    if not values:
        return "--"
    values = np.array(values) * multiplier
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def bold_mean_in_latex(mean_std_str):
    """Bold the mean and std in $\mu \pm \sigma$, bold $< number$, or bold $X.XX\times$."""
    if not mean_std_str or mean_std_str == "--":
        return mean_std_str
    import re
    # Handle $< number$ format
    match_less = re.match(r'(\$<\s*)([\.0-9]+)(\$)', mean_std_str)
    if match_less:
        return f"{match_less.group(1)}\\mathbf{{{match_less.group(2)}}}{match_less.group(3)}"
    # Handle $X.XX\times$ format (speed-up)
    match_speedup = re.match(r'(\$)([\d.]+)(\\times)(\$)', mean_std_str)
    if match_speedup:
        return f"{match_speedup.group(1)}\\mathbf{{{match_speedup.group(2)}\\times}}{match_speedup.group(4)}"
    # Bold only the mean in $X.XX \pm Y.YY$
    match = re.match(r'(\$)([\d.]+)(\s*\\pm\s*[\d.]+\$)', mean_std_str)
    if match:
        return f"{match.group(1)}\\mathbf{{{match.group(2)}}}{match.group(3)}"
    return mean_std_str


def generate_latex_table(fid_results, runtime_results, seeds):
    """Generate LaTeX table."""
    # Baseline exact FID mean (if available) for comparison
    exact_fid_mean = None
    if 'exact' in fid_results and fid_results['exact']['fid']:
        exact_fid_mean = float(np.mean(fid_results['exact']['fid']))
    
    # Calculate baseline exact runtime mean (if available) for speed-up comparison
    exact_runtime_mean = None
    if 'exact' in runtime_results and runtime_results['exact']:
        exact_runtime_mean = float(np.mean(runtime_results['exact']))
    
    # Define attention algorithms and their display names
    attention_order = [
        ('exact', 'Exact'),
        ('reformer', 'Reformer'),
        ('sblocal', 'ScatterBrain'),
        ('performer', 'Performer'),
        ('kdeformer', 'KDEformer'),
        ('thinformer', 'Thinformer'),
        ('wildcat', '\\textsc{WildCat}'),
    ]
    
    # Start building table
    lines = []
    lines.append("\\begin{tabular}{cccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Attention Algorithm} & \\textbf{Speed-up over Exact} & \\textbf{IS Degradation (\\%)} & \\textbf{FID Degradation (\\%)} \\\\")
    lines.append("\\midrule")
    
    for attention_key, attention_name in attention_order:
        # Skip exact row - it's only used as baseline
        if attention_key == 'exact':
            continue
        
        # Check for exact match or pattern match (e.g., wildcat_r96_b8)
        matching_keys = [k for k in fid_results.keys() if k == attention_key or k.startswith(attention_key + '_')]
        
        if not matching_keys:
            continue
        
        # Use the first matching key (could aggregate if multiple)
        attention_key_actual = matching_keys[0]
        
        # Get IS degradation: mean and std of (exact_is - algo_is) / exact_is across seeds
        exact_is_scores = fid_results.get('exact', {}).get('is_mean', [])
        algo_is_scores = fid_results[attention_key_actual]['is_mean']
        
        if attention_key == 'exact':
            is_str = "$0.00 \\pm 0.00$"
        elif exact_is_scores and algo_is_scores and len(exact_is_scores) == len(algo_is_scores):
            degradations = (np.array(exact_is_scores) - np.array(algo_is_scores)) / np.array(exact_is_scores) * 100
            deg_mean = np.mean(degradations)
            deg_std = np.std(degradations, ddof=1) if len(degradations) > 1 else 0.0
            is_str = f"${deg_mean:.2f} \\pm {deg_std:.2f}$"
        else:
            is_str = "--"
        
        # Get FID degradation: mean and std of max(0, algo_fid - exact_fid) / exact_fid across seeds
        exact_fid_scores = fid_results.get('exact', {}).get('fid', [])
        algo_fid_scores = fid_results[attention_key_actual]['fid']
        
        if attention_key == 'exact':
            fid_str = "$0.00 \\pm 0.00$"
        elif exact_fid_scores and algo_fid_scores and len(exact_fid_scores) == len(algo_fid_scores):
            degradations = np.maximum(0, np.array(algo_fid_scores) - np.array(exact_fid_scores)) / np.array(exact_fid_scores) * 100
            deg_mean = np.mean(degradations)
            deg_std = np.std(degradations, ddof=1) if len(degradations) > 1 else 0.0
            fid_str = f"${deg_mean:.2f} \\pm {deg_std:.2f}$"
        else:
            fid_str = "--"
        
        # Bold means for specific methods
        if attention_key in ['wildcat']:
            is_str = bold_mean_in_latex(is_str)
        if attention_key in ['kdeformer', 'thinformer', 'wildcat']:
            fid_str = bold_mean_in_latex(fid_str)
        
        # Calculate speed-up
        if exact_runtime_mean is not None:
            runtime_values = runtime_results.get(attention_key, [])
            if runtime_values:
                algo_runtime_mean = float(np.mean(runtime_values))
                speedup = exact_runtime_mean / algo_runtime_mean
                runtime_str = f"${speedup:.2f}\\times$"
            else:
                runtime_str = "--"
        else:
            runtime_str = "--"
        
        # Bold speed-up for wildcat (WildCat)
        if attention_key == 'wildcat' and runtime_str != "--":
            runtime_str = bold_mean_in_latex(runtime_str)
        
        # Add spacing to all rows except the last one
        spacing = "[1mm]" if attention_key != attention_order[-1][0] else ""
        lines.append(f"\\textbf{{{attention_name}}} & {runtime_str} & {is_str} & {fid_str} \\\\{spacing}")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX table with FID, IS, and runtime statistics')
    parser.add_argument('--model', type=str, default='biggan-deep-512', help='Model name')
    parser.add_argument('--data_per_class', type=int, default=5, help='Number of samples per class')
    parser.add_argument('--num_splits', type=int, default=10, help='Number of splits for IS calculation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='Seeds to average over')
    parser.add_argument('--fid_file', type=str, default='fid_score_results.txt', help='Path to FID results file')
    parser.add_argument('--times_dir', type=str, default='out/times', help='Directory containing runtime CSV files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used in runtime files')
    parser.add_argument('--dim_bins', type=int, default=1, help='Dimension bins parameter')
    parser.add_argument('--runtime_n', type=int, default=None, help='The n parameter in runtime filenames (if None, will match any)')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Parse FID results
    print(f"Reading FID results from {args.fid_file}...")
    fid_results = parse_fid_results(args.fid_file, args.model, args.data_per_class, args.num_splits)
    
    # Print summary of parsed results
    for attention, metrics in fid_results.items():
        print(f"  {attention}: {len(metrics['fid'])} samples")
    
    # Parse runtime results
    print(f"\nReading runtime results from {args.times_dir}...")
    # If runtime_n is not specified, try to infer from available files
    if args.runtime_n is None:
        # Find what n values are available
        import re
        available_ns = set()
        if os.path.exists(args.times_dir):
            for fname in os.listdir(args.times_dir):
                match = re.match(r'times-n(\d+)-', fname)
                if match:
                    available_ns.add(int(match.group(1)))
        if available_ns:
            args.runtime_n = list(available_ns)[0]  # Use the first available
            print(f"  Using runtime files with n={args.runtime_n}")
    
    runtime_results = parse_runtime_files(args.times_dir, args.runtime_n or args.data_per_class, args.batch_size, args.dim_bins)
    
    # Print summary of runtime results
    for attention, times in runtime_results.items():
        print(f"  {attention}: {len(times)} samples")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...\n")
    latex_table = generate_latex_table(fid_results, runtime_results, args.seeds)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        print(f"Table saved to {args.output}")
    else:
        print(latex_table)


if __name__ == '__main__':
    main()
