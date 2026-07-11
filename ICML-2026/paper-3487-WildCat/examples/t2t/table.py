#!/usr/bin/env python3
"""
Generate a LaTeX table with accuracy and runtime statistics.
"""

import argparse
import os
import re
import csv
import numpy as np
from collections import defaultdict


def parse_accuracy_files(acc_dir, seeds=[1, 2, 3, 4, 5]):
    """Parse accuracy CSV files and extract top-1 accuracy for each attention method."""
    results = defaultdict(list)
    
    if not os.path.exists(acc_dir):
        print(f"Warning: {acc_dir} not found")
        return results
    
    for seed in seeds:
        # Look for CSV files matching pattern: acc-{method}-{method}-cuda-s{seed}.csv
        for filename in os.listdir(acc_dir):
            if not filename.startswith('acc-') or not filename.endswith('.csv'):
                continue
            if f'-s{seed}.csv' not in filename:
                continue
            
            # Extract base method name, stripping suffixes like _eager, _r###, _b###
            match = re.match(r'acc-([a-z]+)(?:_eager)?(?:_r\d+)?(?:_b\d+)?-', filename)
            if not match:
                continue
            
            method = match.group(1)
            filepath = os.path.join(acc_dir, filename)
            
            print(f"Parsing {filename} (seed {seed}, method {method})...")
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    
                if len(lines) < 2:
                    print(f"  Skipping: file is too short")
                    continue
                
                # Parse CSV: skip header, read corrects and cnt columns
                reader = csv.DictReader(lines)
                total_corrects = 0
                total_cnt = 0
                row_count = 0
                
                for row in reader:
                    try:
                        corrects = int(row['corrects'])
                        cnt = int(row['cnt'])
                        total_corrects += corrects
                        total_cnt += cnt
                        row_count += 1
                    except (ValueError, KeyError) as e:
                        print(f"    Error parsing row: {e}")
                        continue
                
                if total_cnt > 0:
                    accuracy = total_corrects / total_cnt
                    results[method].append(accuracy)
                    print(f"  {method}: {accuracy:.4f} ({total_corrects}/{total_cnt} from {row_count} rows)")
                else:
                    print(f"  {method}: No data found")
                    
            except Exception as e:
                print(f"Warning: Error reading {filename}: {e}")
    
    return results


def parse_runtime_files(times_dir, layers=[1, 2], batches=range(1, 51)):
    """Parse runtime files and extract layer runtimes for each attention method."""
    results = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(times_dir):
        print(f"Warning: {times_dir} not found")
        return results
    
    print(f"Looking for files in {times_dir}...")
    all_files = os.listdir(times_dir)
    print(f"Found {len(all_files)} files")
    
    for filename in all_files:
        filepath = os.path.join(times_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                
                # Extract attention method from filename
                # Format: times-n1-{attention}-cuda-bs64-bn{batch}.csv
                match = re.search(r'times-n\d+-([a-z]+)-cuda', filename)
                if not match:
                    continue
                attention = match.group(1)
                
                # Try CSV format
                try:
                    lines = content.split('\n')
                    reader = csv.DictReader(lines)
                    for row in reader:
                        # Columns are: attention1.attn.attention_layer, attention2.attn.attention_layer
                        for layer_num in [1, 2]:
                            col_name = f'attention{layer_num}.attn.attention_layer'
                            if col_name in row and row[col_name]:
                                try:
                                    runtime = float(row[col_name])
                                    results[attention][layer_num].append(runtime)
                                    print(f"  {attention} layer {layer_num} = {runtime}")
                                except:
                                    pass
                except Exception as e:
                    print(f"  CSV parsing failed: {e}")
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


def generate_latex_table(acc_results, runtime_results, attention_methods=None):
    """Generate LaTeX table."""
    
    # If attention methods not specified, infer from results
    if attention_methods is None:
        all_methods = set(acc_results.keys()) | set(runtime_results.keys())
        attention_methods = sorted(all_methods)
    
    # Mapping from method names to display names
    method_display_names = {
        'full': 'Exact',
        'reformer': 'Reformer',
        'performer': 'Performer',
        'scatterbrain': 'ScatterBrain',
        'kdeformer': 'KDEformer',
        'thinformer': 'Thinformer',
        'wildcat': '\\textsc{WildCat}',
    }
    
    # Start building table
    lines = []
    lines.append("\\begin{tabular}{cccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Attention Algorithm} & \\textbf{Top-1 Accuracy (\\%)} & \\textbf{Layer 1 Speed-up} & \\textbf{Layer 2 Speed-up} \\\\")
    lines.append("\\midrule")
    
    # Calculate baseline exact runtime means for each layer
    exact_layer1_mean = None
    if 'full' in runtime_results and 1 in runtime_results['full'] and runtime_results['full'][1]:
        exact_layer1_mean = float(np.mean(runtime_results['full'][1]))
    
    exact_layer2_mean = None
    if 'full' in runtime_results and 2 in runtime_results['full'] and runtime_results['full'][2]:
        exact_layer2_mean = float(np.mean(runtime_results['full'][2]))
    
    for idx, attention in enumerate(attention_methods):
        # Get display name
        display_name = method_display_names.get(attention, attention)
        
        # Get accuracy (multiply by 100 for percentage)
        acc_values = acc_results.get(attention, [])
        acc_str = format_mean_std(acc_values, multiplier=100.0, decimals=2)
        
        # Get layer speed-ups
        layer1_values = runtime_results.get(attention, {}).get(1, [])
        if exact_layer1_mean is not None and layer1_values:
            algo_layer1_mean = float(np.mean(layer1_values))
            layer1_speedup = exact_layer1_mean / algo_layer1_mean
            layer1_str = f"${layer1_speedup:.2f}\\times$"
        else:
            layer1_str = "--"
        
        layer2_values = runtime_results.get(attention, {}).get(2, [])
        if exact_layer2_mean is not None and layer2_values:
            algo_layer2_mean = float(np.mean(layer2_values))
            layer2_speedup = exact_layer2_mean / algo_layer2_mean
            layer2_str = f"${layer2_speedup:.2f}\\times$"
        else:
            layer2_str = "--"
        
        # Add spacing except for last row
        spacing = "[0.5mm]" if idx < len(attention_methods) - 1 else ""
        
        # Add row
        lines.append(f"\\textbf{{{display_name}}} & {acc_str} & {layer1_str} & {layer2_str} \\\\{spacing}")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX table with accuracy and runtime statistics')
    parser.add_argument('--acc_dir', type=str, default='out/acc', help='Directory containing accuracy files')
    parser.add_argument('--times_dir', type=str, default='out/times', help='Directory containing runtime files')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='Seeds to average over')
    parser.add_argument('--methods', type=str, nargs='+', default=None, help='Attention methods to include in order')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Parse accuracy results
    print(f"Reading accuracy results from {args.acc_dir}...")
    acc_results = parse_accuracy_files(args.acc_dir, args.seeds)
    
    # Print summary of parsed accuracy results
    for attention, values in sorted(acc_results.items()):
        print(f"  {attention}: {len(values)} samples, mean={np.mean(values):.4f}")
    
    # Parse runtime results
    print(f"\nReading runtime results from {args.times_dir}...")
    runtime_results = parse_runtime_files(args.times_dir)
    
    # Print summary of runtime results
    for attention, layers in sorted(runtime_results.items()):
        for layer, times in sorted(layers.items()):
            print(f"  {attention} layer {layer}: {len(times)} samples, mean={np.mean(times):.4f}")
    
    # If methods not specified, infer and order them
    if args.methods is None:
        # Default order for methods - Performer before Reformer
        all_methods = set(acc_results.keys()) | set(runtime_results.keys())
        method_order = ['full', 'performer', 'reformer', 'kdeformer', 'scatterbrain', 'thinformer', 'wildcat']
        args.methods = [m for m in method_order if m in all_methods]
        # Add any remaining methods
        args.methods.extend([m for m in sorted(all_methods) if m not in args.methods])
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...\n")
    latex_table = generate_latex_table(acc_results, runtime_results, args.methods)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        print(f"Table saved to {args.output}")
    else:
        print(latex_table)


if __name__ == '__main__':
    main()
