#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[92m'      # litegs+dash+reset
    RED = '\033[91m'        # litegs+dash+reset+entropy
    YELLOW = '\033[93m'     # litegs+dash+entropy
    BLUE = '\033[94m'       # litegs+dash
    CYAN = '\033[96m'       # litegs
    MAGENTA = '\033[95m'    # taming
    WHITE = '\033[97m'      # 3dgs
    ORANGE = '\033[38;5;208m'  # msv2
    PURPLE = '\033[38;5;129m'  # msv2.dense
    TEAL = '\033[38;5;45m'  # litegs+dash+alpha
    PINK = '\033[38;5;205m'  # litegs+dash+reset+alpha
    LIGHT_TEAL = '\033[38;5;87m'  # other alpha variants
    BRIGHT_GREEN = '\033[38;5;46m'  # dashgaussian
    BOLD = '\033[1m'        # Bold text
    RESET = '\033[0m'       # Reset to default

def supports_color():
    """Check if terminal supports color output"""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.getenv('TERM') != 'dumb'

def get_strategy_name(full_path):
    """Extract strategy name from full path"""
    path_name = Path(full_path).name.lower()

    # Check for specific strategy patterns
    if '3dgs' in path_name:
        return '3dgs'
    elif 'taming' in path_name:
        return 'taming'
    elif 'msv2' in path_name:
        return 'msv2'
    elif 'dashgaussian' in path_name:
        return 'dashgaussian'
    elif 'litegs' in path_name:
        # Check for litegs variants by looking at what comes after 'litegs'
        parts = path_name.split('litegs', 1)
        if len(parts) > 1 and parts[1]:
            suffix = parts[1].lstrip('+.')  # Remove leading '+' or '.' if present

            strategy = 'litegs'
            if 'dash' in suffix:
                strategy += '+dash'
            if 'reset' in suffix:
                strategy += '+reset'
            if 'entropy' in suffix:
                strategy += '+entropy'
            if 'alpha' in suffix:
                strategy += '+alpha'
            return strategy
        else:
            return 'litegs'
    return 'unknown'

def get_strategy_color(strategy):
    """Get color for a specific strategy"""
    if strategy == 'litegs+dash+reset+alpha':
        return Colors.PINK     # Pink for litegs+dash+reset+alpha
    elif strategy == 'litegs+dash+alpha':
        return Colors.TEAL     # Teal for litegs+dash+alpha
    elif strategy.startswith('litegs') and 'alpha' in strategy:
        return Colors.LIGHT_TEAL  # Fallback for other alpha variants
    elif strategy == '3dgs':
        return Colors.WHITE    # White for 3dgs
    elif strategy == 'taming':
        return Colors.MAGENTA  # Magenta for taming
    elif strategy == 'msv2':
        return Colors.ORANGE   # Orange for msv2
    elif strategy == 'dashgaussian':
        return Colors.BRIGHT_GREEN  # Bright green for dashgaussian
    elif strategy == 'litegs+dash+reset+entropy':
        return Colors.RED      # Red for litegs+dash+reset+entropy
    elif strategy == 'litegs+dash+entropy':
        return Colors.YELLOW   # Yellow for litegs+dash+entropy
    elif strategy == 'litegs+dash+reset':
        return Colors.GREEN    # Green for litegs+dash+reset
    elif strategy == 'litegs+dash':
        return Colors.BLUE     # Blue for litegs+dash
    elif strategy == 'litegs':
        return Colors.CYAN     # Cyan for litegs
    else:
        return Colors.RESET    # No color for unknown

def colorize_row(text, strategy):
    """Colorize entire row based on strategy"""
    if not supports_color():
        return text

    color = get_strategy_color(strategy)
    if color != Colors.RESET:
        return f"{color}{text}{Colors.RESET}"
    return text

def extract_metric_from_file(file_path, pattern, data_type=float):
    """Generic function to extract a metric from a file using regex"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        match = re.search(pattern, content)
        return data_type(match.group(1)) if match else None
    except FileNotFoundError:
        return None

# Metric extraction configuration
METRIC_PATTERNS = {
    'gaussian_count': (r'Final Gaussian count: (\d+)', int),
    'psnr': (r'PSNR:\s+([0-9.]+)', float),
    'ssim': (r'SSIM:\s+([0-9.]+)', float),
    'lpips': (r'LPIPS:\s+([0-9.]+)', float),
    'training_time': (r'Training wall time: ([0-9.]+) seconds', float),
}

# Metric display configuration
METRIC_CONFIGS = [
    ('# Gaussians', 'gaussian_count', '.0f', False),
    ('Training PSNR  ▲', 'training_psnr', '.2f', True),
    ('Training SSIM  ▲', 'training_ssim', '.3f', True),
    ('Training LPIPS ▼', 'training_lpips', '.3f', True),
    ('Testing PSNR  ▲', 'psnr', '.2f', False),
    ('Testing SSIM  ▲', 'ssim', '.3f', False),
    ('Testing LPIPS ▼', 'lpips', '.3f', False),
    ('Training Time (s)', 'training_time', '.2f', False),
]

# All metric keys for calculations
ALL_METRIC_KEYS = ['gaussian_count', 'psnr', 'ssim', 'lpips',
                   'training_psnr', 'training_ssim', 'training_lpips', 'training_time']

def extract_metrics_from_scene(scene_dir):
    """Extract all metrics from a scene directory"""
    metrics = {}

    # File paths
    files = {
        'gaussian_count': scene_dir / 'gaussian_count.txt',
        'test_metrics': scene_dir / 'metrics_testset.txt',
        'train_metrics': scene_dir / 'metrics_trainingset.txt',
        'training_time': scene_dir / 'training_time.txt'
    }

    # Extract basic metrics
    metrics['gaussian_count'] = extract_metric_from_file(
        files['gaussian_count'], *METRIC_PATTERNS['gaussian_count'])
    metrics['training_time'] = extract_metric_from_file(
        files['training_time'], *METRIC_PATTERNS['training_time'])

    # Extract test metrics
    for metric in ['psnr', 'ssim', 'lpips']:
        metrics[metric] = extract_metric_from_file(
            files['test_metrics'], *METRIC_PATTERNS[metric])

    # Extract training metrics
    for metric in ['psnr', 'ssim', 'lpips']:
        metrics[f'training_{metric}'] = extract_metric_from_file(
            files['train_metrics'], *METRIC_PATTERNS[metric])

    return metrics

def get_scene_data(full_path):
    """Extract scene data from a directory"""
    base_dir = Path(full_path)

    if not base_dir.exists():
        print(f"Error: Directory {full_path} does not exist")
        return None, None

    scenes = []
    data = {}

    for scene_dir in base_dir.iterdir():
        if scene_dir.is_dir():
            scene_name = scene_dir.name.split('-')[0]
            scenes.append(scene_name)

            # Extract all metrics for this scene
            metrics = extract_metrics_from_scene(scene_dir)

            if scene_name not in data:
                data[scene_name] = []

            data[scene_name].append(metrics)

    scenes = list(set(scenes))
    scenes.sort()
    return scenes, data

def safe_mean(values):
    """Calculate mean of non-None values, return None if no valid values"""
    valid_values = [v for v in values if v is not None]
    return sum(valid_values) / len(valid_values) if valid_values else None

def calculate_mean_for_metric(experiments, metric_key):
    """Calculate mean for a specific metric across experiments"""
    values = [exp[metric_key] for exp in experiments if exp[metric_key] is not None]
    return safe_mean(values)

def get_mean_metrics(data):
    """Calculate mean metrics for each scene"""
    mean_data = {}
    for scene, experiments in data.items():
        mean_data[scene] = {
            metric_key: calculate_mean_for_metric(experiments, metric_key)
            for metric_key in ALL_METRIC_KEYS
        }

    return mean_data

def calculate_overall_means(mean_data, scenes):
    """Calculate overall means across given scenes"""
    overall_means = {}
    for metric_key in ALL_METRIC_KEYS:
        values = [mean_data[scene][metric_key] for scene in scenes
                 if mean_data[scene][metric_key] is not None]
        overall_means[metric_key] = safe_mean(values)

    return overall_means

def print_single_method_table(full_path, scenes, data, no_training=False):
    """Print table showing individual scenes and overall mean"""
    mean_data = get_mean_metrics(data)
    overall_mean = calculate_overall_means(mean_data, scenes)

    # Print header
    print(f"Data from: {full_path}")

    # Calculate column widths
    scene_col_widths = {scene: max(12, len(scene) + 2) for scene in scenes}
    mean_col_width = max(12, len("Mean") + 2)

    # Print table header
    print(f"{'Metric':<20}|", end='')
    for i, scene in enumerate(scenes):
        separator = "|" if i < len(scenes) - 1 else "|"
        print(f"{scene:<{scene_col_widths[scene]}}{separator}", end='')
    print(f"{'Mean':>{mean_col_width}}")

    # Print separator
    total_width = 21 + sum(scene_col_widths[scene] + 1 for scene in scenes) + mean_col_width + 1
    print('-' * total_width)

    # Filter metrics based on no_training flag
    metrics_to_show = [
        (display_name, metric_key, format_str)
        for display_name, metric_key, format_str, is_training_only
        in METRIC_CONFIGS
        if not (no_training and is_training_only)
    ]

    for metric_name, metric_key, format_str in metrics_to_show:
        print(f"{metric_name:<20}|", end='')
        for i, scene in enumerate(scenes):
            value = mean_data[scene][metric_key]
            separator = "|" if i < len(scenes) - 1 else "|"
            if value is not None:
                if format_str == '.0f':
                    print(f"{value:<{scene_col_widths[scene]},.0f}{separator}", end='')
                else:
                    print(f"{value:<{scene_col_widths[scene]}{format_str}}{separator}", end='')
            else:
                print(f"{'N/A':<{scene_col_widths[scene]}}{separator}", end='')
        
        # Print mean
        overall_value = overall_mean[metric_key]
        if overall_value is not None:
            if format_str == '.0f':
                print(f"{overall_value:>{mean_col_width},.0f}")
            else:
                print(f"{overall_value:>{mean_col_width}{format_str}}")
        else:
            print(f"{'N/A':>{mean_col_width}}")

def format_value(value, format_type):
    """Format value based on type"""
    if value is None:
        return "N/A"
    elif format_type == 'gaussian':
        return f"{value:,.0f}"
    elif format_type == 'psnr':
        return f"{value:.2f}"
    elif format_type == 'ssim':
        return f"{value:.3f}"
    elif format_type == 'lpips':
        return f"{value:.3f}"
    elif format_type == 'time':
        return f"{value:.2f}"

def format_with_ratio(value_b, value_a, format_type):
    """Format value with ratio relative to baseline"""
    if value_b is None:
        return "N/A"
    elif value_a is None or value_a == 0:
        return format_value(value_b, format_type)
    else:
        ratio = value_b / value_a
        base_str = format_value(value_b, format_type)
        if format_type == 'time':
            speedup = 1 / ratio
            return f"{base_str} ({ratio:.2f}x, speedup: {speedup:.2f}x)"
        elif format_type == 'psnr':
            return f"{base_str} ({ratio:.2f}x)"
        elif format_type == 'ssim':
            return f"{base_str} ({ratio:.2f}x)"
        elif format_type == 'lpips':
            return f"{base_str} ({ratio:.2f}x)"
        else:
            return f"{base_str} ({ratio:.2f}x)"

def print_multi_method_comparison_table(all_full_paths, all_scenes, all_data, no_training=False):
    """Print comparison table with first method as baseline and others compared to it"""
    baseline_full_path = all_full_paths[0]
    baseline_scenes = all_scenes[0]
    baseline_data = all_data[0]
    
    # Find common scenes across all methods
    common_scenes = set(baseline_scenes)
    for scenes in all_scenes[1:]:
        common_scenes = common_scenes & set(scenes)
    common_scenes = sorted(list(common_scenes))
    
    if not common_scenes:
        print("No common scenes found across all methods")
        return
    
    # Calculate overall means for all methods
    baseline_mean_data = get_mean_metrics(baseline_data)
    baseline_means = calculate_overall_means(baseline_mean_data, common_scenes)
    
    # Use full paths as method names in comparison table, with baseline annotation
    method_names = []
    for i, full_path in enumerate(all_full_paths):
        if i == 0:
            method_names.append(f"{full_path} (baseline)")
        else:
            method_names.append(full_path)

    # Calculate width for method column in comparison table
    max_method_name_width = max(len(name) for name in method_names)
    method_col_width = max(len("Method"), max_method_name_width) + 1

    # Build header dynamically based on no_training flag
    header_parts = [f"{'Method':<{method_col_width}}", f"{'# Gaussians':<18}"]
    if not no_training:
        header_parts.extend([f"{'Train PSNR ▲':<14}", f"{'Train SSIM ▲':<14}", f"{'Train LPIPS ▼':<14}"])
    header_parts.extend([f"{'Test PSNR ▲':<14}", f"{'Test SSIM ▲':<14}", f"{'Test LPIPS ▼':<14}"])
    if not no_training:
        header_parts.extend([f"{'PSNR (train-test)':<18}", f"{'SSIM (train-test)':<18}", f"{'LPIPS (train-test)':<19}"])
    header_parts.append(f"{'Time (s)':<16}")

    total_table_width = sum(len(part) + 2 for part in header_parts) - 1  # -1 because no separator after last column

    # Print comparison table header
    print("| ".join(header_parts))
    print('-' * total_table_width)
    
    # Print baseline row
    baseline_method_name = method_names[0]

    # Calculate deltas for baseline
    baseline_psnr_delta = None
    baseline_ssim_delta = None
    baseline_lpips_delta = None
    if not no_training:
        baseline_psnr_delta = baseline_means['training_psnr'] - baseline_means['psnr'] if baseline_means['training_psnr'] is not None and baseline_means['psnr'] is not None else None
        baseline_ssim_delta = baseline_means['training_ssim'] - baseline_means['ssim'] if baseline_means['training_ssim'] is not None and baseline_means['ssim'] is not None else None
        baseline_lpips_delta = baseline_means['training_lpips'] - baseline_means['lpips'] if baseline_means['training_lpips'] is not None and baseline_means['lpips'] is not None else None

    # Build baseline row dynamically
    baseline_row_parts = [f"{baseline_method_name:<{method_col_width}}", f"{format_value(baseline_means['gaussian_count'], 'gaussian'):<18}"]
    if not no_training:
        baseline_row_parts.extend([
            f"{format_value(baseline_means['training_psnr'], 'psnr'):<14}",
            f"{format_value(baseline_means['training_ssim'], 'ssim'):<14}",
            f"{format_value(baseline_means['training_lpips'], 'lpips'):<14}"
        ])
    baseline_row_parts.extend([
        f"{format_value(baseline_means['psnr'], 'psnr'):<14}",
        f"{format_value(baseline_means['ssim'], 'ssim'):<14}",
        f"{format_value(baseline_means['lpips'], 'lpips'):<14}"
    ])
    if not no_training:
        baseline_row_parts.extend([
            f"{format_value(baseline_psnr_delta, 'psnr'):<18}",
            f"{format_value(baseline_ssim_delta, 'ssim'):<18}",
            f"{format_value(baseline_lpips_delta, 'lpips'):<19}"
        ])
    baseline_row_parts.append(f"{format_value(baseline_means['training_time'], 'time'):<16}")

    # Color the baseline row based on strategy
    baseline_strategy = get_strategy_name(all_full_paths[0])
    baseline_row = "| ".join(baseline_row_parts)
    print(colorize_row(baseline_row, baseline_strategy))
    
    # Print comparison rows for other methods
    for i, (full_path, scenes, data) in enumerate(zip(all_full_paths[1:], all_scenes[1:], all_data[1:]), 1):
        mean_data = get_mean_metrics(data)
        means = calculate_overall_means(mean_data, common_scenes)

        method_name = method_names[i]
        gaussian_str = format_with_ratio(means['gaussian_count'], baseline_means['gaussian_count'], 'gaussian')
        psnr_str = format_with_ratio(means['psnr'], baseline_means['psnr'], 'psnr')
        ssim_str = format_with_ratio(means['ssim'], baseline_means['ssim'], 'ssim')
        lpips_str = format_with_ratio(means['lpips'], baseline_means['lpips'], 'lpips')
        time_str = format_with_ratio(means['training_time'], baseline_means['training_time'], 'time')

        # Build comparison row dynamically
        comparison_row_parts = [f"{method_name:<{method_col_width}}", f"{gaussian_str:<18}"]

        if not no_training:
            training_psnr_str = format_with_ratio(means['training_psnr'], baseline_means['training_psnr'], 'psnr')
            training_ssim_str = format_with_ratio(means['training_ssim'], baseline_means['training_ssim'], 'ssim')
            training_lpips_str = format_with_ratio(means['training_lpips'], baseline_means['training_lpips'], 'lpips')
            comparison_row_parts.extend([f"{training_psnr_str:<14}", f"{training_ssim_str:<14}", f"{training_lpips_str:<14}"])

        comparison_row_parts.extend([f"{psnr_str:<14}", f"{ssim_str:<14}", f"{lpips_str:<14}"])

        if not no_training:
            # Calculate deltas for comparison method
            psnr_delta = means['training_psnr'] - means['psnr'] if means['training_psnr'] is not None and means['psnr'] is not None else None
            ssim_delta = means['training_ssim'] - means['ssim'] if means['training_ssim'] is not None and means['ssim'] is not None else None
            lpips_delta = means['training_lpips'] - means['lpips'] if means['training_lpips'] is not None and means['lpips'] is not None else None

            # Format deltas with ratios relative to baseline deltas
            psnr_delta_str = format_with_ratio(psnr_delta, baseline_psnr_delta, 'psnr')
            ssim_delta_str = format_with_ratio(ssim_delta, baseline_ssim_delta, 'ssim')
            lpips_delta_str = format_with_ratio(lpips_delta, baseline_lpips_delta, 'lpips')
            comparison_row_parts.extend([f"{psnr_delta_str:<18}", f"{ssim_delta_str:<18}", f"{lpips_delta_str:<19}"])

        comparison_row_parts.append(f"{time_str:<16}")

        # Color the comparison row based on strategy
        strategy = get_strategy_name(full_path)
        comparison_row = "| ".join(comparison_row_parts)
        print(colorize_row(comparison_row, strategy))
    print()

def build_tables(*full_paths, no_training=False, comparison_only=False):
    """Build and display tables for one or more methods"""
    if not full_paths:
        print("Error: No paths provided")
        return

    # Load data for all paths
    all_scenes = []
    all_data = []
    valid_full_paths = []

    for full_path in full_paths:
        scenes, data = get_scene_data(full_path)
        if scenes is not None:
            all_scenes.append(scenes)
            all_data.append(data)
            valid_full_paths.append(full_path)
        else:
            print(f"Skipping invalid path: {full_path}")

    if not valid_full_paths:
        print("Error: No valid paths found")
        return

    if len(valid_full_paths) == 1:
        # Single method mode
        print_single_method_table(valid_full_paths[0], all_scenes[0], all_data[0], no_training=no_training)
    else:
        # Multi-method comparison mode
        print_multi_method_comparison_table(valid_full_paths, all_scenes, all_data, no_training=no_training)

        # Print individual method tables only if comparison_only is False
        if not comparison_only:
            for i, (full_path, scenes, data) in enumerate(zip(valid_full_paths, all_scenes, all_data)):
                print_single_method_table(full_path, scenes, data, no_training=no_training)
                if i < len(valid_full_paths) - 1:  # Don't print extra newline after last table
                    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build tables of scene metrics from output directories')
    parser.add_argument('paths', nargs='+', help='Paths to output directories. First path is used as baseline for comparisons.')
    parser.add_argument('--no_training', action='store_true', help='Skip training metrics in the output tables')
    parser.add_argument('--comparison_only', action='store_true', help='Show only the comparison table, skip individual method tables')

    args = parser.parse_args()
    build_tables(*args.paths, no_training=args.no_training, comparison_only=args.comparison_only)
