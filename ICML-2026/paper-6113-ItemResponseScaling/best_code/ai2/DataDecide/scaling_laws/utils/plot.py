import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SIZE_COLORS = {
    '4M': 'brown',
    '6M': '#7f7f7f',  # gray
    '8M': '#17becf',  # cyan
    '10M': '#bcbd22', # olive
    '14M': '#e377c2', # pink
    '16M': '#8c564b', # brown
    '20M': 'black',
    '60M': 'teal',
    '90M': 'pink',
    '150M': '#1f77b4',
    '300M': '#2ca02c',
    '530M': '#ff7f0e',
    '750M': '#d62728',
    '1B': '#9467bd'
}

# Global dictionary to store colors for labels
LABEL_COLOR_MAP = {}
COLOR_IDX = {'col': 0}


def assign_color(label):
    if label not in LABEL_COLOR_MAP:
        available_colors = list(mcolors.TABLEAU_COLORS.keys())
        assigned_color = available_colors[COLOR_IDX['col'] % len(available_colors)]
        LABEL_COLOR_MAP[label] = assigned_color
        COLOR_IDX['col'] += 1
    return LABEL_COLOR_MAP[label]


def lighten_color(color, amount=0.2):
    r, g, b = mcolors.to_rgb(color)
    new_r = min(r + (1 - r) * amount, 1)
    new_g = min(g + (1 - g) * amount, 1)
    new_b = min(b + (1 - b) * amount, 1)
    return new_r, new_g, new_b


def plot_task_accuracy(ax, two_class_results, task, sizes, show_legend=False, size_colors=SIZE_COLORS):
    # First plot all scatter points
    all_x = []
    all_y = []
    for size in list(size_colors.keys()):
        if size not in two_class_results.index.tolist():
            continue
        data = two_class_results.loc[size]
        x = np.array(two_class_results.columns, dtype=np.float64)
        y = np.array(data.values, dtype=np.float64)
        
        # Plot scatter points with consistent colors
        ax.scatter(x, y, marker='o', label=f'{size}', s=5, color=size_colors[size])
        
        # Collect valid points for overall spline
        mask = ~np.isnan(y) & ~np.isnan(x) & ~np.isneginf(y) & ~np.isneginf(x)
        all_x.extend(x[mask])
        all_y.extend(y[mask])
    
    # Add interpolating spline, ignoring nans
    mask = ~np.isnan(all_y) & ~np.isnan(all_x)
    if np.sum(mask) >= 3:  # Need at least 4 points for cubic spline
        all_x = np.array(np.array(all_x)[mask]) # exclude compute=0
        all_y = np.array(np.array(all_y)[mask]) # exclude compute=0

        x_nonzero = all_x != 0
        all_x = all_x[x_nonzero] # exclude x=0 values
        all_y = all_y[x_nonzero] # exclude x=0 values
        
        # Sort points by x value
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]
        
        # Fit smoothed B-spline with high smoothing parameter
        x_smooth = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), len(all_x))
        # Use UnivariateSpline with high smoothing for a smoother fit
        spline = UnivariateSpline(np.log10(all_x), all_y, s=len(all_x))
        y_smooth = spline(np.log10(x_smooth))

        ax.plot(x_smooth, y_smooth, color='k', linestyle='--', label='spline', linewidth=1)
    
    # Add random baseline
    ax.axhline(y=0.5, color='r', linestyle='-', label='random', linewidth=0.5)
    
    ax.set_xlabel('Compute')
    ax.set_ylabel('2-class Accuracy')
    ax.set_title(f'{task}')
    ax.set_xscale('log', base=10)
    if show_legend: ax.legend(loc='lower right', fontsize=10, ncols=2)

    # Add vertical lines at specific FLOPS values with matching colors and accuracies
    # for flops, size in zip(sizes, ['150M', '300M', '530M', '750M', '1B']):
    for flops, size in zip(sizes, list(size_colors.keys())):
        if size not in two_class_results.index.tolist():
            continue
        try:
            acc = two_class_results.loc[size].get(np.float64(flops), np.nan)
            if not np.isnan(acc) and not np.isneginf(acc):
                ax.axvline(x=flops, color=size_colors[size], linestyle=':', alpha=0.7)
                ax.text(
                    flops, 0.98, ' ' + ('1.' if acc == 1 else f'{acc:.2f}').lstrip('0'), 
                    rotation=0, color=size_colors[size], ha='left', va='bottom', fontsize=8)
            else:
                raise FileNotFoundError(f'Not all results found for task={task}, size={size}')
        except Exception as e:
            # raise RuntimeError(f'Cant graph cheap decisions lines: {e}')
            print(f'Cant graph cheap decisions lines: {e}')