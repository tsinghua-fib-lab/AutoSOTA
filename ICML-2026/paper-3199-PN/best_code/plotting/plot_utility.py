#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

"""
Plotting utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle
from scipy import interpolate, stats
from scipy.stats import gaussian_kde

def setup_plot_style():
    """Setup comprehensive matplotlib style with minor ticks enabled by default."""
    
    # Font sizes
    fsize = 17          # General font size
    tsize = 23          # Title font size  
    lsize = 16          # Legend font size
    asize = 22          # Axis label font size
    ticksize = 21       # Tick label font size
    
    # Tick properties
    tdir = 'in'         # Tick direction
    major = 5.0         # Major tick size
    minor = 3.0         # Minor tick size
    
    # Line and layout properties
    lwidth = 0.8        # Axes line width
    lhandle = 2.0       # Legend handle length
    
    # Figure properties
    figwidth = 7       # Default figure width
    figheight = 5       # Default figure height
    dpi = 200          # High DPI for crisp plots

    # Apply style
    plt.style.use('default')
    
    # Text and fonts
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fsize
    
    # Figure settings
    plt.rcParams['figure.figsize'] = [figwidth, figheight]
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # Axes and labels
    plt.rcParams['axes.titlesize'] = tsize
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = asize
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    
    # Ticks - Major
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['xtick.major.width'] = lwidth
    plt.rcParams['ytick.major.width'] = lwidth
    
    # Ticks - Minor (enabled by default)
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['xtick.minor.width'] = lwidth * 0.6
    plt.rcParams['ytick.minor.width'] = lwidth * 0.6
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    
    # Grid
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['axes.grid.which'] = 'both' 
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    
    # Legend
    plt.rcParams['legend.fontsize'] = lsize
    plt.rcParams['legend.handlelength'] = lhandle
    plt.rcParams['legend.handletextpad'] = 0.5
    plt.rcParams['legend.columnspacing'] = 1.0
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.shadow'] = False
    
    # Lines and markers
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.markeredgewidth'] = 0.5
    
    # # Colors (color cycle)
    # plt.rcParams['axes.prop_cycle'] = plt.cycler('color', [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    # ])
    
    # Set up automatic minor tick locators for new plots
    _setup_minor_ticks_hook()

def _setup_minor_ticks_hook():
    """Internal function to automatically enable minor tick locators."""
    # Store original subplot creation function
    original_subplot = plt.subplot
    original_subplots = plt.subplots
    original_figure = plt.figure
    
    def subplot_with_minor_ticks(*args, **kwargs):
        ax = original_subplot(*args, **kwargs)
        _enable_minor_ticks_on_axes(ax)
        return ax
    
    def subplots_with_minor_ticks(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        if hasattr(axes, '__iter__'):
            for ax in axes.flat:
                _enable_minor_ticks_on_axes(ax)
        else:
            _enable_minor_ticks_on_axes(axes)
        return fig, axes
    
    def figure_with_minor_ticks(*args, **kwargs):
        fig = original_figure(*args, **kwargs)
        # Enable minor ticks on any existing axes
        for ax in fig.get_axes():
            _enable_minor_ticks_on_axes(ax)
        return fig
    
    # Monkey patch matplotlib functions
    plt.subplot = subplot_with_minor_ticks
    plt.subplots = subplots_with_minor_ticks
    plt.figure = figure_with_minor_ticks

def _enable_minor_ticks_on_axes(ax):
    """Enable minor ticks on a specific axes object."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def enable_minor_ticks():
    """Manually enable minor ticks on current axes (if hook doesn't work)."""
    _enable_minor_ticks_on_axes(plt.gca())

def set_figure_size(width=10, height=6):
    """Set figure size for current or next figure."""
    plt.rcParams['figure.figsize'] = [width, height]

def reset_plot_style():
    """Reset to matplotlib defaults."""
    plt.rcParams.update(plt.rcParamsDefault)


def plot_metric_vs_param(
        x_param_list, 
        results, 
        ax, 
        draw_best=True, 
        best='min',
        xscale='log', 
        yscale='linear', 
        xlim=None,
        ylim=None,
        value_labels=None, 
        add_min_curve=False):
    """
    Plot metric vs parameter at different steps (iso-step curves) or thresholds (iso-metric curves).
    
    Generic plotting function that reproduces original subplot() function exactly.
    
    Args:
        x_param_list: List of x parameter values (e.g., p_e values)
        results: 2D array [n_params, n_steps_or_thresholds] with metric values or steps
        ax: Matplotlib axis to plot on
        draw_best: Whether to plot best reached metric (default: True)
        best: 'min' or 'max' to indicate whether best is minimum or maximum (default: 'min')
        xscale: X-axis scale (default: 'log')
        yscale: Y-axis scale (default: 'linear')
        xlim : tuple, optional X-axis limits as (min, max)
        ylim : tuple, optional Y-axis limits as (min, max)
        value_labels: Optional list of labels for each curve (e.g., threshold values for inverted plots)
        add_min_curve: Whether to add min curve curve (for inverted plots)
    """
    #add_interpolation=False,  add_interpolation: Whether to add interpolation curve on last values (for inverted plots)
    
    max_n_steps = np.shape(results)[1]
    min_results = [results[i, -1] for i in range(len(x_param_list))]

    # Determine if we have value labels (for inverted plots)
    use_value_labels = value_labels is not None
    
    # Plot first step/threshold
    ax.plot(x_param_list, results[:, 0], color="black", marker='.')
    s = 1 if not use_value_labels else value_labels[0]
    l = results[3, 0]
    if l is not None and (not use_value_labels or not np.isinf(l)):
        label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
        ax.text(x_param_list[3], l, label_text, 
                bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
    
    # # Plot intermediate steps/thresholds based on max_n_steps
    if max_n_steps < 5:
        ax.plot(x_param_list, results[:,::], color="black", marker='.', alpha=0.1)
        s_step = 1
        for idx in range(s_step, max_n_steps, 1):
            l = results[3, idx]
            s = s_step if not use_value_labels else value_labels[idx]
            if l is not None and (not use_value_labels or not np.isinf(l)):
                label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
                if l < ylim[1]*0.9 and l > ylim[0]*1.1:  # Only label if within y-axis limits
                    ax.text(x_param_list[3], l, label_text,
                       bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
            s_step += 1 

    elif max_n_steps < 25:
        ax.plot(x_param_list, results[:, :], linestyle=":", color="grey", marker='')
        ax.plot(x_param_list, results[:,::5], color="black", marker='.')
        s_step = 5
        for idx in range(s_step, max_n_steps, 5):
            l = results[3, idx]
            s = s_step if not use_value_labels else value_labels[idx]
            if l is not None and (not use_value_labels or not np.isinf(l)):
                label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
                ax.text(x_param_list[3], l, label_text,
                       bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
            s_step += 5
            
    elif max_n_steps < 100:
        ax.plot(x_param_list, results[:, :], color="grey", marker='', alpha=0.1)
        ax.plot(x_param_list, results[:, ::5], linestyle=":", color="grey", marker='')
        ax.plot(x_param_list, results[:, ::25], color="black", marker='.')
        s_step = 25
        for idx in range(s_step, max_n_steps, 25):
            l = results[3, idx]
            s = s_step if not use_value_labels else value_labels[idx]
            if l is not None and (not use_value_labels or not np.isinf(l)):
                label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
                ax.text(x_param_list[3], l, label_text,
                       bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
            s_step += 25
    
    elif max_n_steps < 5000:
        ax.plot(x_param_list, results[:, ::5], color="grey", marker='', alpha=0.1)
        ax.plot(x_param_list, results[:, ::25], linestyle=":", color="grey", marker='')
        ax.plot(x_param_list, results[:, ::100], color="black", marker='.')
        s_step = 200
        for idx in range(s_step, max_n_steps, 200):
            l = results[3, idx]
            s = s_step if not use_value_labels else value_labels[idx]
            if l is not None and (not use_value_labels or not np.isinf(l)):
                label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
                ax.text(x_param_list[3], l, label_text,
                       bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
            s_step += 200

    # Updated Large Step Number Plot
    else:
        # Plot every 50 steps lightly
        ax.plot(x_param_list, results[:,::50], color="grey", marker='', alpha=0.1) # [:,::50] Select every 50 steps starting from index 0
        # Plot every 250 steps with dashed line
        ax.plot(x_param_list, results[:,::250], linestyle=":", color="grey", marker='') # [:,::250] Select every 250 steps starting from index 0
        # Plot every 1000 steps prominently
        ax.plot(x_param_list, results[:,::1000], color="black", marker='.') # [:,::1000] Select every 1000 steps starting from index 0

        # Label every 2000 steps
        s_step = 2000
        for idx in range(s_step, max_n_steps, 2000):
            l = results[3, idx]
            s = s_step if not use_value_labels else value_labels[idx]
            if l is not None and (not use_value_labels or not np.isinf(l)):
                label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
                ax.text(x_param_list[3], l, label_text,
                       bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
            s_step += 2000
    
    # Always plot last step/threshold
    ax.plot(x_param_list, results[:, -1], color="black", marker='.')
    s = max_n_steps if not use_value_labels else value_labels[-1]
    l = results[3, -1]
    if l is not None and (not use_value_labels or not np.isinf(l)):
        label_text = f"{s}" if not use_value_labels else f"{s:.2e}"
        ax.text(x_param_list[3], l, label_text,
               bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=1))
        
    # Draw best
    if draw_best:
        best_x_param = []
        best_results = []


        for j in range(results.shape[1]):

            if best == "max":
                prev_best_y = -np.inf
            else:
                prev_best_y = +np.inf
            prev_best_x = None
            for i in range(len(x_param_list)):
                val = results[i, j]

                if val is not None and ((best == 'min' and val < prev_best_y) or (best == 'max' and val > prev_best_y)):
                    prev_best_x = x_param_list[i]
                    prev_best_y = val
                    
            if prev_best_x is not None:
                best_x_param.append(prev_best_x)
                best_results.append(prev_best_y)
        #print([(best_x_param[i], best_results[i]) for i in range(len(best_x_param))])
        ax.plot(best_x_param, best_results, color="green", marker='', linestyle='dashed', label='Best')
    
    # Add min curve (for inverted plots)
    if add_min_curve:
        argmin_idx = np.argmin(results, axis=-2)
        argmin_x_param = [x_param_list[idx] for idx in argmin_idx]
        min_val = np.min(results, axis=-2)
        ax.plot(argmin_x_param, min_val, color="green")

    # ax.grid(which="both")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Apply limits if specified
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    
    # Clean up annotations outside limits (if limits were set)
    if xlim is not None or ylim is not None:
        _remove_text_outside_limits(ax)

def _remove_text_outside_limits(ax):
    """Remove text annotations outside current axis limits"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Go through all text objects in the axis
    texts_to_remove = []
    for text in ax.texts:
        x, y = text.get_position()
        # Check if text is outside limits
        if not (xlim[0]*1.1 <= x <= xlim[1]*0.9 and ylim[0]*1.1 <= y <= ylim[1]*0.9):
            texts_to_remove.append(text)
    
    # Remove identified texts
    for text in texts_to_remove:
        text.remove()


def compute_steps_to_thresholds(x_param_list, results, n_thresholds=100):
    """
    Compute steps needed to reach different metric thresholds (preprocessing for inverted plots).
    
    This function inverts the data: instead of metric values at each step,
    it computes the step number needed to reach each threshold.
    
    Args:
        x_param_list: List of x parameter values (e.g., p_e values)
        results: 2D array [n_params, n_steps] with metric values (decreasing from ~1 to ~0)
        n_thresholds: Number of metric thresholds to compute
    
    Returns:
        thresholds: Array of threshold values
        result_steps: 2D array [n_params, n_thresholds] with steps to reach each threshold
    """
    # Find min and max metric values to define thresholds
    all_valid = []
    for k in range(len(x_param_list)):
        valid = [v for v in results[k] if v is not None]
        all_valid.extend(valid)
    
    max_metric = max(all_valid) if all_valid else 1.0
    min_metric = min(all_valid) if all_valid else 0.0
    
    # Create thresholds from max to min (metric decreases during training)
    thresholds = np.linspace(max_metric, min_metric, n_thresholds)
    
    # For each parameter value, compute steps to reach each threshold
    result_steps = []
    
    for k in range(len(x_param_list)):
        result = results[k]
        result_threshold_steps = []
        
        for threshold in thresholds:
            # Find first step where metric <= threshold
            final_n_steps = np.inf
            for n_steps in range(len(result)):
                if result[n_steps] is None:
                    break
                if result[n_steps] <= threshold:
                    final_n_steps = n_steps
                    break
            
            result_threshold_steps.append(final_n_steps)
        
        result_steps.append(result_threshold_steps)
    
    result_steps = np.array(result_steps)
    
    return thresholds, result_steps


def extract_data(results_dict, metric='p_diff'):
    """
    Extract data from training results dictionary.
    
    Args:
        results_dict: Dict {x_param: {'training_results': {'history': {...}}}}
        metric: Metric to extract ('p_diff', 'p_epsilon', 'loss')
    
    Returns:
        x_param_list: List of x parameter values (sorted keys from dict)
        results: 2D array [n_params, max_steps] (padded with None)
    """
    x_param_list = sorted(results_dict.keys())
    
    # Extract histories
    all_results = []
    max_n_steps = 0
    
    for x_param in x_param_list:
        result = results_dict[x_param]
        training_result = result.get('training_results', result)
        history = training_result.get('history', {})
        
        if metric in history:
            metric_history = history[metric]
            all_results.append(list(metric_history))
            max_n_steps = max(max_n_steps, len(metric_history))
    
    # Pad to same length
    padded_results = []
    for results_x_param in all_results:
        l = len(results_x_param)
        padded_results_x_param = results_x_param + [None] * (max_n_steps - l)
        padded_results.append(padded_results_x_param)
    
    results = np.array(padded_results)
    
    return x_param_list, results

def plot_qq_with_r2_list(ax, snapshots, colors, weight_key, title, n_sample=500, text_ratio=1.2):
    """Plot Q-Q with undersampling for visibility and full R² computation."""
    

    r2_values = []
    step_values = []

    for i, (snapshot, color) in enumerate(zip(snapshots, colors)):
        weights = snapshot[weight_key].flatten()
        if len(weights) == 0:
            continue

        (theoretical_q, sample_q), (slope, intercept, r_full) = stats.probplot(weights, dist="norm", plot=None)

        if len(theoretical_q) > n_sample:
            indices = np.random.choice(len(theoretical_q), n_sample, replace=False)
            theoretical_q_sampled = theoretical_q[indices]
            sample_q_sampled = sample_q[indices]
        else:
            theoretical_q_sampled = theoretical_q
            sample_q_sampled = sample_q

        ax.scatter(theoretical_q_sampled, sample_q_sampled, color=color, alpha=0.7, s=20,
                   edgecolors='black', linewidth=0.5)
        ax.plot(theoretical_q, slope * theoretical_q + intercept,
                color=color, linewidth=2.5, alpha=0.9, zorder=0)

        r2_values.append(r_full**2)
        step_values.append(snapshot['step'])

    text_x, text_y = 0.02, 0.98
    line_height = 0.07*text_ratio

    for i, (step, r2, color) in enumerate(zip(step_values, r2_values, colors[:len(step_values)])):
        rect_y = text_y - (i + 0.5) * line_height
        rect = Rectangle((text_x - 0.005, rect_y - line_height/3), 0.17*text_ratio, 0.9*line_height,
                          transform=ax.transAxes, facecolor='white', alpha=1,
                          edgecolor=color, linewidth=2,zorder=2)
        ax.add_patch(rect)
        ax.text(text_x, rect_y, f'$R^2$ = {r2:.2f}',
                transform=ax.transAxes, verticalalignment='center', fontsize=plt.rcParams['legend.fontsize']*text_ratio,zorder=3)

    ax.set_title(title)
    ax.set_xlim(-2.5, +2.5)
    ax.set_ylim(-0.5, +1.5)
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.5)
    ax.grid(visible=True, which='minor', color='black', linestyle='-', alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['axes.labelsize']*text_ratio)

def create_subplot_grid(rows, cols):
    """Create subplot grid using default figsize scaled by rows/cols."""
    default_width = plt.rcParams['figure.figsize'][0]
    default_height = plt.rcParams['figure.figsize'][1]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * default_width, rows * default_height))
    return fig, axes

def compute_kde_envelope(weights, x_range):
    """Compute KDE envelope for given weights."""
    if len(weights) > 10:
        kde = gaussian_kde(weights.flatten())
        return kde(x_range)
    return np.zeros_like(x_range)

def plot_distribution_analysis(snapshots,n_oracle_0, n_oracle_1, text_ratio=1.2):
    text_ratio = 1.2
    row, col = 2, 3
    fig, axes = create_subplot_grid(row, col)

    cmap = cm.get_cmap('viridis')
    snapshot_steps = [snapshot['step'] for snapshot in snapshots]
    norm = mcolors.Normalize(vmin=min(snapshot_steps), vmax=max(snapshot_steps))
    colors = [cmap(norm(step)) for step in snapshot_steps]

    x_range = np.linspace(-0.25, 1.25, 300)

    ax = axes[0, 0]
    for i, (snapshot, color) in enumerate(zip(snapshots, colors)):
        weights = snapshot['weights'].flatten()
        ax.hist(weights, bins=40, alpha=0.7, density=True, color=color, histtype='stepfilled', edgecolor='none')

        weights_oracle1 = snapshot['weights_oracle1'].flatten()
        weights_oracle0 = snapshot['weights_oracle0'].flatten()
        total_weights = len(weights_oracle1) + len(weights_oracle0)
        prop1 = len(weights_oracle1) / total_weights
        prop0 = len(weights_oracle0) / total_weights
        combined_envelope = prop1 * compute_kde_envelope(weights_oracle1, x_range) + prop0 * compute_kde_envelope(weights_oracle0, x_range)
        ax.plot(x_range, combined_envelope, color='black', linewidth=4, alpha=1.0)
        ax.plot(x_range, combined_envelope, color=color, linewidth=2, alpha=1.0)


    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('Weight Value', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.set_ylabel('Density', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.set_title(f'Overall ({n_oracle_1 + n_oracle_0} Weights)', fontsize=plt.rcParams['axes.titlesize']*text_ratio)
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.5)
    ax.grid(visible=True, which='minor', color='black', linestyle='-', alpha=0.25)

    ax = axes[0, 1]
    for i, (snapshot, color) in enumerate(zip(snapshots, colors)):
        weights = snapshot['weights_oracle1'].flatten()
        if len(weights) > 0:
            envelope = compute_kde_envelope(weights, x_range)
        else:
            envelope = [None] * len(x_range)
        ax.hist(weights, bins=40, alpha=0.7, density=True, color=color, histtype='stepfilled', edgecolor='none')
        ax.plot(x_range, envelope, color='black', linewidth=4, alpha=1.0)
        ax.plot(x_range, envelope, color=color, linewidth=2, alpha=1.0)
        ax.axvline(np.mean(weights), color=color, linestyle='--', alpha=0.8, linewidth=2)

    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('Weight Value', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.set_title(f'Oracle = 1 ({n_oracle_1} Weights)', fontsize=plt.rcParams['axes.titlesize']*text_ratio)
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.5)
    ax.grid(visible=True, which='minor', color='black', linestyle='-', alpha=0.25)

    ax = axes[0, 2]
    for i, (snapshot, color) in enumerate(zip(snapshots, colors)):
        weights = snapshot['weights_oracle0'].flatten()
        if len(weights) > 0:
            envelope = compute_kde_envelope(weights, x_range)
        else:
            envelope = [None] * len(x_range)
        ax.hist(weights, bins=40, alpha=0.7, density=True, color=color, histtype='stepfilled', edgecolor='none')
        ax.plot(x_range, envelope, color='black', linewidth=4, alpha=1.0)
        ax.plot(x_range, envelope, color=color, linewidth=2, alpha=1.0)
        ax.axvline(np.mean(weights), color=color, linestyle='--', alpha=0.8, linewidth=2)


    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('Weight Value', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.set_title(f'Oracle = 0 ({n_oracle_0} Weights)', fontsize=plt.rcParams['axes.titlesize']*text_ratio)
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['axes.labelsize']*text_ratio)
    ax.grid(visible=True, which='major', color='black', linestyle='-', alpha=0.5)
    ax.grid(visible=True, which='minor', color='black', linestyle='-', alpha=0.25)

    np.random.seed(42)

    plot_qq_with_r2_list(axes[1, 0], snapshots, colors, 'weights', 'Q-Q Plot', n_sample=250, text_ratio=text_ratio)
    axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    axes[1, 0].set_ylabel('Sample Quantiles', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    plot_qq_with_r2_list(axes[1, 1], snapshots, colors, 'weights_oracle1', 'Q-Q Plot', n_sample=250, text_ratio=text_ratio)
    axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    plot_qq_with_r2_list(axes[1, 2], snapshots, colors, 'weights_oracle0', 'Q-Q Plot', n_sample=250, text_ratio=text_ratio)
    axes[1, 2].set_xlabel('Theoretical Quantiles', fontsize=plt.rcParams['axes.labelsize']*text_ratio)

    plt.tight_layout()
    cbar_ax = fig.add_axes([1.01, 0.06, 0.02, 0.875])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Step $k$', fontsize=plt.rcParams['axes.labelsize']*text_ratio)
    cbar.ax.tick_params(labelsize=plt.rcParams['axes.labelsize']*text_ratio)

    return fig, axes

