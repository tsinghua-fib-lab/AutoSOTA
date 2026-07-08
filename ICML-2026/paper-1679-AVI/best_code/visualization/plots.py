
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

IMAGE_DIR = Path(__file__).resolve().parent.parent / "images"


def _save_figure(fig, save_path):

    if not save_path:
        return

    target_path = Path(save_path)
    if not target_path.is_absolute():
        target_path = IMAGE_DIR / target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {target_path}")


def plot_topk_fwer_over_time(topk_fwer_results, title="Top-k FWER over Time", 
                             save_path=None):

    data = topk_fwer_results['topk_fwer_over_time']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(data['time'], data['topk_fwer'], color='blue', linewidth=3.5)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=3)
    
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Top-k FWER', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.4)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_topk_set_size_over_time(topk_fwer_results, 
                                 title="Top-k Confidence Set Size over Time",
                                 save_path=None):

    data = topk_fwer_results['topk_fwer_over_time']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(data['time'], data['avg_set_size'], 
            color='darkgreen', linewidth=3.5, label='Mean set size')
    
    ax.fill_between(data['time'],
                    data['avg_set_size'] - data['sd_set_size'],
                    data['avg_set_size'] + data['sd_set_size'],
                    alpha=0.2, color='green', label='±1 SD')
    
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Set Size', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_topk_comparison_grid(comparison_results, alpha=0.1,
                              title_prefix="Top-k Comparison", save_path=None):

    data = comparison_results['combined_data']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    for method in data['method'].unique():
        for sd in data['sd'].unique():
            subset = data[(data['method'] == method) & (data['sd'] == sd)]
            linestyle = '-' if method == 'random_pair' else '--'
            ax.plot(subset['time'], subset['topk_fwer'], 
                   linestyle=linestyle, linewidth=3,
                   label=f"{method}, sd={sd}")
    
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2.3)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Top-k FWER', fontsize=10)
    ax.set_ylim(0, 0.15)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for method in data['method'].unique():
        for sd in data['sd'].unique():
            subset = data[(data['method'] == method) & (data['sd'] == sd)]
            linestyle = '-' if method == 'random_pair' else '--'
            ax.plot(subset['time'], subset['avg_set_size'],
                   linestyle=linestyle, linewidth=3,
                   label=f"{method}, sd={sd}")
    
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Average Set Size', fontsize=10)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_power_comparison_methods(comparison_results, 
                                  title="Power over Time by Sampling Method",
                                  save_path=None):

    data = comparison_results['combined_data']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in data['method'].unique():
        subset = data[data['method'] == method]
        ax.plot(subset['time'], subset['avg_power'], 
               linewidth=3.5, label=method)
    
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Power', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.tick_params(labelsize=12)
    ax.legend(title='Sampling Method', fontsize=14, title_fontsize=15, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_fwer_comparison_methods(comparison_results, alpha,
                                title="FWER over Time by Sampling Method",
                                save_path=None):

    data = comparison_results['combined_data']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in data['method'].unique():
        subset = data[data['method'] == method]
        ax.plot(subset['time'], subset['fwer'], 
               linewidth=3.5, label=method)
    
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=3)
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Empirical FWER', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.4)
    ax.tick_params(labelsize=12)
    ax.legend(title='Sampling Method', fontsize=14, title_fontsize=15, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_fwer_power_comparison_grid(comparison_results, alpha=0.1,
                                    title_prefix="FWER and Power Comparison",
                                    save_path=None, sd_label="sd"):

    data = comparison_results['combined_data']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    for sd in sorted(data['sd'].unique()):
        subset = data[data['sd'] == sd]
        for method in sorted(data['method'].unique()):
            method_subset = subset[subset['method'] == method]
            linestyle = '-' if method == 'random_pair' else '--'
            ax.plot(method_subset['time'], method_subset['fwer'],
                   linestyle=linestyle, linewidth=3.5,
                   label=f"{sd_label}={sd}, {method}", alpha=0.8)
    
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=3, label=f'α={alpha}')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical FWER', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.15)
    ax.legend(fontsize=14, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for sd in sorted(data['sd'].unique()):
        subset = data[data['sd'] == sd]
        for method in sorted(data['method'].unique()):
            method_subset = subset[subset['method'] == method]
            linestyle = '-' if method == 'random_pair' else '--'
            ax.plot(method_subset['time'], method_subset['avg_power'],
                   linestyle=linestyle, linewidth=3.5,
                   label=f"{sd_label}={sd}, {method}", alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Power', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.02, min(1.02, data['avg_power'].max() * 1.1))
    ax.legend(fontsize=14, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_fwer_power_time_analysis(comparison_results, alpha=0.1, m=10,
                                   title_prefix="FWER and Power Analysis",
                                   save_path=None):

    data = comparison_results['combined_data']
    
    methods = sorted(data['method'].unique()) if 'method' in data.columns else [None]
    sd_values = sorted(data['sd'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(sd_values)))
    
    method_styles = {
        'Original': {'linestyle': '-', 'label': 'Original'},
        'Max': {'linestyle': '--', 'label': 'Max'},
        'Weighted_No_proximity': {'linestyle': ':', 'label': 'Weighted_No_proximity'},
    }
    for method in methods:
        if method and method not in method_styles:
            method_styles[method] = {'linestyle': '-', 'label': method}
    
    ax = axes[0]
    for sd_idx, sd in enumerate(sd_values):
        for method in methods:
            if method:
                subset = data[(data['sd'] == sd) & (data['method'] == method)]
                style = method_styles.get(method, {'linestyle': '-', 'label': method})
                label = f"sd={sd}, {style['label']}"
            else:
                subset = data[data['sd'] == sd]
                style = {'linestyle': '-', 'label': ''}
                label = f"sd={sd}"
            
            if len(subset) > 0:
                ax.plot(subset['time'], subset['fwer'],
                       linewidth=3, color=colors[sd_idx],
                       linestyle=style['linestyle'], label=label, alpha=0.85)
    
    ax.axhline(y=alpha, color='red', linestyle=':', linewidth=2.5, label=f'α={alpha}')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical FWER', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.15)
    ax.legend(fontsize=11, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for sd_idx, sd in enumerate(sd_values):
        for method in methods:
            if method:
                subset = data[(data['sd'] == sd) & (data['method'] == method)]
                style = method_styles.get(method, {'linestyle': '-', 'label': method})
                label = f"sd={sd}, {style['label']}"
            else:
                subset = data[data['sd'] == sd]
                style = {'linestyle': '-', 'label': ''}
                label = f"sd={sd}"
            
            if len(subset) > 0:
                ax.plot(subset['time'], subset['avg_power'],
                       linewidth=3, color=colors[sd_idx],
                       linestyle=style['linestyle'], label=label, alpha=0.85)
                if 'sd_power' in subset.columns:
                    ax.fill_between(subset['time'],
                                   subset['avg_power'] - subset['sd_power'],
                                   subset['avg_power'] + subset['sd_power'],
                                   alpha=0.1, color=colors[sd_idx])
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Power', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_covariate_comparison_fwer_power(summary, time_points, methods_list, 
                                          method_labels, m=10, alpha=0.1, 
                                          num_simulations=100, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'random_pair_original': '#1f77b4',
        'tournament_original': '#ff7f0e',
        'random_pair_covariate': '#2ca02c',
        'tournament_covariate': '#d62728'
    }
    linestyles = {
        'random_pair_original': '-',
        'tournament_original': '-',
        'random_pair_covariate': '--',
        'tournament_covariate': '--'
    }
    
    default_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_idx = 0
    for method in methods_list:
        if method not in colors:
            colors[method] = default_colors[color_idx % len(default_colors)]
            color_idx += 1
        if method not in linestyles:
            linestyles[method] = '-'
    
    ax1 = axes[0]
    for method in methods_list:
        label = method_labels.get(method, method)
        ax1.plot(time_points, summary[method]['fwer'], 
                label=label, 
                color=colors[method],
                linestyle=linestyles[method],
                linewidth=2)
    ax1.axhline(y=alpha, color='red', linestyle=':', linewidth=1.5, label=f'α = {alpha}')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('FWER', fontsize=12)
    ax1.set_title('FWER over Time', fontsize=14)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_ylim(0, max(0.2, max([max(summary[m]['fwer']) for m in methods_list]) * 1.1))
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for method in methods_list:
        label = method_labels.get(method, method)
        ax2.plot(time_points, summary[method]['power_mean'], 
                label=label, 
                color=colors[method],
                linestyle=linestyles[method],
                linewidth=2)
        power_mean = np.array(summary[method]['power_mean'])
        power_std = np.array(summary[method]['power_std'])
        ax2.fill_between(time_points, 
                        power_mean - power_std, 
                        power_mean + power_std,
                        color=colors[method], alpha=0.1)
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_title('Power over Time', fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'SERPANT Algorithm Comparison: Original vs Covariate-Assisted\n(m={m}, α={alpha}, {num_simulations} simulations)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_covariate_sd_x_comparison(comparison_results, alpha=0.1, m=10,
                                    title_prefix="Covariate-Assisted Algorithm",
                                    save_path=None):
    data = comparison_results['combined_data']
    sd_x_values = comparison_results['sd_x_values']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sd_x_values)))
    
    ax = axes[0]
    for idx, sd_x in enumerate(sorted(sd_x_values)):
        subset = data[data['sd_x'] == sd_x]
        ax.plot(subset['time'], subset['fwer'],
               linewidth=3, color=colors[idx],
               label=f"sd_x={sd_x}")
    
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2.5, label=f'α={alpha}')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical FWER', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for idx, sd_x in enumerate(sorted(sd_x_values)):
        subset = data[data['sd_x'] == sd_x]
        ax.plot(subset['time'], subset['avg_power'],
               linewidth=3, color=colors[idx],
               label=f"sd_x={sd_x}")
        if 'sd_power' in subset.columns:
            ax.fill_between(subset['time'],
                           subset['avg_power'] - subset['sd_power'],
                           subset['avg_power'] + subset['sd_power'],
                           alpha=0.15, color=colors[idx])
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Power', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_original_vs_covariate_comparison(comparison_results, alpha=0.1, m=10,
                                           save_path=None):
    data = comparison_results['combined_data']
    sd_x_values = comparison_results['sd_x_values']
    
    n_sd_x = len(sd_x_values)
    fig, axes = plt.subplots(2, n_sd_x, figsize=(6*n_sd_x, 10))
    
    if n_sd_x == 1:
        axes = axes.reshape(2, 1)
    
    colors = {
        'original': '#1f77b4',
        'covariate_assisted': '#d62728'
    }
    linestyles = {
        'original': '-',
        'covariate_assisted': '--'
    }
    labels = {
        'original': 'Original',
        'covariate_assisted': 'Covariate-Assisted'
    }
    
    for col_idx, sd_x in enumerate(sorted(sd_x_values)):
        ax = axes[0, col_idx]
        for algorithm in ['original', 'covariate_assisted']:
            subset = data[(data['sd_x'] == sd_x) & (data['algorithm'] == algorithm)]
            if len(subset) > 0:
                ax.plot(subset['time'], subset['fwer'],
                       linewidth=3, color=colors[algorithm],
                       linestyle=linestyles[algorithm],
                       label=labels[algorithm])
        
        ax.axhline(y=alpha, color='red', linestyle=':', linewidth=2, label=f'α={alpha}')
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Empirical FWER', fontsize=11, fontweight='bold')
        ax.set_title(f'sd_x = {sd_x}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 0.25)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, col_idx]
        for algorithm in ['original', 'covariate_assisted']:
            subset = data[(data['sd_x'] == sd_x) & (data['algorithm'] == algorithm)]
            if len(subset) > 0:
                ax.plot(subset['time'], subset['avg_power'],
                       linewidth=3, color=colors[algorithm],
                       linestyle=linestyles[algorithm],
                       label=labels[algorithm])
                if 'sd_power' in subset.columns:
                    ax.fill_between(subset['time'],
                                   subset['avg_power'] - subset['sd_power'],
                                   subset['avg_power'] + subset['sd_power'],
                                   alpha=0.12, color=colors[algorithm])
        
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Power', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_covariate_sd_x_grid(comparison_results, alpha=0.1,
                              title_prefix="Original vs Covariate-Assisted",
                              save_path=None):
    data = comparison_results['combined_data']
    sd_x_values = sorted(comparison_results['sd_x_values'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(sd_x_values)))
    
    ax = axes[0]
    for idx, sd_x in enumerate(sd_x_values):
        for algorithm in ['original', 'covariate_assisted']:
            subset = data[(data['sd_x'] == sd_x) & (data['algorithm'] == algorithm)]
            if len(subset) > 0:
                linestyle = '-' if algorithm == 'original' else '--'
                label = f"sd_x={sd_x}, {'Original' if algorithm == 'original' else 'Covariate'}"
                ax.plot(subset['time'], subset['fwer'],
                       linewidth=2.5, color=base_colors[idx],
                       linestyle=linestyle, label=label, alpha=0.85)
    
    ax.axhline(y=alpha, color='red', linestyle=':', linewidth=2.5, label=f'α={alpha}')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical FWER', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for idx, sd_x in enumerate(sd_x_values):
        for algorithm in ['original', 'covariate_assisted']:
            subset = data[(data['sd_x'] == sd_x) & (data['algorithm'] == algorithm)]
            if len(subset) > 0:
                linestyle = '-' if algorithm == 'original' else '--'
                label = f"sd_x={sd_x}, {'Original' if algorithm == 'original' else 'Covariate'}"
                ax.plot(subset['time'], subset['avg_power'],
                       linewidth=2.5, color=base_colors[idx],
                       linestyle=linestyle, label=label, alpha=0.85)
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Power', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_covariate_methods_comparison(comparison_results, alpha=0.1,
                                       save_path=None):
    data = comparison_results['combined_data']
    sd_x_values = sorted(comparison_results['sd_x_values'])
    sampling_methods = comparison_results.get('sampling_methods', data['method'].unique().tolist())
    
    n_sd_x = len(sd_x_values)
    fig, axes = plt.subplots(2, n_sd_x, figsize=(6*n_sd_x, 10))
    
    if n_sd_x == 1:
        axes = axes.reshape(2, 1)
    
    method_colors = {
        'random_pair': '#1f77b4',
        'tournament': '#ff7f0e'
    }
    algorithm_linestyles = {
        'original': '-',
        'covariate_assisted': '--'
    }
    algorithm_labels = {
        'original': 'Orig',
        'covariate_assisted': 'Cov'
    }
    method_labels = {
        'random_pair': 'RP',
        'tournament': 'Tour'
    }
    
    for col_idx, sd_x in enumerate(sd_x_values):
        ax = axes[0, col_idx]
        for method in sampling_methods:
            for algorithm in ['original', 'covariate_assisted']:
                subset = data[(data['sd_x'] == sd_x) & 
                             (data['method'] == method) & 
                             (data['algorithm'] == algorithm)]
                if len(subset) > 0:
                    label = f"{method_labels.get(method, method)}-{algorithm_labels[algorithm]}"
                    ax.plot(subset['time'], subset['fwer'],
                           linewidth=2.5, 
                           color=method_colors.get(method, '#333333'),
                           linestyle=algorithm_linestyles[algorithm],
                           label=label, alpha=0.85)
        
        ax.axhline(y=alpha, color='red', linestyle=':', linewidth=2, label=f'α={alpha}')
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Empirical FWER', fontsize=11, fontweight='bold')
        ax.set_title(f'sd_x = {sd_x}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 0.25)
        ax.legend(fontsize=9, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, col_idx]
        for method in sampling_methods:
            for algorithm in ['original', 'covariate_assisted']:
                subset = data[(data['sd_x'] == sd_x) & 
                             (data['method'] == method) & 
                             (data['algorithm'] == algorithm)]
                if len(subset) > 0:
                    label = f"{method_labels.get(method, method)}-{algorithm_labels[algorithm]}"
                    ax.plot(subset['time'], subset['avg_power'],
                           linewidth=2.5,
                           color=method_colors.get(method, '#333333'),
                           linestyle=algorithm_linestyles[algorithm],
                           label=label, alpha=0.85)
                    if 'sd_power' in subset.columns:
                        ax.fill_between(subset['time'],
                                       subset['avg_power'] - subset['sd_power'],
                                       subset['avg_power'] + subset['sd_power'],
                                       alpha=0.08, 
                                       color=method_colors.get(method, '#333333'))
        
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Power', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig


def plot_covariate_methods_grid(comparison_results, alpha=0.1,
                                 save_path=None):
    data = comparison_results['combined_data']
    sd_x_values = sorted(comparison_results['sd_x_values'])
    sampling_methods = comparison_results.get('sampling_methods', data['method'].unique().tolist())
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sd_x_colors = plt.cm.tab10(np.linspace(0, 0.7, len(sd_x_values)))
    
    algorithm_linestyles = {
        'original': '-',
        'covariate_assisted': '--'
    }
    method_markers = {
        'random_pair': 'o',
        'tournament': 's'
    }
    algorithm_labels = {
        'original': 'Orig',
        'covariate_assisted': 'Cov'
    }
    method_labels = {
        'random_pair': 'RP',
        'tournament': 'Tour'
    }
    

    ax = axes[0]
    for idx, sd_x in enumerate(sd_x_values):
        for method in sampling_methods:
            for algorithm in ['original', 'covariate_assisted']:
                subset = data[(data['sd_x'] == sd_x) & 
                             (data['method'] == method) & 
                             (data['algorithm'] == algorithm)]
                if len(subset) > 0:
                    label = f"sd_x={sd_x}, {method_labels.get(method, method)}-{algorithm_labels[algorithm]}"
                    ax.plot(subset['time'], subset['fwer'],
                           linewidth=2.2, color=sd_x_colors[idx],
                           linestyle=algorithm_linestyles[algorithm],
                           marker=method_markers.get(method, 'o'),
                           markevery=max(1, len(subset)//8),
                           markersize=5,
                           label=label, alpha=0.85)
    
    ax.axhline(y=alpha, color='red', linestyle=':', linewidth=2.5, label=f'α={alpha}')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical FWER', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for idx, sd_x in enumerate(sd_x_values):
        for method in sampling_methods:
            for algorithm in ['original', 'covariate_assisted']:
                subset = data[(data['sd_x'] == sd_x) & 
                             (data['method'] == method) & 
                             (data['algorithm'] == algorithm)]
                if len(subset) > 0:
                    label = f"sd_x={sd_x}, {method_labels.get(method, method)}-{algorithm_labels[algorithm]}"
                    ax.plot(subset['time'], subset['avg_power'],
                           linewidth=2.2, color=sd_x_colors[idx],
                           linestyle=algorithm_linestyles[algorithm],
                           marker=method_markers.get(method, 'o'),
                           markevery=max(1, len(subset)//8),
                           markersize=5,
                           label=label, alpha=0.85)
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Power', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig

