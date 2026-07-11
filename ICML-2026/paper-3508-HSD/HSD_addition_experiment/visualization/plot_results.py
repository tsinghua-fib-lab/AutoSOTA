"""
Visualization for ICML rebuttal.
Three focused plot types:
  1. Training curves comparison
  2. Vector field comparison on mesh
  3. Topological contour comparison
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    'font.family': 'serif',
})

COLORS = {'HSD': '#2176AE', 'FNO': '#F0803C', 'DeepONet': '#57A773',
           'GNOT': '#9B59B6', 'ONO': '#E74C3C', 'HAMLET': '#F39C12',
           'GeoFNO': '#1ABC9C', 'MGN': '#34495E'}


def plot_training_curves(train_histories, val_histories, model_names, save_path):
    """Plot train/val loss curves for all methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, t_hist, v_hist in zip(model_names, train_histories, val_histories):
        c = COLORS.get(name, '#333333')
        ax1.plot(t_hist, label=name, color=c, linewidth=1.5)
        ax2.plot(v_hist, label=name, color=c, linewidth=1.5)

    for ax, title in [(ax1, 'Training Loss'), (ax2, 'Validation Loss')]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative L2')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f"{save_path}.{fmt}")
    plt.close()
    print(f"  Saved {save_path}")


def plot_bar_comparison(results_dict, metric_key, title, save_path, ylabel=None):
    """Grouped bar chart comparing methods."""
    names = list(results_dict.keys())
    vals = [results_dict[n].get(metric_key, 0) for n in names]
    colors = [COLORS.get(n, '#999999') for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.5))
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='white', linewidth=0.8)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f"{save_path}.{fmt}")
    plt.close()
    print(f"  Saved {save_path}")


def plot_cross_input_consistency(results_per_mode, save_path):
    """Line plot showing accuracy across input modes."""
    modes = list(results_per_mode.keys())
    rels = [results_per_mode[m]['relative_l2'] for m in modes]
    spread = max(rels) - min(rels)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(modes, rels, 'o-', color=COLORS['HSD'], linewidth=2, markersize=8, label='HSD')
    ax.fill_between(modes, min(rels) - 0.001, max(rels) + 0.001, alpha=0.15, color=COLORS['HSD'])
    ax.set_ylabel('Relative L2')
    ax.set_title(f'Cross-Input Consistency (spread: {spread:.4f})')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f"{save_path}.{fmt}")
    plt.close()
    print(f"  Saved {save_path}")


def plot_ablation_table(results, title, save_path):
    """Create a table-style figure for ablation results."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.5 + 1)))
    ax.axis('off')

    headers = ['Method', 'Params', 'Rel L2']
    rows = []
    for name, r in sorted(results.items(), key=lambda x: x[1].get('rel_l2', x[1].get('relative_l2', 99))):
        rel = r.get('rel_l2', r.get('relative_l2', 0))
        params = r.get('params', 0)
        rows.append([name, f'{params:,}' if params else '-', f'{rel:.4f}'])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Highlight best row
    best_idx = min(range(len(rows)), key=lambda i: float(rows[i][2]))
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor('#d4edda')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f"{save_path}.{fmt}")
    plt.close()
    print(f"  Saved {save_path}")


def generate_all_plots(results_path='./output/results/all_results.json',
                       output_dir='./output/figures'):
    """Generate all plots from saved results."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_path):
        print(f"No results at {results_path}, run experiments first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print("Generating visualizations...")

    # Input modality ablation
    for ds in ['ellipsoid_aero', 'torus_helmholtz']:
        key = f'{ds}_input'
        if key in results:
            plot_cross_input_consistency(results[key], f'{output_dir}/ablation1_input_{ds}')

    # Baseline comparison
    for ds in ['ellipsoid_aero', 'torus_helmholtz']:
        key = f'{ds}_baseline'
        if key in results:
            plot_bar_comparison(results[key], 'rel_l2',
                                f'Baseline Comparison ({ds})',
                                f'{output_dir}/ablation3_baselines_{ds}',
                                ylabel='Relative L2 Error')
            plot_ablation_table(results[key],
                                f'Extended Baselines ({ds})',
                                f'{output_dir}/ablation3_table_{ds}')

    # Layer type ablation
    for ds in ['ellipsoid_aero', 'torus_helmholtz']:
        key = f'{ds}_layer'
        if key in results:
            plot_bar_comparison(results[key], 'relative_l2',
                                f'Layer Ablation ({ds})',
                                f'{output_dir}/ablation2_layer_{ds}',
                                ylabel='Relative L2 Error')

    print("Done generating plots.")


if __name__ == "__main__":
    generate_all_plots()
