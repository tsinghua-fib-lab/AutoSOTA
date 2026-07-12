import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

# Data with real values
models = ['Llama-3.1-8B', 'Mathstral-7B', 'Qwen2.5-Coder-7B', 'DeepSeek-R1-Distill-Qwen-7B']

# v0 data
v0_rps = [10.50, 11.25, 10.83, 10.21]
v0_ttft = [18.20, 32.40, 28.50, 35.80]
v0_tpot = [245.30, 235.60, 175.80, 225.40]

# v1 data
v1_rps = [14.93, 12.85, 13.20, 11.80]
v1_ttft = [29.10, 41.90, 41.88, 44.20]
v1_tpot = [358.04, 221.10, 168.90, 198.70]

# Ours data
ours_rps = [16.37, 14.39, 14.19, 13.75]
ours_ttft = [34.79, 55.26, 22.55, 42.10]
ours_tpot = [188.45, 178.30, 125.91, 165.54]

colors = {'v0': '#8A8A8A', 'v1': '#B23A3A', 'Ours': '#2A6F97'}
markers = {'v0': '^', 'v1': 'o', 'Ours': 's'}


def plot_line_3metrics():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x = np.arange(len(models))
    
    # RPS
    ax0 = axes[0]
    ax0.plot(x, v1_rps, marker=markers['v1'], color=colors['v1'], linewidth=2,
             markersize=10, label=r'v1', linestyle='-')
    ax0.plot(x, v0_rps, marker=markers['v0'], color=colors['v0'], linewidth=2,
             markersize=10, label=r'v0', linestyle='--')
    ax0.plot(x, ours_rps, marker=markers['Ours'], color=colors['Ours'], linewidth=2,
             markersize=10, label=r'EB($\hat{k}^*$)', linestyle='-')
    
    ax0.set_xticks(x)
    ax0.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax0.set_ylabel('RPS')
    ax0.set_title('Throughput (↑ better)', fontweight='bold')
    ax0.legend()
    ax0.grid(axis='y', alpha=0.3)
    
    # TTFT
    ax1 = axes[1]
    ax1.plot(x, v1_ttft, marker=markers['v1'], color=colors['v1'], linewidth=2,
             markersize=10, label=r'v1', linestyle='-')
    ax1.plot(x, v0_ttft, marker=markers['v0'], color=colors['v0'], linewidth=2,
             markersize=10, label=r'v0', linestyle='--')
    ax1.plot(x, ours_ttft, marker=markers['Ours'], color=colors['Ours'], linewidth=2,
             markersize=10, label=r'EB($\hat{k}^*$)', linestyle='-')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax1.set_ylabel('TTFT (s)')
    ax1.set_title('Time to First Token (↓ better)', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # TPOT
    ax2 = axes[2]
    ax2.plot(x, v1_tpot, marker=markers['v1'], color=colors['v1'], linewidth=2,
             markersize=10, label=r'v1', linestyle='-')
    ax2.plot(x, v0_tpot, marker=markers['v0'], color=colors['v0'], linewidth=2,
             markersize=10, label=r'v0', linestyle='--')
    ax2.plot(x, ours_tpot, marker=markers['Ours'], color=colors['Ours'], linewidth=2,
             markersize=10, label=r'EB($\hat{k}^*$)', linestyle='-')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax2.set_ylabel('TPOT (ms)')
    ax2.set_title('Time per Output Token (↓ better)', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    plt.savefig('scalmodel.pdf', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    plot_line_3metrics()
    print("Plot saved to scalmodel.pdf")
