"""
Parameter sensitivity analysis for DropoutTS.
Visualizes the impact of sensitivity parameter (gamma) across different noise levels.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Set ICML academic style (unified format)
sns.set_theme(style="whitegrid", font_scale=1.0, rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": ".15",
    "grid.linestyle": "--",
    "axes.linewidth": 1.2,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 10,
})

# 2. Read data - using Informer_old (trend is more obvious)
df = pd.read_csv('../checkpoints/Informer_old/table2_all_experiments_detailed.csv')

# 3. Data filtering and preprocessing
df_synth = df[df['Dataset'].str.contains('SyntheticTS') & 
              (df['Has_DropoutTS'] == 'Yes') & 
              (df['sparsity_weight'] == 0.00)].copy()

# Extract noise levels
df_synth['Noise Level'] = df_synth['Dataset'].str.extract(r'noise(\d+\.\d+)').astype(float)

# Create string format labels
df_synth['Noise Label'] = df_synth['Noise Level'].apply(lambda x: f'{x:.1f}')
df_synth['Sensitivity Label'] = df_synth['init_sensitivity'].apply(lambda x: f'{x:.1f}')

# 4. Create canvas (suitable for ICML single column)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

# Define color scheme
unique_ws = sorted(df_synth['init_sensitivity'].unique())
unique_noises = sorted(df_synth['Noise Level'].unique())

# Left plot: 3 sensitivity colors
palette_ws = ['#8B4789', '#F4A460', '#4682B4']  # Purple, Orange, Blue
# Right plot: 5 noise level colors (blue to green gradient)
palette_noise = ['#4A5899', '#6B8E99', '#7FA99A', '#A5C4A5', '#C8D9B4']

# ======================== Left plot: Impact of w_s across Noise Levels ========================
# Average across different prediction lengths
df_avg = df_synth.groupby(['Noise Level', 'init_sensitivity']).agg({
    'MSE': ['mean', 'std']
}).reset_index()
df_avg.columns = ['Noise Level', 'init_sensitivity', 'MSE_mean', 'MSE_std']

# Plot line chart
for i, ws in enumerate(unique_ws):
    data_ws = df_avg[df_avg['init_sensitivity'] == ws]
    axes[0].plot(
        data_ws['Noise Level'], 
        data_ws['MSE_mean'],
        marker='o',
        markersize=7,
        linewidth=2.5,
        color=palette_ws[i],
        label=f'{ws:.1f}',
        zorder=3
    )

axes[0].set_title(r'(a) Sensitivity $\gamma$ vs Noise Level', fontweight='bold', fontsize=11, loc='center')
axes[0].set_xlabel(r'Noise Level $\sigma$')
axes[0].set_ylabel('MSE')
axes[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
axes[0].set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
axes[0].legend(title=r'Sensitivity $\gamma$', loc='upper left', framealpha=0.9, edgecolor='gray')
axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.1)
axes[0].grid(True, linestyle='--', alpha=0.25)

# ======================== Right plot: Scatter Plot + Mean Lines ========================
# Approach: Scatter plot shows all original data points + mean lines connecting them

# Create sub-positions for each sensitivity (similar to grouping)
sensitivity_positions = {1.0: 0, 5.0: 1, 10.0: 2}
noise_offset = {0.1: -0.15, 0.3: -0.075, 0.5: 0, 0.7: 0.075, 0.9: 0.15}

# Plot scatter points (all original data points)
for i, noise in enumerate(unique_noises):
    for j, sens in enumerate(unique_ws):
        data_subset = df_synth[(df_synth['Noise Level'] == noise) & 
                               (df_synth['init_sensitivity'] == sens)]
        
        if len(data_subset) > 0:
            # Calculate x position: sensitivity base position + noise level offset
            x_pos = sensitivity_positions[sens] + noise_offset[noise]
            
            # Plot original data points (semi-transparent)
            axes[1].scatter(
                [x_pos] * len(data_subset),
                data_subset['MSE'],
                color=palette_noise[i],
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5,
                zorder=2
            )
            
            # Plot mean markers (solid, more prominent)
            mean_val = data_subset['MSE'].mean()
            axes[1].scatter(
                x_pos,
                mean_val,
                color=palette_noise[i],
                marker='D',  # Diamond
                s=80,
                edgecolors='black',
                linewidth=1.2,
                zorder=3,
                label=f'{noise:.1f}' if j == 0 else None  # Only add legend for first sensitivity
            )

# Draw lines connecting means for each noise level
for i, noise in enumerate(unique_noises):
    x_positions = []
    y_means = []
    
    for sens in unique_ws:
        data_subset = df_synth[(df_synth['Noise Level'] == noise) & 
                               (df_synth['init_sensitivity'] == sens)]
        if len(data_subset) > 0:
            x_pos = sensitivity_positions[sens] + noise_offset[noise]
            x_positions.append(x_pos)
            y_means.append(data_subset['MSE'].mean())
    
    # Draw connecting lines
    if len(x_positions) > 1:
        axes[1].plot(x_positions, y_means, 
                    color=palette_noise[i], 
                    linewidth=1.5, 
                    alpha=0.4,
                    linestyle='--',
                    zorder=1)

# Set X-axis
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['1.0', '5.0', '10.0'])
axes[1].set_xlabel(r'Sensitivity $\gamma$')
axes[1].set_ylabel('MSE')
axes[1].set_title(r'(b) MSE Distribution by $\gamma$', fontweight='bold', fontsize=11, loc='center')

# Legend: show noise levels (diamond markers)
axes[1].legend(title=r'Noise $\sigma$', 
               loc='upper left', 
               framealpha=0.9, 
               edgecolor='gray',
               ncol=1,
               handletextpad=0.5,
               columnspacing=0.8)

axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.1)
axes[1].grid(True, linestyle='--', alpha=0.25)

# Save high-resolution image
plt.savefig('parameter_sensitivity_analysis.pdf', bbox_inches='tight', dpi=300)
plt.show()