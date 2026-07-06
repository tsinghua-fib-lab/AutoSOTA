"""
Visualization of the fixed dropout paradox in Informer and consistent gains in Crossformer.
Shows how DropoutTS restores robustness in unstable models and provides consistent improvements.
"""
import matplotlib.pyplot as plt
import seaborn as sns
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

# 2. Prepare data (extracted from previous LaTeX tables)
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

# Informer: High volatility -> Stable (critical improvement)
informer_raw = [0.966, 0.978, 0.936, 0.841, 0.828]
informer_dt  = [0.514, 0.507, 0.494, 0.471, 0.464]

# Crossformer: Stable -> Lower (consistent enhancement)
cross_raw = [0.450, 0.439, 0.431, 0.411, 0.409]
cross_dt  = [0.386, 0.378, 0.391, 0.379, 0.360]

# 3. Plotting (suitable for ICML single column)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

# --- Left plot: Informer ---
ax1 = axes[0]
ax1.plot(noise_levels, informer_raw, 'o-', color='#D62728', lw=2.0, markersize=6, label='Informer')
ax1.plot(noise_levels, informer_dt, 's-', color='#1F77B4', lw=2.0, markersize=6, label='Informer + DropoutTS')

# Annotate paradox - red box around data points
ax1.fill_between([0.05, 0.95], 0.76, 1.02, color='#D62728', alpha=0.06)
ax1.text(0.31, 0.80, "Fixed Dropout Paradox\nUnstable & High Error", color='#D62728', fontsize=8.5, ha='center', fontweight='bold', style='italic')
ax1.annotate("", xy=(0.62, 0.55), xytext=(0.62, 0.78), arrowprops=dict(arrowstyle="-|>", color='#1F77B4', lw=1.8, mutation_scale=12))
ax1.text(0.64, 0.67, "Restored\nRobustness", color='#1F77B4', fontsize=8.5, fontweight='bold', va='center')

ax1.set_title('(a) Informer', fontweight='bold', fontsize=11, loc='center')
ax1.set_xlabel(r'Noise Level ($\sigma$)')
ax1.set_ylabel('MSE')
ax1.set_xticks(noise_levels)
ax1.set_ylim(0.38, 1.18)
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax1.grid(True, linestyle='--', alpha=0.25)

# --- Right plot: Crossformer ---
ax2 = axes[1]
ax2.plot(noise_levels, cross_raw, 'o-', color='#D62728', lw=2.0, markersize=6, label='Crossformer')
ax2.plot(noise_levels, cross_dt, 's-', color='#1F77B4', lw=2.0, markersize=6, label='Crossformer + DropoutTS')

# Blue background region around blue line
ax2.fill_between([0.05, 0.95], 0.355, 0.395, color='#1F77B4', alpha=0.06)

# Annotate consistent improvement - arrow indicates reduction
avg_imp = np.mean([(r - d)/r for r, d in zip(cross_raw, cross_dt)]) * 100
ax2.annotate("", xy=(0.28, 0.392), xytext=(0.28, 0.428), 
             arrowprops=dict(arrowstyle="-|>", color='#1F77B4', lw=1.8, mutation_scale=12))
ax2.text(0.30, 0.41, f"Consistent Gain\n(~{avg_imp:.1f}%)", color='#1F77B4', fontsize=8.5, ha='left', fontweight='bold', va='center')

ax2.set_title('(b) Crossformer', fontweight='bold', fontsize=11, loc='center')
ax2.set_xlabel(r'Noise Level ($\sigma$)')
ax2.set_ylabel('MSE')
ax2.set_xticks(noise_levels)
ax2.set_ylim(0.35, 0.48) 
ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax2.grid(True, linestyle='--', alpha=0.25)

plt.savefig('paradox_informer_crossformer.pdf', bbox_inches='tight', dpi=300)
plt.show()