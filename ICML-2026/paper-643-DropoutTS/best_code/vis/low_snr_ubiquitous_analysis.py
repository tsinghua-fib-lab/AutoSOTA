"""
Visualize that low signal-to-noise ratios are ubiquitous in real-world time series datasets.
This supports the motivation for adaptive regularization in DropoutTS.
Also shows SFM-SNR correlation analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import os

# Set ICML academic style
sns.set_theme(style="whitegrid", font_scale=1.0, rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": ".15",
    "grid.linestyle": "--",
    "axes.linewidth": 1.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
})

def compute_sfm_log_scale(signal_data):
    """
    Compute Spectral Flatness Measure (SFM) in log scale, also known as Wiener Entropy.
    
    SFM = exp(Geometric Mean(log A) - log(Arithmetic Mean(A)))
    where A_k = |X_k| is the amplitude at frequency k.
    
    Computing in log scale reduces the impact of extreme values and is more
    perceptually meaningful.
    
    Args:
        signal_data: 1D array of signal values
        
    Returns:
        SFM value (float) in range [0, 1]
    """
    # Compute FFT
    fft_vals = np.fft.rfft(signal_data)
    # Compute amplitude (magnitude)
    amplitudes = np.abs(fft_vals)
    
    # Avoid zeros for log calculation
    amplitudes = amplitudes + 1e-10
    
    # Compute in log scale
    log_amplitudes = np.log(amplitudes)
    
    # Geometric mean in log domain = arithmetic mean of logs
    geometric_mean_log = np.mean(log_amplitudes)
    
    # Arithmetic mean: compute in linear domain then take log
    arithmetic_mean_linear = np.mean(amplitudes)
    arithmetic_mean_log = np.log(arithmetic_mean_linear)
    
    # SFM = exp(GM_log - AM_log) = GM / AM
    sfm = np.exp(geometric_mean_log - arithmetic_mean_log)
    
    return np.clip(sfm, 0.0, 1.0)

# Load data
df = pd.read_csv('real_datasets_statistics.csv')

# Filter datasets: only include specified ones, exclude Traffic
target_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Illness']
# Note: 'Illness' in CSV corresponds to 'ILI' dataset
df_filtered = df[df['Dataset'].isin(target_datasets)].copy()
df_filtered = df_filtered.reset_index(drop=True)

# Extract relevant metrics
datasets = df_filtered['Dataset'].values
snr = df_filtered['SNR (dB, top10% freqs)'].values
noise_power_ratio = df_filtered['Noise Power Ratio'].values * 100  # Convert to percentage
recon_rmse_pct = df_filtered['Recon RMSE (% of std)'].values
recon_corr = df_filtered['Recon Correlation'].values
sfm_original = df_filtered['SFM'].values
cv = df_filtered['CV'].values

print(f"Filtered datasets: {list(datasets)}")
print(f"Excluded: Traffic")

# Recompute SFM in log scale from original data
print("Recomputing SFM in log scale from original datasets...")
sfm_log = np.zeros_like(sfm_original)
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../'))
raw_data_dir = os.path.join(base_dir, 'datasets', 'raw_data')

for i, dataset_name in enumerate(datasets):
    dataset_dir = os.path.join(raw_data_dir, dataset_name)
    data_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    
    if os.path.exists(data_path):
        try:
            df_data = pd.read_csv(data_path)
            # Use first numeric column as signal
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                signal = df_data[numeric_cols[0]].values
                # Remove NaN values
                signal = signal[~np.isnan(signal)]
                if len(signal) > 0:
                    sfm_log[i] = compute_sfm_log_scale(signal)
                    print(f"  {dataset_name}: SFM (original) = {sfm_original[i]:.6f}, SFM (log) = {sfm_log[i]:.6f}")
                else:
                    sfm_log[i] = sfm_original[i]
                    print(f"  {dataset_name}: No valid data, using original SFM")
            else:
                sfm_log[i] = sfm_original[i]
                print(f"  {dataset_name}: No numeric columns, using original SFM")
        except Exception as e:
            sfm_log[i] = sfm_original[i]
            print(f"  {dataset_name}: Error computing log-scale SFM ({e}), using original SFM")
    else:
        sfm_log[i] = sfm_original[i]
        print(f"  {dataset_name}: Data file not found, using original SFM")

# Use log-scale SFM
sfm = sfm_log
print(f"\nUsing log-scale SFM values.")
print(f"SFM range: [{np.min(sfm):.6f}, {np.max(sfm):.6f}]")

print("=" * 80)
print("Real-World Datasets: Low SNR is Ubiquitous")
print("=" * 80)
print(f"\nSNR Statistics:")
print(f"  Mean SNR: {np.mean(snr):.2f} dB")
print(f"  Median SNR: {np.median(snr):.2f} dB")
print(f"  Min SNR: {np.min(snr):.2f} dB ({datasets[np.argmin(snr)]})")
print(f"  Max SNR: {np.max(snr):.2f} dB ({datasets[np.argmax(snr)]})")
print(f"  Std SNR: {np.std(snr):.2f} dB")
print(f"\nDatasets with SNR < 15 dB: {np.sum(snr < 15)} / {len(snr)} ({np.sum(snr < 15)/len(snr)*100:.1f}%)")
print(f"Datasets with SNR < 20 dB: {np.sum(snr < 20)} / {len(snr)} ({np.sum(snr < 20)/len(snr)*100:.1f}%)")
print(f"Datasets with SNR < 30 dB: {np.sum(snr < 30)} / {len(snr)} ({np.sum(snr < 30)/len(snr)*100:.1f}%)")
print("=" * 80)

# Calculate SFM-SNR correlations
pearson_r, pearson_p = stats.pearsonr(sfm, snr)
spearman_r, spearman_p = stats.spearmanr(sfm, snr)

# Fit regression line for SFM vs SNR
z = np.polyfit(sfm, snr, 1)
p = np.poly1d(z)
x_line = np.linspace(sfm.min(), sfm.max(), 100)
y_line = p(x_line)

# ============ Create ICML single-column version (1 row, 3 columns) ============
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)

# ============ Left: SNR Distribution by Dataset ============
sorted_idx = np.argsort(snr)[::-1]  # Descending
colors = []
for s in snr[sorted_idx]:
    if s < 10:
        colors.append('#E74C3C')  # Red
    elif s < 20:
        colors.append('#F39C12')  # Orange
    elif s < 30:
        colors.append('#F1C40F')  # Yellow
    else:
        colors.append('#27AE60')  # Green

bars = ax1.barh(np.arange(len(datasets)), snr[sorted_idx], 
                color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

# Add reference line for low SNR threshold
ax1.axvline(x=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

ax1.set_yticks(np.arange(len(datasets)))
ax1.set_yticklabels(datasets[sorted_idx], fontsize=9, fontweight='bold')
ax1.set_xlabel('SNR (dB)', fontweight='bold')
ax1.set_title('(a) SNR in Real Datasets', fontweight='bold', loc='left')
ax1.grid(True, linestyle='--', alpha=0.3, axis='x')

# Add annotation
pct_below_20 = np.sum(snr < 20) / len(snr) * 100
textstr = f'{np.sum(snr < 20)}/{len(snr)} datasets\nSNR < 20 dB'
ax1.text(0.96, 0.96, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                  edgecolor='black', linewidth=0.8))

# ============ Middle: SFM vs SNR Scatter Plot ============
# Use distinct colors for better visibility
color_map = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
for i, (ds, s, n) in enumerate(zip(datasets, sfm, snr)):
    ax2.scatter(s, n, s=80, alpha=0.75, color=color_map[i], 
                edgecolors='black', linewidth=0.8, zorder=3, label=ds)

# Regression line
ax2.plot(x_line, y_line, 'r--', linewidth=2.0, alpha=0.85, 
         label='Linear fit', zorder=2)

ax2.set_xlabel(r'SFM (Spectral Flatness Measure)', fontweight='bold')
ax2.set_ylabel('SNR (dB)', fontweight='bold')
ax2.set_title('(b) SFM vs SNR Correlation', fontweight='bold', loc='left')
ax2.grid(True, linestyle='--', alpha=0.3)

# Add correlation annotation (compact, ICML style)
textstr = f'r = {pearson_r:.3f}\np < {pearson_p:.1e}'
ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', 
         alpha=0.85, edgecolor='black', linewidth=0.8))

# Add legend for datasets (place in upper right corner inside the plot)
ax2.legend(loc='upper right', 
           frameon=True, framealpha=0.9, fontsize=7, 
           edgecolor='gray', ncol=2, handletextpad=0.3, 
           columnspacing=0.5, handlelength=1.0, 
           fancybox=True)

# ============ Right: Ranked Comparison (SFM and SNR) ============
# Sort by SFM (ascending: low SFM = clean, high SFM = noisy)
sorted_idx_sfm = np.argsort(sfm)

# Create dual-axis plot: SFM (left) and SNR (right)
ax3_twin = ax3.twinx()

# Plot SFM bars (left axis) - red for high noise
bars1 = ax3.bar(np.arange(len(datasets)) - 0.2, sfm[sorted_idx_sfm], 0.4,
                color='#E74C3C', alpha=0.75, edgecolor='black', linewidth=0.8,
                label='SFM (↑ noisy)', zorder=2)

# Plot SNR bars (right axis) - green for clean signal
bars2 = ax3_twin.bar(np.arange(len(datasets)) + 0.2, snr[sorted_idx_sfm], 0.4,
                     color='#27AE60', alpha=0.75, edgecolor='black', linewidth=0.8,
                     label='SNR (↑ clean)', zorder=2)

# Configure axes
ax3.set_xticks(np.arange(len(datasets)))
ax3.set_xticklabels(datasets[sorted_idx_sfm], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('SFM', fontweight='bold', color='#E74C3C')
ax3_twin.set_ylabel('SNR (dB)', fontweight='bold', color='#27AE60')
ax3.set_title('(c) Datasets Ranked by SFM', fontweight='bold', loc='left')
ax3.set_ylim(0, 0.9)  # Set SFM axis maximum to 0.9
ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

# Color the y-axis labels and ticks to match bars
ax3.tick_params(axis='y', labelcolor='#E74C3C')
ax3_twin.tick_params(axis='y', labelcolor='#27AE60')

# Add legend (compact) - moved to upper right to avoid obstruction
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
           frameon=True, framealpha=0.95, fontsize=8, edgecolor='gray', 
           handlelength=1.5, handletextpad=0.5)

plt.savefig('low_snr_ubiquitous_analysis.pdf', bbox_inches='tight', facecolor='white', dpi=300)

print("\n" + "=" * 80)
print("SFM vs SNR Correlation Analysis")
print("=" * 80)
print(f"Pearson Correlation:  r = {pearson_r:.4f}, p-value = {pearson_p:.4e}")
print(f"Spearman Correlation: ρ = {spearman_r:.4f}, p-value = {spearman_p:.4e}")
print("=" * 80)

print("\n" + "=" * 80)
print("Summary: Low SNR is Ubiquitous in Real-World Time Series")
print("=" * 80)
print(f"✅ {np.sum(snr < 20)}/{len(snr)} datasets ({np.sum(snr < 20)/len(snr)*100:.0f}%) have SNR < 20 dB")
print(f"✅ Average noise power: {np.mean(noise_power_ratio):.2f}% of total")
print(f"✅ SFM-SNR correlation: r = {pearson_r:.3f} (p < {pearson_p:.1e})")
print(f"✅ This motivates frequency-domain adaptive dropout in DropoutTS!")
print("=" * 80)
print(f"✅ Saved: low_snr_ubiquitous_analysis.pdf")
