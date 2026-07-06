"""
Visualization of signal regimes and noise types in time series.
Demonstrates different non-stationary patterns and noise characteristics.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ==========================================
# 1. Set ICML Academic Style (Unified Format)
# ==========================================
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

# ==========================================
# 2. Data Generation Configuration
# ==========================================
np.random.seed(42)  # Fix seed for reproducibility
t = np.linspace(0, 3, 1000)  # Time axis 0-3 seconds

# --- Base signal parameters ---
freq = 1.0  # Base frequency 1Hz

# --- Define four signal regimes ---
# 1. Stationary (Periodic)
y_stationary = 4 * np.sin(2 * np.pi * freq * t)

# 2. Non-stationary Mean (Trend)
# Linear trend + sine wave
y_trend = 0.5 * t * 2 + 2 * np.sin(2 * np.pi * freq * t) - 1

# 3. Non-stationary Frequency (Chirp)
# Frequency increases linearly with time: f(t) = f0 + k*t
f0 = 0.5
k = 2.0
phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
y_chirp = 4 * np.sin(phase)

# 4. Non-stationary Variance (AM - Amplitude Modulation)
# Carrier * envelope
carrier_freq = 10.0
mod_freq = 0.5
envelope = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
y_am = 2.5 * envelope * np.sin(2 * np.pi * carrier_freq * t)

# --- Define three noise profiles ---
# Base signal for demonstrating noise
y_base = y_stationary.copy()

# 1. Gaussian Noise (Aleatoric)
noise_gaussian = np.random.normal(0, 0.5, size=t.shape)
y_gaussian = y_base + noise_gaussian

# 2. Heavy-tail Noise (Student-t)
# Use t-distribution (df=2.5) to simulate extreme values
noise_heavy = np.random.standard_t(df=2.5, size=t.shape) * 0.3
y_heavy = y_base + noise_heavy

# 3. Missing Values (Failures)
# Randomly mask 40% of data points
mask = np.random.choice([0, 1], size=t.shape, p=[0.4, 0.6])
y_missing = y_base.copy()
y_missing[mask == 0] = np.nan  # Set to NaN, matplotlib will automatically break lines

# ==========================================
# 3. Plotting Configuration
# ==========================================
fig = plt.figure(figsize=(14, 6))

# Use GridSpec to create two-row layout
# Height ratio 1:1, increase row spacing hspace appropriately
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

# --- First row: 4 signal regimes ---
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.25)
regime_titles = [
    "1. Stationary (Periodic)", 
    "2. Non-stationary (Mean)", 
    "3. Non-stationary (Frequency)", 
    "4. Non-stationary (Variance)"
]
regime_data = [y_stationary, y_trend, y_chirp, y_am]

for i in range(4):
    ax = fig.add_subplot(gs_row1[0, i])
    ax.plot(t, regime_data[i], color='#1F77B4', lw=1.8)
    ax.set_title(regime_titles[i], fontweight='bold', fontsize=10)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_ylim(-4.5, 4.5)

# --- Second row: 3 noise profiles ---
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.25)
noise_titles = [
    "Gaussian Noise (Aleatoric)", 
    "Heavy-tail Noise (Student-t)", 
    "Missing Values (Failures)"
]

# Prepare plotting data (Clean, Noisy)
noise_plot_data = [
    (y_base, y_gaussian),
    (y_base, y_heavy),
    (y_base, y_missing)
]

for i in range(3):
    ax = fig.add_subplot(gs_row2[0, i])
    clean_sig, noisy_sig = noise_plot_data[i]
    
    # Special handling for Missing Values legend and plotting
    if i == 2:
        # Draw complete light line for original signal
        ax.plot(t, y_base, color='#1F77B4', alpha=0.4, lw=2.0, label='Clean Signal')
        # Draw broken dark line for observed values
        ax.plot(t, noisy_sig, color='#2C3E50', lw=2.0, label='Observed (Corrupted)')
    else:
        # For first two, draw Clean (Blue) and Noise (Orange)
        ax.plot(t, clean_sig, color='#1F77B4', alpha=0.7, lw=2.0, label='Clean Signal')
        ax.plot(t, noisy_sig, color='#FF7F0E', alpha=0.85, lw=1.2, label='Corrupted Noise')
    
    ax.set_title(noise_titles[i], fontweight='bold', fontsize=10)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    
    # Adjust Y-axis range for Heavy-tail to show spikes, but not too exaggerated
    if i == 1:
        ax.set_ylim(-6, 6)
    else:
        ax.set_ylim(-4.5, 4.5)

# Save and display
plt.savefig('signal_noise_def.pdf', bbox_inches='tight', dpi=300)
plt.show()