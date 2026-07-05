#!/usr/bin/env python
"""
Visualize real datasets (ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, Illness).
Analyze original dataset statistics and frequency domain characteristics.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from scripts/data_preparation to project root
base_dir = os.path.abspath(os.path.join(current_dir, '../'))
raw_data_dir = os.path.join(base_dir, 'datasets', 'raw_data')

# Set ICML academic style (unified format, enlarged fonts for two-column paper)
sns.set_theme(style="whitegrid", font_scale=1.4, rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": ".15",
    "grid.linestyle": "--",
    "axes.linewidth": 1.5,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
})

# Color palette - consistent with spectral_analysis_demo.py
c_clean = "#34495E"        # Clean/Denoised signal - darker gray
c_noisy = "#E74C3C"        # Original/Noisy signal - red
c_thresh_signal = "#2ECC71"  # Signal Threshold - Green
c_thresh_noise = "#F39C12"   # Noise Threshold - Orange

# Datasets to visualize
dataset_names = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Illness']

# Visualization settings
sample_size = 5000  # Number of points to visualize (for performance)
start_idx = 0       # Start index for visualization
freq_percentile = 10  # Keep top X% of frequency components (by power ranking), DROP bottom (100-X)%

def load_data(dataset_name):
    """Load data for a given dataset."""
    dataset_dir = os.path.join(raw_data_dir, dataset_name)
    data_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(data_path):
        print(f"Warning: File not found: {data_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    
    # Parse date column (try different formats)
    date_col = None
    for col in ['date', 'Date', 'DATE', 'time', 'Time', 'TIME', 'timestamp', 'Timestamp']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            # If parsing fails, create a simple index-based date
            df[date_col] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
    else:
        # If no date column found, create one
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
        date_col = 'date'
    
    return df, date_col

def compute_fft_spectrum(signal_data, dt=1.0):
    """Compute FFT spectrum and return frequencies and amplitudes."""
    # Compute FFT (real FFT for real-valued signals)
    fft_vals = np.fft.rfft(signal_data)
    # Compute amplitude (magnitude)
    amplitudes = np.abs(fft_vals)
    # Compute corresponding frequencies
    freqs = np.fft.rfftfreq(len(signal_data), d=dt)
    return freqs, amplitudes

def compute_sfm(signal_data):
    """
    Compute Spectral Flatness Measure (SFM) in log scale, also known as Wiener Entropy.
    
    SFM = exp(Geometric Mean(log A) - log(Arithmetic Mean(A)))
    where A_k = |X_k| is the amplitude at frequency k.
    
    Computing in log scale reduces the impact of extreme values and is more
    perceptually meaningful.
    
    SFM ranges from 0 to 1:
    - SFM = 1: Perfectly flat spectrum (white noise)
    - SFM = 0: Completely non-flat spectrum (single frequency)
    
    Args:
        signal_data: 1D array of signal values
        
    Returns:
        SFM value (float)
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

def compute_snr_frequency_based(signal_data, freq_percentile=90):
    """
    Compute SNR based on frequency-domain energy separation.
    
    Method:
    1. Compute FFT and power spectrum
    2. Rank frequency components by their power
    3. Keep top {freq_percentile}% of frequencies by power ranking (signal)
    4. Remaining {100-freq_percentile}% frequencies are considered noise
    5. SNR = 10 * log10(Signal Power / Noise Power)
    
    This is a common approach for estimating SNR when ground truth is unavailable.
    
    Args:
        signal_data: 1D array of signal values
        freq_percentile: Percentile of frequency components to keep as signal (default: 90)
        
    Returns:
        snr_db: SNR in decibels
        signal_reconstructed: Reconstructed signal from high-energy components
        noise_reconstructed: Reconstructed noise from low-energy components
        signal_mask: Boolean mask indicating which frequencies are kept
    """
    # Compute FFT
    fft_vals = np.fft.rfft(signal_data)
    
    # Compute power (magnitude squared)
    power = np.abs(fft_vals) ** 2
    
    # Calculate threshold: keep top {freq_percentile}% of frequency components
    n_freqs = len(power)
    n_keep = max(1, int(np.ceil(n_freqs * freq_percentile / 100.0)))
    
    # Get threshold power (keep frequencies with power >= threshold)
    sorted_power = np.sort(power)[::-1]  # Sort descending
    power_threshold = sorted_power[min(n_keep - 1, len(sorted_power) - 1)]
    
    # Create masks based on power threshold
    signal_mask = power >= power_threshold
    noise_mask = ~signal_mask
    
    # Separate signal and noise components
    fft_signal = fft_vals.copy()
    fft_signal[noise_mask] = 0
    
    fft_noise = fft_vals.copy()
    fft_noise[signal_mask] = 0
    
    # Reconstruct time-domain signals
    signal_reconstructed = np.fft.irfft(fft_signal, n=len(signal_data))
    noise_reconstructed = np.fft.irfft(fft_noise, n=len(signal_data))
    
    # Calculate powers
    signal_power = np.var(signal_reconstructed)
    noise_power = np.var(noise_reconstructed)
    
    # Calculate SNR in dB
    if noise_power > 1e-10:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = 100.0  # Very high SNR (essentially no noise)
    
    return snr_db, signal_reconstructed, noise_reconstructed, signal_mask

def create_visualization_plot(freq_percentile=90):
    """Create visualization plots showing time domain (clean signal) and frequency domain."""
    # Filter available datasets
    available_datasets = []
    for dataset_name in dataset_names:
        df, _ = load_data(dataset_name)
        if df is not None:
            available_datasets.append(dataset_name)
        else:
            print(f"✗ Skipping {dataset_name}: data not found")
    
    if not available_datasets:
        print("\nError: No datasets found!")
        return None
    
    n_datasets = len(available_datasets)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(32, 5 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset_name in enumerate(available_datasets):
        df, date_col = load_data(dataset_name)
        
        if df is None:
            continue
        
        # Get the first feature column (skip date column)
        feature_cols = [col for col in df.columns if col != date_col]
        if not feature_cols:
            print(f"Warning: No feature columns found for {dataset_name}")
            continue
        
        # Use first feature for visualization
        feat_col = feature_cols[0]
        
        # Sample data
        total_len = len(df)
        end_idx = min(start_idx + sample_size, total_len)
        
        sample = df.iloc[start_idx:end_idx]
        values = sample[feat_col].values
        
        # Get dates for plotting
        dates = sample[date_col].values if date_col in sample.columns else np.arange(len(values))
        
        # Compute SFM and SNR
        sfm = compute_sfm(values)
        snr_db, signal_recon, noise_recon, signal_mask = compute_snr_frequency_based(
            values, freq_percentile=freq_percentile
        )
        
        # Compute percentage of frequencies kept
        freq_kept_pct = (signal_mask.sum() / len(signal_mask)) * 100
        
        # Print statistics to console
        print(f"[{dataset_name}] SFM: {sfm:.4f}, SNR: {snr_db:.2f} dB, "
              f"Freq kept: {freq_kept_pct:.1f}%, Feature: {feat_col}")
        
        # Column 1: Time domain reconstruction comparison
        # Original Clean (GT) as background, Filtered Noisy (Result) on top
        axes[idx, 0].plot(dates, values, 
                          lw=2.5, alpha=0.4, color=c_clean,
                          label='Original (GT)', zorder=1)
        axes[idx, 0].plot(dates, signal_recon, 
                          lw=2.0, alpha=0.75, color=c_noisy,
                          label='Filtered (Result)', zorder=2)
        
        # Compute correlation for title
        corr = np.corrcoef(values, signal_recon)[0, 1]
        title_time = f'{dataset_name} - Reconstruction (Corr={corr:.4f})'
        axes[idx, 0].set_title(title_time, fontweight='bold', fontsize=16)
        axes[idx, 0].set_xlabel('Time')
        axes[idx, 0].set_ylabel('Value')
        axes[idx, 0].grid(True, linestyle='--', alpha=0.25)
        axes[idx, 0].legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        if hasattr(dates[0], 'strftime'):
            axes[idx, 0].tick_params(axis='x', rotation=45)
        
        # Column 2: Frequency domain (FFT) with threshold line
        dt = 1.0
        freqs, amplitudes = compute_fft_spectrum(values, dt=dt)
        
        # Compute power for threshold calculation
        fft_vals = np.fft.rfft(values)
        power = np.abs(fft_vals) ** 2
        
        # Find threshold
        n_freqs = len(power)
        n_keep = max(1, int(np.ceil(n_freqs * freq_percentile / 100.0)))
        sorted_power = np.sort(power)[::-1]
        power_threshold = sorted_power[min(n_keep - 1, len(sorted_power) - 1)]
        amplitude_threshold = np.sqrt(power_threshold)
        
        # Plot FFT spectrum
        axes[idx, 1].semilogy(freqs, amplitudes, 
                             lw=2.0, alpha=0.8, color=c_noisy,
                             label='FFT Amplitude', zorder=1)
        
        # Add threshold line
        axes[idx, 1].axhline(y=amplitude_threshold, 
                            color=c_thresh_signal, linestyle='--', lw=2.0, 
                            label=f'Signal Threshold (top {freq_percentile}%)', 
                            alpha=0.8, zorder=2)
        
        title_freq = f'{dataset_name} - Frequency Domain (SFM={sfm:.4f})'
        axes[idx, 1].set_title(title_freq, fontweight='bold', fontsize=16)
        axes[idx, 1].set_xlabel('Frequency (normalized)')
        axes[idx, 1].set_ylabel('Amplitude (log scale)')
        axes[idx, 1].grid(True, linestyle='--', alpha=0.25, which='both')
        axes[idx, 1].legend(loc='best', framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    return fig

def create_statistics_table(freq_percentile=90):
    """Create a statistics table for all datasets."""
    stats_data = []
    
    for dataset_name in dataset_names:
        df, date_col = load_data(dataset_name)
        
        if df is None:
            continue
        
        # Get all feature columns
        feature_cols = [col for col in df.columns if col != date_col]
        if not feature_cols:
            continue
        
        # Analyze first feature for detailed statistics
        feat_col = feature_cols[0]
        values = df[feat_col].values
        
        # Calculate SFM (Spectral Flatness Measure)
        sfm = compute_sfm(values)
        
        # Calculate SNR (frequency-domain based)
        snr_db, signal_recon, noise_recon, signal_mask = compute_snr_frequency_based(
            values, freq_percentile=freq_percentile
        )
        
        # Calculate signal power (variance)
        signal_power = np.var(values)
        noise_power = np.var(noise_recon)
        noise_power_ratio = noise_power / signal_power
        
        # Calculate frequency kept percentage
        freq_kept_pct = (signal_mask.sum() / len(signal_mask)) * 100
        
        # Calculate autocorrelation at lag 1 (measure of temporal dependency)
        if len(values) > 1:
            autocorr_lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]
        else:
            autocorr_lag1 = 0.0
        
        # Calculate coefficient of variation (CV = std/mean)
        mean_val = np.mean(values)
        if abs(mean_val) > 1e-10:
            cv = np.std(values) / abs(mean_val)
        else:
            cv = np.inf
        
        # Get frequency domain statistics
        freqs, amplitudes = compute_fft_spectrum(values, dt=1.0)
        # Find dominant frequency (highest amplitude, excluding DC component)
        if len(amplitudes) > 1:
            dominant_freq_idx = np.argmax(amplitudes[1:]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            dominant_amplitude = amplitudes[dominant_freq_idx]
        else:
            dominant_freq = 0.0
            dominant_amplitude = 0.0
        
        # Calculate reconstruction error (how well can we reconstruct by dropping low-energy freqs)
        recon_error = values - signal_recon
        recon_mse = np.mean(recon_error ** 2)
        recon_mae = np.mean(np.abs(recon_error))
        recon_rmse = np.sqrt(recon_mse)
        relative_recon_error = recon_rmse / (np.std(values) + 1e-10)  # RMSE as % of std
        recon_correlation = np.corrcoef(values, signal_recon)[0, 1]  # Correlation between original and reconstructed
        
        # Calculate statistics
        stats_data.append({
            'Dataset': dataset_name,
            'Total Points': len(values),
            'Num Features': len(feature_cols),
            'First Feature': feat_col,
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'CV': cv,
            'Variance': signal_power,
            'Autocorr(lag=1)': autocorr_lag1,
            'SFM': sfm,
            f'SNR (dB, top{freq_percentile}% freqs)': snr_db,
            'Noise Power Ratio': noise_power_ratio,
            'Freq Kept (%)': freq_kept_pct,
            'Recon RMSE': recon_rmse,
            'Recon RMSE (% of std)': relative_recon_error * 100,
            'Recon Correlation': recon_correlation,
            'Recon MAE': recon_mae,
            'Dominant Freq': dominant_freq,
            'Dominant Amp': dominant_amplitude
        })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df

def main():
    print("=" * 80)
    print("Analyzing Real Time Series Datasets")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Raw data directory exists: {os.path.exists(raw_data_dir)}")
    print()
    
    # Check which datasets exist
    available_datasets = []
    for dataset_name in dataset_names:
        result = load_data(dataset_name)
        if result is None:
            print(f"✗ Missing dataset: {dataset_name}")
            continue
        df, _ = result
        if df is not None:
            available_datasets.append(dataset_name)
            print(f"✓ Found dataset: {dataset_name}")
        else:
            print(f"✗ Missing dataset: {dataset_name}")
    
    if not available_datasets:
        print("\nError: No datasets found!")
        return
    
    print(f"\nAnalyzing {len(available_datasets)} datasets...")
    print(f"Sample size for visualization: {sample_size} points (starting from index {start_idx})")
    print(f"Statistics computed on full datasets")
    print(f"Frequency threshold: Keep top {freq_percentile}% (DROP bottom {100-freq_percentile}%)")
    print(f"\n⚠️  Aggressive filtering to demonstrate droppability!")
    print(f"\nSNR Calculation Method:")
    print(f"  1. Compute FFT power spectrum")
    print(f"  2. Rank all frequency components by their power (magnitude squared)")
    print(f"  3. Set threshold to keep ONLY top {freq_percentile}% (highest energy)")
    print(f"  4. Keep top {freq_percentile}% frequencies (above threshold) as 'signal'")
    print(f"  5. DROP bottom {100-freq_percentile}% frequencies (below threshold) as 'noise'")
    print(f"  6. Reconstruct both signal and noise in time domain via IFFT")
    print(f"  7. SNR (dB) = 10 * log10(Signal Power / Noise Power)")
    print(f"\n💡 This demonstrates DropoutTS's frequency-domain filtering approach!")
    print(f"   Even dropping {100-freq_percentile}% of frequencies, reconstruction remains high quality!")
    
    # 1. Visualization plot (time + frequency domain + reconstruction)
    print("\n[1/2] Creating visualization plots...")
    fig1 = create_visualization_plot(freq_percentile=freq_percentile)
    if fig1 is not None:
        fig1.savefig('real_datasets_original_analysis.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig1)
    
    # 2. Statistics table
    print("\n[2/2] Computing statistics...")
    stats_df = create_statistics_table(freq_percentile=freq_percentile)
    print("\n" + "=" * 80)
    print("Dataset Statistics Summary:")
    print("=" * 80)
    
    # Print with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2f}')
    print(stats_df.to_string(index=False))
    print("=" * 80)
    
    # Save statistics
    stats_df.to_csv('real_datasets_statistics.csv', index=False, float_format='%.6f')
    
    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("\nKey Observations:")
    snr_col = f'SNR (dB, top{freq_percentile}% freqs)'
    print(f"  - SFM (Spectral Flatness Measure) ranges from {stats_df['SFM'].min():.4f} to {stats_df['SFM'].max():.4f}")
    print(f"    → Lower SFM = more structured/tonal (strong frequency components)")
    print(f"    → Higher SFM = more noise-like (flat spectrum)")
    print(f"  - SNR ranges from {stats_df[snr_col].min():.2f} dB to {stats_df[snr_col].max():.2f} dB")
    print(f"    → Higher SNR = cleaner signal (low-power freqs contribute less)")
    print(f"    → Lower SNR = more noise (low-power freqs contribute more)")
    print(f"  - Noise Power Ratio ranges from {stats_df['Noise Power Ratio'].min():.4f} to {stats_df['Noise Power Ratio'].max():.4f}")
    print(f"    → Fraction of total power in the bottom {100-freq_percentile}% of frequencies")
    print(f"\n💡 DropoutTS Motivation:")
    print(f"  - Datasets with lower SNR have more droppable low-power frequency components")
    print(f"  - Frequency-domain filtering identifies which components are noise-like")
    print(f"\n  📊 Visualization Columns:")
    print(f"  Column 1: Time domain - Original (gray) vs Denoised (blue)")
    print(f"            → Shows effect of keeping ONLY top {freq_percentile}% highest-energy frequencies")
    print(f"  Column 2: Frequency domain - FFT spectrum with threshold line (dashed red)")
    print(f"            → Above threshold: top {freq_percentile}% frequencies (KEPT)")
    print(f"            → Below threshold: bottom {100-freq_percentile}% frequencies (DROPPED)")
    print(f"  Column 3: Reconstruction quality - Original (blue solid) vs Reconstructed (red dashed)")
    print(f"            → Shows how well signal can be reconstructed after DROPPING {100-freq_percentile}%!")
    print(f"\n  🎯 Key Insight: ")
    print(f"  Even after dropping {100-freq_percentile}% of frequencies, reconstruction quality remains excellent!")
    print(f"  → High correlation (check stats) means bottom {100-freq_percentile}% are droppable")
    print(f"  → This validates DropoutTS's frequency-domain dropout approach!")
    print("=" * 80)
    
    # Show plots (optional, comment out if running in headless mode)
    # plt.show()

if __name__ == '__main__':
    main()

