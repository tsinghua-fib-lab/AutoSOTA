"""
Spectral analysis demonstration for different signal types and noise profiles.
Shows time domain, frequency domain, and reconstruction analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# --- 1. Set ICML Academic Style (Unified Format) ---
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

# Color palette - original color scheme (better appearance)
c_clean_orig = "#34495E"     # GT - darker gray
c_noisy_input = "#E74C3C"    # Noisy Input - red
c_noisy_filt = "#E74C3C"     # Filtered Noisy (Result) - red
c_thresh_sig = "#2ECC71"     # Signal Threshold (Green)
c_thresh_noise = "#F39C12"   # Noise Threshold (Orange)

# --- 2. Data Generation ---
def generate_series(length=300, type='periodic', **kwargs):
    t = np.linspace(0, 10, length)
    if type == 'periodic':
        data = 3.0 * np.sin(2 * np.pi * 0.5 * t) + \
               1.0 * np.sin(2 * np.pi * 2.0 * t)
    elif type == 'trend':
        trend = 0.6 * t
        season = 1.5 * np.sin(2 * np.pi * 1.5 * t)
        data = trend + season
    elif type == 'chirp':
        data = signal.chirp(t, f0=0.5, f1=3, t1=10, method='linear') * 2.5
    elif type == 'am':
        carrier = np.sin(2 * np.pi * 2.0 * t)
        modulator = 1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t) 
        data = carrier * modulator * 3.0
    elif type == 'combined':
        # Combined signal: Periodic + Trend + Chirp + AM
        # Support different versions via kwargs
        version = kwargs.get('version', 1)
        
        if version == 1:
            # Version 1: Emphasize periodic and trend
            periodic = 4.0 * np.sin(2 * np.pi * 0.5 * t) + 1.5 * np.sin(2 * np.pi * 2.0 * t)
            trend = 0.8 * t
            season = 1.2 * np.sin(2 * np.pi * 1.5 * t)
            trend_seasonal = trend + season
            chirp_sig = signal.chirp(t, f0=0.5, f1=3, t1=10, method='linear') * 1.5
            carrier = np.sin(2 * np.pi * 2.0 * t)
            modulator = 1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
            am_sig = carrier * modulator * 2.0
            data = periodic + trend_seasonal + chirp_sig + am_sig
        elif version == 2:
            # Version 2: Emphasize chirp and AM
            periodic = 2.0 * np.sin(2 * np.pi * 0.5 * t) + 0.8 * np.sin(2 * np.pi * 2.0 * t)
            trend = 0.4 * t
            season = 1.0 * np.sin(2 * np.pi * 1.5 * t)
            trend_seasonal = trend + season
            chirp_sig = signal.chirp(t, f0=0.5, f1=3, t1=10, method='linear') * 3.5
            carrier = np.sin(2 * np.pi * 2.0 * t)
            modulator = 1.0 + 0.7 * np.sin(2 * np.pi * 0.2 * t)
            am_sig = carrier * modulator * 4.0
            data = periodic + trend_seasonal + chirp_sig + am_sig
        elif version == 3:
            # Version 3: Balanced with different frequencies
            periodic = 3.5 * np.sin(2 * np.pi * 0.6 * t) + 1.2 * np.sin(2 * np.pi * 2.5 * t)
            trend = 0.5 * t
            season = 1.8 * np.sin(2 * np.pi * 1.8 * t)
            trend_seasonal = trend + season
            chirp_sig = signal.chirp(t, f0=0.4, f1=3.5, t1=10, method='linear') * 2.8
            carrier = np.sin(2 * np.pi * 2.2 * t)
            modulator = 1.0 + 0.6 * np.sin(2 * np.pi * 0.25 * t)
            am_sig = carrier * modulator * 3.5
            data = periodic + trend_seasonal + chirp_sig + am_sig
        else:
            # Default: equal weights
            periodic = 3.0 * np.sin(2 * np.pi * 0.5 * t) + 1.0 * np.sin(2 * np.pi * 2.0 * t)
            trend = 0.6 * t
            season = 1.5 * np.sin(2 * np.pi * 1.5 * t)
            trend_seasonal = trend + season
            chirp_sig = signal.chirp(t, f0=0.5, f1=3, t1=10, method='linear') * 2.5
            carrier = np.sin(2 * np.pi * 2.0 * t)
            modulator = 1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
            am_sig = carrier * modulator * 3.0
            data = periodic + trend_seasonal + chirp_sig + am_sig
        
    data = (data - np.mean(data)) / np.std(data)
    return t, data

def add_noise(data, type='gaussian', level=0.6):
    np.random.seed(42)  # Ensure reproducibility
    noise = np.zeros_like(data)
    
    if type == 'gaussian':
        # 1. Gaussian Noise (Additive)
        noise = np.random.normal(0, level, size=len(data))
        return data + noise, noise
        
    elif type == 'heavy_tail':
        # 2. Heavy-tail/Outlier Noise (Additive)
        # Student-t (df=2.5) for heavy tails
        noise = np.random.standard_t(2.5, size=len(data)) * (level * 0.8)
        return data + noise, noise
        
    elif type == 'missing':
        # 3. Missing Values (Replacement)
        # Mask 15% of the data points to make it visible
        mask = np.random.choice([0, 1], size=len(data), p=[0.15, 0.85]) 
        noisy_data = data * mask
        # The "noise" component here is technically the negative of the missing signal
        noise_component = noisy_data - data 
        return noisy_data, noise_component
    
    return data, np.zeros_like(data)

# --- 2.5. SFM Calculation ---
def compute_sfm(amplitudes):
    """
    Compute Spectral Flatness Measure (SFM).
    SFM = Geometric Mean / Arithmetic Mean of power spectrum
    
    Args:
        amplitudes: FFT amplitude spectrum (1D array)
    
    Returns:
        SFM value (scalar, between 0 and 1)
    """
    power_spec = amplitudes ** 2 + 1e-8  # Add small epsilon to avoid log(0)
    geo_mean = np.exp(np.mean(np.log(power_spec)))
    ari_mean = np.mean(power_spec)
    sfm = geo_mean / (ari_mean + 1e-8)
    return np.clip(sfm, 0.0, 1.0)

# --- 3. Plotting Logic ---
def plot_12_scenarios():
    """
    Generate 4 separate subplots, each containing 3 rows (one for each noise type)
    Each subplot corresponds to a different signal type
    """
    n_samples = 300
    
    # 4 Signal Types
    signal_configs = [
        ('periodic', 'Stationary (Periodic)', 99, 'periodic'),
        ('trend', 'Non-stat Mean (Pure Trend)', 99, 'trend'),
        ('chirp', 'Non-stat Freq (Chirp)', 99, 'chirp'),
        ('am', 'Non-stat Var (AM)', 99, 'am')
    ]
    
    # 3 Noise Types
    noise_configs = [
        ('gaussian', 'Gaussian Noise'),
        ('heavy_tail', 'Heavy-tail Noise'),
        ('missing', 'Missing Values')
    ]
    
    # Generate a separate figure for each signal type
    for s_type, s_title, pct, filename_suffix in signal_configs:
        fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
        
        # Generate clean signal once for this signal type
        t, clean = generate_series(n_samples, s_type)
        
        for row_idx, (n_type, n_title) in enumerate(noise_configs):
            ax_t = axes[row_idx, 0]
            ax_lin = axes[row_idx, 1]
            ax_recon = axes[row_idx, 2]
            
            # Generate Noisy Data
            noisy, noise_component = add_noise(clean, n_type, level=0.5)
            
            # FFT
            fft_clean_val = np.fft.rfft(clean)
            fft_noisy_val = np.fft.rfft(noisy)
            fft_noise_only = np.fft.rfft(noise_component)
            
            amp_clean = np.abs(fft_clean_val)
            amp_noisy = np.abs(fft_noisy_val)
            amp_noise_only = np.abs(fft_noise_only)
            freqs = np.fft.rfftfreq(n_samples, d=t[1]-t[0])

            # Compute SFM values
            sfm_clean = compute_sfm(amp_clean)
            sfm_noise = compute_sfm(amp_noise_only)
            sfm_noisy = compute_sfm(amp_noisy)
            
            # Print SFM values to console
            print(f"[{s_title} + {n_title}] SFM - Clean: {sfm_clean:.4f}, Noise: {sfm_noise:.4f}, Noisy: {sfm_noisy:.4f}")

            # Dynamic Thresholding
            thresh_signal = np.percentile(amp_clean, pct)
            thresh_noise = np.percentile(amp_noise_only, pct)
            
            # --- Reconstruction Logic ---
            mask_clean = amp_clean > thresh_signal
            fft_clean_filt = fft_clean_val * mask_clean
            recon_clean = np.fft.irfft(fft_clean_filt, n=n_samples)
            
            # For reconstruction, we apply the same mask logic to the noisy signal
            mask_noisy = amp_noisy > thresh_noise
            fft_noisy_filt = fft_noisy_val * mask_noisy
            recon_noisy = np.fft.irfft(fft_noisy_filt, n=n_samples)

            # --- Col 1: Time Domain ---
            ax_t.plot(t, clean, color=c_clean_orig, alpha=0.85, lw=2.0, label='Original Clean', zorder=1)
            if n_type == 'missing':
                ax_t.plot(t, noisy, color=c_noisy_input, alpha=0.65, lw=1.8, label='Noisy Input (Zeroed)', zorder=2)
            else:
                ax_t.plot(t, noisy, color=c_noisy_input, alpha=0.55, lw=1.8, label='Noisy Input', zorder=2)
            
            title_text = f"{s_title} + {n_title}"
            ax_t.set_title(title_text, fontweight='bold', fontsize=11)
            ax_t.set_ylabel("Amplitude")
            if row_idx == 0: 
                ax_t.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
            ax_t.grid(True, linestyle='--', alpha=0.25)

            # --- Col 2: Freq Domain (Linear) ---
            ax_lin.plot(freqs, amp_clean, color=c_clean_orig, alpha=0.95, lw=2.0, label='Signal Spectrum', zorder=1)
            ax_lin.plot(freqs, amp_noisy, color=c_noisy_input, alpha=0.55, lw=1.8, label='Noisy Spectrum', zorder=2)
            ax_lin.axhline(thresh_signal, color=c_thresh_sig, linestyle='--', lw=2.0, alpha=0.8, label=f'Signal {pct}% Thresh', zorder=3)
            ax_lin.axhline(thresh_noise, color=c_thresh_noise, linestyle=':', lw=2.0, alpha=0.8, label=f'Noise {pct}% Thresh', zorder=3)
            
            ax_lin.set_title("Freq Domain (Linear)", fontweight='bold', fontsize=11)
            if row_idx == 0: 
                ax_lin.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
            ax_lin.grid(True, linestyle='--', alpha=0.25)

            # --- Col 3: Reconstruction Analysis ---
            ax_recon.plot(t, clean, color=c_clean_orig, alpha=0.4, lw=2.5, label='Original Clean (GT)', zorder=1)
            ax_recon.plot(t, recon_noisy, color=c_noisy_filt, alpha=0.75, lw=2.0, label='Filtered Noisy (Result)', zorder=2)
            
            if s_type == 'chirp': 
                ax_recon.text(0.5, 0.05, f"Top {100-pct}% Energy", transform=ax_recon.transAxes, 
                              ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.9, 
                              edgecolor='gray', linewidth=0.8, boxstyle='round,pad=0.5'))
            
            ax_recon.set_title("Reconstruction Analysis", fontweight='bold', fontsize=11)
            if row_idx == 0: 
                ax_recon.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
            ax_recon.grid(True, linestyle='--', alpha=0.25)

        # Add shared x-labels at the bottom
        for j in range(3):
            xlabel = "Time (s)" if j in [0, 2] else "Frequency (Hz)"
            axes[-1, j].set_xlabel(xlabel)

        # Save each figure separately
        output_filename = f'spectral_analysis_{filename_suffix}.pdf'
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved: {output_filename}")
    
    print("All 4 subplots have been saved successfully!")


def plot_paper_version():
    """
    Paper version: Combined signal (4 signal types) × 3 noise types = 3 rows
    Each row shows: Time Domain | Frequency Domain | Reconstruction
    Each row uses a different version of combined signal with different hyperparameters
    """
    n_samples = 300
    pct = 99
    
    # 3 Noise Types
    noise_configs = [
        ('gaussian', 'Gaussian Noise', 1),
        ('heavy_tail', 'Heavy-tail Noise', 2),
        ('missing', 'Missing Values', 3)
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    
    for row_idx, (n_type, n_title, version) in enumerate(noise_configs):
        # Generate different clean signal for each row with different version
        t, clean = generate_series(n_samples, 'combined', version=version)
        ax_t = axes[row_idx, 0]
        ax_lin = axes[row_idx, 1]
        ax_recon = axes[row_idx, 2]
        
        # Generate Noisy Data
        noisy, noise_component = add_noise(clean, n_type, level=0.5)
        
        # FFT
        fft_clean_val = np.fft.rfft(clean)
        fft_noisy_val = np.fft.rfft(noisy)
        fft_noise_only = np.fft.rfft(noise_component)
        
        amp_clean = np.abs(fft_clean_val)
        amp_noisy = np.abs(fft_noisy_val)
        amp_noise_only = np.abs(fft_noise_only)
        freqs = np.fft.rfftfreq(n_samples, d=t[1]-t[0])

        # Compute SFM values
        sfm_clean = compute_sfm(amp_clean)
        sfm_noise = compute_sfm(amp_noise_only)
        sfm_noisy = compute_sfm(amp_noisy)
        
        print(f"[Combined Signal + {n_title}] SFM - Clean: {sfm_clean:.4f}, Noise: {sfm_noise:.4f}, Noisy: {sfm_noisy:.4f}")

        # Dynamic Thresholding
        thresh_signal = np.percentile(amp_clean, pct)
        thresh_noise = np.percentile(amp_noise_only, pct)
        
        # Reconstruction Logic
        mask_noisy = amp_noisy > thresh_noise
        fft_noisy_filt = fft_noisy_val * mask_noisy
        recon_noisy = np.fft.irfft(fft_noisy_filt, n=n_samples)

        # --- Col 1: Time Domain ---
        ax_t.plot(t, clean, color=c_clean_orig, alpha=0.85, lw=2.0, label='Original Clean', zorder=1)
        if n_type == 'missing':
            ax_t.plot(t, noisy, color=c_noisy_input, alpha=0.65, lw=1.8, label='Noisy Input (Zeroed)', zorder=2)
        else:
            ax_t.plot(t, noisy, color=c_noisy_input, alpha=0.55, lw=1.8, label='Noisy Input', zorder=2)
        
        title_text = f"Combined Clean Signal + {n_title}"
        ax_t.set_title(title_text, fontweight='bold', fontsize=11)
        ax_t.set_ylabel("Amplitude")
        if row_idx == 0: 
            ax_t.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        ax_t.grid(True, linestyle='--', alpha=0.25)

        # --- Col 2: Freq Domain (Linear) ---
        ax_lin.plot(freqs, amp_clean, color=c_clean_orig, alpha=0.95, lw=2.0, label='Signal Spectrum', zorder=1)
        ax_lin.plot(freqs, amp_noisy, color=c_noisy_input, alpha=0.55, lw=1.8, label='Noisy Spectrum', zorder=2)
        ax_lin.axhline(thresh_signal, color=c_thresh_sig, linestyle='--', lw=2.0, alpha=0.8, 
                      label=f'Signal {pct}% Thresh', zorder=3)
        ax_lin.axhline(thresh_noise, color=c_thresh_noise, linestyle=':', lw=2.0, alpha=0.8, 
                      label=f'Noise {pct}% Thresh', zorder=3)
        
        ax_lin.set_title("Frequency Domain", fontweight='bold', fontsize=11)
        if row_idx == 0: 
            ax_lin.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        ax_lin.grid(True, linestyle='--', alpha=0.25)

        # --- Col 3: Reconstruction Analysis ---
        ax_recon.plot(t, clean, color=c_clean_orig, alpha=0.4, lw=2.5, label='Original Clean (GT)', zorder=1)
        ax_recon.plot(t, recon_noisy, color=c_noisy_filt, alpha=0.75, lw=2.0, label='Filtered Noisy (Result)', zorder=2)
        
        ax_recon.set_title("Reconstruction", fontweight='bold', fontsize=11)
        if row_idx == 0: 
            ax_recon.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        ax_recon.grid(True, linestyle='--', alpha=0.25)

    # Add shared x-labels at the bottom
    for j in range(3):
        xlabel = "Time (s)" if j in [0, 2] else "Frequency (Hz)"
        axes[-1, j].set_xlabel(xlabel)

    plt.savefig('spectral_analysis_paper.pdf', bbox_inches='tight', dpi=300)
    plt.show()


# Choose which version to run
plot_paper_version()  # Paper version: 4 rows × 3 cols
plot_12_scenarios()  # Full version: 12 rows × 3 cols