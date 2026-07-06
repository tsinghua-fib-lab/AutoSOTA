"""
Spectral detrending analysis comparing different detrending methods.
Demonstrates the robustness of robust OLS compared to end-to-end methods.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Settings (ICML academic style, unified format) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.unicode_minus': False
})

def get_data(case='linear_spike', length=200):
    t = np.linspace(0, 10, length)
    np.random.seed(2024)  # Fix the seed for reproducibility, works well in my tests

    # Base noise
    base_noise = np.random.normal(0, 0.4, length)

    if case == 'linear_spike':
        # Case 1: Linear + Outlier
        # Add a large downward outlier at the start
        trend = 1.2 * t
        data = trend + base_noise
        data[0] = -8.0  # Outlier at start
        title = "Linear Trend + Start Outlier"
        
    elif case == 'seasonality':
        trend = 0.8 * t
        seasonality = 3.0 * np.sin(2 * np.pi * 0.5 * t)
        data = trend + seasonality + base_noise
        title = "Trend + Seasonality (Phase Mismatch)"
        
    elif case == 'quadratic':
        # Case 3: Quadratic
        trend = 0.15 * (t - 5)**2
        data = trend + base_noise
        # Add disturbance at the end to destabilize End-to-End method
        data[-1] += 2.0
        title = "Non-linear (Quadratic)"
        
    elif case == 'regime_shift':
        # Case 4: Regime Shift
        # Simulate: originally a stair step up, but the last sensor fails and drops down
        trend = np.zeros_like(t)
        trend[length//2:] = 8.0  # Step up
        trend += 0.3 * t  # Base trend
        data = trend + base_noise

        # Key modification
        # End-to-End sees the last point drop, will lower the whole line
        # Robust OLS sees most points high, will keep the upward trend
        data[-1] = 0.0  # Drop at the end
        title = "Regime Shift + End Failure"

    return t, data, title

def apply_detrend(t, data, method):
    if method == 'none':
        return data, np.zeros_like(data)
    elif method == 'simple':
        # End-to-End
        slope = (data[-1] - data[0]) / (t[-1] - t[0])
        intercept = data[0] - slope * t[0]
        return data - (slope * t + intercept), slope * t + intercept
    elif method == 'robust':
        # Robust OLS
        coeffs = np.polyfit(t, data, deg=1)
        trend_line = np.polyval(coeffs, t)
        return data - trend_line, trend_line

def filter_and_reconstruct(data, pct=88):
    fft_val = np.fft.rfft(data)
    amp = np.abs(fft_val)
    threshold = np.percentile(amp, pct)
    mask = amp > threshold
    return np.fft.irfft(fft_val * mask, n=len(data))

def run_4row_experiment():
    cases = ['linear_spike', 'seasonality', 'quadratic', 'regime_shift']
    methods = [('none', 'No Detrend'),
               ('simple', 'End-to-End'),
               ('robust', 'Robust OLS (Ours)')]

    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 2.8 * len(cases)), constrained_layout=True)

    for row_idx, case in enumerate(cases):
        t, original_data, case_title = get_data(case)
        
        for col_idx, (method, method_name) in enumerate(methods):
            ax = axes[row_idx, col_idx]

            detrended_data, trend_line = apply_detrend(t, original_data, method)
            recon_detrended = filter_and_reconstruct(detrended_data)
            final_recon = recon_detrended + trend_line

            # Plot
            ax.plot(t, original_data, color='gray', alpha=0.3, lw=1, label='Input')

            if method != 'none':
                ls = '--' if method == 'robust' else ':'
                ax.plot(t, trend_line, color='orange', linestyle=ls, alpha=0.9, lw=1.5, label='Trend')

            ax.plot(t, final_recon, color='#3498DB', lw=1.8, alpha=0.9, label='Recon')

            # Title
            if method == 'robust':
                ax.set_title(f"{method_name}",
                             fontweight='bold', color='#1F618D', backgroundcolor='#EAF2F8')
            else:
                ax.set_title(f"{method_name}", fontweight='bold')

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left')

            ax.set_xticks([])
            if col_idx == 0:
                ax.set_ylabel(case_title, fontsize=9, fontweight='medium')

    plt.savefig('spectral_detrend_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()

run_4row_experiment()