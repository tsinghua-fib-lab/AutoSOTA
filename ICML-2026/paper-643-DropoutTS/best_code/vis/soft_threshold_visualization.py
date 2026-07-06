"""
Visualization of soft threshold mask function.
Demonstrates the effect of threshold location (tau) and sharpness parameter (alpha).
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration (ICML academic style, unified format) ---
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
})
# ICML style color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- Core Functions ---
def sigmoid(x):
    # Numerically stable implementation to prevent overflow
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def softplus(alpha):
    # Softplus ensures the sharpness coefficient is always positive
    # log(1 + exp(alpha))
    return np.log1p(np.exp(-np.abs(alpha))) + np.maximum(alpha, 0)

def soft_threshold_mask(amplitude, threshold_tau, sharpness_alpha):
    """
    Compute soft threshold mask
    amplitude (Â): the input normalized amplitude
    threshold_tau (τ): threshold center
    sharpness_alpha (α): sharpness control parameter (raw, before softplus)
    """
    s = softplus(sharpness_alpha) # Calculate the actual sharpness coefficient
    logits = s * (amplitude - threshold_tau)
    mask = sigmoid(logits)
    return mask

# --- Data Preparation ---
# Simulate normalized amplitude input, range slightly wider than [0, 1] to show boundary effects
x_amplitudes = np.linspace(-0.1, 1.1, 500)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

# ==========================================
# Subplot 1: Effect of threshold location (τ)
# ==========================================
ax1 = axes[0]
fixed_alpha = 4.0  # Use a moderate fixed sharpness
taus_to_test = [0.2, 0.5, 0.8]

for i, tau in enumerate(taus_to_test):
    mask_values = soft_threshold_mask(x_amplitudes, tau, fixed_alpha)
    
    # Plot curve
    ax1.plot(x_amplitudes, mask_values, lw=2.0, color=colors[i], 
             label=f'Threshold $\\tau = {tau}$')
    
    # Draw auxiliary lines at center (where mask=0.5)
    ax1.vlines(tau, 0, 0.5, linestyles=':', color=colors[i], alpha=0.6)
    ax1.hlines(0.5, 0, tau, linestyles=':', color=colors[i], alpha=0.6)

ax1.set_title('(a) Effect of Threshold Location ($\\tau$)', fontweight='bold', loc='left')
ax1.set_xlabel('Normalized Input Amplitude ($\hat{\mathbf{A}}$)', fontweight='bold')
ax1.set_ylabel('Output Mask Value ($\mathbf{M}$)', fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(frameon=True, fancybox=True)
ax1.text(0.02, 1.02, 'Pass Region', transform=ax1.transAxes, color='green', fontweight='bold', fontsize=9)
ax1.text(0.02, -0.08, 'Block Region', transform=ax1.transAxes, color='red', fontweight='bold', fontsize=9)

# ==========================================
# Subplot 2: Effect of sharpness (α)
# ==========================================
ax2 = axes[1]
fixed_tau = 0.5  # Center the threshold
alphas_to_test = [-2.0, 2.0, 10.0] # From very soft to very hard
alpha_labels = ['Very Soft ($\\alpha=-2$)', 'Medium ($\\alpha=2$)', 'Hard-like ($\\alpha=10$)']

# 1. Plot ideal hard threshold (non-differentiable reference)
hard_mask = np.where(x_amplitudes >= fixed_tau, 1.0, 0.0)
ax2.step(x_amplitudes, hard_mask, where='mid', lw=2.5, color='black', linestyle='--',
         label='Ideal Hard Step (Non-diff.)', zorder=1)

# 2. Plot soft thresholds with different α values
for i, alpha in enumerate(alphas_to_test):
    mask_values = soft_threshold_mask(x_amplitudes, fixed_tau, alpha)
    # Compute actual slope value for display
    actual_slope = softplus(alpha)
    label_text = f'{alpha_labels[i]} (slope $\\approx {actual_slope:.1f}$)'
    
    ax2.plot(x_amplitudes, mask_values, lw=2.0, color=colors[i+2], label=label_text, zorder=2)

ax2.set_title('(b) Effect of Sharpness Parameter ($\\alpha$)', fontweight='bold', loc='left')
ax2.set_xlabel('Normalized Input Amplitude ($\hat{\mathbf{A}}$)', fontweight='bold')
ax2.set_ylabel('Output Mask Value ($\mathbf{M}$)', fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(frameon=True, fancybox=True, loc='center right')

# Mark center point
ax2.scatter([fixed_tau], [0.5], color='black', s=50, zorder=3)
ax2.text(fixed_tau+0.02, 0.45, 'Center (0.5, 0.5)', fontsize=8)

# Save and show
plt.savefig('soft_threshold_visualization.pdf', dpi=300, bbox_inches='tight')
plt.show()
