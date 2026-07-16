"""
Regenerate multi-dimensional IRT figures from saved data
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Load saved data
data = np.load('../../result/monkey_analysis/power_beta_bernoulli_multidim_data.npz', allow_pickle=True)

M_values = data['M_values']
d_values = data['d_values']
n_reps = int(data['n_reps'])

print(f"Loaded data with {len(d_values)} dimensions, {len(M_values)} M values, {n_reps} reps")
print(f"Dimensions: {d_values}")
print(f"M values: {M_values}")

# Reconstruct results from saved data
results = {}
for d in d_values:
    results[d] = {
        'rmse_beta_p_mean': data[f'd{d}_rmse_beta_p_mean'],
        'rmse_beta_p_std': data[f'd{d}_rmse_beta_p_std'],
        'rmse_binary_p_mean': data[f'd{d}_rmse_binary_p_mean'],
        'rmse_binary_p_std': data[f'd{d}_rmse_binary_p_std'],
        'corr_beta_p_mean': data[f'd{d}_corr_beta_p_mean'],
        'corr_beta_p_std': data[f'd{d}_corr_beta_p_std'],
        'corr_binary_p_mean': data[f'd{d}_corr_binary_p_mean'],
        'corr_binary_p_std': data[f'd{d}_corr_binary_p_std'],
    }

output_dir = '../../result/monkey_analysis'
M_arr = np.array(M_values)

# Color gradient setup
import matplotlib.colors as mcolors
n_dims = len(d_values)
pink = np.array([233, 30, 99]) / 255
purple = np.array([156, 39, 176]) / 255
blue = np.array([63, 81, 181]) / 255

gradient_colors = []
for i, d in enumerate(d_values):
    t = i / (n_dims - 1) if n_dims > 1 else 0
    if t < 0.5:
        color = pink + (purple - pink) * (t * 2)
    else:
        color = purple + (blue - purple) * ((t - 0.5) * 2)
    gradient_colors.append(color)

colors = {d: gradient_colors[i] for i, d in enumerate(d_values)}
markers = {1: 'p', 2: 'o', 4: 's', 8: '^', 16: 'D', 32: 'v'}

# ============================================================================
# Standard Figure
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Beta RMSE
ax = axes[0, 0]
for d in d_values:
    ax.errorbar(M_arr, results[d]['rmse_beta_p_mean'], yerr=results[d]['rmse_beta_p_std'],
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_ylabel('RMSE of P(correct)')
ax.set_title('Beta IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])
ax.set_ylim(0, 0.05)

# Top-right: Binary RMSE
ax = axes[0, 1]
for d in d_values:
    ax.errorbar(M_arr, results[d]['rmse_binary_p_mean'], yerr=results[d]['rmse_binary_p_std'],
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_title('Binary IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Bottom-left: Beta Correlation
ax = axes[1, 0]
for d in d_values:
    ax.errorbar(M_arr, results[d]['corr_beta_p_mean'], yerr=results[d]['corr_beta_p_std'],
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('Correlation (r)')
ax.legend(loc='lower right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])
ax.set_ylim(0.9, 1.005)

# Bottom-right: Binary Correlation
ax = axes[1, 1]
for d in d_values:
    ax.errorbar(M_arr, results[d]['corr_binary_p_mean'], yerr=results[d]['corr_binary_p_std'],
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xlabel('Number of Test Takers (M)')
ax.legend(loc='lower right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])
ax.set_ylim(0, 1.05)

plt.suptitle('Multi-dimensional IRT: Probability Matrix Recovery', fontsize=14)
plt.tight_layout()
output_path = f"{output_dir}/power_beta_bernoulli_multidim_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# ============================================================================
# Log-Log Scale Figure with SEM error bars
# ============================================================================
# Compute SEM instead of std
def compute_sem(std, n):
    return std / np.sqrt(n)

fig_log, axes_log = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Beta RMSE (log-log)
ax = axes_log[0, 0]
for d in d_values:
    sem = compute_sem(results[d]['rmse_beta_p_std'], n_reps)
    ax.errorbar(M_arr, results[d]['rmse_beta_p_mean'], yerr=sem,
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_ylabel('RMSE of P(correct)')
ax.set_title('Beta IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Top-right: Binary RMSE (log-log)
ax = axes_log[0, 1]
for d in d_values:
    sem = compute_sem(results[d]['rmse_binary_p_std'], n_reps)
    ax.errorbar(M_arr, results[d]['rmse_binary_p_mean'], yerr=sem,
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_title('Binary IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Bottom-left: Beta Correlation (log-x, log for 1-corr)
ax = axes_log[1, 0]
for d in d_values:
    one_minus_corr = 1 - np.array(results[d]['corr_beta_p_mean'])
    sem = compute_sem(results[d]['corr_beta_p_std'], n_reps)
    ax.errorbar(M_arr, one_minus_corr, yerr=sem,
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('1 - Correlation')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Bottom-right: Binary Correlation (log-x, log for 1-corr)
ax = axes_log[1, 1]
for d in d_values:
    one_minus_corr = 1 - np.array(results[d]['corr_binary_p_mean'])
    sem = compute_sem(results[d]['corr_binary_p_std'], n_reps)
    ax.errorbar(M_arr, one_minus_corr, yerr=sem,
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Number of Test Takers (M)')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

plt.suptitle('Multi-dimensional IRT: Probability Matrix Recovery (Log-Log Scale, SEM)', fontsize=14)
plt.tight_layout()
output_path_log = f"{output_dir}/power_beta_bernoulli_multidim_comparison_loglog_sem.png"
plt.savefig(output_path_log, dpi=300, bbox_inches='tight')
print(f"Log-log SEM figure saved to: {output_path_log}")

# ============================================================================
# Clean Log-Log Scale Figure (no error bars)
# ============================================================================
fig_clean, axes_clean = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Beta RMSE (log-log)
ax = axes_clean[0, 0]
for d in d_values:
    ax.plot(M_arr, results[d]['rmse_beta_p_mean'],
            f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6,
            label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_ylabel('RMSE of P(correct)')
ax.set_title('Beta IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Top-right: Binary RMSE (log-log)
ax = axes_clean[0, 1]
for d in d_values:
    ax.plot(M_arr, results[d]['rmse_binary_p_mean'],
            f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6,
            label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_title('Binary IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Bottom-left: Beta Correlation (log-x, log for 1-corr)
ax = axes_clean[1, 0]
for d in d_values:
    one_minus_corr = 1 - np.array(results[d]['corr_beta_p_mean'])
    ax.plot(M_arr, one_minus_corr,
            f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6,
            label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('1 - Correlation')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Bottom-right: Binary Correlation (log-x, log for 1-corr)
ax = axes_clean[1, 1]
for d in d_values:
    one_minus_corr = 1 - np.array(results[d]['corr_binary_p_mean'])
    ax.plot(M_arr, one_minus_corr,
            f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6,
            label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Number of Test Takers (M)')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

plt.suptitle('Multi-dimensional IRT: Probability Matrix Recovery (Log-Log Scale)', fontsize=14)
plt.tight_layout()
output_path_clean = f"{output_dir}/power_beta_bernoulli_multidim_comparison_loglog_clean.png"
plt.savefig(output_path_clean, dpi=300, bbox_inches='tight')
print(f"Log-log clean figure saved to: {output_path_clean}")

print("\nAll figures regenerated!")
