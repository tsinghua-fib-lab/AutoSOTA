"""
Multi-dimensional IRT Model Comparison: Binary vs Beta

Model: p(y_ij = 1 | a_j, z_j, θ_i) = σ(a_j · θ_i + z_j)

Where:
- θ_i ∈ R^d is the d-dimensional ability vector for test-taker i
- a_j ∈ R^d is the d-dimensional discrimination vector for item j
- z_j ∈ R is the scalar difficulty for item j
- a_j · θ_i is the dot product

This generalizes the 2PL model to multiple latent dimensions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from itertools import product

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def prob_multidim_np(theta, a, z):
    """
    Multi-dimensional IRT probability (numpy version for simulation)

    Args:
        theta: (M, d) ability vectors
        a: (N, d) discrimination vectors
        z: (N,) difficulty scalars

    Returns:
        (M, N) probability matrix
    """
    # theta @ a.T gives (M, N) matrix of dot products
    logits = theta @ a.T + z[None, :]  # (M, N)
    return 1.0 / (1.0 + np.exp(-logits))


def simulate_binary_multidim(theta, a, z, rng):
    """Simulate binary responses from multi-dim IRT model"""
    p = prob_multidim_np(theta, a, z)
    y = rng.binomial(1, p)
    return y, p


def simulate_prob_matrix_multidim(theta, a, z, noise_sd, rng):
    """Simulate noisy probability matrix from multi-dim IRT model"""
    p = prob_multidim_np(theta, a, z)
    noisy = p + rng.normal(0.0, noise_sd, size=p.shape)
    noisy = np.clip(noisy, 1e-6, 1 - 1e-6)
    return noisy, p


def fit_multidim_bernoulli_torch_single(Y_t, M, N, d, maxiter=1000, lr=1.0, seed=None):
    """Single optimization run for Binary multi-dim IRT"""
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize parameters (must be leaf tensors for optimizer)
    theta = torch.randn(M, d, device=device) * 0.1
    theta = theta.clone().detach().requires_grad_(True)
    a = torch.randn(N, d, device=device) * 0.1
    a = a.clone().detach().requires_grad_(True)
    z = torch.zeros(N, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([theta, a, z], max_iter=maxiter, lr=lr,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        z_centered = z - z.mean()
        logits = theta @ a.T + z_centered[None, :]
        p = torch.sigmoid(logits)
        eps = 1e-12
        ll = torch.sum(Y_t * torch.log(p + eps) + (1 - Y_t) * torch.log(1 - p + eps))
        # Adaptive regularization: stronger for higher dimensions
        reg_strength = 0.01 * (1 + d / 8)  # Increases with dimension
        reg_theta = reg_strength * torch.sum(theta ** 2)
        reg_a = reg_strength * torch.sum(a ** 2)
        loss = -ll + reg_theta + reg_a
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)

    with torch.no_grad():
        z_centered = z - z.mean()
        # Compute final loss for comparison
        logits = theta @ a.T + z_centered[None, :]
        p = torch.sigmoid(logits)
        eps = 1e-12
        ll = torch.sum(Y_t * torch.log(p + eps) + (1 - Y_t) * torch.log(1 - p + eps))
        final_loss = -ll.item()

    return (theta.detach().cpu().numpy(),
            a.detach().cpu().numpy(),
            z_centered.detach().cpu().numpy(),
            final_loss)


def fit_multidim_bernoulli_torch(Y, d, maxiter=1000, lr=1.0, n_restarts=5):
    """
    Fit multi-dimensional IRT model to binary responses using PyTorch LBFGS on GPU
    with multiple random restarts.

    Args:
        Y: (M, N) binary response matrix
        d: latent dimension
        maxiter: max iterations for LBFGS
        lr: learning rate
        n_restarts: number of random restarts

    Returns:
        theta: (M, d) estimated ability vectors
        a: (N, d) estimated discrimination vectors
        z: (N,) estimated difficulty scalars
    """
    M, N = Y.shape
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    best_loss = float('inf')
    best_result = None

    for restart in range(n_restarts):
        theta, a, z, loss = fit_multidim_bernoulli_torch_single(
            Y_t, M, N, d, maxiter, lr, seed=restart * 1000 + 42
        )
        if loss < best_loss:
            best_loss = loss
            best_result = (theta, a, z)

    return best_result


def fit_multidim_beta_torch_single(Pobs_t, M, N, d, z_init, phi=400.0, maxiter=1000, lr=1.0, seed=None):
    """Single optimization run for Beta multi-dim IRT"""
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize parameters (must be leaf tensors for optimizer)
    theta = torch.randn(M, d, device=device) * 0.1
    theta = theta.clone().detach().requires_grad_(True)
    a = torch.randn(N, d, device=device) * 0.1
    a = a.clone().detach().requires_grad_(True)
    z = torch.tensor(z_init, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([theta, a, z], max_iter=maxiter, lr=lr,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()

        # Center z for identifiability
        z_centered = z - z.mean()

        # Compute logits: (M, N)
        logits = theta @ a.T + z_centered[None, :]
        p = torch.sigmoid(logits)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        alpha = phi * p
        beta_param = phi * (1 - p)

        # Beta log-likelihood
        ll = torch.sum(
            (alpha - 1) * torch.log(Pobs_t) +
            (beta_param - 1) * torch.log(1 - Pobs_t) -
            torch.lgamma(alpha) - torch.lgamma(beta_param) + torch.lgamma(alpha + beta_param)
        )

        # Adaptive regularization: stronger for higher dimensions
        reg_strength = 0.01 * (1 + d / 8)
        reg_theta = reg_strength * torch.sum(theta ** 2)
        reg_a = reg_strength * torch.sum(a ** 2)

        loss = -ll + reg_theta + reg_a
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        z_centered = z - z.mean()
        # Compute final loss for comparison
        logits = theta @ a.T + z_centered[None, :]
        p = torch.sigmoid(logits)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        alpha = phi * p
        beta_param = phi * (1 - p)
        ll = torch.sum(
            (alpha - 1) * torch.log(Pobs_t) +
            (beta_param - 1) * torch.log(1 - Pobs_t) -
            torch.lgamma(alpha) - torch.lgamma(beta_param) + torch.lgamma(alpha + beta_param)
        )
        final_loss = -ll.item()

    return (theta.detach().cpu().numpy(),
            a.detach().cpu().numpy(),
            z_centered.detach().cpu().numpy(),
            final_loss)


def fit_multidim_beta_torch(Pobs, d, phi=400.0, maxiter=1000, lr=1.0, n_restarts=5):
    """
    Fit multi-dimensional IRT model to probability matrix using Beta likelihood
    with multiple random restarts.

    Args:
        Pobs: (M, N) observed probability matrix
        d: latent dimension
        phi: Beta precision parameter
        maxiter: max iterations for LBFGS
        lr: learning rate
        n_restarts: number of random restarts

    Returns:
        theta: (M, d) estimated ability vectors
        a: (N, d) estimated discrimination vectors
        z: (N,) estimated difficulty scalars
    """
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)

    # Better initialization from observed probabilities
    logit_P = np.log(Pobs / (1 - Pobs))
    z_init = -logit_P.mean(axis=0)
    z_init = z_init - z_init.mean()

    best_loss = float('inf')
    best_result = None

    for restart in range(n_restarts):
        theta, a, z, loss = fit_multidim_beta_torch_single(
            Pobs_t, M, N, d, z_init, phi, maxiter, lr, seed=restart * 1000 + 123
        )
        if loss < best_loss:
            best_loss = loss
            best_result = (theta, a, z)

    return best_result


def recovery_z(true_z, est_z):
    """Compute RMSE and correlation for difficulty parameter recovery"""
    true_centered = true_z - true_z.mean()
    est_centered = est_z - est_z.mean()
    rmse = float(np.sqrt(np.mean((est_centered - true_centered) ** 2)))
    corr = float(np.corrcoef(true_centered, est_centered)[0, 1])
    return rmse, corr


def recovery_vectors(true_vecs, est_vecs):
    """
    Compute recovery metrics for multi-dimensional vectors (a or theta).
    Since there's rotational ambiguity, we compare the Gram matrices.

    Returns RMSE of predicted probabilities as a proxy.
    """
    # Compare via correlation of flattened vectors (after centering each)
    true_flat = true_vecs.flatten()
    est_flat = est_vecs.flatten()

    # This is a rough metric - true recovery would need Procrustes alignment
    corr = float(np.corrcoef(true_flat, est_flat)[0, 1]) if len(true_flat) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((true_flat - est_flat) ** 2)))
    return rmse, corr


def recovery_prob_matrix(true_theta, true_a, true_z, est_theta, est_a, est_z):
    """
    Compute RMSE and correlation of predicted probability matrices.
    This is the best metric since it's invariant to rotations.
    """
    p_true = prob_multidim_np(true_theta, true_a, true_z)
    p_est = prob_multidim_np(est_theta, est_a, est_z)

    rmse = float(np.sqrt(np.mean((p_true - p_est) ** 2)))
    corr = float(np.corrcoef(p_true.flatten(), p_est.flatten())[0, 1])
    return rmse, corr


# --- Simulation parameters ---
N = 100  # number of items
n_reps = 100  # repetitions per condition (increased for smoother curves)
noise_sd = 0.01

# Range of test taker counts: 2^n for n=1 to 7
M_values = [2, 4, 8, 16, 32, 64, 128]

# Latent dimensions to test (including d=1 for Rasch)
d_values = [1, 2, 4, 8, 16, 32]

# For high dimensions, skip M=2 (underdetermined problem)
def get_valid_M_values(d):
    """Get valid M values for a given dimension - skip small M for high d"""
    if d >= 8:
        return [m for m in M_values if m >= 4]  # Skip M=2 for d >= 8
    return M_values

# Main RNG for generating true parameters
rng_main = np.random.default_rng(42)

print(f"Running multi-dimensional IRT simulations...", flush=True)
print(f"Dimensions: {d_values}", flush=True)
print(f"M values: {M_values}", flush=True)
print(f"Repetitions: {n_reps}", flush=True)

# Count total simulations
total_sims = sum(len(get_valid_M_values(d)) * n_reps for d in d_values)
print(f"Total simulations: {total_sims}", flush=True)

start_time = time.time()

# Store results: dict[d] -> arrays of shape (n_reps, len(M_values))
# Note: For high d, some M values are skipped (set to NaN)
results = {}

for d in d_values:
    valid_M = get_valid_M_values(d)
    print(f"\n=== Dimension d={d} (M values: {valid_M}) ===", flush=True)

    rmse_binary_z_all = np.full((n_reps, len(M_values)), np.nan)
    rmse_beta_z_all = np.full((n_reps, len(M_values)), np.nan)
    corr_binary_z_all = np.full((n_reps, len(M_values)), np.nan)
    corr_beta_z_all = np.full((n_reps, len(M_values)), np.nan)

    # Probability matrix recovery (invariant to rotation)
    rmse_binary_p_all = np.full((n_reps, len(M_values)), np.nan)
    rmse_beta_p_all = np.full((n_reps, len(M_values)), np.nan)
    corr_binary_p_all = np.full((n_reps, len(M_values)), np.nan)
    corr_beta_p_all = np.full((n_reps, len(M_values)), np.nan)

    for rep in range(n_reps):
        # Generate true parameters for this repetition
        a_true = rng_main.normal(0, 1, size=(N, d))
        z_true = rng_main.normal(0, 1, size=N)

        for i, M in enumerate(M_values):
            # Skip invalid M values for this dimension
            if M not in valid_M:
                continue

            # Create separate RNG for this simulation
            rng = np.random.default_rng(42 + rep * 1000 + M + d * 10000)
            theta_true = rng.normal(0, 1, size=(M, d))

            # Binary multi-dim IRT
            Y, _ = simulate_binary_multidim(theta_true, a_true, z_true, rng)
            (theta_hat_bin, a_hat_bin, z_hat_bin) = fit_multidim_bernoulli_torch(Y, d)

            rmse_z, corr_z = recovery_z(z_true, z_hat_bin)
            rmse_binary_z_all[rep, i] = rmse_z
            corr_binary_z_all[rep, i] = corr_z

            rmse_p, corr_p = recovery_prob_matrix(theta_true, a_true, z_true,
                                                   theta_hat_bin, a_hat_bin, z_hat_bin)
            rmse_binary_p_all[rep, i] = rmse_p
            corr_binary_p_all[rep, i] = corr_p

            # Beta multi-dim IRT
            P_obs, _ = simulate_prob_matrix_multidim(theta_true, a_true, z_true, noise_sd, rng)
            (theta_hat_beta, a_hat_beta, z_hat_beta) = fit_multidim_beta_torch(P_obs, d, phi=400.0)

            rmse_z, corr_z = recovery_z(z_true, z_hat_beta)
            rmse_beta_z_all[rep, i] = rmse_z
            corr_beta_z_all[rep, i] = corr_z

            rmse_p, corr_p = recovery_prob_matrix(theta_true, a_true, z_true,
                                                   theta_hat_beta, a_hat_beta, z_hat_beta)
            rmse_beta_p_all[rep, i] = rmse_p
            corr_beta_p_all[rep, i] = corr_p

        if (rep + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  d={d}: Completed {rep + 1}/{n_reps} repetitions ({elapsed:.1f}s)", flush=True)

    # Store results for this dimension
    results[d] = {
        'rmse_binary_z_all': rmse_binary_z_all,
        'rmse_beta_z_all': rmse_beta_z_all,
        'corr_binary_z_all': corr_binary_z_all,
        'corr_beta_z_all': corr_beta_z_all,
        'rmse_binary_p_all': rmse_binary_p_all,
        'rmse_beta_p_all': rmse_beta_p_all,
        'corr_binary_p_all': corr_binary_p_all,
        'corr_beta_p_all': corr_beta_p_all,
        'rmse_binary_z_mean': rmse_binary_z_all.mean(axis=0),
        'rmse_binary_z_std': rmse_binary_z_all.std(axis=0),
        'rmse_beta_z_mean': rmse_beta_z_all.mean(axis=0),
        'rmse_beta_z_std': rmse_beta_z_all.std(axis=0),
        'corr_binary_z_mean': corr_binary_z_all.mean(axis=0),
        'corr_binary_z_std': corr_binary_z_all.std(axis=0),
        'corr_beta_z_mean': corr_beta_z_all.mean(axis=0),
        'corr_beta_z_std': corr_beta_z_all.std(axis=0),
        'rmse_binary_p_mean': rmse_binary_p_all.mean(axis=0),
        'rmse_binary_p_std': rmse_binary_p_all.std(axis=0),
        'rmse_beta_p_mean': rmse_beta_p_all.mean(axis=0),
        'rmse_beta_p_std': rmse_beta_p_all.std(axis=0),
        'corr_binary_p_mean': corr_binary_p_all.mean(axis=0),
        'corr_binary_p_std': corr_binary_p_all.std(axis=0),
        'corr_beta_p_mean': corr_beta_p_all.mean(axis=0),
        'corr_beta_p_std': corr_beta_p_all.std(axis=0),
    }

total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.1f}s")

# Print summary
print("\n" + "=" * 100)
print("RESULTS SUMMARY - Difficulty (z) Recovery")
print("=" * 100)
for d in d_values:
    print(f"\nDimension d={d}:")
    for i, M in enumerate(M_values):
        print(f"  M={M:4d}: Binary RMSE={results[d]['rmse_binary_z_mean'][i]:.4f}±{results[d]['rmse_binary_z_std'][i]:.4f}  |  "
              f"Beta RMSE={results[d]['rmse_beta_z_mean'][i]:.4f}±{results[d]['rmse_beta_z_std'][i]:.4f}")

print("\n" + "=" * 100)
print("RESULTS SUMMARY - Probability Matrix Recovery (rotation-invariant)")
print("=" * 100)
for d in d_values:
    print(f"\nDimension d={d}:")
    for i, M in enumerate(M_values):
        print(f"  M={M:4d}: Binary RMSE={results[d]['rmse_binary_p_mean'][i]:.4f}±{results[d]['rmse_binary_p_std'][i]:.4f}  |  "
              f"Beta RMSE={results[d]['rmse_beta_p_mean'][i]:.4f}±{results[d]['rmse_beta_p_std'][i]:.4f}")

# Save results
output_dir = "../../result/monkey_analysis"
os.makedirs(output_dir, exist_ok=True)
data_path = f"{output_dir}/power_beta_bernoulli_multidim_data.npz"

save_dict = {
    'M_values': np.array(M_values),
    'd_values': np.array(d_values),
    'n_reps': n_reps,
    'N': N,
}
for d in d_values:
    for key, val in results[d].items():
        save_dict[f'd{d}_{key}'] = val

np.savez(data_path, **save_dict)
print(f"\nData saved to: {data_path}")

# Plotting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# Create figure: 2 rows (RMSE, Corr) x 2 columns (Beta, Binary)
# Each panel shows different dimensions as separate lines
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
M_arr = np.array(M_values)

# Gradient color palette from pink to blue (based on dimension)
# Pink (#E91E63) -> Purple (#9C27B0) -> Blue (#3F51B5)
import matplotlib.colors as mcolors
n_dims = len(d_values)
# Create gradient from pink to purple to blue
pink = np.array([233, 30, 99]) / 255  # #E91E63
purple = np.array([156, 39, 176]) / 255  # #9C27B0
blue = np.array([63, 81, 181]) / 255  # #3F51B5

gradient_colors = []
for i, d in enumerate(d_values):
    t = i / (n_dims - 1) if n_dims > 1 else 0
    if t < 0.5:
        # Pink to purple
        color = pink + (purple - pink) * (t * 2)
    else:
        # Purple to blue
        color = purple + (blue - purple) * ((t - 0.5) * 2)
    gradient_colors.append(color)

colors = {d: gradient_colors[i] for i, d in enumerate(d_values)}
markers = {1: 'p', 2: 'o', 4: 's', 8: '^', 16: 'D', 32: 'v'}  # p = pentagon for d=1

# Top-left: Beta RMSE (zoomed to 0-0.05)
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

# Bottom-left: Beta Correlation (zoomed to 0.9-1.0)
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
print(f"\nFigure saved to: {output_path}")

# ============================================================================
# Log-Log Scale Figure
# ============================================================================
fig_log, axes_log = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Beta RMSE (log-log)
ax = axes_log[0, 0]
for d in d_values:
    ax.errorbar(M_arr, results[d]['rmse_beta_p_mean'], yerr=results[d]['rmse_beta_p_std'],
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
    ax.errorbar(M_arr, results[d]['rmse_binary_p_mean'], yerr=results[d]['rmse_binary_p_std'],
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_title('Binary IRT')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Bottom-left: Beta Correlation (log-x, linear-y for 1-corr)
ax = axes_log[1, 0]
for d in d_values:
    # Plot 1 - correlation to show improvement on log scale
    one_minus_corr = 1 - np.array(results[d]['corr_beta_p_mean'])
    ax.errorbar(M_arr, one_minus_corr, yerr=results[d]['corr_beta_p_std'],
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
    ax.errorbar(M_arr, one_minus_corr, yerr=results[d]['corr_binary_p_std'],
                fmt=f'{markers[d]}-', color=colors[d], linewidth=2, markersize=6, capsize=3, capthick=1.2,
                label=f'd={d}')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Number of Test Takers (M)')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

plt.suptitle('Multi-dimensional IRT: Probability Matrix Recovery (Log-Log Scale)', fontsize=14)
plt.tight_layout()
output_path_log = f"{output_dir}/power_beta_bernoulli_multidim_comparison_loglog.png"
plt.savefig(output_path_log, dpi=300, bbox_inches='tight')
print(f"Log-log figure saved to: {output_path_log}")
