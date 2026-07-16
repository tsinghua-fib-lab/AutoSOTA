import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def rasch_prob_np(theta, b):
    """Rasch probability (numpy version for simulation)"""
    return 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))

def fit_rasch_bernoulli_torch(Y, maxiter=500, lr=1.0):
    """Fit Rasch model to binary responses using PyTorch LBFGS on GPU"""
    M, N = Y.shape
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    # Initialize parameters
    theta = torch.zeros(M, device=device, requires_grad=True)
    b = torch.zeros(N, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([theta, b], max_iter=maxiter, lr=lr,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # Center b for identifiability
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()

        p = torch.sigmoid(theta_centered[:, None] - b_centered[None, :])
        eps = 1e-12
        ll = torch.sum(Y_t * torch.log(p + eps) + (1 - Y_t) * torch.log(1 - p + eps))
        loss = -ll
        loss.backward()
        return loss

    optimizer.step(closure)

    # Extract centered parameters
    with torch.no_grad():
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()

    return theta_centered.cpu().numpy(), b_centered.cpu().numpy()

def fit_rasch_beta_torch(Pobs, phi=400.0, maxiter=500, lr=1.0):
    """Fit Rasch model to probability matrix using Beta likelihood with PyTorch LBFGS on GPU"""
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)

    # Initialize parameters
    theta = torch.zeros(M, device=device, requires_grad=True)
    b = torch.zeros(N, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([theta, b], max_iter=maxiter, lr=lr,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # Center b for identifiability
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()

        p = torch.sigmoid(theta_centered[:, None] - b_centered[None, :])
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        alpha = phi * p
        beta_param = phi * (1 - p)

        # Beta log-likelihood
        ll = torch.sum(
            (alpha - 1) * torch.log(Pobs_t) +
            (beta_param - 1) * torch.log(1 - Pobs_t) -
            torch.lgamma(alpha) - torch.lgamma(beta_param) + torch.lgamma(alpha + beta_param)
        )
        loss = -ll
        loss.backward()
        return loss

    optimizer.step(closure)

    # Extract centered parameters
    with torch.no_grad():
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()

    return theta_centered.cpu().numpy(), b_centered.cpu().numpy()

def recovery(true_b, est_b):
    """Compute RMSE and correlation for parameter recovery"""
    true_b = true_b - true_b.mean()
    est_b = est_b - est_b.mean()
    rmse = float(np.sqrt(np.mean((est_b - true_b) ** 2)))
    corr = float(np.corrcoef(true_b, est_b)[0, 1])
    return rmse, corr

def run_single_simulation(args):
    """Run a single simulation for given M and repetition"""
    M, rep, b_true, N, noise_sd = args

    # Create separate RNG for this simulation
    rng = np.random.default_rng(42 + rep * 1000 + M)
    theta = rng.normal(0, 1, size=M)

    # Simulate data
    p_true = rasch_prob_np(theta, b_true)
    Y = rng.binomial(1, p_true)
    P_obs = p_true + rng.normal(0, noise_sd, size=p_true.shape)
    P_obs = np.clip(P_obs, 1e-6, 1 - 1e-6)

    # Fit models
    _, b_hat_bin = fit_rasch_bernoulli_torch(Y)
    _, b_hat_beta = fit_rasch_beta_torch(P_obs)

    # Compute recovery metrics
    rmse_bin, corr_bin = recovery(b_true, b_hat_bin)
    rmse_beta, corr_beta = recovery(b_true, b_hat_beta)

    return {
        'M': M, 'rep': rep,
        'rmse_binary': rmse_bin, 'corr_binary': corr_bin,
        'rmse_beta': rmse_beta, 'corr_beta': corr_beta
    }

# --- Simulation parameters ---
N = 100
n_reps = 10
noise_sd = 0.01

# Range of test taker counts: 2^n for n=1 to 7
M_values = [2, 4, 8, 16, 32, 64, 128]

# Generate true parameters (shared across all simulations within a rep)
# We'll generate per-rep b_true inside the loop
rng_main = np.random.default_rng(7)

print(f"Running {n_reps} repetitions for each of {len(M_values)} M values...")
print(f"Total simulations: {n_reps * len(M_values)}")

start_time = time.time()

# Store results
rmse_binary_all = np.zeros((n_reps, len(M_values)))
rmse_beta_all = np.zeros((n_reps, len(M_values)))
corr_binary_all = np.zeros((n_reps, len(M_values)))
corr_beta_all = np.zeros((n_reps, len(M_values)))

# Run simulations - GPU jobs run sequentially but are much faster
for rep in range(n_reps):
    # Generate b_true for this repetition
    b_true = rng_main.normal(0, 1, size=N)

    for i, M in enumerate(M_values):
        result = run_single_simulation((M, rep, b_true, N, noise_sd))
        rmse_binary_all[rep, i] = result['rmse_binary']
        rmse_beta_all[rep, i] = result['rmse_beta']
        corr_binary_all[rep, i] = result['corr_binary']
        corr_beta_all[rep, i] = result['corr_beta']

    if (rep + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Completed {rep + 1}/{n_reps} repetitions ({elapsed:.1f}s)")

total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.1f}s")

# Compute mean and std
rmse_binary_mean = rmse_binary_all.mean(axis=0)
rmse_binary_std = rmse_binary_all.std(axis=0)
rmse_beta_mean = rmse_beta_all.mean(axis=0)
rmse_beta_std = rmse_beta_all.std(axis=0)

corr_binary_mean = corr_binary_all.mean(axis=0)
corr_binary_std = corr_binary_all.std(axis=0)
corr_beta_mean = corr_beta_all.mean(axis=0)
corr_beta_std = corr_beta_all.std(axis=0)

# Print summary
print(f"\nResults (mean ± std over {n_reps} repetitions):")
print("-" * 80)
for i, M in enumerate(M_values):
    print(f"M={M:4d}:  Binary RMSE={rmse_binary_mean[i]:.4f}±{rmse_binary_std[i]:.4f}  |  "
          f"Beta RMSE={rmse_beta_mean[i]:.4f}±{rmse_beta_std[i]:.4f}")

# Save results
output_dir = "../../result/monkey_analysis"
os.makedirs(output_dir, exist_ok=True)
data_path = f"{output_dir}/power_beta_bernoulli_data.npz"
np.savez(data_path,
         M_values=np.array(M_values),
         rmse_binary_all=rmse_binary_all,
         rmse_beta_all=rmse_beta_all,
         corr_binary_all=corr_binary_all,
         corr_beta_all=corr_beta_all,
         rmse_binary_mean=rmse_binary_mean,
         rmse_binary_std=rmse_binary_std,
         rmse_beta_mean=rmse_beta_mean,
         rmse_beta_std=rmse_beta_std,
         corr_binary_mean=corr_binary_mean,
         corr_binary_std=corr_binary_std,
         corr_beta_mean=corr_beta_mean,
         corr_beta_std=corr_beta_std,
         n_reps=n_reps,
         N=N)
print(f"\nData saved to: {data_path}")

# Plotting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 14
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
M_arr = np.array(M_values)

# Colors: blue for Binary, pink/magenta for Beta
color_binary = '#4169E1'  # Royal blue
color_beta = '#E91E63'    # Pink/magenta

# Panel 1: RMSE vs M with error bars
ax = axes[0]
ax.errorbar(M_arr, rmse_binary_mean, yerr=rmse_binary_std, fmt='o-', color=color_binary,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary Rasch')
ax.errorbar(M_arr, rmse_beta_mean, yerr=rmse_beta_std, fmt='s-', color=color_beta,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta Rasch')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('RMSE of Item Difficulty (b)')
ax.set_title('Parameter Recovery vs. Sample Size (1PL)')
ax.legend(loc='upper right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])

# Panel 2: Correlation vs M with error bars
ax = axes[1]
ax.errorbar(M_arr, corr_binary_mean, yerr=corr_binary_std, fmt='o-', color=color_binary,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary Rasch')
ax.errorbar(M_arr, corr_beta_mean, yerr=corr_beta_std, fmt='s-', color=color_beta,
            linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta Rasch')
ax.set_xlabel('Number of Test Takers (M)')
ax.set_ylabel('Correlation (r)')
ax.set_title('Correlation vs. Sample Size (1PL)')
ax.legend(loc='lower right')
ax.set_xticks(M_arr)
ax.set_xticklabels([str(m) for m in M_arr])
ax.set_ylim(0, 1.05)

plt.tight_layout()
output_path = f"{output_dir}/power_beta_bernoulli_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
