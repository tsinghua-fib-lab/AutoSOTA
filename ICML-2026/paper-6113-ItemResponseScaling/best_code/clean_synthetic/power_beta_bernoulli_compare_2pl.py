import numpy as np
import torch
import matplotlib.pyplot as plt
from tueplots import bundles as bundle
import os
import time

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def prob_2pl_np(theta, a, b):
    """2PL probability (numpy version for simulation)"""
    return 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))

def simulate_binary_2pl(theta, a, b, rng):
    """Simulate binary responses from 2PL model"""
    p = prob_2pl_np(theta, a, b)
    y = rng.binomial(1, p)
    return y, p

def simulate_prob_matrix_2pl(theta, a, b, noise_sd, rng):
    """Simulate noisy probability matrix from 2PL model"""
    p = prob_2pl_np(theta, a, b)
    noisy = p + rng.normal(0.0, noise_sd, size=p.shape)
    noisy = np.clip(noisy, 1e-6, 1 - 1e-6)
    return noisy, p

def fit_2pl_bernoulli_torch(Y, maxiter=1000, lr=1.0):
    """Fit 2PL model to binary responses using PyTorch LBFGS on GPU"""
    M, N = Y.shape
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    # Better initialization from observed response rates
    p_obs = Y.mean(axis=0)
    p_obs = np.clip(p_obs, 0.01, 0.99)
    b_init = -np.log(p_obs / (1 - p_obs))
    b_init = b_init - b_init.mean()

    theta_obs = Y.mean(axis=1)
    theta_obs = np.clip(theta_obs, 0.01, 0.99)
    theta_init = np.log(theta_obs / (1 - theta_obs))
    theta_init = theta_init - theta_init.mean()

    # Initialize parameters
    theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(b_init, dtype=torch.float32, device=device, requires_grad=True)
    log_a = torch.zeros(N, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([theta, b, log_a], max_iter=maxiter, lr=lr,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # Center b for identifiability
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()

        # Constrain log_a
        log_a_clipped = torch.clamp(log_a, -1.5, 1.5)
        a = torch.exp(log_a_clipped)

        p = torch.sigmoid(a[None, :] * (theta_centered[:, None] - b_centered[None, :]))
        eps = 1e-12
        ll = torch.sum(Y_t * torch.log(p + eps) + (1 - Y_t) * torch.log(1 - p + eps))

        # Light regularization on log_a
        reg = 0.1 * torch.sum(log_a ** 2)
        loss = -ll + reg
        loss.backward()
        return loss

    optimizer.step(closure)

    # Extract parameters
    with torch.no_grad():
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()
        log_a_clipped = torch.clamp(log_a, -1.5, 1.5)
        a_out = torch.exp(log_a_clipped)

    return (theta_centered.cpu().numpy(), a_out.cpu().numpy(), b_centered.cpu().numpy())

def fit_2pl_beta_torch(Pobs, phi=400.0, maxiter=1000, lr=1.0):
    """Fit 2PL model to probability matrix using Beta likelihood with PyTorch LBFGS on GPU"""
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)

    # Better initialization from observed probabilities
    logit_P = np.log(Pobs / (1 - Pobs))
    theta_init = logit_P.mean(axis=1)
    b_init = -logit_P.mean(axis=0)
    theta_init = theta_init - theta_init.mean()
    b_init = b_init - b_init.mean()

    # Initialize parameters
    theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(b_init, dtype=torch.float32, device=device, requires_grad=True)
    log_a = torch.zeros(N, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([theta, b, log_a], max_iter=maxiter, lr=lr,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # Center b for identifiability
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()

        # Constrain log_a
        log_a_clipped = torch.clamp(log_a, -1.5, 1.5)
        a = torch.exp(log_a_clipped)

        p = torch.sigmoid(a[None, :] * (theta_centered[:, None] - b_centered[None, :]))
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        alpha = phi * p
        beta_param = phi * (1 - p)

        # Beta log-likelihood
        ll = torch.sum(
            (alpha - 1) * torch.log(Pobs_t) +
            (beta_param - 1) * torch.log(1 - Pobs_t) -
            torch.lgamma(alpha) - torch.lgamma(beta_param) + torch.lgamma(alpha + beta_param)
        )

        # Stronger regularization on log_a
        reg = 0.5 * torch.sum(log_a ** 2)
        loss = -ll + reg
        loss.backward()
        return loss

    optimizer.step(closure)

    # Extract parameters
    with torch.no_grad():
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()
        log_a_clipped = torch.clamp(log_a, -1.5, 1.5)
        a_out = torch.exp(log_a_clipped)

    return (theta_centered.cpu().numpy(), a_out.cpu().numpy(), b_centered.cpu().numpy())

def recovery(true_vals, est_vals):
    """Compute RMSE and correlation for parameter recovery"""
    true_centered = true_vals - true_vals.mean()
    est_centered = est_vals - est_vals.mean()
    rmse = float(np.sqrt(np.mean((est_centered - true_centered) ** 2)))
    corr = float(np.corrcoef(true_centered, est_centered)[0, 1])
    return rmse, corr

def recovery_a(true_a, est_a):
    """Compute RMSE and correlation for discrimination parameter recovery (no centering)"""
    rmse = float(np.sqrt(np.mean((est_a - true_a) ** 2)))
    corr = float(np.corrcoef(true_a, est_a)[0, 1])
    return rmse, corr

LOAD_FROM_NPZ = True
output_dir = "."
os.makedirs(output_dir, exist_ok=True)
data_path = f"{output_dir}/power_beta_bernoulli_2pl_data.npz"

if LOAD_FROM_NPZ:
    data = np.load(data_path)
    M_values = data["M_values"].tolist()
    rmse_binary_b_mean = data["rmse_binary_b_mean"]
    rmse_binary_b_std = data["rmse_binary_b_std"]
    rmse_beta_b_mean = data["rmse_beta_b_mean"]
    rmse_beta_b_std = data["rmse_beta_b_std"]
    corr_binary_b_mean = data["corr_binary_b_mean"]
    corr_binary_b_std = data["corr_binary_b_std"]
    corr_beta_b_mean = data["corr_beta_b_mean"]
    corr_beta_b_std = data["corr_beta_b_std"]
    rmse_binary_a_mean = data["rmse_binary_a_mean"]
    rmse_binary_a_std = data["rmse_binary_a_std"]
    rmse_beta_a_mean = data["rmse_beta_a_mean"]
    rmse_beta_a_std = data["rmse_beta_a_std"]
    corr_binary_a_mean = data["corr_binary_a_mean"]
    corr_binary_a_std = data["corr_binary_a_std"]
    corr_beta_a_mean = data["corr_beta_a_mean"]
    corr_beta_a_std = data["corr_beta_a_std"]
    n_reps = int(data["n_reps"]) if "n_reps" in data else None
    N = int(data["N"]) if "N" in data else None
    print(f"\nLoaded data from: {data_path}")
else:
    # --- Simulation parameters ---
    N = 100
    n_reps = 10
    noise_sd = 0.01

    # Range of test taker counts: 2^n for n=1 to 7
    M_values = [2, 4, 8, 16, 32, 64, 128]

    # Main RNG for generating true parameters
    rng_main = np.random.default_rng(42)

    print(f"Running {n_reps} repetitions for each of {len(M_values)} M values (2PL model)...")
    print(f"Total simulations: {n_reps * len(M_values)}")

    start_time = time.time()

    # Store results for b parameter
    rmse_binary_b_all = np.zeros((n_reps, len(M_values)))
    rmse_beta_b_all = np.zeros((n_reps, len(M_values)))
    corr_binary_b_all = np.zeros((n_reps, len(M_values)))
    corr_beta_b_all = np.zeros((n_reps, len(M_values)))

    # Store results for a parameter
    rmse_binary_a_all = np.zeros((n_reps, len(M_values)))
    rmse_beta_a_all = np.zeros((n_reps, len(M_values)))
    corr_binary_a_all = np.zeros((n_reps, len(M_values)))
    corr_beta_a_all = np.zeros((n_reps, len(M_values)))

    for rep in range(n_reps):
        # Generate true parameters for this repetition
        b_true = rng_main.normal(0, 1, size=N)
        a_true = rng_main.lognormal(0, 0.5, size=N)

        for i, M in enumerate(M_values):
            # Create separate RNG for this simulation
            rng = np.random.default_rng(42 + rep * 1000 + M)
            theta = rng.normal(0, 1, size=M)

            # Binary 2PL
            Y, _ = simulate_binary_2pl(theta, a_true, b_true, rng)
            (_, a_hat_bin, b_hat_bin) = fit_2pl_bernoulli_torch(Y)
            rmse_b, corr_b = recovery(b_true, b_hat_bin)
            rmse_a, corr_a = recovery_a(a_true, a_hat_bin)
            rmse_binary_b_all[rep, i] = rmse_b
            corr_binary_b_all[rep, i] = corr_b
            rmse_binary_a_all[rep, i] = rmse_a
            corr_binary_a_all[rep, i] = corr_a

            # Beta 2PL
            P_obs, _ = simulate_prob_matrix_2pl(theta, a_true, b_true, noise_sd, rng)
            (_, a_hat_beta, b_hat_beta) = fit_2pl_beta_torch(P_obs, phi=400.0)
            rmse_b, corr_b = recovery(b_true, b_hat_beta)
            rmse_a, corr_a = recovery_a(a_true, a_hat_beta)
            rmse_beta_b_all[rep, i] = rmse_b
            corr_beta_b_all[rep, i] = corr_b
            rmse_beta_a_all[rep, i] = rmse_a
            corr_beta_a_all[rep, i] = corr_a

        if (rep + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Completed {rep + 1}/{n_reps} repetitions ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    # Compute mean and std for b
    rmse_binary_b_mean = rmse_binary_b_all.mean(axis=0)
    rmse_binary_b_std = rmse_binary_b_all.std(axis=0)
    rmse_beta_b_mean = rmse_beta_b_all.mean(axis=0)
    rmse_beta_b_std = rmse_beta_b_all.std(axis=0)
    corr_binary_b_mean = corr_binary_b_all.mean(axis=0)
    corr_binary_b_std = corr_binary_b_all.std(axis=0)
    corr_beta_b_mean = corr_beta_b_all.mean(axis=0)
    corr_beta_b_std = corr_beta_b_all.std(axis=0)

    # Compute mean and std for a
    rmse_binary_a_mean = rmse_binary_a_all.mean(axis=0)
    rmse_binary_a_std = rmse_binary_a_all.std(axis=0)
    rmse_beta_a_mean = rmse_beta_a_all.mean(axis=0)
    rmse_beta_a_std = rmse_beta_a_all.std(axis=0)
    corr_binary_a_mean = corr_binary_a_all.mean(axis=0)
    corr_binary_a_std = corr_binary_a_all.std(axis=0)
    corr_beta_a_mean = corr_beta_a_all.mean(axis=0)
    corr_beta_a_std = corr_beta_a_all.std(axis=0)

    # Print summary
    print(f"\nResults (mean +/- std over {n_reps} repetitions):")
    print("-" * 100)
    print("Difficulty (b) recovery:")
    for i, M in enumerate(M_values):
        print(f"M={M:4d}:  Binary RMSE={rmse_binary_b_mean[i]:.4f}+/-{rmse_binary_b_std[i]:.4f}  |  "
              f"Beta RMSE={rmse_beta_b_mean[i]:.4f}+/-{rmse_beta_b_std[i]:.4f}")
    print("\nDiscrimination (a) recovery:")
    for i, M in enumerate(M_values):
        print(f"M={M:4d}:  Binary RMSE={rmse_binary_a_mean[i]:.4f}+/-{rmse_binary_a_std[i]:.4f}  |  "
              f"Beta RMSE={rmse_beta_a_mean[i]:.4f}+/-{rmse_beta_a_std[i]:.4f}")

    # Save results
    np.savez(data_path,
             M_values=np.array(M_values),
             rmse_binary_b_all=rmse_binary_b_all,
             rmse_beta_b_all=rmse_beta_b_all,
             corr_binary_b_all=corr_binary_b_all,
             corr_beta_b_all=corr_beta_b_all,
             rmse_binary_a_all=rmse_binary_a_all,
             rmse_beta_a_all=rmse_beta_a_all,
             corr_binary_a_all=corr_binary_a_all,
             corr_beta_a_all=corr_beta_a_all,
             rmse_binary_b_mean=rmse_binary_b_mean,
             rmse_binary_b_std=rmse_binary_b_std,
             rmse_beta_b_mean=rmse_beta_b_mean,
             rmse_beta_b_std=rmse_beta_b_std,
             rmse_binary_a_mean=rmse_binary_a_mean,
             rmse_binary_a_std=rmse_binary_a_std,
             rmse_beta_a_mean=rmse_beta_a_mean,
             rmse_beta_a_std=rmse_beta_a_std,
             corr_binary_b_mean=corr_binary_b_mean,
             corr_binary_b_std=corr_binary_b_std,
             corr_beta_b_mean=corr_beta_b_mean,
             corr_beta_b_std=corr_beta_b_std,
             corr_binary_a_mean=corr_binary_a_mean,
             corr_binary_a_std=corr_binary_a_std,
             corr_beta_a_mean=corr_beta_a_mean,
             corr_beta_a_std=corr_beta_a_std,
             n_reps=n_reps,
             N=N)
    print(f"\nData saved to: {data_path}")

# Plotting
rc = bundle.icml2024()
rc.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 24
})

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    M_arr = np.array(M_values)

    # Colors: blue for Binary, pink/magenta for Beta
    color_binary = '#4169E1'  # Royal blue
    color_beta = '#E91E63'    # Pink/magenta

    # Panel 1: RMSE of b vs M
    ax = axes[0]
    ax.errorbar(M_arr, rmse_binary_b_mean, yerr=rmse_binary_b_std, fmt='o-', color=color_binary,
                linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary-IRT 2PL')
    ax.errorbar(M_arr, rmse_beta_b_mean, yerr=rmse_beta_b_std, fmt='s-', color=color_beta,
                linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta-IRT 2PL')
    ax.set_xlabel(r'Number of Test Takers ($M$)')
    ax.set_ylabel('RMSE')
    # ax.set_title('Difficulty Recovery vs. Sample Size (2PL)')
    ax.legend(loc='upper right')
    ax.set_xticks(M_arr)
    ax.set_xticklabels([str(m) for m in M_arr])

    # Panel 2: Correlation of b vs M
    ax = axes[1]
    ax.errorbar(M_arr, corr_binary_b_mean, yerr=corr_binary_b_std, fmt='o-', color=color_binary,
                linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Binary-IRT 2PL')
    ax.errorbar(M_arr, corr_beta_b_mean, yerr=corr_beta_b_std, fmt='s-', color=color_beta,
                linewidth=2, markersize=6, capsize=4, capthick=1.5, label='Beta-IRT 2PL')
    ax.set_xlabel(r'Number of Test Takers ($M$)')
    ax.set_ylabel(r'Correlation ($\rho$)')
    # ax.set_title('Correlation vs. Sample Size (2PL)')
    ax.legend(loc='lower right')
    ax.set_xticks(M_arr)
    ax.set_xticklabels([str(m) for m in M_arr])
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = f"{output_dir}/power_beta_bernoulli_2pl_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
