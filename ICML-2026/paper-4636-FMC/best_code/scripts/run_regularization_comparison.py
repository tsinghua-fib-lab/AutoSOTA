#!/usr/bin/env python3
"""Run ELBO training with different regularization settings for comparison."""

import torch
import sys
import os

sys.path.insert(0, "/home/pruhlman/Doctorat/project/ropefm")

from scripts.simple_elbo_vs_fm import (
    SimpleGaussianModel,
    train_source_elbo_simple,
    train_source_fm_simple,
    visualize_results,
    Standardizer,
)

# Configuration
n_train = 50000
n_test = 5000
theta_dim = 2
x_dim = 3
y_dim = 3
device = "cpu"
seed = 42
epochs = 200

torch.manual_seed(seed)

print("=" * 80)
print("REGULARIZATION COMPARISON FOR LATEX SUMMARY")
print("=" * 80)

# Create model
model = SimpleGaussianModel(theta_dim, x_dim, y_dim, seed=seed)

# Generate dataset
theta_train, x_train, y_train = model.sample(n_train)
theta_test, x_test, y_test = model.sample(n_test)

# Standardize data
print("\nStandardizing data...")
theta_std = Standardizer(theta_train)
x_std = Standardizer(x_train)
y_std = Standardizer(y_train)

theta_train_std = theta_std.standardize(theta_train)
x_train_std = x_std.standardize(x_train)
y_train_std = y_std.standardize(y_train)

print(f"✓ Theta: mean={theta_std.mean.squeeze().numpy()}, std={theta_std.std.squeeze().numpy()}")
print(f"✓ X: mean={x_std.mean.squeeze().numpy()}, std={x_std.std.squeeze().numpy()}")
print(f"✓ Y: mean={y_std.mean.squeeze().numpy()}, std={y_std.std.squeeze().numpy()}")

# Create output directory
os.makedirs("logs/elbo_source", exist_ok=True)

# ============================================================================
# 1. NO REGULARIZATION
# ============================================================================
print("\n" + "=" * 80)
print("1. NO REGULARIZATION")
print("=" * 80)

source_flow_none, epsilon_encoder_none, losses_none = train_source_elbo_simple(
    model,
    theta_train_std,
    y_train_std,
    epochs=epochs,
    lr=1e-5,
    batch_size=256,
    device=device,
    beta_kl=1.0,
    lambda_std=0.0,  # No std penalty
    lambda_diversity=0.0,  # No diversity
    use_kl_annealing=False,
    patience=10,
)

# Train Flow Matching for comparison
print("\nTraining Flow Matching (same for all experiments)...")
flow_matching, losses_fm = train_source_fm_simple(
    model,
    x_train_std,
    y_train_std,
    epochs=epochs,
    lr=1e-4,
    batch_size=256,
    device=device,
)

# Save visualization
print("\nGenerating visualization...")
import matplotlib.pyplot as plt
from scripts.simple_elbo_vs_fm import sample_from_source, sample_from_fm

y_single = y_test[:1]
y_single_std = y_std.standardize(y_single)
n_samples = 1000

# Sample from both methods (in standardized space)
x_samples_elbo_std = sample_from_source(
    source_flow_none, y_single_std, n_samples, device,
    epsilon_encoder=epsilon_encoder_none, model=model, theta_std=theta_std
)
x_samples_elbo_std = x_samples_elbo_std.squeeze(1)
x_samples_elbo = x_std.unstandardize(x_samples_elbo_std).cpu().numpy()

x_samples_fm_std = sample_from_fm(flow_matching, y_single_std, n_samples, device)
x_samples_fm = x_std.unstandardize(x_samples_fm_std).cpu().numpy()

# Get theta samples via p(theta|x) (in original space)
mu_post_elbo, Sigma_post = model.true_posterior_theta_given_x(torch.from_numpy(x_samples_elbo))
theta_samples_elbo = (
    mu_post_elbo + torch.randn(n_samples, model.theta_dim) @ torch.linalg.cholesky(Sigma_post).T
)
theta_samples_elbo = theta_samples_elbo.numpy()

mu_post_fm, _ = model.true_posterior_theta_given_x(torch.from_numpy(x_samples_fm))
theta_samples_fm = (
    mu_post_fm + torch.randn(n_samples, model.theta_dim) @ torch.linalg.cholesky(Sigma_post).T
)
theta_samples_fm = theta_samples_fm.numpy()

# Get true posteriors
mu_true_theta, Sigma_true_theta = model.true_posterior_theta_given_y(y_single)
theta_samples_true = (
    mu_true_theta
    + torch.randn(n_samples, model.theta_dim) @ torch.linalg.cholesky(Sigma_true_theta).T
).numpy()

mu_true_x, Sigma_true_x = model.true_posterior_x_given_y(y_single)
x_samples_true = (
    mu_true_x + torch.randn(n_samples, model.x_dim) @ torch.linalg.cholesky(Sigma_true_x).T
).numpy()

# Get grids
theta0_grid, theta1_grid, theta_density_grid = model.eval_posterior_theta_given_y_on_grid(y_single)
x0_grid, x1_grid, x_density_grid = model.eval_posterior_x_given_y_on_grid(y_single)

# Create figure
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle(f"NO REGULARIZATION (β=1.0, λ_std=0.0, λ_div=0.0)", fontsize=14, fontweight="bold")

# Row 1: x samples
axes[0, 0].contour(
    x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[0, 0].scatter(
    x_samples_elbo[:, 0], x_samples_elbo[:, 1], alpha=0.5, s=15, c="blue", label="ELBO samples"
)
axes[0, 0].set_xlabel("x[0]")
axes[0, 0].set_ylabel("x[1]")
axes[0, 0].set_title("ELBO: x samples vs true p(x|y)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].contour(
    x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[0, 1].scatter(
    x_samples_fm[:, 0], x_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
)
axes[0, 1].set_xlabel("x[0]")
axes[0, 1].set_ylabel("x[1]")
axes[0, 1].set_title("Flow Matching: x samples vs true p(x|y)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(x_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue")
axes[0, 2].hist(x_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange")
axes[0, 2].hist(
    x_samples_true[:, 0], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
)
axes[0, 2].set_xlabel("x[0]")
axes[0, 2].set_ylabel("Density")
axes[0, 2].set_title("x[0] Distribution")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[0, 3].hist(x_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue")
axes[0, 3].hist(x_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange")
axes[0, 3].hist(
    x_samples_true[:, 1], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
)
axes[0, 3].set_xlabel("x[1]")
axes[0, 3].set_ylabel("Density")
axes[0, 3].set_title("x[1] Distribution")
axes[0, 3].legend()
axes[0, 3].grid(True, alpha=0.3)

# Row 2: theta samples
axes[1, 0].contour(
    theta0_grid, theta1_grid, theta_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[1, 0].contourf(
    theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
)
axes[1, 0].scatter(
    theta_samples_elbo[:, 0],
    theta_samples_elbo[:, 1],
    alpha=0.5,
    s=15,
    c="blue",
    label="ELBO samples",
)
axes[1, 0].set_xlabel("θ[0]")
axes[1, 0].set_ylabel("θ[1]")
axes[1, 0].set_title("ELBO: θ samples vs true p(θ|y)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].contour(
    theta0_grid, theta1_grid, theta_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[1, 1].contourf(
    theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
)
axes[1, 1].scatter(
    theta_samples_fm[:, 0], theta_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
)
axes[1, 1].set_xlabel("θ[0]")
axes[1, 1].set_ylabel("θ[1]")
axes[1, 1].set_title("Flow Matching: θ samples vs true p(θ|y)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(
    theta_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
)
axes[1, 2].hist(
    theta_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange"
)
axes[1, 2].hist(
    theta_samples_true[:, 0], bins=30, alpha=0.6, label="True p(θ|y)", density=True, color="green"
)
axes[1, 2].set_xlabel("θ[0]")
axes[1, 2].set_ylabel("Density")
axes[1, 2].set_title("θ[0] Distribution")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

axes[1, 3].hist(
    theta_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
)
axes[1, 3].hist(
    theta_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange"
)
axes[1, 3].hist(
    theta_samples_true[:, 1], bins=30, alpha=0.6, label="True p(θ|y)", density=True, color="green"
)
axes[1, 3].set_xlabel("θ[1]")
axes[1, 3].set_ylabel("Density")
axes[1, 3].set_title("θ[1] Distribution")
axes[1, 3].legend()
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logs/elbo_source/comparison_no_reg.pdf", dpi=300, bbox_inches="tight")
print("✓ Saved: logs/elbo_source/comparison_no_reg.pdf")
plt.close()

# ============================================================================
# 2. LOW REGULARIZATION
# ============================================================================
print("\n" + "=" * 80)
print("2. LOW REGULARIZATION")
print("=" * 80)

source_flow_low, epsilon_encoder_low, losses_low = train_source_elbo_simple(
    model,
    theta_train_std,
    y_train_std,
    epochs=epochs,
    lr=1e-5,
    batch_size=256,
    device=device,
    beta_kl=0.5,  # Relax KL
    lambda_std=0.1,  # Light std penalty
    lambda_diversity=0.01,  # Light diversity
    use_kl_annealing=False,
    patience=10,
)

# Sample and visualize
x_samples_elbo_std = sample_from_source(
    source_flow_low, y_single_std, n_samples, device,
    epsilon_encoder=epsilon_encoder_low, model=model, theta_std=theta_std
).squeeze(1)
x_samples_elbo = x_std.unstandardize(x_samples_elbo_std).cpu().numpy()
mu_post_elbo, _ = model.true_posterior_theta_given_x(torch.from_numpy(x_samples_elbo))
theta_samples_elbo = (
    mu_post_elbo + torch.randn(n_samples, 2) @ torch.linalg.cholesky(Sigma_post).T
).numpy()

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle(f"LOW REGULARIZATION (β=0.5, λ_std=0.1, λ_div=0.01)", fontsize=14, fontweight="bold")

# [Same plotting code as above]
axes[0, 0].contour(
    x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[0, 0].scatter(
    x_samples_elbo[:, 0], x_samples_elbo[:, 1], alpha=0.5, s=15, c="blue", label="ELBO samples"
)
axes[0, 0].set_xlabel("x[0]")
axes[0, 0].set_ylabel("x[1]")
axes[0, 0].set_title("ELBO: x samples vs true p(x|y)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].contour(
    x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[0, 1].scatter(
    x_samples_fm[:, 0], x_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
)
axes[0, 1].set_xlabel("x[0]")
axes[0, 1].set_ylabel("x[1]")
axes[0, 1].set_title("Flow Matching: x samples vs true p(x|y)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(x_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue")
axes[0, 2].hist(x_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange")
axes[0, 2].hist(
    x_samples_true[:, 0], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
)
axes[0, 2].set_xlabel("x[0]")
axes[0, 2].set_ylabel("Density")
axes[0, 2].set_title("x[0] Distribution")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[0, 3].hist(x_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue")
axes[0, 3].hist(x_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange")
axes[0, 3].hist(
    x_samples_true[:, 1], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
)
axes[0, 3].set_xlabel("x[1]")
axes[0, 3].set_ylabel("Density")
axes[0, 3].set_title("x[1] Distribution")
axes[0, 3].legend()
axes[0, 3].grid(True, alpha=0.3)

axes[1, 0].contour(
    theta0_grid, theta1_grid, theta_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[1, 0].contourf(
    theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
)
axes[1, 0].scatter(
    theta_samples_elbo[:, 0],
    theta_samples_elbo[:, 1],
    alpha=0.5,
    s=15,
    c="blue",
    label="ELBO samples",
)
axes[1, 0].set_xlabel("θ[0]")
axes[1, 0].set_ylabel("θ[1]")
axes[1, 0].set_title("ELBO: θ samples vs true p(θ|y)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].contour(
    theta0_grid, theta1_grid, theta_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[1, 1].contourf(
    theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
)
axes[1, 1].scatter(
    theta_samples_fm[:, 0], theta_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
)
axes[1, 1].set_xlabel("θ[0]")
axes[1, 1].set_ylabel("θ[1]")
axes[1, 1].set_title("Flow Matching: θ samples vs true p(θ|y)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(
    theta_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
)
axes[1, 2].hist(
    theta_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange"
)
axes[1, 2].hist(
    theta_samples_true[:, 0], bins=30, alpha=0.6, label="True p(θ|y)", density=True, color="green"
)
axes[1, 2].set_xlabel("θ[0]")
axes[1, 2].set_ylabel("Density")
axes[1, 2].set_title("θ[0] Distribution")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

axes[1, 3].hist(
    theta_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
)
axes[1, 3].hist(
    theta_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange"
)
axes[1, 3].hist(
    theta_samples_true[:, 1], bins=30, alpha=0.6, label="True p(θ|y)", density=True, color="green"
)
axes[1, 3].set_xlabel("θ[1]")
axes[1, 3].set_ylabel("Density")
axes[1, 3].set_title("θ[1] Distribution")
axes[1, 3].legend()
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logs/elbo_source/comparison_low_reg.pdf", dpi=300, bbox_inches="tight")
print("✓ Saved: logs/elbo_source/comparison_low_reg.pdf")
plt.close()

# ============================================================================
# 3. HIGH REGULARIZATION
# ============================================================================
print("\n" + "=" * 80)
print("3. HIGH REGULARIZATION")
print("=" * 80)

source_flow_high, epsilon_encoder_high, losses_high = train_source_elbo_simple(
    model,
    theta_train_std,
    y_train_std,
    epochs=epochs,
    lr=1e-5,
    batch_size=256,
    device=device,
    beta_kl=0.1,  # Strong KL relaxation
    lambda_std=1.0,  # Strong std penalty
    lambda_diversity=0.1,  # Strong diversity
    use_kl_annealing=True,
    patience=10,
)

# Sample and visualize
x_samples_elbo_std = sample_from_source(
    source_flow_high, y_single_std, n_samples, device,
    epsilon_encoder=epsilon_encoder_high, model=model, theta_std=theta_std
).squeeze(1)
x_samples_elbo = x_std.unstandardize(x_samples_elbo_std).cpu().numpy()
mu_post_elbo, _ = model.true_posterior_theta_given_x(torch.from_numpy(x_samples_elbo))
theta_samples_elbo = (
    mu_post_elbo + torch.randn(n_samples, 2) @ torch.linalg.cholesky(Sigma_post).T
).numpy()

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle(
    f"HIGH REGULARIZATION (β=0.1→1.0, λ_std=1.0, λ_div=0.1, KL annealing)",
    fontsize=14,
    fontweight="bold",
)

# [Same plotting code]
axes[0, 0].contour(
    x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[0, 0].scatter(
    x_samples_elbo[:, 0], x_samples_elbo[:, 1], alpha=0.5, s=15, c="blue", label="ELBO samples"
)
axes[0, 0].set_xlabel("x[0]")
axes[0, 0].set_ylabel("x[1]")
axes[0, 0].set_title("ELBO: x samples vs true p(x|y)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].contour(
    x0_grid, x1_grid, x_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[0, 1].scatter(
    x_samples_fm[:, 0], x_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
)
axes[0, 1].set_xlabel("x[0]")
axes[0, 1].set_ylabel("x[1]")
axes[0, 1].set_title("Flow Matching: x samples vs true p(x|y)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(x_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue")
axes[0, 2].hist(x_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange")
axes[0, 2].hist(
    x_samples_true[:, 0], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
)
axes[0, 2].set_xlabel("x[0]")
axes[0, 2].set_ylabel("Density")
axes[0, 2].set_title("x[0] Distribution")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[0, 3].hist(x_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue")
axes[0, 3].hist(x_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange")
axes[0, 3].hist(
    x_samples_true[:, 1], bins=30, alpha=0.6, label="True p(x|y)", density=True, color="green"
)
axes[0, 3].set_xlabel("x[1]")
axes[0, 3].set_ylabel("Density")
axes[0, 3].set_title("x[1] Distribution")
axes[0, 3].legend()
axes[0, 3].grid(True, alpha=0.3)

axes[1, 0].contour(
    theta0_grid, theta1_grid, theta_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[1, 0].contourf(
    theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
)
axes[1, 0].scatter(
    theta_samples_elbo[:, 0],
    theta_samples_elbo[:, 1],
    alpha=0.5,
    s=15,
    c="blue",
    label="ELBO samples",
)
axes[1, 0].set_xlabel("θ[0]")
axes[1, 0].set_ylabel("θ[1]")
axes[1, 0].set_title("ELBO: θ samples vs true p(θ|y)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].contour(
    theta0_grid, theta1_grid, theta_density_grid, levels=8, colors="black", alpha=0.4, linewidths=1
)
axes[1, 1].contourf(
    theta0_grid, theta1_grid, theta_density_grid, levels=20, cmap="Greys", alpha=0.3
)
axes[1, 1].scatter(
    theta_samples_fm[:, 0], theta_samples_fm[:, 1], alpha=0.5, s=15, c="orange", label="FM samples"
)
axes[1, 1].set_xlabel("θ[0]")
axes[1, 1].set_ylabel("θ[1]")
axes[1, 1].set_title("Flow Matching: θ samples vs true p(θ|y)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(
    theta_samples_elbo[:, 0], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
)
axes[1, 2].hist(
    theta_samples_fm[:, 0], bins=30, alpha=0.6, label="FM", density=True, color="orange"
)
axes[1, 2].hist(
    theta_samples_true[:, 0], bins=30, alpha=0.6, label="True p(θ|y)", density=True, color="green"
)
axes[1, 2].set_xlabel("θ[0]")
axes[1, 2].set_ylabel("Density")
axes[1, 2].set_title("θ[0] Distribution")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

axes[1, 3].hist(
    theta_samples_elbo[:, 1], bins=30, alpha=0.6, label="ELBO", density=True, color="blue"
)
axes[1, 3].hist(
    theta_samples_fm[:, 1], bins=30, alpha=0.6, label="FM", density=True, color="orange"
)
axes[1, 3].hist(
    theta_samples_true[:, 1], bins=30, alpha=0.6, label="True p(θ|y)", density=True, color="green"
)
axes[1, 3].set_xlabel("θ[1]")
axes[1, 3].set_ylabel("Density")
axes[1, 3].set_title("θ[1] Distribution")
axes[1, 3].legend()
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logs/elbo_source/comparison_high_reg.pdf", dpi=300, bbox_inches="tight")
print("✓ Saved: logs/elbo_source/comparison_high_reg.pdf")
plt.close()

print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 80)
print("Generated plots:")
print("  - logs/elbo_source/comparison_no_reg.pdf")
print("  - logs/elbo_source/comparison_low_reg.pdf")
print("  - logs/elbo_source/comparison_high_reg.pdf")
