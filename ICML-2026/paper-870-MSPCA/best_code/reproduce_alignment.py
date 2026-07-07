"""
Reproduction script for MS-PCA Alignment metric (Table 2).
Settings: n=1000, d=900 (c=0.9), pi_1=0.15, n_trials=200, seed=233.
Matches rubric: model_name=MS-PCA;benchmark=GaussianMixture-OneSpike;pi_1=0.15;dimension=900;sample_size=1000;n_trials=200
"""
import numpy as np
import numpy.linalg as LA
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import time

rng = np.random.default_rng(seed=233)

def estimate_theta_square(lambda_hat, c):
    """Reverse function of l |-> 1 + l + c * (1 + l)/l."""
    nabla = np.square(c + 1 - lambda_hat) - 4*c
    nabla_sqrt = np.sqrt(np.abs(nabla))
    linear_w = (c - lambda_hat + 1)
    solution1 = (-linear_w + nabla_sqrt)/2
    return solution1

def ms_pca(X_tilde, max_k_r=10, C=None):
    """Mean-Shift PCA algorithm from the paper."""
    d, n = X_tilde.shape
    c = d / n
    if C is None:
        C = 1/c
    max_k_r = min(d, max_k_r)

    svd_tilde = TruncatedSVD(n_components=max_k_r, n_iter=7, random_state=42)
    svd_tilde.fit(X_tilde.T/np.sqrt(n))
    S_tilde = svd_tilde.singular_values_
    U_tilde = svd_tilde.components_.T

    theta_square_prime = estimate_theta_square(S_tilde[0], c)

    # Generate knockoff noise
    noise_proportion_prime = 1
    gamma_prime = rng.binomial(1, noise_proportion_prime, n)
    noise_norm_prime = 2 * (theta_square_prime / noise_proportion_prime)**0.5

    m_prime, _ = LA.qr(rng.normal(size=(d, 1)))
    m_prime = noise_norm_prime * m_prime
    A_prime = m_prime * gamma_prime
    X_prime = X_tilde + A_prime

    svd_prime = TruncatedSVD(n_components=max_k_r, n_iter=7, random_state=42)
    svd_prime.fit(X_prime.T/np.sqrt(n))
    S_prime = svd_prime.singular_values_

    radius = C * n**(-1/2)

    # Invariance check
    stable_indices = []
    for i in range(max_k_r):
        for j in range(max_k_r):
            if abs(S_tilde[i] - S_prime[j]) < radius:
                stable_indices.append(i)
                break

    stable_eigenvalues = S_tilde[stable_indices]**2
    components = U_tilde[:, stable_indices]
    return stable_eigenvalues, components

# Paper settings matching the rubric
k = 1
magnitude_a = 2
n = 1000
c = 0.9
d = int(n * c)  # = 900
spike_base = np.sqrt(c)
spike = [2 * spike_base]

noise_proportions = [0.05, 0.1, 0.15, 0.25]
n_trials = 200

print(f"Reproducing MS-PCA Alignment")
print(f"Settings: n={n}, d={d}, c={c}, n_trials={n_trials}")
print(f"Noise proportions: {noise_proportions}")
print(f"Seed: 233")
print()

all_alignments = []
all_timings = []

for trial in tqdm(range(n_trials), desc="Trials"):
    trial_alignments = []
    trial_timings = []

    for noise_proportion in noise_proportions:
        # Generate data
        d_actual = int(n * c)
        spiked_vector, _ = LA.qr(rng.normal(size=(d_actual, k)))
        sigma = np.identity(d_actual)
        for i in range(k):
            sigma += spike[i] * np.outer(spiked_vector[:, i], spiked_vector[:, i])
        X = rng.multivariate_normal(np.zeros(d_actual), sigma, n).T
        U, S, Vh = LA.svd(X/np.sqrt(n), full_matrices=False)

        # Contaminate with mean-shift
        theta_bar_square = np.sqrt(c)
        noise_norm_base = np.sqrt(theta_bar_square / noise_proportion)
        noise_norm = magnitude_a * noise_norm_base

        m1, _ = LA.qr(rng.normal(size=(d_actual, 1)))
        m1 = noise_norm * m1
        gamma = rng.binomial(1, noise_proportion, n)
        A = m1 * gamma
        X_tilde = A + X

        # Run MS-PCA and time it
        t_start = time.perf_counter()
        stable_eigenvalues, components = ms_pca(X_tilde, C=1/c)
        t_end = time.perf_counter()

        # Compute alignment (cosine similarity between true and recovered PC)
        if len(components.shape) == 2 and components.shape[1] > 0:
            ms_alignment = abs(U[:, 0] @ components[:, 0])
        else:
            ms_alignment = 0.0

        trial_alignments.append(ms_alignment)
        trial_timings.append((t_end - t_start) * 1000)  # ms

    all_alignments.append(trial_alignments)
    all_timings.append(trial_timings)

# Compute statistics
stats_array = np.array(all_alignments)
timing_array = np.array(all_timings)

print()
print("=== MS-PCA Alignment Results (%, mean +/- std) ===")
for i, np_ in enumerate(noise_proportions):
    mean_val = stats_array[:, i].mean() * 100
    std_val = stats_array[:, i].std() * 100
    print(f"  pi_1={np_:.2f}: {mean_val:.2f} +/- {std_val:.2f}%")

print()
print("=== MS-PCA Runtime Results (ms, mean +/- std) ===")
for i, np_ in enumerate(noise_proportions):
    mean_val = timing_array[:, i].mean()
    std_val = timing_array[:, i].std()
    print(f"  pi_1={np_:.2f}: {mean_val:.4f} +/- {std_val:.4f} ms")

# Save results
fname = "/repo/result/rebuttal/pc1_reproduce"
df_mean = pd.DataFrame(
    stats_array.mean(axis=0).reshape(1, -1) * 100,
    columns=[f"pi_1={np_}" for np_ in noise_proportions],
    index=["MS-PCA"]
)
df_std = pd.DataFrame(
    stats_array.std(axis=0).reshape(1, -1) * 100,
    columns=[f"pi_1={np_}" for np_ in noise_proportions],
    index=["MS-PCA"]
)
df_mean.to_csv(f"{fname}_alignment_mean.csv")
df_std.to_csv(f"{fname}_alignment_std.csv")

df_time_mean = pd.DataFrame(
    timing_array.mean(axis=0).reshape(1, -1),
    columns=[f"pi_1={np_}" for np_ in noise_proportions],
    index=["MS-PCA"]
)
df_time_std = pd.DataFrame(
    timing_array.std(axis=0).reshape(1, -1),
    columns=[f"pi_1={np_}" for np_ in noise_proportions],
    index=["MS-PCA"]
)
df_time_mean.to_csv(f"{fname}_runtime_mean.csv")
df_time_std.to_csv(f"{fname}_runtime_std.csv")

print()
print(f"Results saved to {fname}_*")
print("Done.")
