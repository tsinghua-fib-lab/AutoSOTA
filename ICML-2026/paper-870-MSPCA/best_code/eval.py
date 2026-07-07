"""
Evaluation script for MS-PCA reproduction.
Matches rubric: model_name=MS-PCA;benchmark=GaussianMixture-OneSpike;pi_1=0.15;dimension=900;sample_size=1000;n_trials=200

Usage: python3 eval.py
Output: Prints alignment (%, mean +/- std) and runtime (ms, mean +/- std) at pi_1=0.15.
Also saves CSV results to result/rebuttal/pc1_reproduce_*.csv
"""
import numpy as np
import numpy.linalg as LA
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import time
import sys

rng = np.random.default_rng(seed=233)


def estimate_theta_square(lambda_hat, c):
    """Reverse function of l |-> 1 + l + c * (1 + l)/l."""
    nabla = np.square(c + 1 - lambda_hat) - 4 * c
    nabla_sqrt = np.sqrt(np.abs(nabla))
    linear_w = c - lambda_hat + 1
    return (-linear_w + nabla_sqrt) / 2


def ms_pca(X_tilde, max_k_r=3, C=None):
    """Mean-Shift PCA algorithm (Algorithm 1 from the paper)."""
    d, n = X_tilde.shape
    c = d / n
    max_k_r = min(d, max_k_r)

    # Step 1: Exact SVD of contaminated data (ARPACK top-k)
    U_tilde, S_tilde, Vh_tilde = svds(X_tilde / np.sqrt(n), k=max_k_r, which='LM')
    # svds returns ascending order; reverse to descending
    S_tilde = S_tilde[::-1]
    U_tilde = U_tilde[:, ::-1]

    # Adaptive C from singular value eigengap (if not explicitly provided)
    if C is None:
        gap_ratio = (S_tilde[0] - S_tilde[1]) / max(S_tilde[1], 1e-10)
        adaptive_factor = max(0.5, min(2.0, 2.0 / (1.0 + gap_ratio)))
        C = (1 / c) * adaptive_factor

    # Step 2: Estimate theta from leading singular value
    theta_square_prime = estimate_theta_square(S_tilde[0], c)

    # Step 3: Generate knockoff mean-shift perturbation
    noise_proportion_prime = 1
    gamma_prime = rng.binomial(1, noise_proportion_prime, n)
    noise_norm_prime = 2 * (theta_square_prime / noise_proportion_prime) ** 0.5
    m_prime, _ = LA.qr(rng.normal(size=(d, 1)))
    m_prime = noise_norm_prime * m_prime
    A_prime = m_prime * gamma_prime
    X_prime = X_tilde + A_prime

    # Step 4: Exact SVD of knockoff-perturbed data (ARPACK top-k)
    U_prime, S_prime, Vh_prime = svds(X_prime / np.sqrt(n), k=max_k_r, which='LM')
    # svds returns ascending order; reverse to descending
    S_prime = S_prime[::-1]

    # Step 5: Invariance check to identify stable components
    radius = C * n ** (-1 / 2)
    stable_indices = []
    for i in range(max_k_r):
        for j in range(max_k_r):
            if abs(S_tilde[i] - S_prime[j]) < radius:
                stable_indices.append(i)
                break
            if S_prime[j] < S_tilde[i] - 10 * radius:
                break

    stable_eigenvalues = S_tilde[stable_indices] ** 2
    components = U_tilde[:, stable_indices]
    return stable_eigenvalues, components


def run_experiment(n_trials=200, n=1000, c=0.9, magnitude_a=2, k=1,
                   noise_proportions=None, seed=233):
    """Run the MS-PCA alignment experiment."""
    if noise_proportions is None:
        noise_proportions = [0.05, 0.1, 0.15, 0.25]

    global rng
    rng = np.random.default_rng(seed=seed)

    d = int(n * c)
    spike_base = np.sqrt(c)
    spike = [magnitude_a * spike_base]

    all_alignments = []
    all_timings = []

    for trial in tqdm(range(n_trials), desc="Trials"):
        trial_alignments = []
        trial_timings = []

        for noise_proportion in noise_proportions:
            # Generate uncontaminated data
            spiked_vector, _ = LA.qr(rng.normal(size=(d, k)))
            sigma = np.identity(d)
            for i in range(k):
                sigma += spike[i] * np.outer(spiked_vector[:, i], spiked_vector[:, i])
            X = rng.multivariate_normal(np.zeros(d), sigma, n).T
            U, S, Vh = LA.svd(X / np.sqrt(n), full_matrices=False)

            # Add mean-shift contamination
            theta_bar_square = np.sqrt(c)
            noise_norm_base = np.sqrt(theta_bar_square / noise_proportion)
            noise_norm = magnitude_a * noise_norm_base
            m1, _ = LA.qr(rng.normal(size=(d, 1)))
            m1 = noise_norm * m1
            gamma = rng.binomial(1, noise_proportion, n)
            X_tilde = m1 * gamma + X

            # Run MS-PCA and measure time
            t_start = time.perf_counter()
            stable_eigenvalues, components = ms_pca(X_tilde, C=1 / c)
            t_end = time.perf_counter()

            # Compute alignment
            if len(components.shape) == 2 and components.shape[1] > 0:
                alignment = abs(U[:, 0] @ components[:, 0])
            else:
                alignment = 0.0

            trial_alignments.append(alignment)
            trial_timings.append((t_end - t_start) * 1000)  # ms

        all_alignments.append(trial_alignments)
        all_timings.append(trial_timings)

    return np.array(all_alignments), np.array(all_timings), noise_proportions


if __name__ == "__main__":
    n_trials = 200
    noise_proportions = [0.05, 0.1, 0.15, 0.25]

    # Allow overriding n_trials from command line
    if len(sys.argv) > 1:
        n_trials = int(sys.argv[1])

    print(f"MS-PCA Reproduction: n={1000}, d={900}, c={0.9}, n_trials={n_trials}")
    print(f"Noise proportions: {noise_proportions}")
    print(f"Seed: 233")
    print()

    stats_array, timing_array, noise_proportions = run_experiment(
        n_trials=n_trials, noise_proportions=noise_proportions
    )

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
    import os
    os.makedirs("/repo/result/rebuttal", exist_ok=True)
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

    # Print summary for easy parsing
    pi15_idx = noise_proportions.index(0.15)
    align_mean = stats_array[:, pi15_idx].mean() * 100
    align_std = stats_array[:, pi15_idx].std() * 100
    runtime_mean = timing_array[:, pi15_idx].mean()
    runtime_std = timing_array[:, pi15_idx].std()

    print()
    print("=== FINAL SUMMARY (pi_1=0.15) ===")
    print(f"ALIGNMENT={align_mean:.4f}")
    print(f"ALIGNMENT_STD={align_std:.4f}")
    print(f"RUNTIME_MS={runtime_mean:.4f}")
    print(f"RUNTIME_STD={runtime_std:.4f}")
    print("Done.")
