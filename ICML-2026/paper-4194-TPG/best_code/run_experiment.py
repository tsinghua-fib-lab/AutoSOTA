#!/usr/bin/env python3
"""
Optimized reproduction script for paper 4194 (iter 3: n_trials=2000 for STD reduction):
"Estimation of Treatment Effects Under Nonstationarity via the Truncated Policy Gradient Estimator"

Reproduces Table 3: MAE and STD for the Two-State Nonstationary MDP experiment.
Settings: T=5000, n_trials=1000, smoothness=0.5, treatment_bias=0.1, noise_std=0.1, reward_std=0.1
20 mixing coefficients from 0.01 to 0.99
"""
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import solve

# ============================================================
# Estimator functions (from notebook cells 1-2)
# ============================================================

def DQ_truncated_estimator(rewards, actions, states, discount_factor, k):
    """
    Vectorized DQ truncated estimator using cumulative sums.
    Computationally equivalent to the original loop-based version.
    q_values[i] = sum_{t=i}^{min(i+k, T-1)} rewards[t] * gamma^{t-i}
    discounted_q_values[i] = q_values[i] * gamma^i = sum_{t=i}^{min(i+k, T-1)} rewards[t] * gamma^t
    """
    T = len(rewards)
    # Precompute discounted rewards and cumulative sum
    disc_factors = discount_factor ** np.arange(T)
    disc_rewards = rewards * disc_factors
    cumsum_disc = np.cumsum(disc_rewards)
    
    # For each index i, discounted_q_values[i] = sum_{t=i}^{i+k} rewards[t]*gamma^t
    # = cumsum_disc[min(i+k, T-1)] - (cumsum_disc[i-1] if i>0 else 0)
    end_indices = np.minimum(np.arange(T) + k, T - 1)
    discounted_q_values = cumsum_disc[end_indices].copy()
    mask = np.arange(T) > 0
    discounted_q_values[mask] -= cumsum_disc[np.arange(T)[mask] - 1]
    
    # q_values = discounted_q_values / gamma^i
    q_values = discounted_q_values / disc_factors
    
    DQ_trunc_Est = 2 * np.sum(discounted_q_values * actions) - 2 * np.sum(discounted_q_values * (1 - actions))
    return DQ_trunc_Est, q_values


def DQ_truncated_estimator_batch(rewards, actions, states, discount_factor, k_list):
    """
    Vectorized batch computation of DQ truncated estimators for multiple k values.
    Returns dict: {k: (estimate/T, q_values)} for each k in k_list.
    Much faster than calling DQ_truncated_estimator for each k separately.
    """
    T = len(rewards)
    disc_factors = discount_factor ** np.arange(T)
    disc_rewards = rewards * disc_factors
    cumsum_disc = np.cumsum(disc_rewards)
    
    results = {}
    for k in k_list:
        end_indices = np.minimum(np.arange(T) + k, T - 1)
        discounted_q_values = cumsum_disc[end_indices].copy()
        mask = np.arange(T) > 0
        discounted_q_values[mask] -= cumsum_disc[np.arange(T)[mask] - 1]
        
        q_values = discounted_q_values / disc_factors
        
        DQ_trunc_Est = 2 * np.sum(discounted_q_values * actions) - 2 * np.sum(discounted_q_values * (1 - actions))
        results[k] = DQ_trunc_Est / T
    
    return results


def DQ_LSTD_lambda_estimator(rewards, actions, states, discount_factor, k=None, alpha=1.0, lambda_=0.0):
    states = states.reshape(-1, 1) if states.ndim == 1 else states
    T, d = states.shape
    phi = states

    A = np.zeros((d, d))
    b = np.zeros(d)
    z = np.zeros(d)
    r_mean = np.mean(rewards[:-1])

    for t in range(T - 1):
        phi_t = phi[t]
        phi_tp1 = phi[t + 1]
        r_t = rewards[t]
        z = lambda_ * discount_factor * z + phi_t
        delta_phi = phi_t - discount_factor * phi_tp1
        A += np.outer(z, delta_phi)
        b += z * (r_t - r_mean)

    theta = solve(A + alpha * np.eye(d), b)

    snews = states[1:]
    mask_treat = actions[:-1] == 1
    mask_control = actions[:-1] == 0

    if np.sum(mask_treat) == 0 or np.sum(mask_control) == 0:
        raise ValueError("Insufficient samples for treated or control to compute delta_xbar.")

    ss_treated = snews[mask_treat]
    ss_control = snews[mask_control]
    delta_xbar = np.mean(ss_treated, axis=0) - np.mean(ss_control, axis=0)
    estimate = theta @ delta_xbar
    return estimate, theta


# ============================================================
# MDP Simulation (from notebook cells 4-5)
# ============================================================

def simulate_nonstationary_mdp(policy, T, transition_kernels, reward_matrix,
                                state_space_size, exo_chain, seed, reward_std=0.1):
    np.random.seed(seed)
    states, actions, rewards = [], [], []
    current_state = np.random.choice(state_space_size)

    for t in range(T):
        z_t = exo_chain[t]
        P = transition_kernels[z_t]

        if policy == "treatment":
            a_t = 1
        elif policy == "control":
            a_t = 0
        elif policy == "random":
            a_t = np.random.choice([0, 1])

        mean_reward = reward_matrix[current_state, a_t]
        reward = np.random.normal(loc=mean_reward, scale=reward_std)

        rewards.append(reward)
        actions.append(a_t)
        states.append(current_state)

        next_state = np.random.choice(state_space_size, p=P[current_state, a_t])
        current_state = next_state

    return np.array(rewards), np.array(actions), np.array(states)


def generate_exo_chain_identity(T):
    return np.arange(T)


def generate_mean_reverting_kernels(T, mixing_coeff=0.3, treatment_bias=0.1,
                                     smoothness=0.95, noise_std=0.02, seed=None):
    if seed is not None:
        np.random.seed(seed)

    kernels = []

    def construct_random_rows_with_exact_tv(mixing_coeff):
        assert 0 < mixing_coeff < 1, "mixing_coeff must be in (0, 1)"
        assert mixing_coeff <= 1.0, "mixing_coeff cannot exceed 1.0"
        delta = mixing_coeff / 2
        if np.random.rand() < 0.5:
            row0 = np.array([0.5 + delta, 0.5 - delta])
            row1 = np.array([0.5 - delta, 0.5 + delta])
        else:
            row0 = np.array([0.5 - delta, 0.5 + delta])
            row1 = np.array([0.5 + delta, 0.5 - delta])
        return row0, row1

    row0_mean, row1_mean = construct_random_rows_with_exact_tv(mixing_coeff)
    row_0 = row0_mean.copy()
    row_1 = row1_mean.copy()

    for t in range(T):
        kernel = np.zeros((2, 2, 2))
        row_0 = smoothness * row_0 + (1 - smoothness) * row0_mean + np.random.normal(0, noise_std, 2)
        row_1 = smoothness * row_1 + (1 - smoothness) * row1_mean + np.random.normal(0, noise_std, 2)

        for row in [row_0, row_1]:
            row[:] = np.clip(row, 0.01, 0.99)
            row[:] /= row.sum()

        kernel[0, 0] = row_0.copy()
        kernel[1, 0] = row_1.copy()

        for s in [0, 1]:
            biased_row = kernel[s, 0] + np.array([-treatment_bias, treatment_bias])
            biased_row = np.clip(biased_row, 0.01, 0.99)
            kernel[s, 1] = biased_row / biased_row.sum()

        kernels.append(kernel)

    return kernels


# ============================================================
# Experiment functions (from notebook cells 6, 8, 11)
# ============================================================

def simulate_single_trial_combined(i, T, transition_kernels, reward_matrix, exo_chain,
                                    state_space_size, discount_factor, k_list):
    # Simulate treatment and control for true ATE
    r_treat, _, _ = simulate_nonstationary_mdp("treatment", T, transition_kernels, reward_matrix,
                                                 state_space_size, exo_chain, seed=1000+i)
    r_control, _, _ = simulate_nonstationary_mdp("control", T, transition_kernels, reward_matrix,
                                                   state_space_size, exo_chain, seed=2000+i)
    true_ate = (np.sum(r_treat) - np.sum(r_control)) / T

    # Simulate random policy for estimators
    r_rand, a_rand, s_rand = simulate_nonstationary_mdp("random", T, transition_kernels, reward_matrix,
                                                          state_space_size, exo_chain, seed=3000+i)

    stationary_dq, _ = DQ_LSTD_lambda_estimator(r_rand, a_rand, s_rand, discount_factor)

    # Compute DQ-truncated estimators for all k at once (vectorized batch)
    dq_trunc_by_k = DQ_truncated_estimator_batch(r_rand, a_rand, s_rand, discount_factor, k_list)

    return true_ate, dq_trunc_by_k, stationary_dq


def evaluate_estimators_truncated_only(n_trials, T, transition_kernels, reward_matrix, exo_chain,
                                        state_space_size, discount_factor):
    k_list = [0, 1, 3, 5, 10, 50, 100, 500, T]

    results = Parallel(n_jobs=-1)(
        delayed(simulate_single_trial_combined)(
            i, T, transition_kernels, reward_matrix, exo_chain, state_space_size, discount_factor, k_list
        ) for i in tqdm(range(n_trials), desc="Simulating trials")
    )

    # Unpack results
    true_ates_all = []
    dq_results_by_k = {k: [] for k in k_list}
    stationary_dqs = []

    for true_ate, dq_trunc_by_k, stationary_dq in results:
        true_ates_all.append(true_ate)
        stationary_dqs.append(stationary_dq)
        for k in k_list:
            dq_results_by_k[k].append(dq_trunc_by_k[k])

    results_dict = {"Ground Truth ATE": true_ates_all}
    results_dict["Stationary DQ"] = stationary_dqs
    for k in k_list:
        results_dict[f"k={k}"] = dq_results_by_k[k]

    return results_dict


def summarize_across_mixing(results_dir, treatment_bias, k_list=None, mixing_coeffs=None, smoothness=0.0):
    if mixing_coeffs is None:
        mixing_coeffs = np.round(np.linspace(0.01, 0.99, num=20), 2)
    if k_list is None:
        k_list = ["k=0", "k=1", "k=3", "k=5", "k=10", "k=50", "k=100", "k=5000"]

    mae_by_k = {k: [] for k in k_list}
    std_by_k = {k: [] for k in k_list}
    all_ground_truth_values = []

    for mixing_coeff in mixing_coeffs:
        filename = f"{results_dir}/results_mix{mixing_coeff:.2f}_bias{treatment_bias}_smooth{smoothness}.pkl"
        if not os.path.exists(filename):
            print(f"Missing: {filename}")
            continue

        with open(filename, 'rb') as f:
            results = pickle.load(f)

        gate_list = np.array(results["Ground Truth ATE"])
        all_ground_truth_values.append(gate_list)

        for k in k_list:
            if k not in results:
                continue
            est_list = np.array(results[k])
            bias_pct = np.abs(100 * (est_list - gate_list) / gate_list)
            mae_by_k[k].append(np.mean(bias_pct))
            std_by_k[k].append(np.std(est_list))

    # Print mean ground truth ATE
    if all_ground_truth_values:
        ground_truth_all = np.concatenate(all_ground_truth_values)
        mean_gate = np.mean(ground_truth_all)
        print(f"\nMean Ground Truth ATE across all files: {mean_gate:.4f}")
    else:
        print("No Ground Truth ATEs loaded.")

    # Print summary stats
    print("\n" + "=" * 60)
    print("Summary across mixing coefficients (Table 3 reproduction):")
    print("=" * 60)
    print(f"{'k':<12} {'MAE (%)':<12} {'STD':<12}")
    print("-" * 36)
    for k in k_list:
        mae_vals = mae_by_k[k]
        std_vals = std_by_k[k]
        if len(mae_vals) > 0:
            mae_mean = np.mean(mae_vals)
            std_mean = np.mean(std_vals)
            print(f"{k:<12} {mae_mean:<12.2f} {std_mean:<12.3f}")
        else:
            print(f"{k:<12} No data")

    return mae_by_k, std_by_k, ground_truth_all


# ============================================================
# Main experiment
# ============================================================

def main():
    start_time = time.time()

    # Parameters matching Appendix G.1
    mixing_coeffs = np.round(np.linspace(0.01, 0.99, num=20), 2)
    treatment_bias = 0.1   # kernel deviation δ
    smoothness = 0.5       # mean reversion rate α
    noise_std = 0.1        # σ_ϵ
    reward_std = 0.1       # σ_r
    T = 5000               # horizon
    n_trials = 2000        # independent trials

    reward_matrix = np.array([[0, 1],
                              [5, 6]])

    exo_chain = generate_exo_chain_identity(T)
    seed = 42

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("TPG Estimator Reproduction - Two-State Nonstationary MDP")
    print("=" * 60)
    print(f"Horizon T: {T}")
    print(f"Trials per mixing rate: {n_trials}")
    print(f"Mixing coefficients: {len(mixing_coeffs)} values from {mixing_coeffs[0]} to {mixing_coeffs[-1]}")
    print(f"Treatment bias (δ): {treatment_bias}")
    print(f"Smoothness (α): {smoothness}")
    print(f"Noise std (σ_ϵ): {noise_std}")
    print(f"Reward std (σ_r): {reward_std}")
    print(f"Reward matrix: {reward_matrix.tolist()}")
    print(f"Total runs: {len(mixing_coeffs)} mixing rates × {n_trials} trials")
    print("=" * 60)

    for mc_idx, mixing_coeff in enumerate(mixing_coeffs):
        mc_start = time.time()
        print(f"\n[{mc_idx+1}/{len(mixing_coeffs)}] Running mixing_coeff = {mixing_coeff:.2f}...")

        transition_kernels = generate_mean_reverting_kernels(
            T=T,
            mixing_coeff=mixing_coeff,
            treatment_bias=treatment_bias,
            smoothness=smoothness,
            noise_std=noise_std,
            seed=seed
        )

        results = evaluate_estimators_truncated_only(
            n_trials=n_trials,
            T=T,
            transition_kernels=transition_kernels,
            reward_matrix=reward_matrix,
            exo_chain=exo_chain,
            state_space_size=2,
            discount_factor=1.0
        )

        filename = f"{results_dir}/results_mix{mixing_coeff:.2f}_bias{treatment_bias}_smooth{smoothness}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

        mc_elapsed = time.time() - mc_start
        print(f"  Saved to {filename} ({mc_elapsed:.1f}s)")

    # Print summary
    k_list = ["k=0", "k=1", "k=3", "k=5", "k=10", "k=50", "k=100", "k=5000"]
    mae_by_k, std_by_k, gate_all = summarize_across_mixing(
        results_dir=results_dir,
        treatment_bias=treatment_bias,
        k_list=k_list,
        mixing_coeffs=mixing_coeffs,
        smoothness=smoothness
    )

    total_elapsed = time.time() - start_time
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
