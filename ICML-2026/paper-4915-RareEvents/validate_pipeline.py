#!/usr/bin/env python3
"""Quick validation: load ARI data, run MBAR, compute histogram."""
import json
import sys
import os
import time
import numpy as np
import pymbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("data", "paper")
OBS_NAME = "ari"

# ------------------------------------------------------------
# Classes from notebook
# ------------------------------------------------------------
class Trajectories:
    def __init__(self, trajectories, biases, steps_per_bias):
        self.trajectories = trajectories
        self.biases = biases
        self.steps_per_bias = steps_per_bias

    def get_all_biases(self):
        all_biases = []
        for biases_per_traj in self.biases:
            for bias in biases_per_traj:
                if bias not in all_biases:
                    all_biases.append(bias)
        return sorted(all_biases)

    def get_all_trajs_per_bias(self):
        all_biases = []
        all_trajs_per_bias = []
        for biases_for_trajs, trajs_per_biases in zip(self.biases, self.trajectories):
            cum_steps = 0
            for bias in biases_for_trajs:
                if bias not in all_biases:
                    all_biases.append(bias)
                    all_trajs_per_bias.append([])
                    for traj in trajs_per_biases:
                        all_trajs_per_bias[-1].append(traj[cum_steps: cum_steps + self.steps_per_bias])
                else:
                    for traj in trajs_per_biases:
                        index = all_biases.index(bias)
                        all_trajs_per_bias[index].append(traj[cum_steps: cum_steps + self.steps_per_bias])
                cum_steps += self.steps_per_bias

        sorted_indices = np.argsort(all_biases)
        all_biases = [all_biases[i] for i in sorted_indices]
        all_trajs_per_bias = [all_trajs_per_bias[i] for i in sorted_indices]
        return all_biases, np.array(all_trajs_per_bias)

class Scores:
    def __init__(self, scores_dict):
        self.scores_dict = scores_dict

    def __add__(self, other):
        new_scores_dict = dict(self.scores_dict)
        for key, value in other.scores_dict.items():
            if key in new_scores_dict:
                new_scores_dict[key].extend(value)
            else:
                new_scores_dict[key] = value
        return Scores(new_scores_dict)

    def get_all_biases(self):
        return sorted(self.scores_dict.keys())

# ------------------------------------------------------------
# Core functions
# ------------------------------------------------------------
def apply_burnin(trajectories, burnin=0.1):
    burnin_steps = int(trajectories.steps_per_bias * burnin)
    new_trajectories = []
    for biases, trajs_per_biases in zip(trajectories.biases, trajectories.trajectories):
        new_trajs_per_biases = []
        for traj in trajs_per_biases:
            new_traj = []
            for i in range(len(biases)):
                start = i * trajectories.steps_per_bias + burnin_steps
                end = (i + 1) * trajectories.steps_per_bias
                new_traj += traj[start:end]
            new_trajs_per_biases.append(new_traj)
        new_trajectories.append(new_trajs_per_biases)
    return Trajectories(new_trajectories, trajectories.biases, trajectories.steps_per_bias - burnin_steps)

def gelman_rubin(trajectories, cutoff=1.1):
    biases, all_trajs_per_bias = trajectories.get_all_trajs_per_bias()
    grs = []
    scores_dict = {}
    for bias, trajs_per_bias in zip(biases, all_trajs_per_bias):
        j, L = trajs_per_bias.shape
        means = np.mean(trajs_per_bias, axis=1)
        mean_of_means = np.mean(means)
        B = L / (j - 1) * np.sum((means - mean_of_means) ** 2)
        W = 1 / j * np.sum(np.var(trajs_per_bias, axis=1, ddof=1))
        var_hat = (L - 1) / L * W + B / L
        R_hat = np.sqrt(var_hat / W)
        grs.append(R_hat)
        if R_hat < cutoff:
            scores_dict[bias] = trajs_per_bias.flatten().tolist()
    return Scores(scores_dict), grs

def subsample_scores(scores, num_samples, random=False):
    subsample_dict = {}
    if random:
        for bias in scores.get_all_biases():
            subsample_dict[bias] = np.random.choice(scores.scores_dict[bias], num_samples).tolist()
    else:
        for bias in scores.get_all_biases():
            subsample_dict[bias] = scores.scores_dict[bias][:num_samples]
    return Scores(subsample_dict)

def mbar_fn(scores, return_overlap=False):
    biases = scores.get_all_biases()
    if 0 not in biases:
        raise Exception("0 must be in the scores dictionary to perform mbar.")
    zero_index = biases.index(0)
    all_scores = np.array([])
    Ns = np.array([])
    for bias in biases:
        all_scores = np.append(all_scores, scores.scores_dict[bias])
        Ns = np.append(Ns, len(scores.scores_dict[bias]))
    u_kn = np.outer(biases, all_scores)
    mbar = pymbar.MBAR(u_kn, Ns)
    mbar_results = mbar.compute_free_energy_differences()
    res = (-mbar_results["Delta_f"][zero_index]).tolist()
    if return_overlap:
        overlap = mbar.compute_overlap()
        return res, overlap
    else:
        return res

def get_weights(scores, log_Zs):
    scores_list = []
    weights_list = []
    for i, bias in enumerate(scores.get_all_biases()):
        scores_list += scores.scores_dict[bias]
        weights_list += np.exp(log_Zs[i] + bias * np.array(scores.scores_dict[bias])).tolist()
    return scores_list, weights_list

def generate_trajectories_bootstrap(trajectories):
    new_trajectories = []
    for biases, trajs in zip(trajectories.biases, trajectories.trajectories):
        indices = np.random.randint(0, len(trajs), len(trajs))
        new_trajectories.append([trajs[i] for i in indices])
    return Trajectories(new_trajectories, trajectories.biases, trajectories.steps_per_bias)

def bootstrap_conf_interval(trajectories, unbiased_scores, bins, num_bootstraps=100, lower_idx=2, upper_idx=97):
    if upper_idx > num_bootstraps - 1:
        raise Exception("upper_idx must be <= num_bootstraps - 1")
    if lower_idx >= upper_idx:
        raise Exception("lower_idx must be strictly less than upper_idx")

    trajs_burnin = apply_burnin(trajectories)
    num_unbiased = len(unbiased_scores.scores_dict[0]) // 2

    boot_histogram_heights = np.zeros((num_bootstraps, len(bins) - 1))
    for i in range(num_bootstraps):
        t0 = time.time()
        bootstrap_trajs = generate_trajectories_bootstrap(trajs_burnin)
        accepted_scores, _ = gelman_rubin(bootstrap_trajs)
        accepted_scores += subsample_scores(unbiased_scores, num_unbiased, random=True)
        log_Zs = mbar_fn(accepted_scores)
        scores_list, weights_list = get_weights(accepted_scores, log_Zs)
        histogram_heights, _ = np.histogram(scores_list, weights=weights_list, bins=bins, density=True)
        boot_histogram_heights[i, :] = histogram_heights
        elapsed = time.time() - t0
        print(f"  Bootstrap {i+1}/{num_bootstraps}: {elapsed:.1f}s", flush=True)

    sorted_boot_heights = np.sort(boot_histogram_heights, axis=0)
    lower_bounds = sorted_boot_heights[lower_idx]
    upper_bounds = sorted_boot_heights[upper_idx]
    return lower_bounds, upper_bounds, boot_histogram_heights

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
print("=" * 60)
print("Loading ARI data...")
print("=" * 60)

trajs_dir = os.path.join(DATA_DIR, f"{OBS_NAME}_trajectories.json")
unbiased_dir = os.path.join(DATA_DIR, f"{OBS_NAME}_unbiased_samples.json")

with open(trajs_dir, "r") as f:
    trajectories_saved = json.load(f)
    trajectories = Trajectories(trajectories_saved[0], trajectories_saved[1], trajectories_saved[2])

with open(unbiased_dir, "r") as f:
    unbiased_scores_list = json.load(f)
    unbiased_scores = Scores({0: unbiased_scores_list})

print(f"Number of annealing schedules: {len(trajectories.trajectories)}")
print(f"Trajectories per schedule: {len(trajectories.trajectories[0])}")
print(f"Steps per bias: {trajectories.steps_per_bias}")
print(f"All biases: {trajectories.get_all_biases()}")
print(f"Unbiased samples: {len(unbiased_scores_list)}")

# Apply burnin
trajectories_burnin = apply_burnin(trajectories, 0.1)
print(f"Steps per bias after burnin: {trajectories_burnin.steps_per_bias}")

# Gelman-Rubin
accepted_scores, grs = gelman_rubin(trajectories_burnin)
print(f"Gelman-Rubin values: {[f'{g:.4f}' for g in grs]}")
print(f"Accepted biases after GR: {accepted_scores.get_all_biases()}")

# Subsample unbiased
num_trajs_per_schedule = len(trajectories.trajectories[0])
num_biases_per_anneal = len(trajectories.biases[0])
unbiased_samples_for_biased = int(trajectories.steps_per_bias * num_trajs_per_schedule / 2)
accepted_scores += subsample_scores(unbiased_scores, num_samples=unbiased_samples_for_biased)

num_samples_for_unbiased = trajectories.steps_per_bias * num_trajs_per_schedule * num_biases_per_anneal // 2 + unbiased_samples_for_biased
unbiased_samples_for_histogram = unbiased_scores_list[:num_samples_for_unbiased]
print(f"Unbiased samples for biased: {unbiased_samples_for_biased}")
print(f"Total unbiased histogram samples: {len(unbiased_samples_for_histogram)}")

# MBAR
print("\nComputing MBAR...")
t0 = time.time()
log_Zs = mbar_fn(accepted_scores)
print(f"log_Zs: {[f'{z:.4f}' for z in log_Zs]}")
print(f"MBAR took {time.time() - t0:.1f}s")

# Histogram
lower, upper, num_bins = (-8, 15, 80)
bins = np.linspace(lower, upper, num_bins)
bin_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2

scores_list, weights_list = get_weights(accepted_scores, log_Zs)
histogram_heights, _ = np.histogram(scores_list, weights=weights_list, bins=bins, density=True)
unbiased_histogram_heights, _ = np.histogram(unbiased_samples_for_histogram, bins=bins, density=True)

print("\nMBAR Histogram (first 20 bins):")
for i in range(min(20, len(bin_centers))):
    print(f"  Bin {bin_centers[i]:.2f}: MBAR={histogram_heights[i]:.6e}, Direct={unbiased_histogram_heights[i]:.6e}")

print("\nMBAR Histogram (last 20 bins):")
for i in range(max(0, len(bin_centers)-20), len(bin_centers)):
    print(f"  Bin {bin_centers[i]:.2f}: MBAR={histogram_heights[i]:.6e}, Direct={unbiased_histogram_heights[i]:.6e}")

# Find minimum non-zero MBAR density (the Minimum Accessible Probability Density)
nonzero_mask = histogram_heights > 0
if np.any(nonzero_mask):
    min_density = np.min(histogram_heights[nonzero_mask])
    min_bin_idx = np.where(histogram_heights == min_density)[0][0]
    print(f"\nMinimum MBAR density: {min_density:.6e} at bin center {bin_centers[min_bin_idx]:.2f}")
else:
    print("\nNo non-zero MBAR densities!")

# Find density at ARI ≈ 14 (the tail region)
ari_14_idx = np.argmin(np.abs(bin_centers - 14))
print(f"MBAR density at ARI≈{bin_centers[ari_14_idx]:.2f}: {histogram_heights[ari_14_idx]:.6e}")
print(f"Direct density at ARI≈{bin_centers[ari_14_idx]:.2f}: {unbiased_histogram_heights[ari_14_idx]:.6e}")

# Quick bootstrap (5 iterations for validation)
print("\n" + "=" * 60)
print("Quick bootstrap (5 iterations) for validation...")
print("=" * 60)
unbiased_scores_boot = subsample_scores(unbiased_scores, unbiased_samples_for_biased)
t0 = time.time()
lower_bound, upper_bound, boot_heights = bootstrap_conf_interval(
    trajectories, unbiased_scores_boot, bins,
    num_bootstraps=5, lower_idx=1, upper_idx=3
)
print(f"5 bootstraps took {time.time() - t0:.1f}s")

# Compute CI half-widths
ci_half_width = (upper_bound - lower_bound) / 2
relative_ci_half_width = ci_half_width / histogram_heights

print("\nRelative CI half-widths (MBAR, key bins):")
for idx in [0, 10, 20, 30, 40, 50, 60, 70]:
    if idx < len(bin_centers):
        print(f"  ARI≈{bin_centers[idx]:.2f}: height={histogram_heights[idx]:.6e}, "
              f"CI_hw={ci_half_width[idx]:.6e}, rel_CI_hw={relative_ci_half_width[idx]:.6e}")

# Find minimum relative CI half-width (at the peak region)
peak_region = (bin_centers >= 0) & (bin_centers <= 5)
if np.any(peak_region):
    min_rel_ci = np.min(relative_ci_half_width[peak_region])
    min_idx = np.where(peak_region)[0][np.argmin(relative_ci_half_width[peak_region])]
    print(f"\nMinimum relative CI half-width (ARI 0-5): {min_rel_ci:.6e} at ARI≈{bin_centers[min_idx]:.2f}")

print("\nPipeline validation complete!")
