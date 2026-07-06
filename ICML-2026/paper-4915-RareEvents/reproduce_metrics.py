#!/usr/bin/env python3
"""Reproduce paper metrics for ARI observable.

Metrics:
1. Minimum Accessible Probability Density: minimum MBAR density in tails where direct sampling has zero samples
2. Relative CI Half Width: CI^(1/2)/h for MBAR and direct histograms at the peak (ARI 0-5)
"""
import json
import sys
import os
import time
import warnings
import numpy as np
import pymbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join("data", "paper")
OBS_NAME = "ari"
OUTPUT_DIR = "reproduction_output"

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
        raise Exception(f"upper_idx {upper_idx} must be <= num_bootstraps - 1 = {num_bootstraps - 1}")
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

def wilson_interval(data, bins, zscore, density=True):
    n = len(data)
    n_s, _ = np.histogram(data, bins)
    n_f = n - n_s
    p = (n_s + 0.5 * (zscore**2)) / (n + zscore**2)
    diff = (zscore / (n + (zscore**2))) * np.sqrt(((n_s * n_f) / n) + (zscore**2 / 4))
    if density:
        upper = (p - diff) / (bins[1:] - bins[:-1])
        lower = (p + diff) / (bins[1:] - bins[:-1])
        return lower, upper
    else:
        return p - diff, p + diff

# ------------------------------------------------------------
# Main reproduction
# ------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("REPRODUCTION: Rare Event Analysis of LLMs - ARI Observable")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1/6] Loading ARI data...")
    trajs_dir = os.path.join(DATA_DIR, f"{OBS_NAME}_trajectories.json")
    unbiased_dir = os.path.join(DATA_DIR, f"{OBS_NAME}_unbiased_samples.json")

    with open(trajs_dir, "r") as f:
        trajectories_saved = json.load(f)
        trajectories = Trajectories(trajectories_saved[0], trajectories_saved[1], trajectories_saved[2])

    with open(unbiased_dir, "r") as f:
        unbiased_scores_list = json.load(f)
        unbiased_scores = Scores({0: unbiased_scores_list})

    print(f"  Annealing schedules: {len(trajectories.trajectories)}")
    print(f"  Trajectories per schedule: {len(trajectories.trajectories[0])}")
    print(f"  Steps per bias: {trajectories.steps_per_bias}")
    print(f"  Biases: {trajectories.get_all_biases()}")
    print(f"  Unbiased (direct) samples: {len(unbiased_scores_list)}")

    # ---- Preprocessing ----
    print("\n[2/6] Preprocessing (burn-in + Gelman-Rubin)...")
    t0 = time.time()

    trajectories_burnin = apply_burnin(trajectories, 0.1)
    accepted_scores, grs = gelman_rubin(trajectories_burnin)
    print(f"  GR values: {[f'{g:.4f}' for g in grs]}")
    print(f"  Accepted biases: {accepted_scores.get_all_biases()}")

    # Add unbiased samples
    num_trajs_per_schedule = len(trajectories.trajectories[0])
    num_biases_per_anneal = len(trajectories.biases[0])
    unbiased_samples_for_biased = int(trajectories.steps_per_bias * num_trajs_per_schedule / 2)
    accepted_scores += subsample_scores(unbiased_scores, num_samples=unbiased_samples_for_biased)

    num_samples_for_unbiased = (trajectories.steps_per_bias * num_trajs_per_schedule *
                                num_biases_per_anneal // 2 + unbiased_samples_for_biased)
    unbiased_samples_for_histogram = unbiased_scores_list[:num_samples_for_unbiased]

    print(f"  Unbiased for biased histogram: {unbiased_samples_for_biased}")
    print(f"  Direct histogram samples: {len(unbiased_samples_for_histogram)}")
    print(f"  Preprocessing took {time.time() - t0:.1f}s")

    # ---- MBAR ----
    print("\n[3/6] Computing MBAR...")
    t0 = time.time()
    log_Zs = mbar_fn(accepted_scores)
    print(f"  log_Zs: {[f'{z:.4f}' for z in log_Zs]}")
    print(f"  MBAR took {time.time() - t0:.1f}s")

    # ---- Histograms ----
    print("\n[4/6] Computing histograms...")
    # Paper settings for ARI: lower=-8, upper=15, num_bins=80
    lower, upper, num_bins = (-8, 15, 80)
    bins = np.linspace(lower, upper, num_bins)
    bin_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    bin_width = (upper - lower) / (num_bins - 1)

    scores_list, weights_list = get_weights(accepted_scores, log_Zs)
    histogram_heights, _ = np.histogram(scores_list, weights=weights_list, bins=bins, density=True)
    unbiased_histogram_heights, _ = np.histogram(unbiased_samples_for_histogram, bins=bins, density=True)

    print(f"  Bin count: {num_bins - 1}, Bin width: {bin_width:.4f}")

    # ---- Bootstrap CI ----
    num_bootstraps = 100
    lower_idx = 2    # For 100 bootstraps, index 2 = 3rd percentile
    upper_idx = 97   # index 97 = 98th percentile -> 96% CI

    print(f"\n[5/6] Bootstrap CI ({num_bootstraps} iterations, 96% CI)...")
    print(f"  Estimated time: ~{num_bootstraps * 170 / 60:.0f} minutes")
    t0 = time.time()

    unbiased_scores_boot = subsample_scores(unbiased_scores, unbiased_samples_for_biased)
    lower_bound, upper_bound, boot_heights = bootstrap_conf_interval(
        trajectories, unbiased_scores_boot, bins,
        num_bootstraps=num_bootstraps,
        lower_idx=lower_idx,
        upper_idx=upper_idx
    )
    print(f"  Bootstrap took {time.time() - t0:.1f}s ({(time.time() - t0)/60:.1f} min)")

    # ---- Compute metrics ----
    print("\n[6/6] Computing metrics...")

    # Metric 1: Minimum Accessible Probability Density
    # Find minimum non-zero MBAR density in the right tail (ARI > 0 where direct has zero counts)
    right_tail_mask = (bin_centers > 5) & (histogram_heights > 0) & (unbiased_histogram_heights == 0)
    if np.any(right_tail_mask):
        min_density_beyond_direct = np.min(histogram_heights[right_tail_mask])
        min_idx = np.where(right_tail_mask)[0][np.argmin(histogram_heights[right_tail_mask])]
    else:
        nonzero_mask = histogram_heights > 0
        min_density_beyond_direct = np.min(histogram_heights[nonzero_mask])
        min_idx = np.where(histogram_heights == min_density_beyond_direct)[0][0]

    # Also get density at ARI≈14 specifically
    ari_14_idx = np.argmin(np.abs(bin_centers - 14))
    density_at_ari_14 = histogram_heights[ari_14_idx]

    print(f"\n  Metric 1: Minimum Accessible Probability Density")
    print(f"    Minimum MBAR density in right tail (ARI>5, direct=0): {min_density_beyond_direct:.6e} at ARI≈{bin_centers[min_idx]:.2f}")
    print(f"    MBAR density at ARI≈{bin_centers[ari_14_idx]:.2f}: {density_at_ari_14:.6e}")
    print(f"    Direct density at ARI≈{bin_centers[ari_14_idx]:.2f}: {unbiased_histogram_heights[ari_14_idx]:.6e}")
    print(f"    Paper value: ~1e-10 (from Fig 3b log-scale axes)")
    print(f"    Baseline (direct sampling limit): ~1e-6")

    # Metric 2: Relative CI Half Width at peak region (ARI 0-5)
    ci_half_width = (upper_bound - lower_bound) / 2
    relative_ci_half_width = ci_half_width / np.where(histogram_heights > 0, histogram_heights, np.inf)

    peak_mask = (bin_centers >= 0) & (bin_centers <= 5) & (histogram_heights > 0)
    if np.any(peak_mask):
        min_rel_ci_mbar = np.min(relative_ci_half_width[peak_mask])
        min_rel_ci_idx = np.where(peak_mask)[0][np.argmin(relative_ci_half_width[peak_mask])]
        mean_rel_ci_mbar = np.mean(relative_ci_half_width[peak_mask])
    else:
        min_rel_ci_mbar = np.nan
        mean_rel_ci_mbar = np.nan

    # Wilson interval for direct histogram
    zscore = 2.0537  # 96% confidence
    lower_wilson, upper_wilson = wilson_interval(unbiased_samples_for_histogram, bins, zscore)
    ci_hw_direct = (upper_wilson - lower_wilson) / 2
    rel_ci_hw_direct = ci_hw_direct / np.where(unbiased_histogram_heights > 0, unbiased_histogram_heights, np.inf)

    if np.any(peak_mask):
        min_rel_ci_direct = np.min(rel_ci_hw_direct[peak_mask])
        min_rel_ci_direct_idx = np.where(peak_mask)[0][np.argmin(rel_ci_hw_direct[peak_mask])]
        mean_rel_ci_direct = np.mean(rel_ci_hw_direct[peak_mask])
    else:
        min_rel_ci_direct = np.nan
        mean_rel_ci_direct = np.nan

    print(f"\n  Metric 2: Relative CI Half Width at peak (ARI 0-5)")
    print(f"    MBAR min relative CI half-width: {min_rel_ci_mbar:.6e} at ARI≈{bin_centers[min_rel_ci_idx]:.2f}")
    print(f"    MBAR mean relative CI half-width (peak): {mean_rel_ci_mbar:.6e}")
    print(f"    Direct min relative CI half-width: {min_rel_ci_direct:.6e}")
    print(f"    Direct mean relative CI half-width (peak): {mean_rel_ci_direct:.6e}")
    print(f"    Paper value (MBAR): ~0.003 (from Fig 4a)")
    print(f"    Paper value (direct): ~0.03 (from Fig 4a)")

    # ---- Save results ----
    results = {
        "paper_id": 4915,
        "observable": OBS_NAME,
        "model": "TinyStories-8M",
        "prompt_tokens": 16,
        "completion_tokens": 100,
        "temperature": 1.0,
        "decoding": "ancestral",
        "ari_cap": 15,
        "bins": {"lower": lower, "upper": upper, "num_bins": num_bins, "bin_width": bin_width},
        "num_bootstraps": num_bootstraps,
        "ci_level": 0.96,
        "ci_lower_idx": lower_idx,
        "ci_upper_idx": upper_idx,
        "metrics": {
            "minimum_accessible_probability_density": float(min_density_beyond_direct),
            "minimum_accessible_probability_density_bin_center": float(bin_centers[min_idx]),
            "density_at_ari_14": float(density_at_ari_14),
            "density_at_ari_14_bin_center": float(bin_centers[ari_14_idx]),
            "mbar_min_relative_ci_half_width": float(min_rel_ci_mbar),
            "mbar_mean_relative_ci_half_width": float(mean_rel_ci_mbar),
            "mbar_min_rel_ci_hw_bin_center": float(bin_centers[min_rel_ci_idx]) if not np.isnan(min_rel_ci_mbar) else None,
            "direct_min_relative_ci_half_width": float(min_rel_ci_direct),
            "direct_mean_relative_ci_half_width": float(mean_rel_ci_direct),
        },
        "histogram_heights_mbar": histogram_heights.tolist(),
        "histogram_heights_direct": unbiased_histogram_heights.tolist(),
        "ci_lower_bound": lower_bound.tolist(),
        "ci_upper_bound": upper_bound.tolist(),
        "bin_centers": bin_centers.tolist(),
        "gelman_rubin_values": [float(g) for g in grs],
        "log_Zs": [float(z) for z in log_Zs],
        "num_samples": {
            "total_mbar": len(scores_list),
            "total_direct": len(unbiased_samples_for_histogram),
            "unbiased_for_biased": unbiased_samples_for_biased,
        }
    }

    output_path = os.path.join(OUTPUT_DIR, "reproduction_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ---- Generate plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig 3(b) equivalent: Density histogram
    ax = axes[0, 0]
    ax.step(bin_centers, histogram_heights, where="mid", label="MBAR", color="C0")
    ax.fill_between(bin_centers, lower_bound, upper_bound, step="mid",
                    label="96% CI", color="C0", alpha=0.3)
    ax.step(bin_centers, unbiased_histogram_heights, where="mid", label="Direct", color="C1")
    ax.set_xlabel("ARI")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("Fig 3(b): ARI Distribution (MBAR vs Direct)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoom on right tail
    ax = axes[0, 1]
    tail_mask = bin_centers > 8
    ax.step(bin_centers[tail_mask], histogram_heights[tail_mask], where="mid", label="MBAR", color="C0")
    ax.fill_between(bin_centers[tail_mask], lower_bound[tail_mask], upper_bound[tail_mask],
                    step="mid", label="96% CI", color="C0", alpha=0.3)
    ax.step(bin_centers[tail_mask], unbiased_histogram_heights[tail_mask], where="mid", label="Direct", color="C1")
    ax.set_xlabel("ARI")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("Right Tail Zoom (ARI > 8)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative CI half-width (Fig 4a)
    ax = axes[1, 0]
    valid_mask = histogram_heights > 0
    ax.plot(bin_centers[valid_mask], relative_ci_half_width[valid_mask], '.-', label="MBAR", color="C0")
    valid_direct = unbiased_histogram_heights > 0
    ax.plot(bin_centers[valid_direct], rel_ci_hw_direct[valid_direct], '.-', label="Direct", color="C1")
    ax.set_xlabel("ARI")
    ax.set_ylabel("Relative CI Half-Width")
    ax.set_yscale("log")
    ax.set_title("Fig 4(a): Relative CI Half-Width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(5, color='grey', linestyle='--', alpha=0.5)

    # GR statistics
    ax = axes[1, 1]
    all_biases = trajectories.get_all_biases()
    ax.plot(all_biases, np.array(grs) - 1, 'o-', label="GR - 1")
    ax.axhline(0.1, linestyle='--', color='grey', label="Cutoff (0.1)")
    ax.set_xlabel("Bias λ")
    ax.set_ylabel("GR - 1")
    ax.set_yscale("log")
    ax.set_title("Gelman-Rubin Convergence Diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "reproduction_plots.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")

    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
