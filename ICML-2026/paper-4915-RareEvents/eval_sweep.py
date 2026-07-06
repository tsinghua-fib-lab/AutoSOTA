#!/usr/bin/env python3
"""Parameter sweep: test burnin, GR cutoff, bin count/range combos with fast bootstrap."""
import json, os, time, warnings
from multiprocessing import Pool, cpu_count
import numpy as np

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import pymbar
import matplotlib
matplotlib.use("Agg")

DATA_DIR = os.path.join("data", "paper")
OBS_NAME = "ari"
NUM_WORKERS = 32
FAST_BOOTSTRAPS = 20

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
        all_biases, all_trajs_per_bias = [], []
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

def mbar_fn(scores):
    biases = scores.get_all_biases()
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
    return res

def get_weights(scores, log_Zs):
    scores_list, weights_list = [], []
    for i, bias in enumerate(scores.get_all_biases()):
        scores_list += scores.scores_dict[bias]
        weights_list += np.exp(log_Zs[i] + bias * np.array(scores.scores_dict[bias])).tolist()
    return scores_list, weights_list

def generate_trajectories_bootstrap(trajectories):
    new_trajectories = []
    for biases, trajs in zip(trajectories.biases, trajectories.trajectories):
        indices = np.random.randint(0, len(trajs), len(trajs))
        new_trajectories.append([trajs[i] for i in indices])
    return Trajectories(new_trajectories, biases, trajectories.steps_per_bias)

def _single_bootstrap(args):
    seed, trajs_burnin_data, unbiased_scores_list, num_unbiased_samples, bins = args
    np.random.seed(seed)
    trajs_burnin = Trajectories(trajs_burnin_data["trajectories"], trajs_burnin_data["biases"], trajs_burnin_data["steps_per_bias"])
    unbiased_scores = Scores({0: unbiased_scores_list})
    bootstrap_trajs = generate_trajectories_bootstrap(trajs_burnin)
    accepted_scores, _ = gelman_rubin(bootstrap_trajs)
    accepted_scores += subsample_scores(unbiased_scores, num_unbiased_samples, random=True)
    log_Zs = mbar_fn(accepted_scores)
    scores_list, weights_list = get_weights(accepted_scores, log_Zs)
    histogram_heights, _ = np.histogram(scores_list, weights=weights_list, bins=bins, density=True)
    return histogram_heights

def evaluate_config(trajs_burnin, unbiased_scores_list, config, base_seed=42):
    """Fast evaluation with reduced bootstraps."""
    burnin = config["burnin"]
    cutoff = config["gr_cutoff"]
    lower = config["lower"]
    upper = config["upper"]
    num_bins = config["num_bins"]
    n_boot = config.get("n_bootstraps", FAST_BOOTSTRAPS)

    bins = np.linspace(lower, upper, num_bins)
    bin_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2

    accepted_scores, _ = gelman_rubin(trajs_burnin, cutoff)
    num_trajs_per_schedule = len(trajs_burnin.trajectories[0])
    unbiased_samples_for_biased = int(trajs_burnin.steps_per_bias * num_trajs_per_schedule / 2)
    accepted_scores += subsample_scores(Scores({0: unbiased_scores_list}), num_samples=unbiased_samples_for_biased)

    log_Zs = mbar_fn(accepted_scores)
    scores_list, weights_list = get_weights(accepted_scores, log_Zs)
    histogram_heights, _ = np.histogram(scores_list, weights=weights_list, bins=bins, density=True)
    unbiased_histogram_heights, _ = np.histogram(unbiased_scores_list, bins=bins, density=True)

    trajs_burnin_data = {
        "trajectories": trajs_burnin.trajectories,
        "biases": trajs_burnin.biases,
        "steps_per_bias": trajs_burnin.steps_per_bias,
    }
    seeds = [base_seed + i for i in range(n_boot)]
    args_list = [(seeds[i], trajs_burnin_data, unbiased_scores_list, unbiased_samples_for_biased, bins) for i in range(n_boot)]

    with Pool(NUM_WORKERS) as pool:
        boot_heights_list = pool.map(_single_bootstrap, args_list)

    boot_histogram_heights = np.array(boot_heights_list)
    sorted_boot_heights = np.sort(boot_histogram_heights, axis=0)
    li = max(0, int(0.02 * n_boot))
    ui = min(n_boot - 1, int(0.98 * n_boot))
    lower_bound = sorted_boot_heights[li]
    upper_bound = sorted_boot_heights[ui]

    ci_half_width = (upper_bound - lower_bound) / 2
    relative_ci_hw_mbar = ci_half_width / np.where(histogram_heights > 0, histogram_heights, np.inf)

    peak_mask = (bin_centers >= 0) & (bin_centers <= 5) & (histogram_heights > 0)
    if np.any(peak_mask):
        min_rel_ci_mbar = np.min(relative_ci_hw_mbar[peak_mask])
    else:
        min_rel_ci_mbar = np.nan

    ari_14_idx = np.argmin(np.abs(bin_centers - 14))
    density_at_ari_14 = histogram_heights[ari_14_idx]

    right_tail_mask = (bin_centers > 5) & (histogram_heights > 0) & (unbiased_histogram_heights == 0)
    if np.any(right_tail_mask):
        min_density_tail = np.min(histogram_heights[right_tail_mask])
    else:
        min_density_tail = np.nan

    n_accepted = len(accepted_scores.get_all_biases()) - 1  # minus unbiased

    return {
        "config": config,
        "min_rel_ci_hw": float(min_rel_ci_mbar) if not np.isnan(min_rel_ci_mbar) else None,
        "density_at_14": float(density_at_ari_14),
        "min_density_tail": float(min_density_tail) if not np.isnan(min_density_tail) else None,
        "n_accepted_biases": n_accepted,
    }

def main():
    print("=" * 70)
    print("PARAMETER SWEEP: fast eval with reduced bootstraps")
    print(f"Fast bootstraps: {FAST_BOOTSTRAPS}")
    print("=" * 70)

    # Load data
    trajs_dir = os.path.join(DATA_DIR, f"{OBS_NAME}_trajectories.json")
    unbiased_dir = os.path.join(DATA_DIR, f"{OBS_NAME}_unbiased_samples.json")
    with open(trajs_dir) as f:
        trajectories_saved = json.load(f)
        trajectories = Trajectories(trajectories_saved[0], trajectories_saved[1], trajectories_saved[2])
    with open(unbiased_dir) as f:
        unbiased_scores_list = json.load(f)

    # Define sweep parameters
    burnin_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    gr_cutoff_values = [1.05, 1.10, 1.15, 1.20, 1.30]
    bin_configs = [
        {"lower": -8, "upper": 15, "num_bins": 80},   # baseline
        {"lower": -8, "upper": 18, "num_bins": 100},
        {"lower": -10, "upper": 20, "num_bins": 120},
        {"lower": -8, "upper": 20, "num_bins": 150},
    ]

    # First: sweep burnin × GR cutoff with baseline bins
    print("\n--- Phase 1: Burnin + GR cutoff sweep ---")
    results = []
    for burnin in burnin_values:
        trajs_burnin = apply_burnin(trajectories, burnin)
        for cutoff in gr_cutoff_values:
            config = {
                "burnin": burnin, "gr_cutoff": cutoff,
                "lower": -8, "upper": 15, "num_bins": 80,
                "n_bootstraps": FAST_BOOTSTRAPS,
            }
            t0 = time.time()
            r = evaluate_config(trajs_burnin, unbiased_scores_list, config)
            elapsed = time.time() - t0
            r["elapsed"] = elapsed
            results.append(r)
            if r["min_rel_ci_hw"]:
                print(f"  burnin={burnin:.2f} cutoff={cutoff:.2f}: CI_hw={r[min_rel_ci_hw]:.6f} dens14={r[density_at_14]:.2e} biases={r[n_accepted_biases]} ({elapsed:.0f}s)")
            else:
                print(f"  burnin={burnin:.2f} cutoff={cutoff:.2f}: NO VALID PEAK biases={r[n_accepted_biases]} ({elapsed:.0f}s)")

    # Sort by CI half-width (primary metric)
    valid = [r for r in results if r["min_rel_ci_hw"] is not None]
    valid.sort(key=lambda r: r["min_rel_ci_hw"])
    best_burnin_gr = valid[0] if valid else None

    if best_burnin_gr:
        print(f"\nBest burnin/GR: burnin={best_burnin_gr[config][burnin]} cutoff={best_burnin_gr[config][gr_cutoff]}: CI_hw={best_burnin_gr[min_rel_ci_hw]:.6f}")

        # Phase 2: sweep bin configs with best burnin/GR
        print("\n--- Phase 2: Bin config sweep ---")
        best_burnin = best_burnin_gr["config"]["burnin"]
        best_cutoff = best_burnin_gr["config"]["gr_cutoff"]
        trajs_burnin = apply_burnin(trajectories, best_burnin)

        bin_results = []
        for bc in bin_configs:
            config = {
                "burnin": best_burnin, "gr_cutoff": best_cutoff,
                "lower": bc["lower"], "upper": bc["upper"], "num_bins": bc["num_bins"],
                "n_bootstraps": FAST_BOOTSTRAPS,
            }
            t0 = time.time()
            r = evaluate_config(trajs_burnin, unbiased_scores_list, config)
            elapsed = time.time() - t0
            r["elapsed"] = elapsed
            bin_results.append(r)
            print(f"  bins=[{bc[lower]},{bc[upper]}]{bc[num_bins]}: CI_hw={r[min_rel_ci_hw]:.6f} dens14={r[density_at_14]:.2e} ({elapsed:.0f}s)")

        # Save all results
        with open("reproduction_output/sweep_results.json", "w") as f:
            json.dump({"burnin_gr_sweep": results, "bin_sweep": bin_results}, f, indent=2)

        # Print top 3
        all_valid = [r for r in results + bin_results if r["min_rel_ci_hw"] is not None]
        all_valid.sort(key=lambda r: r["min_rel_ci_hw"])
        print("\n--- Top 5 Configurations ---")
        for i, r in enumerate(all_valid[:5]):
            c = r["config"]
            print(f"  #{i+1}: burnin={c[burnin]} cutoff={c[gr_cutoff]} bins=[{c[lower]},{c[upper]}]{c[num_bins]}: CI_hw={r[min_rel_ci_hw]:.6f} dens14={r[density_at_14]:.2e}")

        best = all_valid[0]
        print(f"\nBEST: burnin={best[config][burnin]} cutoff={best[config][gr_cutoff]} bins=[{best[config][lower]},{best[config][upper]}]{best[config][num_bins]}")
        print(f"  CI half-width: {best[min_rel_ci_hw]:.6f}")
        print(f"  Density at ARI=14: {best[density_at_14]:.2e}")

        # Write best config for full run
        with open("reproduction_output/best_config.json", "w") as f:
            json.dump(best["config"], f, indent=2)
    else:
        print("No valid configurations found!")
        with open("reproduction_output/sweep_results.json", "w") as f:
            json.dump({"burnin_gr_sweep": results}, f, indent=2)

if __name__ == "__main__":
    main()
