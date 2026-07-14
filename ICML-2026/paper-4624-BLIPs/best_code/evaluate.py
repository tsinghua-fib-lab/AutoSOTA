"""
Evaluation script for BLIP N-body experiment.
Computes MSE, NLL, and CRPS from saved test predictions.
Usage: python3 evaluate.py [model_id] [seed1,seed2,...]
"""
import torch
import numpy as np
from scipy.stats import norm
import sys
import os

def compute_metrics(stats_file):
    """Compute MSE, NLL, CRPS from saved test stats."""
    stats = torch.load(stats_file, map_location='cpu', weights_only=False)

    true_pos = stats['true_positions']  # [N, n_nodes, 3]
    mean_pos = stats['mean_positions']  # [N, n_nodes, 3]
    var_pos = stats.get('var_positions', None)  # [N, n_nodes, 3] or None

    # Flatten
    true_flat = true_pos.reshape(-1).float()
    mean_flat = mean_pos.reshape(-1).float()
    squared_errors = (true_flat - mean_flat) ** 2

    # MSE
    mse = squared_errors.mean().item()
    results = {'MSE': mse, 'MSE_x10^{-1}': mse * 10}

    if var_pos is not None:
        var_flat = var_pos.reshape(-1).float()
        var_flat = torch.clamp(var_flat, min=1e-12)

        # NLL
        log_var = torch.log(2 * np.pi * var_flat)
        precision_weighted_error = squared_errors / var_flat
        nll_per_element = 0.5 * (log_var + precision_weighted_error)
        nll = nll_per_element.mean().item()
        results['NLL'] = nll

        # CRPS for Gaussian
        std_flat = torch.sqrt(var_flat)
        z_np = ((true_flat - mean_flat) / std_flat).numpy()
        sigma_np = std_flat.numpy()

        phi = norm.pdf(z_np)
        Phi = norm.cdf(z_np)
        crps_per_element = sigma_np * (
            z_np * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi)
        )
        crps = crps_per_element.mean()
        results['CRPS'] = float(crps)
    else:
        results['NLL'] = None
        results['CRPS'] = None

    return results


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "BayesGNN"
    seeds_str = sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3"
    seeds = [int(s.strip()) for s in seeds_str.split(",")]

    all_results = {'MSE': [], 'MSE_x10^{-1}': [], 'NLL': [], 'CRPS': []}

    for seed in seeds:
        stats_file = f"/repo/nbody_results_{model_id}_{seed}.pt"
        if not os.path.exists(stats_file):
            print(f"WARNING: {stats_file} not found, skipping seed {seed}")
            continue

        metrics = compute_metrics(stats_file)
        nll_str = f"{metrics['NLL']:.4f}" if metrics['NLL'] is not None else "N/A"
        crps_str = f"{metrics['CRPS']:.6f}" if metrics['CRPS'] is not None else "N/A"
        print(f"Seed {seed}: MSE={metrics['MSE']:.6f}, MSE×10⁻¹={metrics['MSE_x10^{-1}']:.4f}, "
              f"NLL={nll_str}, CRPS={crps_str}")

        for k in all_results:
            if metrics.get(k) is not None:
                all_results[k].append(metrics[k])

    n_seeds = len(all_results['MSE'])
    if n_seeds == 0:
        print("\nNo results found!")
        return

    print(f"\n{'='*60}")
    print(f"Summary over {n_seeds} seeds:")
    for metric_name in ['MSE_x10^{-1}', 'NLL', 'CRPS']:
        values = all_results[metric_name]
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")

    print(f"\n{'='*60}")
    print("Paper targets for BLIP BayesGNN on N-body:")
    print("  MSE×10⁻¹: 0.092 ± 0.004  (CI [0.088, 0.096])")
    print("  NLL:     -9.03 ± 0.50  (CI [-9.53, -8.53])")
    print("  CRPS:     0.032 ± 0.001  (CI [0.031, 0.033])")

    mse_values = all_results['MSE_x10^{-1}']
    if mse_values:
        mse_mean = np.mean(mse_values)
        mse_in = 0.088 <= mse_mean <= 0.096
        nll_values = all_results['NLL']
        nll_mean = np.mean(nll_values) if nll_values else None
        nll_in = nll_mean is not None and (-9.53 <= nll_mean <= -8.53)
        crps_values = all_results['CRPS']
        crps_mean = np.mean(crps_values) if crps_values else None
        crps_in = crps_mean is not None and (0.031 <= crps_mean <= 0.033)

        print(f"\nResults check:")
        print(f"  MSE×10⁻¹ = {mse_mean:.4f} -> {'✓ WITHIN' if mse_in else '✗ OUTSIDE'} CI")
        if nll_mean is not None:
            print(f"  NLL = {nll_mean:.4f} -> {'✓ WITHIN' if nll_in else '✗ OUTSIDE'} CI")
        if crps_mean is not None:
            print(f"  CRPS = {crps_mean:.6f} -> {'✓ WITHIN' if crps_in else '✗ OUTSIDE'} CI")


if __name__ == "__main__":
    main()
