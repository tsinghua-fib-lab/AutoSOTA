#!/usr/bin/env python3
"""SBC Rank Histograms and Coverage Curves for ICML Rebuttal.

Runs SBC and empirical coverage diagnostics for all methods on benchmark tasks.
Uses the DGP (y, not x) to evaluate calibration under misspecification.

For tasks with analytical posteriors (Gaussian), uses the true p(θ|y).
For other tasks, uses SBC with the DGP simulator.

Addresses Reviewers xKrR (Q4), Qmdc (Point 4), uvU7.

Usage:
    python scripts/run_sbc_coverage.py --experiment comparison_benchmark
    python scripts/run_sbc_coverage.py --experiment comparison_benchmark --tasks high_dim_gaussian
    python scripts/run_sbc_coverage.py --experiment comparison_benchmark --num_sims 500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpointing import CheckpointManager, ResultsPath  # noqa: E402
from simulator import get_simulator  # noqa: E402
from ropefm.validation.base import compute_ranks, compute_credible_interval_coverage  # noqa: E402
from utils.metrics import acauc  # noqa: E402


# Methods to evaluate
METHOD_DISPLAY = {
    "npe": "NPE",
    "fm_post_transform": "FMCPE",
    "mf_npe": "MF-NPE",
    "rope": "RoPE",
    "dpe": "DPE",
}

METHOD_COLORS = {
    "NPE": "#2ca02c",
    "FMCPE": "#1f77b4",
    "MF-NPE": "#ff7f0e",
    "RoPE": "#8c564b",
    "DPE": "#9467bd",
}


def sample_from_model(model, y_obs, n_samples, device, batch_size=50):
    """Sample from model in batches, returning (n_samples, n_obs, theta_dim)."""
    n_obs = y_obs.shape[0]
    all_samples = []

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        y_batch = y_obs[start:end].to(device)
        with torch.no_grad():
            samples = model.sample(y_batch, n_samples, device)
        # Ensure shape is (n_samples, batch_size, theta_dim)
        if samples.ndim == 3:
            if samples.shape[0] != n_samples:
                samples = samples.permute(1, 0, 2)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=1)  # (n_samples, n_obs, theta_dim)


def _get_theta_y(simulator, num_sims, test_data=None):
    """Get (theta_true, y_sim) for SBC/coverage.

    For generative simulators: sample fresh from prior + true DGP.
    For file-based simulators (callable_dgp=False): use saved test set
    to avoid overlap with calibration/training data.
    """
    if hasattr(simulator, "callable_dgp") and not simulator.callable_dgp:
        if test_data is None:
            raise ValueError("test_data required for file-based simulator")
        theta = test_data["theta"]
        y = test_data["y"]
        # Subsample if needed (test set may be larger than num_sims)
        if len(theta) > num_sims:
            idx = torch.randperm(len(theta))[:num_sims]
            theta, y = theta[idx], y[idx]
        return theta, y
    else:
        theta_true = simulator.sample_prior(num_sims)
        dgp = simulator.get_simulator(misspecified=False)
        y_sim = dgp(theta_true)
        return theta_true, y_sim


def run_sbc_dgp(
    model,
    simulator,
    num_sims: int,
    num_samples: int,
    device: torch.device,
    batch_size: int = 50,
    test_data=None,
):
    """Run SBC using the DGP (y, not x).

    Generates (theta, y) from the DGP and computes ranks of theta
    in posterior samples from the model.

    Returns:
        ranks: (num_sims, theta_dim) - ranks of true theta in posterior samples
    """
    theta_true, y_sim = _get_theta_y(simulator, num_sims, test_data)

    all_ranks = []
    for start in tqdm(range(0, num_sims, batch_size), desc="SBC"):
        end = min(start + batch_size, num_sims)
        y_batch = y_sim[start:end].to(device)
        theta_batch = theta_true[start:end]

        with torch.no_grad():
            samples = model.sample(y_batch, num_samples, device)

        # Ensure shape (num_samples, batch_size, theta_dim)
        if samples.ndim == 3 and samples.shape[0] != num_samples:
            samples = samples.permute(1, 0, 2)

        ranks = compute_ranks(theta_batch.to(samples.device), samples)
        all_ranks.append(ranks.cpu())

    return torch.cat(all_ranks, dim=0)


def run_coverage_dgp(
    model,
    simulator,
    num_sims: int,
    num_samples: int,
    device: torch.device,
    alpha_levels: List[float] = None,
    batch_size: int = 50,
    test_data=None,
):
    """Run coverage analysis using the DGP (y, not x).

    Returns:
        dict with empirical_coverage, acauc, mce
    """
    if alpha_levels is None:
        alpha_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    theta_true, y_sim = _get_theta_y(simulator, num_sims, test_data)

    all_samples = []
    for start in tqdm(range(0, num_sims, batch_size), desc="Coverage"):
        end = min(start + batch_size, num_sims)
        y_batch = y_sim[start:end].to(device)

        with torch.no_grad():
            samples = model.sample(y_batch, num_samples, device)

        if samples.ndim == 3 and samples.shape[0] != num_samples:
            samples = samples.permute(1, 0, 2)

        all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=1)  # (num_samples, num_sims, theta_dim)

    # Compute coverage
    empirical_coverage = {}
    for alpha in alpha_levels:
        cov = compute_credible_interval_coverage(theta_true, all_samples, alpha)
        empirical_coverage[alpha] = cov

    # ACAUC
    acauc_val = acauc(theta_true, all_samples)

    # Mean Calibration Error
    mce = np.mean([abs(empirical_coverage[a] - a) for a in alpha_levels])

    return {
        "empirical_coverage": empirical_coverage,
        "acauc": acauc_val,
        "mce": mce,
    }


def plot_sbc_histograms(
    results: Dict[str, torch.Tensor],
    num_samples: int,
    num_sims: int,
    save_path: Path,
    title: str = "",
):
    """Plot SBC rank histograms for all methods side by side.

    Args:
        results: {method_display_name: ranks tensor (num_sims, theta_dim)}
        num_samples: Number of posterior samples (determines max rank)
        num_sims: Number of SBC simulations
        save_path: Path to save the plot
        title: Plot title
    """
    methods = list(results.keys())
    n_methods = len(methods)
    first_ranks = list(results.values())[0]
    theta_dim = first_ranks.shape[1]

    fig, axes = plt.subplots(
        theta_dim, n_methods,
        figsize=(3.5 * n_methods, 3 * theta_dim),
        squeeze=False,
    )

    num_bins = min(20, num_samples)

    for col, method in enumerate(methods):
        ranks = results[method]
        normalized = ranks.float() / num_samples

        for row in range(theta_dim):
            ax = axes[row, col]
            dim_ranks = normalized[:, row].numpy()

            color = METHOD_COLORS.get(method, "#333333")
            ax.hist(dim_ranks, bins=num_bins, range=(0, 1),
                    density=True, alpha=0.7, color=color)

            # Expected uniform
            ax.axhline(y=1.0, color="black", linestyle="--", lw=1)

            # 95% CI band
            n_per_bin = num_sims / num_bins
            std = np.sqrt(n_per_bin * (1 - 1/num_bins)) / num_sims * num_bins
            ax.axhspan(1 - 1.96 * std, 1 + 1.96 * std,
                       alpha=0.15, color="gray")

            if row == 0:
                ax.set_title(method, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"$\\theta_{{{row+1}}}$")
            if row == theta_dim - 1:
                ax.set_xlabel("Rank")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_coverage_curves(
    results: Dict[str, Dict],
    save_path: Path,
    title: str = "",
):
    """Plot expected vs empirical coverage curves for all methods.

    Args:
        results: {method_display_name: {"empirical_coverage": {alpha: cov}, ...}}
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=0)

    for method, res in results.items():
        cov = res["empirical_coverage"]
        alphas = sorted(cov.keys())
        empirical = [cov[a] for a in alphas]
        color = METHOD_COLORS.get(method, None)

        label = method

        ax.plot(alphas, empirical, "o-", label=label, markersize=5,
                color=color, alpha=0.8)

    ax.set_xlabel("Expected Coverage ($\\alpha$)", fontsize=11)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    if title:
        ax.set_title(title, fontsize=13)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def load_simulator_from_task(task: str):
    """Load simulator from task config."""
    from omegaconf import OmegaConf
    task_cfg_path = PROJECT_ROOT / "configs" / "task" / f"{task}.yaml"
    task_cfg = OmegaConf.load(task_cfg_path)
    return get_simulator(
        task_cfg.simulator.name,
        **OmegaConf.to_container(task_cfg.simulator.params, resolve=True)
    )


def run_diagnostics(
    experiment: str,
    results_dir: str = "results",
    tasks: Optional[List[str]] = None,
    ncals: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    num_sims: int = 500,
    num_samples: int = 500,
    variant_name: str = "fm_pt_base",
    output_dir: Optional[str] = None,
):
    """Run SBC and coverage diagnostics for all methods."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = ResultsPath(Path(results_dir), experiment)
    cm = CheckpointManager(paths)

    if output_dir is None:
        output_dir = Path(results_dir) / experiment / "rebuttal_plots" / "sbc_coverage"
    else:
        output_dir = Path(output_dir)

    if tasks is None:
        tasks = ["high_dim_gaussian", "pendulum", "wind_tunnel", "light_tunnel"]
    if ncals is None:
        ncals = [200]
    if seeds is None:
        seeds = [33]

    all_results = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        try:
            simulator = load_simulator_from_task(task)
        except Exception as e:
            print(f"  Could not load simulator: {e}")
            continue

        # Load test set once per task (used for file-based simulators)
        test_data = cm.load_shared_test_data(task)
        if test_data is not None:
            print(f"  Loaded test set: {test_data['theta'].shape[0]} samples")
        else:
            print("  No test set found; will generate fresh samples")

        for seed in seeds:
            for ncal in ncals:
                print(f"\n  seed={seed}, ncal={ncal}")
                task_key = f"{task}/seed{seed}_ncal{ncal}"

                sbc_results = {}
                coverage_results = {}

                # Load models
                models = {}

                # NPE (uncorrected baseline)
                npe_model = cm.load_shared_simulation_model(task, "npe", device)
                if npe_model is not None:
                    models["NPE"] = npe_model

                # FMCPE (corrected) — try requested variant, fall back to fm_pt_fmpe
                fm_pt = cm.load_variant_calibration_model(
                    variant_name, task, seed, "fm_post_transform", ncal, device
                )
                if fm_pt is None and variant_name != "fm_pt_fmpe":
                    fm_pt = cm.load_variant_calibration_model(
                        "fm_pt_fmpe", task, seed, "fm_post_transform", ncal, device
                    )
                if fm_pt is not None:
                    models["FMCPE"] = fm_pt

                # MF-NPE
                mf_npe = cm.load_shared_baseline(task, seed, "mf_npe", ncal, device)
                if mf_npe is not None:
                    models["MF-NPE"] = mf_npe

                # RoPE (limit to 1000 simulations to avoid OOM)
                rope_model = cm.load_shared_baseline(task, seed, "rope", ncal, device)
                if rope_model is not None:
                    sim_data = cm.load_shared_simulations(task)
                    if sim_data is not None:
                        n_rope = min(1000, sim_data["x"].shape[0])
                        rope_model.set_sim_data(sim_data["x"][:n_rope], sim_data["theta"][:n_rope])
                    models["RoPE"] = rope_model

                # DPE
                dpe_model = cm.load_shared_baseline(task, seed, "dpe", ncal, device)
                if dpe_model is not None:
                    models["DPE"] = dpe_model

                if not models:
                    print("    No models loaded, skipping")
                    continue

                # Run SBC and Coverage for each model
                for method_name, model in models.items():
                    print(f"\n    {method_name}:")

                    # SBC
                    try:
                        ranks = run_sbc_dgp(
                            model, simulator, num_sims, num_samples, device,
                            test_data=test_data,
                        )
                        sbc_results[method_name] = ranks
                        print(f"      SBC done, ranks shape: {ranks.shape}")
                    except Exception as e:
                        print(f"      SBC failed: {e}")

                    # Coverage
                    try:
                        cov = run_coverage_dgp(
                            model, simulator, num_sims, num_samples, device,
                            test_data=test_data,
                        )
                        coverage_results[method_name] = cov
                        print(f"      Coverage done, ACAUC={cov['acauc']:.4f}, MCE={cov['mce']:.4f}")
                    except Exception as e:
                        print(f"      Coverage failed: {e}")

                # Plot SBC histograms
                if sbc_results:
                    plot_sbc_histograms(
                        sbc_results,
                        num_samples=num_samples,
                        num_sims=num_sims,
                        save_path=output_dir / task / f"sbc_seed{seed}_ncal{ncal}.pdf",
                        title=f"{task} — SBC Rank Histograms (N_cal={ncal})",
                    )

                # Plot coverage curves
                if coverage_results:
                    plot_coverage_curves(
                        coverage_results,
                        save_path=output_dir / task / f"coverage_seed{seed}_ncal{ncal}.pdf",
                        title=f"{task} — Coverage (N_cal={ncal})",
                    )

                # Save metrics
                metrics_out = {}
                for method_name in coverage_results:
                    metrics_out[method_name] = {
                        "acauc": coverage_results[method_name]["acauc"],
                        "mce": coverage_results[method_name]["mce"],
                        "empirical_coverage": {
                            str(k): v
                            for k, v in coverage_results[method_name]["empirical_coverage"].items()
                        },
                    }

                metrics_path = output_dir / task / f"metrics_seed{seed}_ncal{ncal}.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump(metrics_out, f, indent=2)
                print(f"\n  Saved metrics: {metrics_path}")

                all_results[task_key] = metrics_out

    # Summary table
    print("\n\n" + "="*80)
    print("SUMMARY: ACAUC and MCE across all methods and tasks")
    print("="*80)
    for task_key, metrics in all_results.items():
        print(f"\n{task_key}:")
        for method, m in metrics.items():
            print(f"  {method:15s}  ACAUC={m['acauc']:.4f}  MCE={m['mce']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SBC and Coverage Diagnostics")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--ncals", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num_sims", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--variant", default="fm_pt_base")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_diagnostics(
        experiment=args.experiment,
        results_dir=args.results_dir,
        tasks=args.tasks,
        ncals=args.ncals,
        seeds=args.seeds,
        num_sims=args.num_sims,
        num_samples=args.num_samples,
        variant_name=args.variant,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
