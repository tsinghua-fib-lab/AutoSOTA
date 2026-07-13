#!/usr/bin/env python3
"""Full Corner Plots for ICML Rebuttal.

Generates full D×D corner plots for all methods at multiple N_cal values.
- LightTunnel: 4×4 corner plot (R, G, B, α)
- Gaussian: 3×3 corner plot (θ₁, θ₂, θ₃)

Addresses Reviewer Qmdc (figure issues, Point 7).

Usage:
    python scripts/plot_corner_plots.py --experiment comparison_benchmark
    python scripts/plot_corner_plots.py --experiment comparison_benchmark --tasks light_tunnel
    python scripts/plot_corner_plots.py --experiment comparison_benchmark --tasks high_dim_gaussian --ncals 50 200 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpointing import CheckpointManager, ResultsPath  # noqa: E402
from simulator import get_simulator  # noqa: E402


# Task-specific parameter labels
TASK_PARAM_LABELS = {
    "light_tunnel": ["$R$", "$G$", "$B$", r"$\alpha$"],
    "high_dim_gaussian": [f"$\\theta_{{{i+1}}}$" for i in range(3)],
    "pure_gaussian": [f"$\\theta_{{{i+1}}}$" for i in range(2)],
    "no_misspec_gaussian": [f"$\\theta_{{{i+1}}}$" for i in range(3)],
    "pendulum": [r"$g$", r"$\ell$"],
    "wind_tunnel": [r"$v$"],
}

METHOD_DISPLAY = {
    "npe": "NPE",
    "fmpe": "FMPE",
    "fm_post_transform": "FMCPE",
    "mf_npe": "MF-NPE",
    "rope": "RoPE",
    "dpe": "DPE",
}

METHOD_COLORS = {
    "NPE": "#2ca02c",
    "FMPE": "#d62728",
    "FMCPE": "#1f77b4",
    "MF-NPE": "#ff7f0e",
    "RoPE": "#8c564b",
    "DPE": "#9467bd",
    "True posterior": "#7f7f7f",
}


def sample_from_model(model, y_obs, n_samples, device):
    """Sample from a posterior model, handling shape conventions."""
    with torch.no_grad():
        samples = model.sample(y_obs.to(device), n_samples, device).cpu()
        if samples.ndim == 3:
            if samples.shape[0] == n_samples:
                return samples
            else:
                return samples.permute(1, 0, 2)
        return samples


def make_full_corner(
    theta_true_val: np.ndarray,
    samples_dict: Dict[str, np.ndarray],
    param_labels: List[str],
    save_path: Path,
    title: str = "",
    max_samples: int = 2000,
):
    """Create a full D×D corner plot with multiple methods overlaid.

    Args:
        theta_true_val: True parameter values, shape (D,)
        samples_dict: {method_name: samples array (n_samples, D)}
        param_labels: List of parameter labels
        save_path: Where to save
        title: Plot title
        max_samples: Max samples per method
    """
    D = len(param_labels)

    dfs = []
    for name, samples in samples_dict.items():
        if len(samples) > max_samples:
            idx = np.random.choice(len(samples), max_samples, replace=False)
            samples = samples[idx]
        df = pd.DataFrame(samples, columns=param_labels)
        df["Method"] = name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    palette = {k: v for k, v in METHOD_COLORS.items() if k in samples_dict}

    g = sns.pairplot(
        df_all,
        hue="Method",
        diag_kind="kde",
        palette=palette if palette else None,
        plot_kws=dict(alpha=0.25, s=2, rasterized=True),
        diag_kws=dict(fill=True, alpha=0.4),
        corner=True,
    )

    # Mark true parameter values
    for i in range(D):
        g.axes[i, i].axvline(theta_true_val[i], color="black", linestyle="--", lw=1.5)
        for j in range(i):
            if g.axes[i, j] is not None:
                g.axes[i, j].scatter(
                    theta_true_val[j], theta_true_val[i],
                    color="black", marker="x", s=80, zorder=10,
                )

    # Clean up individual legends, add one global
    for ax in g.axes.flatten():
        if ax is not None and ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = [], []
    for name in samples_dict:
        color = METHOD_COLORS.get(name, "#333333")
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markersize=8))
        labels.append(name)
    # Add true value marker
    handles.append(plt.Line2D([0], [0], marker="x", color="w",
                              markeredgecolor="black", markersize=8, markeredgewidth=2))
    labels.append("True $\\theta$")

    g.fig.legend(handles, labels, loc="upper right", fontsize=9,
                 framealpha=0.9, bbox_to_anchor=(0.98, 0.98))

    if title:
        g.fig.suptitle(title, fontsize=13, y=1.02)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(g.figure)
    print(f"  Saved: {save_path}")


def run_corner_plots(
    experiment: str,
    results_dir: str = "results",
    tasks: Optional[List[str]] = None,
    ncals: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    n_obs: int = 3,
    n_samples: int = 2000,
    variant_name: str = "fm_pt_base",
    output_dir: Optional[str] = None,
    methods: Optional[List[str]] = None,
    device: Optional[str] = None,
):
    """Generate full corner plots for specified tasks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    paths = ResultsPath(Path(results_dir), experiment)
    cm = CheckpointManager(paths)

    if output_dir is None:
        output_dir = Path(results_dir) / experiment / "rebuttal_plots" / "corner_plots"
    else:
        output_dir = Path(output_dir)

    if tasks is None:
        tasks = ["light_tunnel", "high_dim_gaussian"]
    if ncals is None:
        ncals = [10, 50, 200, 1000]
    if seeds is None:
        seeds = [33]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        test_data = cm.load_shared_test_data(task)
        if test_data is None:
            print(f"  No test data, skipping")
            continue

        y_obs = test_data["y"][:n_obs]
        theta_true = test_data["theta"][:n_obs]
        theta_dim = theta_true.shape[-1]

        # Get parameter labels
        param_labels = TASK_PARAM_LABELS.get(
            task, [f"$\\theta_{{{i+1}}}$" for i in range(theta_dim)]
        )

        # Check if task has reference posterior
        has_reference = False
        simulator = None
        try:
            from omegaconf import OmegaConf
            task_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "task" / f"{task}.yaml")
            simulator = get_simulator(
                task_cfg.simulator.name,
                **OmegaConf.to_container(task_cfg.simulator.params, resolve=True)
            )
            has_reference = hasattr(simulator, "sample_reference_posterior")
        except Exception:
            pass

        for seed in seeds:
            for ncal in ncals:
                print(f"\n  seed={seed}, ncal={ncal}")

                models = {}

                def _include(name):
                    return methods is None or name in methods

                # Load all models
                if _include("NPE"):
                    npe = cm.load_shared_simulation_model(task, "npe", device)
                    if npe is not None:
                        models["NPE"] = npe

                if _include("FMPE"):
                    fmpe = cm.load_shared_simulation_model(task, "fmpe", device)
                    if fmpe is not None:
                        models["FMPE"] = fmpe

                if _include("FMCPE"):
                    fm_pt = cm.load_variant_calibration_model(
                        variant_name, task, seed, "fm_post_transform", ncal, device
                    )
                    if fm_pt is not None:
                        models["FMCPE"] = fm_pt

                if _include("MF-NPE"):
                    mf_npe = cm.load_shared_baseline(task, seed, "mf_npe", ncal, device)
                    if mf_npe is not None:
                        models["MF-NPE"] = mf_npe

                if _include("RoPE"):
                    rope = cm.load_shared_baseline(task, seed, "rope", ncal, device)
                    if rope is not None:
                        sim_data = cm.load_shared_simulations(task)
                        if sim_data is not None:
                            max_rope_sims = 1000
                            rope.set_sim_data(sim_data["x"][:max_rope_sims], sim_data["theta"][:max_rope_sims])
                        models["RoPE"] = rope

                if not models:
                    print("    No models loaded")
                    continue

                # Sample from each model
                for obs_i in range(n_obs):
                    y_single = y_obs[obs_i:obs_i+1]
                    obs_samples = {}

                    for method_name, model in models.items():
                        try:
                            s = sample_from_model(model, y_single, n_samples, device)
                            # s shape: (n_samples, 1, theta_dim) -> (n_samples, theta_dim)
                            s = s[:, 0, :].numpy() if s.ndim == 3 else s.numpy()
                            obs_samples[method_name] = s
                        except Exception as e:
                            print(f"    {method_name} sampling failed: {e}")

                    # Add reference posterior if available
                    if has_reference and simulator is not None:
                        try:
                            ref = simulator.sample_reference_posterior(
                                num_samples=n_samples,
                                observations=y_single,
                                misspecified=False,
                            )
                            if ref.ndim == 3:
                                ref = ref[:, 0, :]
                            if hasattr(simulator, "transform"):
                                ref = simulator.transform(ref)
                            obs_samples["True posterior"] = ref.numpy()
                        except Exception as e:
                            print(f"    Reference failed: {e}")

                    if not obs_samples:
                        continue

                    save_path = (
                        output_dir / task
                        / f"corner_seed{seed}_ncal{ncal}_obs{obs_i}.pdf"
                    )
                    make_full_corner(
                        theta_true_val=theta_true[obs_i].numpy(),
                        samples_dict=obs_samples,
                        param_labels=param_labels[:theta_dim],
                        save_path=save_path,
                        title=f"{task} — $N_{{cal}}$={ncal}, obs #{obs_i}",
                    )


def main():
    parser = argparse.ArgumentParser(description="Full Corner Plots")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--ncals", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n_obs", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--variant", default="fm_pt_base")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to include (e.g. NPE FMCPE MF-NPE RoPE). Default: all.")
    parser.add_argument("--device", default=None, help="Device override (e.g. cpu, cuda)")
    args = parser.parse_args()

    run_corner_plots(
        experiment=args.experiment,
        results_dir=args.results_dir,
        tasks=args.tasks,
        ncals=args.ncals,
        seeds=args.seeds,
        n_obs=args.n_obs,
        n_samples=args.n_samples,
        variant_name=args.variant,
        output_dir=args.output_dir,
        methods=args.methods,
        device=args.device,
    )


if __name__ == "__main__":
    main()
