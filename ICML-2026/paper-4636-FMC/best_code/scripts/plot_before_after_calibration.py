#!/usr/bin/env python3
"""Before/After Calibration Plots for ICML Rebuttal.

For given test observations, shows side-by-side:
  (a) Raw NPE posterior p(theta|y) — uncorrected
  (b) FMCPE-corrected posterior

Generates KDE corner plots for Pendulum and Gaussian tasks
at multiple N_cal values. When a reference posterior exists (Gaussian),
overlays the true posterior.

Addresses Reviewer Qmdc's explicit request.

Usage:
    python scripts/plot_before_after_calibration.py --experiment comparison_benchmark
    python scripts/plot_before_after_calibration.py --experiment comparison_benchmark --tasks pendulum
    python scripts/plot_before_after_calibration.py --experiment comparison_benchmark --ncals 10 50 200
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
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpointing import CheckpointManager, ResultsPath  # noqa: E402
from simulator import get_simulator  # noqa: E402


# Method display names and colors
METHOD_DISPLAY = {
    "npe": "NPE (uncorrected)",
    "fmpe": "FMPE (uncorrected)",
    "fm_post_transform": "FMCPE (corrected)",
    "mf_npe": "MF-NPE",
    "rope": "RoPE",
    "reference": "True posterior",
}

METHOD_COLORS = {
    "NPE (uncorrected)": "#2ca02c",
    "FMPE (uncorrected)": "#d62728",
    "FMCPE (corrected)": "#1f77b4",
    "MF-NPE": "#ff7f0e",
    "RoPE": "#8c564b",
    "True posterior": "#7f7f7f",
}


def sample_from_model(model, y_obs, n_samples, device):
    """Sample from a posterior model, handling shape conventions."""
    with torch.no_grad():
        samples = model.sample(y_obs.to(device), n_samples, device).cpu()
        if samples.ndim == 3:
            # Shape: (n_samples, n_obs, theta_dim) or (n_obs, n_samples, theta_dim)
            if samples.shape[0] == n_samples:
                return samples  # (n_samples, n_obs, theta_dim)
            else:
                return samples.permute(1, 0, 2)  # transpose to (n_samples, n_obs, theta_dim)
        return samples


def sample_with_oom_fallback(model, y_obs, n_samples, device, min_samples=200):
    """Sample with automatic n_samples reduction on CUDA OOM."""
    current = n_samples
    while current >= min_samples:
        try:
            torch.cuda.empty_cache()
            return sample_from_model(model, y_obs, current, device)
        except torch.cuda.OutOfMemoryError:
            current //= 2
            print(f"    OOM: retrying with n_samples={current}")
    raise RuntimeError(f"OOM persists at n_samples={current}")


def make_corner_plot(
    theta_true: torch.Tensor,
    samples_dict: Dict[str, np.ndarray],
    obs_idx: int,
    theta_dim: int,
    save_path: Path,
    task: str,
    title: str = "",
    max_samples: int = 2000,
):
    """Create a corner plot comparing multiple methods for one observation.

    Args:
        theta_true: True parameter values, shape (n_obs, theta_dim)
        samples_dict: {display_name: samples array (n_samples, theta_dim)}
        obs_idx: Which observation this is
        theta_dim: Parameter dimensionality
        save_path: Where to save the plot
        title: Plot title
        max_samples: Max samples per method for clarity
    """
    if task == "pendulum":
        theta_labels = [r"$\omega_0$", "$A$"]
    else:
        theta_labels = [f"$\\theta_{{{d + 1}}}$" for d in range(theta_dim)]

    dfs = []
    for name, samples in samples_dict.items():
        if len(samples) > max_samples:
            idx = np.random.choice(len(samples), max_samples, replace=False)
            samples = samples[idx]
        df = pd.DataFrame(samples, columns=theta_labels)
        df["Method"] = name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Build palette from predefined colors
    palette = {}
    for name in samples_dict:
        palette[name] = METHOD_COLORS.get(name, None)
    palette = {k: v for k, v in palette.items() if v is not None}

    g = sns.pairplot(
        df_all,
        hue="Method",
        diag_kind="kde",
        palette=palette if palette else None,
        plot_kws=dict(alpha=0.3, s=3, rasterized=True),
        diag_kws=dict(fill=True, alpha=0.4),
    )

    # Overlay true theta
    t = theta_true[obs_idx].numpy()
    for i in range(theta_dim):
        g.axes[i, i].axvline(t[i], color="black", linestyle="--", lw=1.5)
        for j in range(theta_dim):
            if i != j:
                g.axes[i, j].scatter(
                    t[j],
                    t[i],
                    color="black",
                    marker="x",
                    s=60,
                    zorder=10,
                )

    # Remove all legends (axis-level and figure-level seaborn legend)
    for ax in g.axes.flatten():
        if ax is not None and ax.get_legend() is not None:
            ax.get_legend().remove()
    for leg in g.fig.legends:
        leg.remove()

    # Add single legend with solid patch markers (visible regardless of scatter alpha or 1D layout)
    handles = [
        Patch(facecolor=palette[name], label=name, alpha=0.85)
        for name in samples_dict
        if name in palette
    ]
    if handles:
        g.fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    if title:
        g.fig.suptitle(title, fontsize=14, y=1.02)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(g.figure)
    print(f"    Saved: {save_path}")


def plot_before_after(
    experiment: str,
    results_dir: str = "results",
    tasks: Optional[List[str]] = None,
    ncals: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    n_obs: int = 3,
    n_samples: int = 2000,
    variant_name: str = "fm_pt_base",
    output_dir: Optional[str] = None,
):
    """Generate before/after calibration corner plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = ResultsPath(Path(results_dir), experiment)
    cm = CheckpointManager(paths)

    if output_dir is None:
        output_dir = Path(results_dir) / experiment / "rebuttal_plots" / "before_after"
    else:
        output_dir = Path(output_dir)

    if tasks is None:
        tasks = ["pendulum", "high_dim_gaussian"]
    if ncals is None:
        ncals = [10, 50, 200, 1000]
    if seeds is None:
        seeds = [33]

    # Conditional tasks with analytical reference posteriors
    conditional_tasks = {
        "high_dim_gaussian",
        "pure_gaussian",
        "no_misspec_gaussian",
        "adaptive_gaussian",
        "gaussian",
        "ou_process",
    }

    for task in tasks:
        print(f"\n=== Task: {task} ===")

        # Load test data
        test_data = cm.load_shared_test_data(task)
        if test_data is None:
            print(f"  No test data for {task}, skipping")
            continue

        y_obs = test_data["y"][:n_obs]
        theta_true = test_data["theta"][:n_obs]

        # Get simulator for reference posterior
        has_reference = task in conditional_tasks
        simulator = None
        if has_reference:
            try:
                from omegaconf import OmegaConf

                task_cfg_path = PROJECT_ROOT / "configs" / "task" / f"{task}.yaml"
                task_cfg = OmegaConf.load(task_cfg_path)
                simulator = get_simulator(
                    task_cfg.simulator.name,
                    **OmegaConf.to_container(task_cfg.simulator.params, resolve=True),
                )
            except Exception as e:
                print(f"  Could not load simulator for reference: {e}")
                has_reference = False

        for seed in seeds:
            for ncal in ncals:
                print(f"\n  seed={seed}, ncal={ncal}")

                samples_by_method = {}

                # 1. NPE (uncorrected) — simulation model applied to y
                npe_model = cm.load_shared_simulation_model(task, "npe", device)
                if npe_model is not None:
                    try:
                        s = sample_with_oom_fallback(npe_model, y_obs, n_samples, device)
                        samples_by_method["NPE (uncorrected)"] = s
                        print(f"    NPE: {s.shape}")
                    except Exception as e:
                        print(f"    NPE sampling failed: {e}")

                # 2. FMPE (uncorrected) — simulation model applied to y
                # fmpe_model = cm.load_shared_simulation_model(task, "fmpe", device)
                # if fmpe_model is not None:
                #     try:
                #         s = sample_from_model(fmpe_model, y_obs, n_samples, device)
                #         samples_by_method["FMPE (uncorrected)"] = s
                #         print(f"    FMPE: {s.shape}")
                #     except Exception as e:
                #         print(f"    FMPE sampling failed: {e}")

                # 3. FMCPE (corrected) — variant calibration model
                fm_pt = cm.load_variant_calibration_model(
                    variant_name, task, seed, "fm_post_transform", ncal, device
                )
                if fm_pt is not None:
                    try:
                        s = sample_with_oom_fallback(fm_pt, y_obs, n_samples, device)
                        samples_by_method["FMCPE (corrected)"] = s
                        print(f"    FMCPE: {s.shape}")
                    except Exception as e:
                        print(f"    FMCPE sampling failed: {e}")

                # 4. MF-NPE baseline
                # mf_npe = cm.load_shared_baseline(task, seed, "mf_npe", ncal, device)
                # if mf_npe is not None:
                #     try:
                #         s = sample_from_model(mf_npe, y_obs, n_samples, device)
                #         samples_by_method["MF-NPE"] = s
                #         print(f"    MF-NPE: {s.shape}")
                #     except Exception as e:
                #         print(f"    MF-NPE sampling failed: {e}")

                # 5. RoPE baseline
                # rope_model = cm.load_shared_baseline(task, seed, "rope", ncal, device)
                # if rope_model is not None:
                #     try:
                #         sim_data = cm.load_shared_simulations(task)
                #         if sim_data is not None:
                #             rope_model.set_sim_data(sim_data["x"], sim_data["theta"])
                #         s = sample_from_model(rope_model, y_obs, n_samples, device)
                #         samples_by_method["RoPE"] = s
                #         print(f"    RoPE: {s.shape}")
                #     except Exception as e:
                #         print(f"    RoPE sampling failed: {e}")

                # 6. Reference posterior (if available)
                if has_reference and simulator is not None:
                    try:
                        ref = simulator.sample_reference_posterior(
                            num_samples=n_samples, observations=y_obs, misspecified=False
                        )
                        if ref.ndim == 3:
                            ref = ref[:, :n_obs, :]  # (n_samples, n_obs, theta_dim)
                        if hasattr(simulator, "transform"):
                            ref = simulator.transform(ref)
                        samples_by_method["True posterior"] = ref
                        print(f"    Reference: {ref.shape}")
                    except Exception as e:
                        print(f"    Reference posterior failed: {e}")

                if not samples_by_method:
                    print("    No models loaded, skipping")
                    continue

                # Generate corner plots for each observation
                theta_dim = theta_true.shape[-1]
                for obs_i in range(n_obs):
                    obs_samples = {}
                    for name, s in samples_by_method.items():
                        if isinstance(s, torch.Tensor):
                            s = s.detach().cpu()
                        obs_s = s[:, obs_i, :].numpy() if s.ndim == 3 else s.numpy()
                        obs_samples[name] = obs_s

                    save_path = output_dir / task / f"seed{seed}_ncal{ncal}_obs{obs_i}.pdf"
                    make_corner_plot(
                        theta_true=theta_true,
                        samples_dict=obs_samples,
                        obs_idx=obs_i,
                        theta_dim=theta_dim,
                        save_path=save_path,
                        task=task,
                        title=f"{task} — N_cal={ncal}, obs #{obs_i}",
                    )


def main():
    parser = argparse.ArgumentParser(description="Before/After Calibration Plots")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--ncals", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n_obs", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--variant", default="fm_pt_base")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    plot_before_after(
        experiment=args.experiment,
        results_dir=args.results_dir,
        tasks=args.tasks,
        ncals=args.ncals,
        seeds=args.seeds,
        n_obs=args.n_obs,
        n_samples=args.n_samples,
        variant_name=args.variant,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
