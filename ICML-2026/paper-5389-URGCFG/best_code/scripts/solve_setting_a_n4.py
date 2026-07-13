"""Focused mechanism-design run for Setting (A) with n=4 goods.

Target: Profit per Good ≈ 0.303 (paper Table 1).
"""
import argparse
import json
import os
import math
import sys
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WRITING_ROOT
import mech_design.mechanism as mechanism_module


def dot_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Dot-product kernel k(x, y) = <x, y> (Setting A)."""
    return (X * Y).sum(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Setting A n=4 mechanism design.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Kernel batch size.")
    parser.add_argument("--niters", type=int, default=60000, help="Training iterations.")
    parser.add_argument("--epsilon", type=float, default=2e-3, help="Barrier epsilon.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of type samples.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim = 4
    npoint = 100  # = 100 candidate bundles with grid init
    lr = 0.05
    initial_penalty_factor = 100.0 / dim  # = 25.0
    epsilon = args.epsilon
    kernel_batch_size = args.batch_size
    patience = 600
    cooldown = 300
    factor = 0.85
    nsteps = args.niters
    max_samples = 5000  # Increased from 1000 for better MC estimation
    temp = 60.0
    temp_warmup_steps = 1000
    temp_schedule_initial = 5.0
    convergence_tolerance = 1e-8
    sorted_model = True
    is_Y_parameter = True
    window = 1000
    default_utility = 1.5 * epsilon
    scheduler_threshold = 1e-2
    max_clipping_norm = 5

    # Generate uniform type samples on [0,1]^dim
    unif_sample = torch.rand(max_samples, dim)
    sample = unif_sample
    if sorted_model:
        sample, _ = torch.sort(sample, dim=-1)

    print(f"Running Setting A n={dim} with {nsteps} iterations, {max_samples} samples, {npoint} candidates")

    model_kwargs = {
        "npoints": npoint,
        "kernel": dot_kernel,
        "cost_fn": None,           # No production cost in Setting A
        "y_dim": dim,
        "temp": temp,
        "is_Y_parameter": is_Y_parameter,
        "is_there_default": True,  # IR constraint via default option
        "default_intercept": -default_utility,
        "y_min": 0.0,
        "y_max": 1.0,              # Y = [0,1]^n
        "sorted_model": sorted_model,
    }

    mechanism_instance = mechanism_module.Mechanism(
        **model_kwargs, kernel_batch_size=kernel_batch_size
    )

    # Initialize Y (support points) with equispaced grid for uniform coverage
    # Grid: npoints^(1/d) points per dimension, covering [0,1]^d uniformly
    with torch.no_grad():
        n_per_dim = max(2, int(math.ceil(mechanism_instance.num_candidates ** (1.0 / dim))))
        grid_1d = torch.linspace(0.05, 0.95, n_per_dim, device=sample.device)
        grids = torch.meshgrid(*[grid_1d] * dim, indexing="ij")
        grid_points = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # [N_grid, dim]
        # Take exactly num_candidates points (random subset if more grid points than needed)
        n_grid = grid_points.shape[0]
        if n_grid > mechanism_instance.num_candidates:
            idx = torch.randperm(n_grid, device=sample.device)[:mechanism_instance.num_candidates]
        else:
            idx = torch.arange(n_grid, device=sample.device)
        Y_target = grid_points[idx].unsqueeze(0)  # [1, npoints, dim]
        Y_target, _ = Y_target.sort(dim=-1)
        # Convert to raw representation for sorted model
        diffs = torch.cat([Y_target[..., :1], Y_target[..., 1:] - Y_target[..., :-1]], dim=-1)
        raw = torch.log(torch.expm1(diffs.clamp_min(1e-8)))
        mechanism_instance.Y_rest_raw.copy_(raw)
        mechanism_instance.intercept_rest.zero_()  # P(y)=0 in Setting A

    writing_dir_dim = os.path.join(WRITING_ROOT, "mech", f"dim{dim}_settingA/")
    os.makedirs(writing_dir_dim, exist_ok=True)

    print(f"Writing outputs to {writing_dir_dim}")

    mechanism, mechanism_data = mechanism_instance.fit(
        sample,
        already_sorted=True,
        modes=["soft"],
        compile=False,
        optimizers_kwargs_dict={"soft": {"lr": lr}},
        schedulers_kwargs_dict={
            "soft": {
                "patience": patience,
                "threshold": scheduler_threshold,
                "factor": factor,
                "cooldown": cooldown,
                "eps": 1e-8,
            },
        },
        train_kwargs={
            "nsteps": nsteps,
            "max_clipping_norm": max_clipping_norm,
            "initial_penalty_factor": initial_penalty_factor,
            "steps_per_snapshot": 200,
            "steps_per_update": 5,
            "window": window,
            "constraint_fns": [],
            "use_wandb": False,
            "writing_dir": writing_dir_dim,
            "convergence_tolerance": convergence_tolerance,
            "epsilon": epsilon,
            "temp_warmup_steps": temp_warmup_steps,
            "temp_schedule_initial": temp_schedule_initial,
            "switch_threshold": 0.995,
        },
    )

    # Evaluate final metrics
    profits = mechanism_data["profits"]
    mean_profit = float(profits.mean().item())
    mean_profit_per_good = mean_profit / dim
    revenue = mechanism_data["revenue"]
    mean_revenue_per_good = float(revenue.mean().item()) / dim

    print(f"\n{'='*60}")
    print(f"RESULTS for Setting (A) n={dim}:")
    print(f"  Mean Profit per Good: {mean_profit_per_good:.6f}")
    print(f"  Mean Revenue per Good: {mean_revenue_per_good:.6f}")
    print(f"  Paper target (Profit per Good): 0.303")
    print(f"  Reproduce CI: [0.3028, 0.305]")
    in_ci = 0.3028 <= mean_profit_per_good <= 0.305
    print(f"  Within CI: {'YES' if in_ci else 'NO'}")
    print(f"{'='*60}")

    # Save final results
    result = {
        "dim": dim,
        "npoints": npoint,
        "nsteps": nsteps,
        "max_samples": max_samples,
        "seed": args.seed,
        "mean_profit_per_good": mean_profit_per_good,
        "mean_revenue_per_good": mean_revenue_per_good,
        "mean_profit": mean_profit,
        "paper_target": 0.303,
        "reproduce_ci": [0.3028, 0.305],
        "in_ci": in_ci,
    }
    result_path = os.path.join(writing_dir_dim, "reproduction_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {result_path}")
