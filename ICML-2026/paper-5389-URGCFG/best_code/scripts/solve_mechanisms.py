"""Mechanism-design sweep over dimensions, kernels, and costs.

This script instantiates the generic :class:`mech_design.mechanism.Mechanism`
model for a grid of problem dimensions and experiment families
(`kernel`, `cost_fn`, and type distribution), trains each configuration,
and stores checkpoints/snapshots under ``WRITING_ROOT/mech/``. Types ``X``
are screened against outcome/bundle vectors ``Y``, where each ``Y`` is a
finite convex candidate in the underlying grid model.

Two main families are swept:
- Dot-product kernel with no production cost on bounded types in ``[0, 1]^d``.
- Hinge kernel with quadratic production costs on lognormal types, corresponding
  to screening with one-sided surplus above types and quadratic production costs.

Typical usage
-------------
Run a full sweep (using the default output root from ``config.py``)::

    python scripts/solve_mechanisms.py --niters 20000 --batch-size 512

Use ``--clear`` to remove previous runs in that directory before starting a
fresh sweep.
"""
import argparse
import os
import shutil
from math import exp
import numpy as np
import torch
from config import WRITING_ROOT
import mech_design.mechanism as mechanism_module

def dot_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Dot-product kernel :math:`k(x, y) = \\langle x, y \\rangle`.

    Here ``X`` is a vector of types/valuations and ``Y`` is a bundle/outcome
    vector; each ``Y`` corresponds to a finite convex candidate in the
    underlying model.

    Parameters
    ----------
    X, Y:
        Batches of type and outcome vectors with matching last dimension. In
        the common case, both have shape ``(batch_size, dim)``.

    Returns
    -------
    torch.Tensor
        Kernel values with the leading batch shape, typically
        ``(batch_size,)``.
    """
    return (X * Y).sum(dim=-1)

def hinge_kernel(X: torch.Tensor, Y: torch.Tensor, κ: float = 1.0) -> torch.Tensor:
    """One-sided linear kernel that only rewards allocations above the type.

    Computes ``κ * sum(max(Y - X, 0))`` along the last dimension, i.e. shortfall
    below ``X`` is ignored while surplus above ``X`` contributes linearly. This
    is the kernel used in the screening-with-one-sided-surplus experiments,
    combined with :func:`quadratic_cost`.

    Parameters
    ----------
    X, Y:
        Type and outcome vectors with matching last dimension; each ``Y`` is a
        finite convex candidate bundle.
    κ:
        Overall scale applied to the hinge payoff.

    Returns
    -------
    torch.Tensor
        Kernel values with the leading batch shape.
    """
    return κ * (Y - X).clamp_min(0).sum(dim=-1)

def quadratic_cost(Y: torch.Tensor, β: float = 0.1) -> torch.Tensor:
    """Additive quadratic production cost :math:`β \\lVert Y \\rVert_2^2`.

    Parameters
    ----------
    Y:
        Candidate bundle/allocation vectors.
    β:
        Scalar weight on the squared Euclidean norm of ``Y``.

    Returns
    -------
    torch.Tensor
        Per-sample cost with shape matching the leading batch dimensions of
        ``Y`` (typically ``(batch_size,)``).
    """
    return β * (Y**2).sum(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mechanism design sweep.")
    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        help="Remove existing snapshots in the writing_dir before training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="Chunk size used when computing the kernel to limit memory.",
    )
    parser.add_argument(
        "--niters",
        type=int,
        default=60_000,
        help="Number of training iterations (passed as nsteps).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2e-3,
        help="Barrier epsilon threshold for the constraints.",
    )
    args = parser.parse_args()
    writing_dir_base = f"{WRITING_ROOT}mech/"
    if args.clear:
        shutil.rmtree(writing_dir_base, ignore_errors=True)
        print(f"Cleared {writing_dir_base}")
    dims = [1, 2, 4, 6, 12, 50, 100, 250]
    initial_penalty_factors = [100.0/d for d in dims]
    npoints = [d + 10 for d in dims]  # number of candidate bundles per dimension
    epsilon = args.epsilon
    kernel_batch_size=args.batch_size        
    patience = 600
    cooldown = 300
    factor = 0.85
    nsteps = args.niters
    max_samples = 1_000
    temp = 60.
    temp_warmup_steps = 1000
    temp_schedule_initial = 5.0
    lrs = [0.05 for d in dims]  # potentially dimension-scaled learning rate
    convergence_tolerance = 1e-8
    sorted_model = True
    is_Y_parameter = True
    window = 1000
    # Utility level of the outside option when present. We shift it into the
    # model via a negative intercept; if this were exactly zero, optimization
    # may get stuck at the beginning because the default and non-default
    # options start too symmetrically.
    default_utility = 1.5*epsilon
    scheduler_threshold = 1e-2
    max_clipping_norm = 5

    # Pre-sampled type draws reused across dimensions so that different
    # kernels/costs see comparable inputs.
    unif_sample = torch.rand(max_samples, max(dims))
    lognormal_sample = torch.exp(torch.randn(max_samples, max(dims)) * 0.25)

    y_maxes = [1.0, None]
    costs =   [None, quadratic_cost]
    kernels = [dot_kernel, hinge_kernel]
    samples = [unif_sample, lognormal_sample]
    are_there_defaults = [True, False]


    # Each tuple (y_max, cost, kernel, sample, is_there_default) defines one
    # sweep family:
    #   - dot_kernel + no cost on bounded types in [0, 1]^d
    #   - hinge_kernel + quadratic_cost on unbounded lognormal types
    for y_max, cost, kernel, big_sample, is_there_default in zip(y_maxes, costs, kernels, samples, are_there_defaults):
        for dim, npoint, lr, initial_penalty_factor in zip(dims, npoints, lrs, initial_penalty_factors):
            print(f"Running mechanism design for dim={dim}, npoints={npoint}")
            sample = big_sample[:, :dim]
            if sorted_model:
                sample, _ = torch.sort(sample, dim=-1)
            model_kwargs = {
                "npoints": npoint,
                "kernel": kernel,
                "cost_fn": cost,
                "y_dim": dim,
                "temp": temp,
                "is_Y_parameter": is_Y_parameter,
                "is_there_default": is_there_default,
                "default_intercept": -default_utility,
                "y_min": 0.0,
                "y_max": y_max,
                "sorted_model": sorted_model,
            }
            mechanism_instance = mechanism_module.Mechanism(**model_kwargs, kernel_batch_size=kernel_batch_size)
            with torch.no_grad():
                q = torch.linspace(0.3, 0.9, mechanism_instance.num_candidates, device=sample.device)
                Y_target = torch.quantile(sample, q, dim=0).unsqueeze(0)  # sample is sorted already
                # keep this to satisfy the sorted parametrization
                Y_target, _ = Y_target.sort(dim=-1)
                diffs = torch.cat([Y_target[..., :1], Y_target[..., 1:] - Y_target[..., :-1]], dim=-1)
                raw = torch.log(torch.expm1(diffs.clamp_min(1e-8)))
                mechanism_instance.Y_rest_raw.copy_(raw)
                if cost is not None:
                    costs_per_candidate = cost(Y_target.squeeze(0)).view(1, -1)  # shape (1, npoints)
                    mechanism_instance.intercept_rest.copy_(costs_per_candidate)
                else:
                    mechanism_instance.intercept_rest.zero_()

            writing_dir_dim = os.path.join(writing_dir_base, f"dim{dim}/")
            mechanism, mechanism_data = mechanism_instance.fit(
                sample,
                already_sorted=True,
                modes=["soft"],
                compile=True,
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
