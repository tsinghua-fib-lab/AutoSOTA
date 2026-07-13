"""
RP-GSSM DAVIS-Tracking experiment.

Reproduces the DAVIS-Tracking results from the paper:
  - Linear Regression R²: 0.726
  - Kernel Ridge Regression R²: 0.917

Hyperparameters from Table S1:
  - DZ = 32
  - CRC = Yes (constant recognition covariance)
  - DRC = No (full Cholesky covariance, NOT diagonal)
  - Training: 5000 iterations, batch_size=32, lr=1e-3
  - 3 seeds per the paper

Usage:
  cd /repo
  pdm run python3 run_tracking_experiment.py --seed 0
"""

import os
import argparse
import pickle
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as np
import jax.random as jr
import optax
import numpy as onp

from rp_ssm import (
    datasets, utils, dists, recognition, distmaps, rpm, config, training
)


def build_model(latent_dim: int = 32):
    """Build RP-GSSM model for tracking task (64x64x3 RGB images)."""

    # Prior: stable LGSSM with standardized parametrization
    prior = dists.LGStationaryParam(
        start_from_invariant=True,
        stay_at_invariant=True,
        opt_params=["A"],
        A=0.5 * np.eye(latent_dim),
    )

    # Recognition: CNN for 64x64 RGB images
    # Following the paper's architecture (3-layer CNN)
    cnn_network = recognition.CNN([
        {"features": 32, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
        {"features": 64, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
        {"features": 64, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
    ])

    # Full Cholesky covariance (not diagonal per Table S1)
    rec = [
        recognition.RPMRecognition(
            network=cnn_network,
            dist_map=distmaps.MVNCholesky(latent_dim),
            constant_cov=True,  # CRC = Yes per Table S1
        )
    ]

    model = rpm.RPSSM(prior=prior, recognition=rec)
    return model


def build_config(seed: int = 0, num_iter: int = 5000):
    """Build training config."""
    lr_schedule = optax.schedules.exponential_decay(1e-3, 5000, 0.2)

    cfg = config.Config(
        num_iter=num_iter,
        prior_lr=lr_schedule,
        rec_lr=(lr_schedule,),
        batch_size=32,
        jit=True,
        stabilize_A="clip",
        seed=seed,
    )
    return cfg


def build_logger(data, eval_interval: int = 50):
    """Build logger that computes R² scores."""
    def logger(trainer, aux, batch_indices):
        info = {}
        if trainer.itr % eval_interval == 0:
            posterior = aux["posterior"]
            batch_states = data.train_states[batch_indices]
            means = posterior.params["means"]

            # Linear regression R² for each target dimension
            r2_linear, _ = utils.linear_r2(means, batch_states)
            info["R2_lin"] = f"{r2_linear:.4f}"

            # Track history
            r2_linear_hist = getattr(trainer, "r2_linear_hist", [])
            r2_linear_hist.append(float(r2_linear))
            setattr(trainer, "r2_linear_hist", r2_linear_hist)

        return info
    return logger


def evaluate(trainer, data, split: str = "val"):
    """Evaluate model on train/val split."""
    if split == "train":
        eval_data = data.train_data
        eval_states = data.train_states
    else:
        eval_data = data.val_data
        eval_states = data.val_states

    # Get posterior means
    _, posterior = trainer.apply(eval_data)
    means = posterior.params["means"]

    # Linear regression R²
    r2_linear, _ = utils.linear_r2(means, eval_states)

    # Kernel ridge regression R²
    # Convert to numpy for sklearn
    means_np = onp.array(means).reshape(-1, means.shape[-1])
    states_np = onp.array(eval_states).reshape(-1, eval_states.shape[-1])
    r2_krr, _ = utils.krr_r2(np.array(means_np), np.array(states_np))

    return {
        "linear_r2": float(r2_linear),
        "kernel_ridge_r2": float(r2_krr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_iter", type=int, default=5000, help="Training iterations")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    print(f"=== RP-GSSM DAVIS-Tracking Experiment ===")
    print(f"  Seed: {args.seed}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Iterations: {args.num_iter}")
    print(f"  JAX devices: {jax.devices()}")

    # Load data
    print("\n--- Loading dataset ---")
    t0 = time.time()
    data = datasets.load_dataset("tracking", args.seed)
    print(f"  Train obs shape: {data.train_obs[0].shape}")
    print(f"  Train states shape: {data.train_states.shape}")
    print(f"  Val obs shape: {data.val_obs[0].shape}")
    print(f"  Val states shape: {data.val_states.shape}")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # Build model
    print("\n--- Building model ---")
    model = build_model(latent_dim=args.latent_dim)

    # Build config
    cfg = build_config(seed=args.seed, num_iter=args.num_iter)

    # Build logger
    logger_fn = build_logger(data)

    # Build free energy
    free_energy = rpm.ConstrainedIVFreeEnergy(model=model)

    # Train
    print("\n--- Training ---")
    t0 = time.time()
    trainer = training.Trainer(
        free_energy=free_energy,
        config=cfg,
        logger=logger_fn,
    )
    trainer.fit(data.train_data)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Evaluate
    print("\n--- Evaluation ---")
    train_metrics = evaluate(trainer, data, split="train")
    val_metrics = evaluate(trainer, data, split="val")

    print(f"\n  Training set:")
    print(f"    Linear Regression R²:      {train_metrics['linear_r2']:.4f}")
    print(f"    Kernel Ridge Regression R²: {train_metrics['kernel_ridge_r2']:.4f}")
    print(f"\n  Validation set:")
    print(f"    Linear Regression R²:      {val_metrics['linear_r2']:.4f}")
    print(f"    Kernel Ridge Regression R²: {val_metrics['kernel_ridge_r2']:.4f}")

    # Paper targets for comparison
    print(f"\n  Paper targets (DAVIS-Tracking):")
    print(f"    Linear Regression R²:      0.726")
    print(f"    Kernel Ridge Regression R²: 0.917")

    # Save results
    results = {
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "num_iter": args.num_iter,
        "train_time_s": train_time,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "loss_history": [float(l) for l in trainer.loss_tot],
        "r2_linear_history": getattr(trainer, "r2_linear_hist", []),
    }

    # Save params and results
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("/repo/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"results_seed{args.seed}.pkl"
    params_path = out_dir / f"params_seed{args.seed}.pkl"

    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    trainer.save_params(str(params_path))

    print(f"\n  Results saved to: {results_path}")
    print(f"  Params saved to: {params_path}")

    return results


if __name__ == "__main__":
    main()
