#!/usr/bin/env python3
"""
RP-GSSM DAVIS-Tracking evaluation script.
Trains model from scratch and reports Linear Regression R2 and Kernel Ridge Regression R2.

Usage:
    cd /repo
    pdm run python3 eval_tracking.py --seed 0 --num_iter 5000 --latent_dim 32
"""

import os, argparse, pickle, time, json
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax, jax.numpy as np, jax.random as jr, optax, numpy as onp
from rp_ssm import datasets, utils, dists, recognition, distmaps, rpm, config, training


def build_model(latent_dim: int = 32):
    prior = dists.LGStationaryParam(
        start_from_invariant=True, stay_at_invariant=True,
        opt_params=["A"], A=0.5 * np.eye(latent_dim),
    )
    cnn_network = recognition.CNN([
        {"features": 32, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
        {"features": 64, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
        {"features": 64, "kernel_size": (5, 5), "strides": (2, 2), "padding": "SAME"},
    ])
    rec = [recognition.RPMRecognition(
        network=cnn_network, dist_map=distmaps.MVNCholesky(latent_dim), constant_cov=True
    )]
    return rpm.RPSSM(prior=prior, recognition=rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iter", type=int, default=12000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="/repo/results")
    args = parser.parse_args()

    print(json.dumps({
        "event": "start", "seed": args.seed, "latent_dim": args.latent_dim,
        "num_iter": args.num_iter, "devices": [str(d) for d in jax.devices()]
    }))

    # Load data
    t0 = time.time()
    data = datasets.load_dataset("tracking", args.seed)
    print(json.dumps({
        "event": "data_loaded",
        "train_shape": list(data.train_obs[0].shape),
        "val_shape": list(data.val_obs[0].shape),
        "load_time_s": round(time.time() - t0, 1)
    }))

    # Build and train
    model = build_model(latent_dim=args.latent_dim)
    lr_schedule = optax.schedules.exponential_decay(1e-3, 12000, 0.08)
    beta_schedule = lambda i: (i % 1000) / 1000.0  # 12 cycles of beta 0->1 over 12000 steps
    cfg = config.Config(
        num_iter=args.num_iter, prior_lr=lr_schedule, rec_lr=(lr_schedule,),
        batch_size=32, jit=True, stabilize_A="clip", seed=args.seed,
        beta_schedule=beta_schedule,
    )
    free_energy = rpm.ConstrainedIVFreeEnergy(model=model)
    trainer = training.Trainer(free_energy=free_energy, config=cfg)

    t0 = time.time()
    trainer.fit(data.train_data)
    train_time = time.time() - t0

    # Evaluate
    _, posterior = trainer.apply(data.val_data)
    means = posterior.params["means"]
    r2_linear, _ = utils.linear_r2(means, data.val_states)
    r2_krr, _ = utils.krr_r2(means, data.val_states)

    results = {
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "num_iter": args.num_iter,
        "train_time_s": round(train_time, 1),
        "linear_r2": round(float(r2_linear), 4),
        "kernel_ridge_r2": round(float(r2_krr), 4),
    }

    print(json.dumps({"event": "results", **results}))

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    trainer.save_params(str(out_dir / f"params_seed{args.seed}.pkl"))

    return results


if __name__ == "__main__":
    main()
