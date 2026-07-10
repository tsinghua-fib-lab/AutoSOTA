"""
Training script for consistency bridges

Usage:
    python train_bridge.py --config configs/ou.yaml --gpu 0
    python train_bridge.py --config configs/ou.yaml --gpu 0 --train.lr 3e-4 --train.num_outer_iterations 5000
"""

import os
import sys
import argparse
import yaml
import json
import pickle
from datetime import datetime


# helpers

def apply_cli_overrides(cfg, overrides):
    """
    Modifies cfg with overrides
    """
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        d[keys[-1]] = value


def parse_args():
    parser = argparse.ArgumentParser(description="Train a consistency bridge")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs/<problem_name>/<timestamp>_run)")
    # Collect any extra --section.key value pairs as overrides
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i].lstrip("-")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    return args, overrides


def _build_model(model_cfg, d):
    """Build the neural network from the model config."""
    import flax.linen as nn
    from src.models import ScoreMLP, ConservativeMLP

    activation_map = {"relu": nn.relu, "gelu": nn.gelu, "silu": nn.silu, "tanh": nn.tanh}
    activation = activation_map[model_cfg.get("activation", "gelu")]
    dim_hidden = tuple(model_cfg["dim_hidden"])
    emb_dim_hidden = tuple(model_cfg["emb_dim_hidden"])

    model_type = model_cfg.get("type", "conservative")
    if model_type == "conservative":
        return ConservativeMLP(
            dim_hidden=dim_hidden,
            emb_dim_hidden=emb_dim_hidden,
            activation=activation,
        )
    elif model_type == "score":
        return ScoreMLP(
            dim_hidden=dim_hidden,
            emb_dim_hidden=emb_dim_hidden,
            activation=activation,
            out_dim=d,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


###################################
# Single training run
###################################

def run_training(cfg, outdir):
    """Run a single training, evaluate KL metrics, and save artefacts under `outdir`.

    Returns a dict of final scalar metrics.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from src.consistency_bridge import ConsistencyBridge
    from src.plotting import plot_multiple_trajectories, plot_two_dim_trajectories, plot_trajectories_1d
    from src.evaluation import (
        compute_KL_to_reference, compute_KL_to_reference_sigma_fn,
        compute_KL_to_ground_truth, compute_KL_to_ground_truth_sigma_fn,
    )
    from problems import PROBLEMS

    plot_fn_map = {
        "plot_multiple_trajectories": plot_multiple_trajectories,
        "plot_two_dim_trajectories": plot_two_dim_trajectories,
        "plot_trajectories_1d": plot_trajectories_1d,
    }

    os.makedirs(outdir, exist_ok=True)

    # save config
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("=" * 60)
    print(f"Output directory: {outdir}")

    # build problem
    problem_name = cfg["problem"]["name"]
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem '{problem_name}'. Available: {list(PROBLEMS.keys())}")
    problem = PROBLEMS[problem_name](cfg["problem"])

    # build model
    d = problem["shape"][0]
    model = _build_model(cfg["model"], d)

    # build consistency bridge
    bridge_config = cfg["bridge"]
    bridge = ConsistencyBridge(
        shape=problem["shape"],
        x_0=problem["x_0"],
        x_T=problem["x_T"],
        base_drift_fn=problem["base_drift"],
        sigma_fn=problem["sigma_fn"],
        model=model,
        bridge_config=bridge_config,
        T=problem["T"],
    )

    # train
    train_config = dict(cfg["train"])  # copy so cfg stays intact for saving
    seed = train_config.pop("seed", 0)
    key = jax.random.PRNGKey(seed)
    state, ema_params_lst, ema_grad_norms = bridge.train(key, train_config)

    # save checkpoint
    ckpt_path = os.path.join(outdir, "checkpoint.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "state": state,
            "ema_params": ema_params_lst[-1],
            "ema_params_lst": ema_params_lst,
        }, f)
    print(f"Saved checkpoint to {ckpt_path}")

    # plot grad norms
    plt.figure()
    plt.plot(ema_grad_norms)
    plt.xlabel("Outer iteration")
    plt.ylabel("EMA grad norm")
    fig_path = os.path.join(outdir, "grad_norms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    # sample trajectories
    num_trajs = 20
    num_steps = cfg["train"]["num_steps"]
    sample_keys = jax.random.split(key, num_trajs)
    ema_params = ema_params_lst[-1]

    x_traj, _, _ = jax.vmap(
        bridge.sample_controlled_sde, in_axes=(0, None, None)
    )(sample_keys, ema_params, num_steps)

    jnp.save(os.path.join(outdir, "sampled_trajectories.npy"), x_traj)

    plot_fn_name = cfg.get("plotting", {}).get("plot_fn", None)
    if plot_fn_name is not None:
        if plot_fn_name not in plot_fn_map:
            raise ValueError(f"Unknown plot function {plot_fn_name}")
        plot_fn = plot_fn_map[plot_fn_name]
        plot_fn(x_traj, savedir=os.path.join(outdir, "sampled_trajectories.png"), show=False)

    print(f"Sampled {num_trajs} trajectories, shape: {x_traj.shape}")

    # evaluate KL metrics
    eval_key = jax.random.PRNGKey(seed + 999)

    def learned_drift_fn(x, t):
        return bridge.controlled_drift(ema_params, x, t)

    sigma = problem["sigma_fn"]
    use_sigma_fn = callable(sigma)
    T = problem["T"]

    ref_fn = compute_KL_to_reference_sigma_fn if use_sigma_fn else compute_KL_to_reference
    kl_ref_learned, _ = ref_fn(
        eval_key, bridge, learned_drift_fn, problem["base_drift"], sigma, T, num_steps=num_steps,
    )
    results = {"kl_to_reference_learned": float(kl_ref_learned)}

    if "true_drift_fn" in problem:
        gt_fn = compute_KL_to_ground_truth_sigma_fn if use_sigma_fn else compute_KL_to_ground_truth
        kl_sol, _ = gt_fn(
            eval_key, bridge, problem["true_drift_fn"], learned_drift_fn, sigma, T, num_steps=num_steps,
        )
        kl_ref_truth, _ = ref_fn(
            eval_key, bridge, problem["true_drift_fn"], problem["base_drift"], sigma, T, num_steps=num_steps,
        )
        results["kl_to_solution"] = float(kl_sol)
        results["kl_to_reference_truth"] = float(kl_ref_truth)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
    print(f"All outputs saved to {outdir}")
    print("Done.")

    return results


###################################
# Main
###################################

def main():
    args, overrides = parse_args()

    # use a particular GPU — set before JAX imports
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # apply any CLI overrides
    apply_cli_overrides(cfg, overrides)

    # set output directory
    problem_name = cfg["problem"]["name"]
    if args.outdir is not None:
        outdir = args.outdir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("outputs", problem_name, f"{timestamp}_run")

    run_training(cfg, outdir)


if __name__ == "__main__":
    main()
