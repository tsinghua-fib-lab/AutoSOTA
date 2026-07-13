"""Pendulum independence experiment: FMCPE vs RoPE under CI violation.

Compares methods when calibration data is generated via:
- "transitive": y = decay(x) + noise → CI holds (y ⊥ θ | x)
- "independent": x, y generated independently from θ → CI violated

Evaluation uses SBC/coverage (no MCMC reference posterior).

Usage:
    python scripts/run_pendulum_independence.py --generations transitive --seeds 33 --sbc_sims 50 --sbc_samples 100
    python scripts/run_pendulum_independence.py --generations transitive independent --seeds 33 43 53
"""

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from simulator.pendulum import Pendulum
from training.base import TrainerConfig
from training.trainers.simulation import NPETrainer, FMPETrainer
from training.trainers.calibration import FMPostTransformTrainer
from training.trainers.baselines import RoPETrainer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from scripts.run_sbc_coverage import run_sbc_dgp, run_coverage_dgp


# ============================================================================
# Task config matching configs/task/pendulum.yaml
# ============================================================================

FLOW_X_CONFIG = {
    "space": "data",
    "conditional": True,
    "probability_path": "ot2",
    "prior": "uniform",
    "base_dist": "data_eps",
    "params": {
        "probability_path_params": {"sigma_min": 1e-4},
        "base_dist_params": {"eps": 5e-2},
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": 200,
                "dropout": 0.0,
                "batch_norm": True,
                "context_dim": 10,
                "theta_with_glu": False,
                "context_with_glu": False,
                "activation": "gelu",
                "hidden_dims": [32, 64, 128, 256, 512, 256, 128, 64, 32],
            },
            "theta_embedding_kwargs": {
                "name": "conv1d",
                "time_embedding": True,
                "output_dim": 10,
                "n_freqs": 5,
            },
            "embedding_kwargs": {
                "name": "conv1d",
                "output_dim": 10,
            },
        },
    },
}

FLOW_THETA_CONFIG = {
    "space": "data",
    "conditional": True,
    "probability_path": "ot2",
    "prior": "uniform",
    "base_dist": "gaussian",
    "params": {
        "probability_path_params": {"sigma_min": 1e-4},
        "base_dist_params": {},
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": 2,
                "dropout": 0.0,
                "batch_norm": False,
                "context_dim": 10,
                "theta_with_glu": True,
                "context_with_glu": False,
                "activation": "gelu",
                "hidden_dims": [32, 64, 128, 256, 256, 128, 64, 32],
            },
            "embedding_kwargs": {
                "name": "conv1d",
                "output_dim": 10,
            },
        },
    },
}

FMPE_CONFIG = {
    "space": "data",
    "conditional": True,
    "probability_path": "ot2",
    "prior": "power",
    "base_dist": "gaussian",
    "params": {
        "probability_path_params": {"sigma_min": 1e-4},
        "prior_params": {"rate": 2.0},
        "base_dist_params": {},
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": 2,
                "dropout": 0.0,
                "batch_norm": False,
                "context_dim": 10,
                "theta_with_glu": True,
                "context_with_glu": False,
                "activation": "gelu",
                "hidden_dims": [32, 64, 128, 256, 256, 128, 64, 32],
            },
            "embedding_kwargs": {
                "name": "conv1d",
                "output_dim": 10,
            },
        },
    },
}

TASK_CONFIG = {
    "name": "pendulum",
    "rescale": "none",
    "npe": {
        "training": {"rescale": "none"},
        "params": {
            "embedding_net": {
                "output_dim": 10,
                "image_size": 200,
            },
            "npe_params": {
                "embedding_dim": 10,
                "ntransform": 1,
            },
        },
    },
    "fmpe": {
        "config": FMPE_CONFIG,
        "training_params": {"rescale": "none"},
    },
    "fm_post_transform": {
        "config": {
            "npe": "fmpe",
            "flow_theta": FLOW_THETA_CONFIG,
            "flow_x": FLOW_X_CONFIG,
        },
        "training_params": {"rescale": "none"},
    },
    "rope": {
        "config": {"tau": 1.0, "gamma": 1.0},
        "training_params": {"rescale": "none"},
    },
}


# ============================================================================
# Joint metrics
# ============================================================================

def joint_mmd(theta_true, theta_samples, y_obs, sigma=1.0):
    """Joint MMD between (theta_true, y) and (theta_samples, y).

    Normalizes each dimension to unit variance before computing kernel.
    """
    y_flat = y_obs.reshape(len(y_obs), -1)
    joint_true = torch.cat([theta_true, y_flat], dim=1).float()
    joint_post = torch.cat([theta_samples, y_flat], dim=1).float()

    # Normalize to unit variance
    std = joint_true.std(dim=0).clamp(min=1e-6)
    joint_true = joint_true / std
    joint_post = joint_post / std

    # Gaussian kernel MMD
    def kernel(x, y):
        return torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))

    xx = kernel(joint_true, joint_true).mean()
    yy = kernel(joint_post, joint_post).mean()
    xy = kernel(joint_true, joint_post).mean()
    return float(xx + yy - 2 * xy)


def joint_c2st_with_embedding(theta_true, theta_samples, y_obs, npe_posterior, device):
    """Joint C2ST using NPE embedding to compress y before classification.

    Compares (theta_true, embed(y)) vs (theta_samples, embed(y)).
    Perfect calibration = 0.5.
    """
    # Extract embedding net from NPE
    if hasattr(npe_posterior, 'posterior'):
        npe_model = npe_posterior.posterior
    else:
        npe_model = npe_posterior

    embed_net = deepcopy(npe_model.embedding_net).to(device).eval()
    cond_rescaler = npe_model.cond_rescaler

    with torch.no_grad():
        y_rescaled = cond_rescaler.transform(y_obs.to(device))
        y_embed = embed_net(y_rescaled).cpu()

    # Build joint: (theta, embed(y))
    joint_true = torch.cat([theta_true, y_embed], dim=1).numpy()
    joint_post = torch.cat([theta_samples, y_embed], dim=1).numpy()

    n = len(theta_true)
    X = np.concatenate([joint_true, joint_post], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


# ============================================================================
# Data generation
# ============================================================================

def generate_calibration_data(sim, ncal, generation):
    """Generate calibration triples (theta, x, y) for a given generation mode."""
    theta = sim.sample_prior(ncal)
    x = sim.get_simulator(misspecified=True)(theta)

    if generation == "transitive":
        noisy_process = sim.get_noisy_process()
        y = noisy_process(x)
    else:  # independent
        y = sim.get_simulator(misspecified=False)(theta)

    return theta, x, y


# ============================================================================
# Foundation model training (once)
# ============================================================================

def train_foundation_models(sim, n_sim, device, model_path):
    """Train NPE and FMPE on simulation data. Only needs to be done once."""
    torch.manual_seed(0)
    np.random.seed(0)

    theta_sim = sim.sample_prior(n_sim)
    x_sim = sim.get_simulator(misspecified=True)(theta_sim)

    print("  Training NPE...")
    npe_config = TrainerConfig(
        epochs=500, lr=1e-4, batch_size=256, train_size=0.8,
        max_patience=20, rescale="none", save=True, load=True,
    )
    npe_trainer = NPETrainer(npe_config, model_path, TASK_CONFIG)
    npe_posterior = npe_trainer.train((theta_sim, x_sim), device, logname="npe")

    print("  Training FMPE...")
    fmpe_config = TrainerConfig(
        epochs=300, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=20, rescale="none", save=True, load=True,
    )
    fmpe_trainer = FMPETrainer(fmpe_config, model_path, TASK_CONFIG)
    fmpe_trainer.train((theta_sim, x_sim), device, logname="npe_fmpe")

    return npe_posterior


# ============================================================================
# Calibration training + evaluation for one setting
# ============================================================================

def run_one_setting(generation: str, ncal: int, seed: int, device: torch.device,
                    sim, npe_posterior, model_path: Path,
                    n_rope_pool: int = 2000,
                    sbc_sims: int = 500, sbc_samples: int = 500):
    """Train FMCPE and RoPE and evaluate for one (generation, seed) setting."""
    print(f"\n{'='*60}")
    print(f"Generation={generation}, ncal={ncal}, seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Calibration data (only this varies by generation + seed)
    theta_cal, x_cal, y_cal = generate_calibration_data(sim, ncal, generation)

    results = {"generation": generation, "ncal": ncal, "seed": seed}

    # --- Train FMCPE ---
    print("  Training FMCPE...")
    fmcpe_config = TrainerConfig(
        epochs=300, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=20, rescale="none", save=False, load=False,
    )
    fmcpe_trainer = FMPostTransformTrainer(fmcpe_config, model_path, TASK_CONFIG)
    fmcpe_posterior = fmcpe_trainer.train((theta_cal, x_cal, y_cal), device, logname="fmcpe")

    # --- Train RoPE ---
    print("  Training RoPE...")
    rope_config = TrainerConfig(
        epochs=500, lr=1e-5, batch_size=100, train_size=0.8,
        max_patience=20, rescale="none", save=False, load=False,
    )
    rope_trainer = RoPETrainer(rope_config, model_path, TASK_CONFIG)
    # deepcopy NPE to avoid grad state pollution across calls
    rope_posterior = rope_trainer.train((theta_cal, x_cal, y_cal), device, logname="rope",
                                       npe=deepcopy(npe_posterior))

    # Set RoPE simulation pool
    theta_pool = sim.sample_prior(n_rope_pool)
    x_pool = sim.get_simulator(misspecified=True)(theta_pool)
    rope_posterior.set_sim_data(x_pool, theta_pool)

    # --- Evaluate ---
    print("  Evaluating...")

    # Generate test (theta, y) pairs for joint metrics
    n_joint = sbc_sims
    theta_test = sim.sample_prior(n_joint)
    dgp = sim.get_simulator(misspecified=False)
    y_test = dgp(theta_test)

    for name, posterior in [("fmcpe", fmcpe_posterior), ("rope", rope_posterior)]:
        print(f"    {name} - SBC...")
        posterior.to(device)

        # Re-set sim data on device for RoPE
        if name == "rope":
            rope_posterior.set_sim_data(x_pool, theta_pool)

        try:
            ranks = run_sbc_dgp(posterior, sim, sbc_sims, sbc_samples, device)
            results[f"{name}_sbc_ranks_mean"] = float(ranks.float().mean())
            results[f"{name}_sbc_ranks_std"] = float(ranks.float().std())
        except Exception as e:
            print(f"      SBC failed for {name}: {e}")

        print(f"    {name} - Coverage...")
        try:
            cov = run_coverage_dgp(posterior, sim, sbc_sims, sbc_samples, device)
            results[f"{name}_acauc"] = cov["acauc"]
            results[f"{name}_mce"] = cov["mce"]
            results[f"{name}_coverage"] = {str(k): v for k, v in cov["empirical_coverage"].items()}
            print(f"      ACAUC={cov['acauc']:.4f}, MCE={cov['mce']:.4f}")
        except Exception as e:
            print(f"      Coverage failed for {name}: {e}")

        # Joint metrics: sample 1 theta per y_test
        print(f"    {name} - Joint metrics...")
        try:
            with torch.no_grad():
                theta_post = posterior.sample(y_test.to(device), 1, device).squeeze(0).cpu()

            mmd_val = joint_mmd(theta_test, theta_post, y_test)
            c2st_val = joint_c2st_with_embedding(theta_test, theta_post, y_test, npe_posterior, device)
            results[f"{name}_joint_mmd"] = mmd_val
            results[f"{name}_joint_c2st"] = c2st_val
            print(f"      Joint MMD={mmd_val:.4f}, Joint C2ST={c2st_val:.4f}")
        except Exception as e:
            print(f"      Joint metrics failed for {name}: {e}")

        posterior.cpu()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", nargs="+", type=str, default=["transitive", "independent"])
    parser.add_argument("--ncal", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 43, 53])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_sim", type=int, default=30000)
    parser.add_argument("--n_rope_pool", type=int, default=2000)
    parser.add_argument("--sbc_sims", type=int, default=500)
    parser.add_argument("--sbc_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="results/rebuttal_pendulum_independence")
    args = parser.parse_args()

    device = torch.device(args.device)

    sim = Pendulum()
    model_path = Path("/tmp/pendulum_independence/foundation")
    model_path.mkdir(parents=True, exist_ok=True)

    # Train foundation models once
    print("=" * 60)
    print("Training foundation models (NPE + FMPE) — once")
    print("=" * 60)
    npe_posterior = train_foundation_models(sim, args.n_sim, device, model_path)

    all_results = []

    for generation in args.generations:
        for seed in args.seeds:
            result = run_one_setting(
                generation=generation, ncal=args.ncal, seed=seed, device=device,
                sim=sim, npe_posterior=npe_posterior, model_path=model_path,
                n_rope_pool=args.n_rope_pool,
                sbc_sims=args.sbc_sims, sbc_samples=args.sbc_samples,
            )
            all_results.append(result)

    # Save raw results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Generation':<15} {'Method':<10} {'ACAUC':<18} {'MCE':<18} {'J-MMD':<18} {'J-C2ST':<18}")
    print("-" * 97)

    for generation in args.generations:
        gen_results = [r for r in all_results if r["generation"] == generation]

        for method in ["fmcpe", "rope"]:
            label = "FMCPE" if method == "fmcpe" else "RoPE"
            parts = [f"{generation:<15} {label:<10}"]

            for key in [f"{method}_acauc", f"{method}_mce", f"{method}_joint_mmd", f"{method}_joint_c2st"]:
                vals = [r[key] for r in gen_results if key in r]
                if vals:
                    parts.append(f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
                else:
                    parts.append("N/A")

            print("  ".join(parts))
        print()

    print(f"Results saved to {out_dir / 'raw_results.json'}")


if __name__ == "__main__":
    main()
