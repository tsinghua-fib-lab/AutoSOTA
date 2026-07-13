"""FMCPE ablation: full FMCPE vs flow_x-only (no flow_theta).

When CI holds (transitive), flow_x alone should suffice since p(θ|x,y) = p(θ|x).
When CI is violated (independent), flow_theta is needed to capture direct y→θ dependency.

Evaluation: joint C2ST (with embedding), joint MMD, joint Wasserstein.

Usage:
    python scripts/run_fmcpe_ablation.py --generations transitive independent --seeds 33 43 53
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from posteriors import DualFlowPosteriorEstimator
from simulator.pendulum import Pendulum
from training.base import TrainerConfig
from training.trainers.simulation import FMPETrainer
from training.trainers.calibration import FMPostTransformTrainer


# ============================================================================
# Task config (from configs/task/pendulum.yaml)
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
            "embedding_kwargs": {"name": "conv1d", "output_dim": 10},
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
            "embedding_kwargs": {"name": "conv1d", "output_dim": 10},
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
            "embedding_kwargs": {"name": "conv1d", "output_dim": 10},
        },
    },
}

TASK_CONFIG = {
    "name": "pendulum",
    "rescale": "none",
    "fmpe": {"config": FMPE_CONFIG, "training_params": {"rescale": "none"}},
    "fm_post_transform": {
        "config": {
            "npe": "fmpe",
            "flow_theta": FLOW_THETA_CONFIG,
            "flow_x": FLOW_X_CONFIG,
        },
        "training_params": {"rescale": "none"},
    },
}


# ============================================================================
# Joint metrics
# ============================================================================

def joint_mmd(theta_true, theta_samples, y_obs, sigma=1.0):
    """Joint MMD with per-dimension normalization."""
    y_flat = y_obs.reshape(len(y_obs), -1)
    joint_true = torch.cat([theta_true, y_flat], dim=1).float()
    joint_post = torch.cat([theta_samples, y_flat], dim=1).float()

    std = joint_true.std(dim=0).clamp(min=1e-6)
    joint_true = joint_true / std
    joint_post = joint_post / std

    def kernel(x, y):
        return torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))

    return float(kernel(joint_true, joint_true).mean()
                 + kernel(joint_post, joint_post).mean()
                 - 2 * kernel(joint_true, joint_post).mean())


def joint_c2st(theta_true, theta_samples, y_obs, n_components=10):
    """Joint C2ST with PCA-reduced y (200-dim → n_components)."""
    from sklearn.decomposition import PCA

    y_flat = y_obs.reshape(len(y_obs), -1).numpy()
    pca = PCA(n_components=n_components).fit(y_flat)
    y_reduced = pca.transform(y_flat)

    joint_true = np.concatenate([theta_true.numpy(), y_reduced], axis=1)
    joint_post = np.concatenate([theta_samples.numpy(), y_reduced], axis=1)

    n = len(theta_true)
    X = np.concatenate([joint_true, joint_post], axis=0)
    labels = np.concatenate([np.zeros(n), np.ones(n)])

    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    return float(scores.mean())


def joint_wasserstein(theta_true, theta_samples, y_obs):
    """Sliced joint Wasserstein (average over dimensions) with normalization."""
    y_flat = y_obs.reshape(len(y_obs), -1)
    joint_true = torch.cat([theta_true, y_flat], dim=1).float()
    joint_post = torch.cat([theta_samples, y_flat], dim=1).float()

    # Normalize
    std = joint_true.std(dim=0).clamp(min=1e-6)
    joint_true = (joint_true / std).numpy()
    joint_post = (joint_post / std).numpy()

    distances = []
    for d in range(joint_true.shape[1]):
        distances.append(wasserstein_distance(joint_true[:, d], joint_post[:, d]))
    return float(np.mean(distances))


# ============================================================================
# Helpers
# ============================================================================

def generate_calibration_data(sim, ncal, generation):
    theta = sim.sample_prior(ncal)
    x = sim.get_simulator(misspecified=True)(theta)
    if generation == "transitive":
        y = sim.get_noisy_process()(x)
    else:
        y = sim.get_simulator(misspecified=False)(theta)
    return theta, x, y


def make_flow_x_only_posterior(fmcpe_posterior):
    """Create a posterior that uses only flow_x (no flow_theta correction).

    Sampling: y → flow_x → x̃ → FMPE → θ (skip flow_theta stage).
    """
    return _FlowXOnlyPosterior(fmcpe_posterior)


class _FlowXOnlyPosterior:
    """Wrapper: uses FMCPE's flow_x + base FMPE, skips flow_theta."""

    def __init__(self, fmcpe: DualFlowPosteriorEstimator):
        self.denoiser = fmcpe.denoiser       # flow_x
        self.proposal = fmcpe.proposal        # FMPE
        self.theta_dim = fmcpe.theta_dim

    def to(self, device):
        self.denoiser.to(device)
        self.proposal.to(device)
        return self

    def cpu(self):
        self.denoiser.cpu()
        self.proposal.cpu()
        return self

    def sample(self, y, nsamples, device, **kwargs):
        """Sample x̃ from flow_x(·|y), then θ from FMPE(·|x̃)."""
        # Stage 1: y → x̃ via flow_x
        source = self.denoiser.sample_base(y, nsamples)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)

        source_flat = source.reshape(-1, *source.shape[2:])
        cond_flat = cond.reshape(-1, *cond.shape[2:])
        x_tilde = self.denoiser.sample(source_flat, cond_flat, device,
                                       only_last=True, num_steps=20)
        x_tilde = x_tilde.reshape(nsamples, y.shape[0], *self.denoiser.dim)
        x_flat = x_tilde.reshape(-1, *x_tilde.shape[2:])

        # Stage 2: x̃ → θ via FMPE (no flow_theta)
        theta = self.proposal.sample(x_flat, 1, device).squeeze(0)
        return theta.reshape(nsamples, y.shape[0], *self.theta_dim)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", nargs="+", default=["transitive", "independent"])
    parser.add_argument("--ncal", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 43, 53])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_sim", type=int, default=30000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--output", type=str, default="results/rebuttal_fmcpe_ablation")
    args = parser.parse_args()

    device = torch.device(args.device)
    sim = Pendulum()

    model_path = Path("/tmp/fmcpe_ablation/foundation")
    model_path.mkdir(parents=True, exist_ok=True)

    # Train FMPE once
    print("=" * 60)
    print("Training FMPE (foundation) — once")
    print("=" * 60)
    torch.manual_seed(0)
    np.random.seed(0)
    theta_sim = sim.sample_prior(args.n_sim)
    x_sim = sim.get_simulator(misspecified=True)(theta_sim)

    fmpe_config = TrainerConfig(
        epochs=300, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=20, rescale="none", save=True, load=True,
    )
    fmpe_trainer = FMPETrainer(fmpe_config, model_path, TASK_CONFIG)
    fmpe_trainer.train((theta_sim, x_sim), device, logname="npe_fmpe")

    all_results = []

    for generation in args.generations:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Generation={generation}, ncal={args.ncal}, seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            theta_cal, x_cal, y_cal = generate_calibration_data(sim, args.ncal, generation)

            # Train full FMCPE
            print("  Training FMCPE (full)...")
            fmcpe_config = TrainerConfig(
                epochs=300, lr=1e-3, batch_size=256, train_size=0.8,
                max_patience=20, rescale="none", save=False, load=False,
            )
            fmcpe_trainer = FMPostTransformTrainer(fmcpe_config, model_path, TASK_CONFIG)
            fmcpe = fmcpe_trainer.train((theta_cal, x_cal, y_cal), device, logname="fmcpe")

            # Create flow_x-only variant (reuses trained flow_x, skips flow_theta)
            fmcpe_x_only = make_flow_x_only_posterior(fmcpe)

            # Generate test data
            theta_test = sim.sample_prior(args.n_test)
            y_test = sim.get_simulator(misspecified=False)(theta_test)

            result = {"generation": generation, "ncal": args.ncal, "seed": seed}

            # Evaluate both variants
            for name, posterior in [("fmcpe", fmcpe), ("fmcpe_x_only", fmcpe_x_only)]:
                print(f"  Evaluating {name}...")
                posterior.to(device)

                with torch.no_grad():
                    theta_post = posterior.sample(y_test.to(device), 1, device).squeeze(0).cpu()

                mmd = joint_mmd(theta_test, theta_post, y_test)
                c2st = joint_c2st(theta_test, theta_post, y_test)
                wass = joint_wasserstein(theta_test, theta_post, y_test)

                result[f"{name}_joint_mmd"] = mmd
                result[f"{name}_joint_c2st"] = c2st
                result[f"{name}_joint_wass"] = wass
                print(f"    J-C2ST={c2st:.4f}, J-MMD={mmd:.6f}, J-Wass={wass:.4f}")

                posterior.cpu()

            all_results.append(result)

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n\n" + "=" * 90)
    print("RESULTS SUMMARY: FMCPE (full) vs FMCPE (flow_x only)")
    print("=" * 90)
    print(f"{'Generation':<15} {'Method':<18} {'J-C2ST':<18} {'J-MMD':<18} {'J-Wass':<18}")
    print("-" * 87)

    for generation in args.generations:
        gen_results = [r for r in all_results if r["generation"] == generation]
        for method in ["fmcpe", "fmcpe_x_only"]:
            label = "FMCPE (full)" if method == "fmcpe" else "FMCPE (x only)"
            parts = [f"{generation:<15} {label:<18}"]
            for key in [f"{method}_joint_c2st", f"{method}_joint_mmd", f"{method}_joint_wass"]:
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
