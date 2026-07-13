"""FMCPE flow_theta ablation on a synthetic CI-controlled task.

Synthetic DGP:
    θ ~ Uniform([-3,3]^5)
    x = Aθ + ε_x,       ε_x ~ N(0, σ_x²I),  A ∈ R^{20×5}
    y = Cx + α·Dθ + ε_y, ε_y ~ N(0, σ_y²I),  C ∈ R^{20×20}, D ∈ R^{20×5}

α controls CI violation:
    α=0  → y ⊥ θ | x  (CI holds, flow_theta should be unnecessary)
    α>0  → CI violated  (flow_theta captures direct θ→y effect)

Compares: FMCPE (full = flow_x + flow_theta) vs FMCPE (flow_x only).
Metrics: joint C2ST, joint MMD, joint Wasserstein.

Usage:
    python scripts/run_fmcpe_flow_theta_ablation.py
    python scripts/run_fmcpe_flow_theta_ablation.py --alphas 0.0 1.0 --seeds 33 43 --ncal 500
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from flow_matching.torch_flow import FlowMatching
from posteriors import DualFlowPosteriorEstimator, DirectFlowMatchingPosterior
from training.base import TrainerConfig
from training.trainers.calibration import FMPostTransformTrainer, _create_flow_from_config


# ============================================================================
# Synthetic task
# ============================================================================

class SyntheticCITask:
    """Synthetic task with tunable CI violation."""

    def __init__(self, theta_dim=5, obs_dim=20, alpha=0.0,
                 sigma_x=0.5, sigma_y=0.3, seed=0):
        rng = np.random.RandomState(seed)
        self.theta_dim = theta_dim
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # Fixed random matrices
        self.A = torch.tensor(rng.randn(obs_dim, theta_dim) / np.sqrt(theta_dim),
                              dtype=torch.float32)
        self.C = torch.tensor(rng.randn(obs_dim, obs_dim) * 0.3 / np.sqrt(obs_dim),
                              dtype=torch.float32)
        # Add identity to C so x→y is well-conditioned
        self.C = self.C + torch.eye(obs_dim)
        self.D = torch.tensor(rng.randn(obs_dim, theta_dim) / np.sqrt(theta_dim),
                              dtype=torch.float32)

    def sample_prior(self, n):
        return torch.rand(n, self.theta_dim) * 6 - 3  # Uniform[-3, 3]

    def simulate_x(self, theta):
        """x = Aθ + ε_x (misspecified simulator)."""
        return theta @ self.A.T + torch.randn(len(theta), self.obs_dim) * self.sigma_x

    def simulate_y(self, theta):
        """y from true DGP: sample x, then y = Cx + α·Dθ + ε_y."""
        x = self.simulate_x(theta)
        return x, self.y_from_x_theta(x, theta)

    def y_from_x_theta(self, x, theta):
        """y = Cx + α·Dθ + ε_y."""
        mean = x @ self.C.T + self.alpha * (theta @ self.D.T)
        return mean + torch.randn(len(x), self.obs_dim) * self.sigma_y

    def generate_sim_data(self, n):
        """Simulation data: (θ, x)."""
        theta = self.sample_prior(n)
        x = self.simulate_x(theta)
        return theta, x

    def generate_cal_data(self, n):
        """Calibration data: (θ, x, y) — independent generation."""
        theta = self.sample_prior(n)
        x = self.simulate_x(theta)
        y = self.y_from_x_theta(x, theta)
        return theta, x, y

    def generate_test_data(self, n):
        """Test data: (θ_true, y) from true DGP."""
        theta = self.sample_prior(n)
        _, y = self.simulate_y(theta)
        return theta, y


# ============================================================================
# Architecture config (resmlp, matches independence_ci.yaml)
# ============================================================================

RESMLP_FLOW = {
    "space": "data",
    "conditional": True,
    "probability_path": "ot2",
    "prior": "power",
    "base_dist": "gaussian",
    "params": {
        "probability_path_params": {},
        "prior_params": {"rate": 2.0},
        "base_dist_params": {},
        "drift": {
            "architecture": "resmlp",
            "hidden_dim": [64, 64, 64, 64, 64],
            "mlp_params": {},
        },
    },
}


def make_task_config(theta_dim, obs_dim):
    return {
        "name": "synthetic_ci",
        "rescale": "z_score",
        "fmpe": {
            "config": RESMLP_FLOW,
            "training_params": {"rescale": "z_score"},
        },
        "fm_post_transform": {
            "config": {
                "npe": "fmpe",
                "flow_theta": RESMLP_FLOW,
                "flow_x": RESMLP_FLOW,
            },
            "training_params": {"rescale": "z_score"},
        },
    }


# ============================================================================
# Joint metrics
# ============================================================================

def joint_mmd(theta_true, theta_samples, y_obs, sigma=1.0):
    joint_true = torch.cat([theta_true, y_obs], dim=1).float()
    joint_post = torch.cat([theta_samples, y_obs], dim=1).float()
    std = joint_true.std(dim=0).clamp(min=1e-6)
    joint_true, joint_post = joint_true / std, joint_post / std

    def kernel(x, y):
        return torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))

    return float(kernel(joint_true, joint_true).mean()
                 + kernel(joint_post, joint_post).mean()
                 - 2 * kernel(joint_true, joint_post).mean())


def joint_c2st(theta_true, theta_samples, y_obs):
    joint_true = torch.cat([theta_true, y_obs], dim=1).numpy()
    joint_post = torch.cat([theta_samples, y_obs], dim=1).numpy()
    n = len(theta_true)
    X = np.concatenate([joint_true, joint_post], axis=0)
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    return float(scores.mean())


def joint_wasserstein(theta_true, theta_samples, y_obs):
    joint_true = torch.cat([theta_true, y_obs], dim=1).float()
    joint_post = torch.cat([theta_samples, y_obs], dim=1).float()
    std = joint_true.std(dim=0).clamp(min=1e-6)
    joint_true, joint_post = (joint_true / std).numpy(), (joint_post / std).numpy()
    return float(np.mean([wasserstein_distance(joint_true[:, d], joint_post[:, d])
                          for d in range(joint_true.shape[1])]))


# ============================================================================
# Flow-x-only posterior wrapper
# ============================================================================

class FlowXOnlyPosterior:
    """Uses FMCPE's flow_x + base FMPE, skips flow_theta."""

    def __init__(self, fmcpe: DualFlowPosteriorEstimator):
        self.denoiser = fmcpe.denoiser
        self.proposal = fmcpe.proposal
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
        source = self.denoiser.sample_base(y, nsamples)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)
        source_flat = source.reshape(-1, *source.shape[2:])
        cond_flat = cond.reshape(-1, *cond.shape[2:])
        x_tilde = self.denoiser.sample(source_flat, cond_flat, device,
                                       only_last=True, num_steps=20)
        x_tilde = x_tilde.reshape(nsamples, y.shape[0], *self.denoiser.dim)
        x_flat = x_tilde.reshape(-1, *x_tilde.shape[2:])
        theta = self.proposal.sample(x_flat, 1, device).squeeze(0)
        return theta.reshape(nsamples, y.shape[0], *self.theta_dim)


# ============================================================================
# Train FMPE (foundation)
# ============================================================================

def train_fmpe(task, n_sim, device, model_path, task_config):
    theta_sim, x_sim = task.generate_sim_data(n_sim)

    from training.trainers.simulation import FMPETrainer
    config = TrainerConfig(
        epochs=200, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=20, rescale="z_score", save=True, load=True,
    )
    trainer = FMPETrainer(config, model_path, task_config)
    trainer.train((theta_sim, x_sim), device, logname="npe_fmpe")


# ============================================================================
# Train FMCPE
# ============================================================================

def train_fmcpe(cal_data, device, model_path, task_config):
    config = TrainerConfig(
        epochs=300, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=40, rescale="z_score", save=False, load=False,
    )
    trainer = FMPostTransformTrainer(config, model_path, task_config)
    return trainer.train(cal_data, device, logname="fmcpe")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--ncal", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 43, 53])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_sim", type=int, default=30000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--theta_dim", type=int, default=5)
    parser.add_argument("--obs_dim", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/rebuttal_flow_theta_ablation")
    args = parser.parse_args()

    device = torch.device(args.device)
    task_config = make_task_config(args.theta_dim, args.obs_dim)
    all_results = []

    for alpha in args.alphas:
        # Task with fixed matrices (seed=0 for matrices, varies for data)
        task = SyntheticCITask(
            theta_dim=args.theta_dim, obs_dim=args.obs_dim,
            alpha=alpha, seed=0,
        )

        # Train FMPE once per alpha (simulation data doesn't depend on alpha,
        # but model_path is per-alpha to avoid stale checkpoints)
        model_path = Path(f"/tmp/flow_theta_ablation/alpha_{alpha}")
        model_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"α = {alpha} — Training FMPE (foundation)")
        print(f"{'='*60}")
        torch.manual_seed(0)
        np.random.seed(0)
        train_fmpe(task, args.n_sim, device, model_path, task_config)

        for seed in args.seeds:
            print(f"\n  --- α={alpha}, seed={seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Calibration data
            theta_cal, x_cal, y_cal = task.generate_cal_data(args.ncal)

            # Train full FMCPE
            print("  Training FMCPE (full)...")
            fmcpe = train_fmcpe((theta_cal, x_cal, y_cal), device, model_path, task_config)

            # Create flow_x-only variant
            fmcpe_x_only = FlowXOnlyPosterior(fmcpe)

            # Test data
            theta_test, y_test = task.generate_test_data(args.n_test)

            result = {"alpha": alpha, "seed": seed, "ncal": args.ncal}

            for name, posterior in [("fmcpe", fmcpe), ("fmcpe_x_only", fmcpe_x_only)]:
                print(f"  Evaluating {name}...")
                posterior.to(device)

                with torch.no_grad():
                    theta_post = posterior.sample(y_test.to(device), 1, device).squeeze(0).cpu()

                c2st = joint_c2st(theta_test, theta_post, y_test)
                mmd = joint_mmd(theta_test, theta_post, y_test)
                wass = joint_wasserstein(theta_test, theta_post, y_test)

                result[f"{name}_c2st"] = c2st
                result[f"{name}_mmd"] = mmd
                result[f"{name}_wass"] = wass
                print(f"    C2ST={c2st:.4f}, MMD={mmd:.6f}, Wass={wass:.4f}")

                posterior.cpu()

            all_results.append(result)

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n\n" + "=" * 95)
    print("RESULTS: FMCPE (full) vs FMCPE (flow_x only)")
    print("=" * 95)
    print(f"{'α':<8} {'Method':<18} {'J-C2ST':<20} {'J-MMD':<20} {'J-Wass':<20}")
    print("-" * 86)

    for alpha in args.alphas:
        a_results = [r for r in all_results if r["alpha"] == alpha]
        for method in ["fmcpe", "fmcpe_x_only"]:
            label = "FMCPE (full)" if method == "fmcpe" else "FMCPE (x only)"
            c2st_vals = [r[f"{method}_c2st"] for r in a_results]
            mmd_vals = [r[f"{method}_mmd"] for r in a_results]
            wass_vals = [r[f"{method}_wass"] for r in a_results]
            print(f"{alpha:<8} {label:<18} "
                  f"{np.mean(c2st_vals):.4f} ± {np.std(c2st_vals):.4f}  "
                  f"{np.mean(mmd_vals):.6f} ± {np.std(mmd_vals):.6f}  "
                  f"{np.mean(wass_vals):.4f} ± {np.std(wass_vals):.4f}")
        print()

    print(f"Results saved to {out_dir / 'raw_results.json'}")


if __name__ == "__main__":
    main()
