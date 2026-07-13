"""Run the conditional independence experiment: FMCPE vs RoPE across D scales.

Trains NPE, FMPE, FMCPE, and RoPE on PureGaussian tasks with varying degrees
of conditional independence violation, then evaluates against the analytical
true posterior p(θ|y).

Usage:
    python scripts/run_independence_experiment.py
    python scripts/run_independence_experiment.py --ncal 200 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from simulator.pure_gaussian import PureGaussian
from training.base import TrainerConfig
from training.trainers.simulation import NPETrainer, FMPETrainer
from training.trainers.calibration import FMPostTransformTrainer
from training.trainers.baselines import RoPETrainer


# ============================================================================
# Config
# ============================================================================

SIMULATOR_PARAMS = dict(
    theta_dim=3, obs_dim=10,
    prior_var_scale=1, likelihood_var_scale=1, noisy_var_scale=0.1,
    seed=0,
)

FMPE_CONFIG = {
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

TASK_CONFIG = {
    "name": "independence",
    "rescale": "none",
    "npe": {
        "training": {"rescale": "none"},
        "params": {
            "embedding_net": {},
            "npe_params": {"transforms": 5, "hidden_features": [64, 64], "bins": 8},
        },
    },
    "fmpe": {"config": FMPE_CONFIG, "training_params": {"rescale": "none"}},
    "fm_post_transform": {
        "config": {
            "npe": "fmpe",
            "flow_theta": FMPE_CONFIG,
            "flow_x": FMPE_CONFIG,
        },
        "training_params": {"rescale": "none"},
    },
    "rope": {
        "config": {"tau": 1.0, "gamma": 1.0},
        "training_params": {"rescale": "none"},
    },
}


# ============================================================================
# Metrics
# ============================================================================

def c2st(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Classifier 2-Sample Test using logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n_a, n_b = len(samples_a), len(samples_b)
    X = np.concatenate([samples_a, samples_b], axis=0)
    y = np.concatenate([np.zeros(n_a), np.ones(n_b)])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


def mean_wasserstein(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Mean 1D Wasserstein distance across dimensions."""
    d = samples_a.shape[1]
    return float(np.mean([
        wasserstein_distance(samples_a[:, i], samples_b[:, i]) for i in range(d)
    ]))


def evaluate_posterior(posterior, y_test, true_samples, device, sim=None,
                       n_samples=1000, method_name=""):
    """Evaluate a posterior against true posterior samples.

    Args:
        posterior: trained posterior object
        y_test: (n_obs, obs_dim) test observations
        true_samples: (n_samples, n_obs, theta_dim) reference samples
        device: torch device
        sim: simulator (needed for RoPE to set sim data)
        n_samples: number of samples per observation
        method_name: for logging
    """
    n_obs = y_test.shape[0]
    posterior.to(device)

    # For RoPE, set simulation data
    if hasattr(posterior, "set_sim_data") and sim is not None:
        sim_theta = sim.sample_prior(5000)
        sim_x = sim.get_simulator(misspecified=True)(sim_theta)
        posterior.set_sim_data(sim_x, sim_theta)

    with torch.no_grad():
        try:
            model_samples = posterior.sample(y_test.to(device), n_samples, device)
        except Exception as e:
            print(f"    Sampling failed for {method_name}: {e}")
            return {"c2st": float("nan"), "wasserstein": float("nan")}

    # Ensure shape is (n_samples, n_obs, theta_dim)
    if model_samples.dim() == 2:
        model_samples = model_samples.unsqueeze(0)
    model_samples = model_samples.cpu().numpy()
    true_np = true_samples.numpy()

    # Per-observation metrics, then average
    c2st_vals, w_vals = [], []
    for i in range(n_obs):
        ms = model_samples[:, i, :]
        ts = true_np[:, i, :]
        c2st_vals.append(c2st(ms, ts))
        w_vals.append(mean_wasserstein(ms, ts))

    posterior.cpu()

    return {
        "c2st": float(np.mean(c2st_vals)),
        "c2st_std": float(np.std(c2st_vals)),
        "wasserstein": float(np.mean(w_vals)),
        "wasserstein_std": float(np.std(w_vals)),
    }


# ============================================================================
# Training
# ============================================================================

def run_one_setting(d_scale: float, ncal: int, seed: int, device: torch.device,
                    n_sim: int = 30000, n_test_obs: int = 50,
                    n_posterior_samples: int = 1000):
    """Train and evaluate all methods for one (d_scale, ncal, seed) setting."""
    print(f"\n{'='*60}")
    print(f"D scale={d_scale}, ncal={ncal}, seed={seed}")
    print(f"{'='*60}")

    sim = PureGaussian(**SIMULATOR_PARAMS, direct_theta_scale=d_scale)
    mi = sim.conditional_mutual_information()
    print(f"  I(y; θ | x) = {mi:.4f} nats")

    # Seed AFTER simulator construction (PureGaussian resets torch.manual_seed internally)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    theta_sim = sim.sample_prior(n_sim)
    x_sim = sim.get_simulator(misspecified=True)(theta_sim)

    theta_cal = sim.sample_prior(ncal)
    x_cal = sim.get_simulator(misspecified=True)(theta_cal)
    y_cal = sim.get_simulator(misspecified=False)(theta_cal)

    theta_test = sim.sample_prior(n_test_obs)
    y_test = sim.get_simulator(misspecified=False)(theta_test)

    # Reference posterior samples
    true_samples = sim.sample_reference_posterior(n_posterior_samples, y_test, misspecified=False)

    # Temporary model path
    model_path = Path(f"/tmp/independence_exp/d{d_scale}_ncal{ncal}_seed{seed}")
    model_path.mkdir(parents=True, exist_ok=True)

    results = {"d_scale": d_scale, "ncal": ncal, "seed": seed, "mi": mi}

    # --- Train NPE (SBI-style, needed for RoPE) ---
    print("  Training NPE...")
    npe_config = TrainerConfig(
        epochs=500, lr=1e-4, batch_size=256, train_size=0.8,
        max_patience=20, rescale="none", save=True, load=False,
    )
    npe_trainer = NPETrainer(npe_config, model_path, TASK_CONFIG)
    npe_posterior = npe_trainer.train((theta_sim, x_sim), device, logname="npe")

    # --- Train FMPE (needed for FMCPE) ---
    print("  Training FMPE...")
    fmpe_config = TrainerConfig(
        epochs=200, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=20, rescale="none", save=True, load=False,
    )
    fmpe_trainer = FMPETrainer(fmpe_config, model_path, TASK_CONFIG)
    fmpe_posterior = fmpe_trainer.train((theta_sim, x_sim), device, logname="npe_fmpe")

    # --- Train FMCPE (fm_post_transform) ---
    print("  Training FMCPE...")
    fmcpe_config = TrainerConfig(
        epochs=200, lr=1e-3, batch_size=256, train_size=0.8,
        max_patience=None, rescale="none", save=False, load=False,
    )
    fmcpe_trainer = FMPostTransformTrainer(fmcpe_config, model_path, TASK_CONFIG)
    fmcpe_posterior = fmcpe_trainer.train((theta_cal, x_cal, y_cal), device, logname="fmcpe")

    # --- Train RoPE ---
    print("  Training RoPE...")
    rope_config = TrainerConfig(
        epochs=500, lr=1e-5, batch_size=100, train_size=0.8,
        max_patience=10, rescale="none", save=False, load=False,
    )
    rope_trainer = RoPETrainer(rope_config, model_path, TASK_CONFIG)
    rope_posterior = rope_trainer.train((theta_cal, x_cal, y_cal), device, logname="rope",
                                       npe=npe_posterior)

    # --- Evaluate ---
    print("  Evaluating...")
    for name, posterior in [("fmcpe", fmcpe_posterior), ("rope", rope_posterior)]:
        print(f"    {name}...")
        metrics = evaluate_posterior(
            posterior, y_test, true_samples, device,
            sim=sim, n_samples=n_posterior_samples, method_name=name,
        )
        results[name] = metrics
        print(f"      C2ST={metrics['c2st']:.4f}, W={metrics['wasserstein']:.4f}")

    # Clean up
    import shutil
    shutil.rmtree(model_path, ignore_errors=True)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--ncal", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 43, 53])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_sim", type=int, default=30000)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/rebuttal_independence")
    args = parser.parse_args()

    device = torch.device(args.device)
    all_results = []

    for d_scale in args.d_scales:
        for seed in args.seeds:
            result = run_one_setting(
                d_scale=d_scale, ncal=args.ncal, seed=seed, device=device,
                n_sim=args.n_sim, n_test_obs=args.n_test,
                n_posterior_samples=args.n_samples,
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
    print(f"{'D scale':<10} {'MI (nats)':<12} {'Method':<10} {'C2ST':<18} {'Wasserstein':<18}")
    print("-" * 68)

    for d_scale in args.d_scales:
        scale_results = [r for r in all_results if r["d_scale"] == d_scale]
        mi = scale_results[0]["mi"]

        for method in ["fmcpe", "rope"]:
            c2st_vals = [r[method]["c2st"] for r in scale_results]
            w_vals = [r[method]["wasserstein"] for r in scale_results]

            c2st_mean = np.mean(c2st_vals)
            c2st_std = np.std(c2st_vals)
            w_mean = np.mean(w_vals)
            w_std = np.std(w_vals)

            label = "FMCPE" if method == "fmcpe" else "RoPE"
            print(f"{d_scale:<10.1f} {mi:<12.2f} {label:<10} "
                  f"{c2st_mean:.4f} ± {c2st_std:.4f}  {w_mean:.4f} ± {w_std:.4f}")
        print()

    print(f"Results saved to {out_dir / 'raw_results.json'}")


if __name__ == "__main__":
    main()
