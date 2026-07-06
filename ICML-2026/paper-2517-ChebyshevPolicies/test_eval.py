import sys
sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/algorithms")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import exp_run

# Train a quick policy
kwargs = dict(
    algo="ars",
    env_name="MountainCarContinuous-v0",
    degree=3,
    basis="chebyshev",
    tensorboard_log_dir="/repo/tensorboard_logs/test2/",
    delta_std=0.1,
    n_delta=4,
    n_top=1,
    learning_rate=0.018,
    steps=5000,
    evaluate_every_n_steps=0,
    normalize_actions=True,
    min_action=-1.0,
    max_action=1.0,
    initialization="random",
    zero_policy=False,
    seed=42,
    verbose=0,
    name="test_eval",
)

print("Training quick policy (5000 steps)...")
result = exp_run.run_sb3_polyagent_training(kwargs)
name, duration, coeffs = result
print(f"Trained in {duration}s")

# Test evaluation with options
print("\nTesting evaluation with deterministic starts...")
model, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
    basis="chebyshev", env_name="MountainCarContinuous-v0", coeffs=coeffs, algo="ars")
r_mean, r_std = exp_run.run_sb3_model(model, eval_env, options={"low": -0.5, "high": -0.5})
print(f"  Reward at x0=-0.5: {r_mean:.4f} +/- {r_std:.4f}")

# Test on multiple positions
xs = np.linspace(-0.6, -0.4, 10)
rewards = []
for x in xs:
    model2, eval_env2 = exp_run.get_sb3_polynomial_model_and_eval_env(
        basis="chebyshev", env_name="MountainCarContinuous-v0", coeffs=coeffs, algo="ars")
    r_mean, _ = exp_run.run_sb3_model(model2, eval_env2, options={"low": x, "high": x})
    rewards.append(r_mean)

print(f"\n10-point evaluation:")
print(f"  Mean R: {np.mean(rewards):.4f}")
print(f"  Min R:  {np.min(rewards):.4f}")
print(f"  Max R:  {np.max(rewards):.4f}")

print("\nSUCCESS: Evaluation pipeline working!")
