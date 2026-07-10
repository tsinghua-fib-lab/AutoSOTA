"""Run a single Double Well training with the rubric settings and report KL divergence."""
import os, sys, yaml, json, pickle, jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from src.consistency_bridge import ConsistencyBridge
from src.evaluation import compute_KL_to_ground_truth, compute_KL_to_reference
from problems import PROBLEMS
from train_bridge import _build_model

# Load config
with open("configs/double_well.yaml") as f:
    cfg = yaml.safe_load(f)

outdir = "/repo/outputs/dw_single2"
os.makedirs(outdir, exist_ok=True)

# Build problem and model
problem = PROBLEMS[cfg["problem"]["name"]](cfg["problem"])
d = problem["shape"][0]
model = _build_model(cfg["model"], d)

# Build bridge
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

# Train
train_config = dict(cfg["train"])
seed = train_config.pop("seed", 0)
key = jax.random.PRNGKey(seed)
print(f"Starting training with {train_config['num_outer_iterations']} iterations...")
state, ema_params_lst, ema_grad_norms = bridge.train(key, train_config)
print("Training done.")

# Evaluate
ema_params = ema_params_lst[-1]
eval_key = jax.random.PRNGKey(seed + 999)

def learned_drift_fn(x, t):
    return bridge.controlled_drift(ema_params, x, t)

sigma = problem["sigma_fn"]
T = problem["T"]
num_steps = train_config["num_steps"]

if "true_drift_fn" in problem:
    kl_sol, _ = compute_KL_to_ground_truth(
        eval_key, bridge, problem["true_drift_fn"], learned_drift_fn, sigma, T, num_steps=num_steps,
    )
    print(f"kl_to_solution: {kl_sol:.4f}")

    kl_ref_truth, _ = compute_KL_to_reference(
        eval_key, bridge, problem["true_drift_fn"], problem["base_drift"], sigma, T, num_steps=num_steps,
    )
    print(f"kl_to_reference_truth: {kl_ref_truth:.4f}")

kl_ref_learned, _ = compute_KL_to_reference(
    eval_key, bridge, learned_drift_fn, problem["base_drift"], sigma, T, num_steps=num_steps,
)
print(f"kl_to_reference_learned: {kl_ref_learned:.4f}")

results = {
    "kl_to_reference_learned": float(kl_ref_learned),
    "kl_to_solution": float(kl_sol),
    "kl_to_reference_truth": float(kl_ref_truth),
}
with open(os.path.join(outdir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Done.", results)
