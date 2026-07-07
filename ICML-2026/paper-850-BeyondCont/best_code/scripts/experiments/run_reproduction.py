"""
Reproduction script for paper 850 USB - Simulation (2D) benchmark.

Rubric configuration:
- model_architecture: MLP_256_hidden_5_layers_LeakyReLU
- benchmark: Simulation Gene, dim=2
- time_points: 5
- epochs: 1000 (gradient steps)
- lr_schedule: constant
- optimizer: Adam
- growth_penalty: 1.3 (delta)
- nu: 0.001
- n_runs: 5 (different random seeds)
- Metrics: W1 (lower_better, target 0.019), RME (lower_better, target 0.002)
"""

from pathlib import Path
import sys
import json
import os

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from utils import compute_uot_plans, USB, SDE, wasserstein, wasserstein_with_weights, sample_from_ot_plan
import time as TIME

# ==================================================
# Configuration (matching rubric exactly)
# ==================================================
CONFIG = {
    "batch_size": 256,
    "nu": 0.001,
    "steps": 1000,           # rubric: epochs=1000
    "delta": 1.3,            # rubric: growth_penalty=1.3
    "lr": 0.001,             # Adam default lr
    "lr_schedule": "constant",  # rubric: lr_schedule=constant
    "seeds": [42, 123, 456, 789, 1024],  # rubric: n_runs=5
    "model_dims": None,      # will be set as [dim+1, 256, 256, 256, 256, 256]
    "activation": "leakyrelu",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

# ==================================================
# Data loading
# ==================================================
print("Begin data loading...")
data_name = "simulation"
data = pd.read_csv("data/simulation_gene_data.csv")

output_dir = Path("results/simulation_reproduction")
output_dir.mkdir(parents=True, exist_ok=True)

Xs = []
mass_ratio = []
max_sample_time = int(np.max(data["samples"]))
for k in range(max_sample_time + 1):
    Xs.append(np.array(data[data["samples"] == k])[:, 1:])
    mass_ratio.append(Xs[k].shape[0] / Xs[0].shape[0])

dim = Xs[0].shape[1]
print(f"Data dim: {dim}, time points: {max_sample_time + 1}")
print(f"Samples per time point: {[len(x) for x in Xs]}")
print(f"Mass ratios: {[round(m, 3) for m in mass_ratio]}")

# Set model dims: [dim+1, 256, 256, 256, 256, 256] = 5 hidden layers of 256
CONFIG["model_dims"] = [dim + 1, 256, 256, 256, 256, 256]
print(f"Model architecture: {CONFIG['model_dims']} with {CONFIG['activation']}")

t_train = np.arange(len(Xs)).tolist()
samples_per_interval = np.array(data).shape[0]

# ==================================================
# Compute UOT Plans (done once, shared across seeds)
# ==================================================
delta = CONFIG["delta"]
print(f"\n=== Delta (growth_penalty) = {delta} ===")
start_time = TIME.perf_counter()

print("Computing UOT plans...")
uot_plans, gamma0_plans, gamma1_plans, true_action = compute_uot_plans(
    Xs, t_train, delta=delta, cuda=True
)
print(f"UOT plans computed. True action: {true_action:.4f}")

# ==================================================
# Prepare training data (same for all seeds)
# ==================================================
print("Pre-sampling training data on CPU -> GPU...")

all_x0, all_x1, all_m0, all_m1, all_t_start, all_dt = [], [], [], [], [], []

for k in range(len(t_train) - 1):
    gamma0_plan = gamma0_plans[k]
    gamma1_plan = gamma1_plans[k]

    x0_np, x1_np, idx_0, idx_1 = sample_from_ot_plan(
        gamma0_plan, Xs[k], Xs[k + 1], samples_per_interval
    )
    ratio = (gamma1_plan[idx_0, idx_1] / gamma0_plan[idx_0, idx_1]).reshape(-1, 1)
    m1_np = np.log(1e-8 + ratio)
    m0_np = np.zeros_like(m1_np)

    t_s = t_train[k]
    d_t = t_train[k + 1] - t_train[k]

    all_x0.append(torch.tensor(x0_np, dtype=torch.float32))
    all_x1.append(torch.tensor(x1_np, dtype=torch.float32))
    all_m0.append(torch.tensor(m0_np, dtype=torch.float32))
    all_m1.append(torch.tensor(m1_np, dtype=torch.float32))
    all_t_start.append(torch.full((len(x0_np), 1), t_s, dtype=torch.float32))
    all_dt.append(torch.full((len(x0_np), 1), d_t, dtype=torch.float32))

train_x0 = torch.cat(all_x0).to(device)
train_x1 = torch.cat(all_x1).to(device)
train_m0 = torch.cat(all_m0).to(device)
train_m1 = torch.cat(all_m1).to(device)
train_t_start = torch.cat(all_t_start).to(device)
train_dt = torch.cat(all_dt).to(device)

train_dataset = TensorDataset(train_x0, train_x1, train_m0, train_m1, train_t_start, train_dt)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)

print(f"Data prepared. Total samples: {len(train_dataset)}. Batches per epoch: {len(train_loader)}")

# ==================================================
# Run across seeds
# ==================================================
all_results = []

for run_idx, seed in enumerate(CONFIG["seeds"]):
    print(f"\n{'='*60}")
    print(f"RUN {run_idx + 1}/{len(CONFIG['seeds'])}: seed={seed}")
    print(f"{'='*60}")

    run_start = TIME.perf_counter()

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = USB(CONFIG["model_dims"], nu=CONFIG["nu"], activation=CONFIG["activation"]).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer: Adam with constant LR (no scheduler)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    # Constant LR schedule: no scheduler
    print(f"Optimizer: Adam, lr={CONFIG['lr']}, schedule={CONFIG['lr_schedule']}")

    vlosses, slosses, klosses, losses = [], [], [], []

    model.train()
    global_step = 0
    keep_training = True

    print(f"Training for {CONFIG['steps']} steps...")

    while keep_training:
        for batch_idx, (b_x0, b_x1, b_m0, b_m1, b_ts, b_dt) in enumerate(train_loader):
            optimizer.zero_grad()

            relative_t = torch.rand((b_x0.shape[0], 1), device=device)
            real_t = b_dt * relative_t + b_ts

            xts_samp, vts_samp, kts_samp, eps_samp, mts_samp = model.sample_comditional_flow(
                b_x0, b_x1, b_m0, b_m1, relative_t
            )

            # Time rescaling
            kts_samp = kts_samp / b_dt
            vts_samp = vts_samp / b_dt

            v, s, k_out = model.forward(xts_samp, real_t)

            # Loss computation with mass weights
            weights = torch.exp(mts_samp)
            v_loss = torch.mean(torch.pow(v - vts_samp, 2) * weights)

            bridge_term = 2 * torch.sqrt(relative_t * (1 - relative_t)) / (CONFIG["nu"] + 1e-8)
            s_loss = torch.mean(torch.pow(bridge_term * s + eps_samp, 2) * weights)

            k_loss = torch.mean(torch.pow(k_out - kts_samp, 2) * weights)

            loss = v_loss + s_loss + k_loss

            loss.backward()
            optimizer.step()
            # No scheduler step - constant LR

            global_step += 1

            if global_step % 200 == 0:
                print(f"  Step {global_step}/{CONFIG['steps']} | Loss: {loss.item():.4f} "
                      f"(v: {v_loss.item():.4f}, s: {s_loss.item():.4f}, k: {k_loss.item():.4f})")
                vlosses.append(v_loss.item())
                slosses.append(s_loss.item())
                klosses.append(k_loss.item())
                losses.append(loss.item())

            if global_step >= CONFIG["steps"]:
                keep_training = False
                break

    train_time = TIME.perf_counter() - run_start
    print(f"Training completed in {train_time:.1f}s")

    # ==================================================
    # Inference & Evaluation
    # ==================================================
    print("Running inference...")
    model.eval()
    model.to("cpu")

    simulator = SDE(model, CONFIG["nu"], mode=1, positive=False)

    x_source = torch.tensor(Xs[0], dtype=torch.float32, device="cpu")

    xs, ms, action = simulator.trajectory(
        x=x_source,
        m=torch.zeros([x_source.size(0), 1], device="cpu"),
        delta=CONFIG["delta"],
        T=max_sample_time,
        N=max_sample_time * 100,
    )

    # Evaluate metrics
    wa_unnormalized_list = []
    wa_normalized_list = []
    RME_list = []

    for k in range(1, len(Xs)):
        timestep_idx = k * 100
        if timestep_idx >= xs.shape[0]:
            timestep_idx = xs.shape[0] - 1

        x1s = xs[timestep_idx, :]
        m1s = np.exp(ms[timestep_idx, :])

        # W1 (Wasserstein-1 distance)
        w1_unnorm = wasserstein(
            torch.Tensor(np.array(x1s)),
            torch.tensor(Xs[k], dtype=torch.float32),
            power=1,
        )
        w1_norm = wasserstein_with_weights(
            torch.Tensor(np.array(x1s)),
            np.array(m1s),
            torch.Tensor(Xs[k]),
            np.ones(Xs[k].shape[0]),
            power=1,
        )

        # RME (Relative Mass Error)
        RME = np.abs((np.mean(m1s) - mass_ratio[k]) / mass_ratio[k])

        wa_unnormalized_list.append(w1_unnorm)
        wa_normalized_list.append(w1_norm)
        RME_list.append(RME)

    total_time = TIME.perf_counter() - start_time

    # Compute the mean W1 and RME across all time intervals
    mean_w1 = float(np.mean(wa_normalized_list))
    mean_rme = float(np.mean(RME_list))

    run_result = {
        "seed": seed,
        "run": run_idx + 1,
        "w1_per_timestep": [float(w) for w in wa_normalized_list],
        "w1_unnorm_per_timestep": [float(w) for w in wa_unnormalized_list],
        "rme_per_timestep": [float(r) for r in RME_list],
        "mean_w1": mean_w1,
        "mean_rme": mean_rme,
        "train_time_s": train_time,
        "action": float(action),
    }
    all_results.append(run_result)

    print(f"\nResults for seed {seed}:")
    print(f"  W1 (normalized) per timestep: {[f'{w:.4f}' for w in wa_normalized_list]}")
    print(f"  W1 mean: {mean_w1:.4f}")
    print(f"  RME per timestep: {[f'{r:.4f}' for r in RME_list]}")
    print(f"  RME mean: {mean_rme:.4f}")

# ==================================================
# Summary across runs
# ==================================================
w1_values = [r["mean_w1"] for r in all_results]
rme_values = [r["mean_rme"] for r in all_results]

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"W1 values across {len(CONFIG['seeds'])} runs: {[f'{v:.4f}' for v in w1_values]}")
print(f"RME values across {len(CONFIG['seeds'])} runs: {[f'{v:.4f}' for v in rme_values]}")
print(f"W1 mean ± std: {np.mean(w1_values):.4f} ± {np.std(w1_values):.4f}")
print(f"RME mean ± std: {np.mean(rme_values):.4f} ± {np.std(rme_values):.4f}")
print(f"Rubric target W1: 0.019")
print(f"Rubric target RME: 0.002")

# Save results
summary = {
    "config": CONFIG,
    "results": all_results,
    "summary": {
        "w1_mean": float(np.mean(w1_values)),
        "w1_std": float(np.std(w1_values)),
        "w1_values": [float(v) for v in w1_values],
        "rme_mean": float(np.mean(rme_values)),
        "rme_std": float(np.std(rme_values)),
        "rme_values": [float(v) for v in rme_values],
    },
}

summary_path = output_dir / "reproduction_results.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {summary_path}")

total_elapsed = TIME.perf_counter() - start_time
print(f"Total elapsed time: {total_elapsed:.1f}s")
