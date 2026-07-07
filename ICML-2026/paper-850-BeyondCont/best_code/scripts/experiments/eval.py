#!/usr/bin/env python3
"""Evaluation script for paper 850 USB on Simulation (2D).
Reproduces W1 and RME metrics from the official repository.
Usage: python3 scripts/experiments/eval.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from utils import compute_uot_plans, USB, SDE, wasserstein, wasserstein_with_weights, sample_from_ot_plan

# Configuration matching fast_simulation.py (official code)
NU = 0.001
STEPS = 10000
BATCH_SIZE = 256
DELTA = 1.3
LR = 0.001
SEED = 113

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load data
data = pd.read_csv("data/simulation_gene_data.csv")
Xs, mass_ratio = [], []
max_sample_time = int(np.max(data["samples"]))
for k in range(max_sample_time + 1):
    Xs.append(np.array(data[data["samples"] == k])[:, 1:])
    mass_ratio.append(Xs[k].shape[0] / Xs[0].shape[0])

dim = Xs[0].shape[1]
t_train = np.arange(len(Xs)).tolist()
samples_per_interval = np.array(data).shape[0]

# Compute UOT plans
print("Computing UOT plans...")
uot_plans, gamma0_plans, gamma1_plans, true_action = compute_uot_plans(
    Xs, t_train, delta=DELTA, cuda=True
)

# Prepare training data
all_x0, all_x1, all_m0, all_m1, all_t_start, all_dt = [], [], [], [], [], []
for k in range(len(t_train) - 1):
    x0_np, x1_np, idx_0, idx_1 = sample_from_ot_plan(
        gamma0_plans[k], Xs[k], Xs[k + 1], samples_per_interval
    )
    ratio = (gamma1_plans[k][idx_0, idx_1] / gamma0_plans[k][idx_0, idx_1]).reshape(-1, 1)
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Train model
torch.manual_seed(SEED)
np.random.seed(SEED)
model = USB([dim + 1, 256, 256, 256, 256, 256], nu=NU).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)

print("Training ({} steps, CosineAnnealingLR)...".format(STEPS))
model.train()
global_step = 0
keep_training = True
while keep_training:
    for b_x0, b_x1, b_m0, b_m1, b_ts, b_dt in train_loader:
        optimizer.zero_grad()
        relative_t = torch.rand((b_x0.shape[0], 1), device=device)
        real_t = b_dt * relative_t + b_ts
        xts_samp, vts_samp, kts_samp, eps_samp, mts_samp = model.sample_comditional_flow(
            b_x0, b_x1, b_m0, b_m1, relative_t
        )
        kts_samp = kts_samp / b_dt
        vts_samp = vts_samp / b_dt
        v, s, k_out = model.forward(xts_samp, real_t)
        weights = torch.exp(mts_samp)
        v_loss = torch.mean(torch.pow(v - vts_samp, 2) * weights)
        bridge_term = 2 * torch.sqrt(relative_t * (1 - relative_t)) / (NU + 1e-8)
        s_loss = torch.mean(torch.pow(bridge_term * s + eps_samp, 2) * weights)
        k_loss = torch.mean(torch.pow(k_out - kts_samp, 2) * weights)
        loss = v_loss + s_loss + k_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        if global_step >= STEPS:
            keep_training = False
            break

# Evaluate
print("Evaluating...")
model.eval()
model.to("cpu")
simulator = SDE(model, NU, mode=1, positive=False)
x_source = torch.tensor(Xs[0], dtype=torch.float32, device="cpu")
xs, ms, action = simulator.trajectory(
    x=x_source,
    m=torch.zeros([x_source.size(0), 1], device="cpu"),
    delta=DELTA,
    T=max_sample_time,
    N=max_sample_time * 100,
)

wa_norm_list, rme_list = [], []
for k in range(1, len(Xs)):
    timestep_idx = k * 100
    if timestep_idx >= xs.shape[0]:
        timestep_idx = xs.shape[0] - 1
    x1s = xs[timestep_idx, :]
    m1s = np.exp(ms[timestep_idx, :])
    w1_norm = wasserstein_with_weights(
        torch.Tensor(np.array(x1s)), np.array(m1s),
        torch.Tensor(Xs[k]), np.ones(Xs[k].shape[0]), power=1
    )
    RME = np.abs((np.mean(m1s) - mass_ratio[k]) / mass_ratio[k])
    wa_norm_list.append(w1_norm)
    rme_list.append(RME)

mean_w1 = float(np.mean(wa_norm_list))
mean_rme = float(np.mean(rme_list))

print("\n=== RESULTS ===")
print("W1 (norm) per timestep:", [round(float(w), 4) for w in wa_norm_list])
print("RME per timestep:", [round(float(r), 4) for r in rme_list])
print("W1 mean: {:.4f}".format(mean_w1))
print("RME mean: {:.4f}".format(mean_rme))
print("Paper W1: 0.019")
print("Paper RME: 0.002")

# JSON output for parsing
import json
result = {
    "W1": mean_w1,
    "RME": mean_rme,
    "W1_per_timestep": [float(w) for w in wa_norm_list],
    "RME_per_timestep": [float(r) for r in rme_list],
}
print("\nMETRICS_JSON:", json.dumps(result))
