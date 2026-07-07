import pandas as pd
import numpy as np
import torch
import sys, json, os
sys.path.insert(0, ".")
from utils import compute_uot_plans, USB, SDE, wasserstein, wasserstein_with_weights, sample_from_ot_plan
from torch.utils.data import TensorDataset, DataLoader
import time as TIME

batch_size = 256
nu = 0.001
steps = 3000
device = torch.device("cuda")

data = pd.read_csv("data/simulation_gene_data.csv")
Xs, mass_ratio = [], []
max_sample_time = int(np.max(data["samples"]))
for k in range(max_sample_time + 1):
    Xs.append(np.array(data[data["samples"] == k])[:, 1:])
    mass_ratio.append(Xs[k].shape[0]/Xs[0].shape[0])

dim = Xs[0].shape[1]
t_train = np.arange(len(Xs)).tolist()
samples_per_interval = np.array(data).shape[0]
delta = 1.3

os.makedirs("results/simulation_reproduction", exist_ok=True)

uot_plans, gamma0_plans, gamma1_plans, true_action = compute_uot_plans(Xs, t_train, delta=delta, cuda=True)

all_x0, all_x1, all_m0, all_m1, all_t_start, all_dt = [], [], [], [], [], []
for k in range(len(t_train)-1):
    gamma0_plan = gamma0_plans[k]
    gamma1_plan = gamma1_plans[k]
    x0_np, x1_np, idx_0, idx_1 = sample_from_ot_plan(gamma0_plan, Xs[k], Xs[k+1], samples_per_interval)
    ratio = (gamma1_plan[idx_0, idx_1] / gamma0_plan[idx_0, idx_1]).reshape(-1, 1)
    m1_np = np.log(1e-8 + ratio)
    m0_np = np.zeros_like(m1_np)
    t_s, d_t = t_train[k], t_train[k+1] - t_train[k]
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

seeds = [42, 113, 256, 512, 1024]
all_results = []

for run_idx, seed in enumerate(seeds):
    run_start = TIME.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = USB([dim + 1, 256, 256, 256, 256, 256], nu=nu).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # CONSTANT LR - no scheduler

    model.train()
    gs, keep = 0, True
    while keep:
        for b_x0, b_x1, b_m0, b_m1, b_ts, b_dt in train_loader:
            optimizer.zero_grad()
            rt = torch.rand((b_x0.shape[0], 1), device=device)
            real_t = b_dt * rt + b_ts
            xts_samp, vts_samp, kts_samp, eps_samp, mts_samp = model.sample_comditional_flow(b_x0, b_x1, b_m0, b_m1, rt)
            kts_samp = kts_samp / b_dt
            vts_samp = vts_samp / b_dt
            v, s, k_out = model.forward(xts_samp, real_t)
            weights = torch.exp(mts_samp)
            v_loss = torch.mean(torch.pow(v - vts_samp, 2) * weights)
            bt = 2 * torch.sqrt(rt * (1 - rt)) / (nu + 1e-8)
            s_loss = torch.mean(torch.pow(bt * s + eps_samp, 2) * weights)
            k_loss = torch.mean(torch.pow(k_out - kts_samp, 2) * weights)
            loss = v_loss + s_loss + k_loss
            loss.backward()
            optimizer.step()
            gs += 1
            if gs >= steps:
                keep = False
                break

    model.eval()
    model.to("cpu")
    simulator = SDE(model, nu, mode=1, positive=False)
    x_source = torch.tensor(Xs[0], dtype=torch.float32, device="cpu")
    xs, ms, action = simulator.trajectory(x=x_source, m=torch.zeros([x_source.size(0),1], device="cpu"), delta=delta, T=max_sample_time, N=max_sample_time*100)

    wa_norm_list, rme_list = [], []
    for k in range(1, len(Xs)):
        ti = k * 100
        if ti >= xs.shape[0]: ti = xs.shape[0] - 1
        x1s = xs[ti, :]
        m1s = np.exp(ms[ti, :])
        w1_norm = wasserstein_with_weights(torch.Tensor(np.array(x1s)), np.array(m1s), torch.Tensor(Xs[k]), np.ones(Xs[k].shape[0]), power=1)
        RME = np.abs((np.mean(m1s)-mass_ratio[k])/mass_ratio[k])
        wa_norm_list.append(w1_norm)
        rme_list.append(RME)

    mw1 = float(np.mean(wa_norm_list))
    mrme = float(np.mean(rme_list))
    rt = TIME.perf_counter() - run_start
    all_results.append({"seed": seed, "w1": mw1, "rme": mrme})
    print("Seed {}: W1={:.4f}, RME={:.4f}, time={:.1f}s".format(seed, mw1, mrme, rt))

w1s = [r["w1"] for r in all_results]
rmes = [r["rme"] for r in all_results]
print("\n=== CONSTANT LR (3000 steps) ===")
print("W1 values:", [round(w, 4) for w in w1s])
print("W1 mean+std: {:.4f}+-{:.4f}".format(np.mean(w1s), np.std(w1s)))
print("RME values:", [round(r, 4) for r in rmes])
print("RME mean+std: {:.4f}+-{:.4f}".format(np.mean(rmes), np.std(rmes)))
print("Paper W1: 0.019, Paper RME: 0.002")
