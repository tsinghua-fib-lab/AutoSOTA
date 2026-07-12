#!/usr/bin/env python3
"""
Reproduction eval script for paper 4799:
  Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution

Target rubric metric:
  Dataset: SYNTHETIC
  Mechanism: FD-MNAR, mu=0.50
  Model: GNNmim (backbone=GCN, transductive, binary classification)
  n_runs: 5  (seeds = [1, 43, 15, 118, 222])

This script replicates the exact code path from main.py for the target (model, mechanism, prob).
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, "/repo")
from utils import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── Load synthetic dataset ──────────────────────────────────────────
base_dir = os.path.join("/repo", "data")
data = torch.load(os.path.join(base_dir, "synthetic.pt"))
data.num_classes = len(torch.unique(data.y))
print(f"Dataset: nodes={data.num_nodes}, features={data.num_features}, "
      f"classes={data.num_classes}", flush=True)

# Set up adjacency (identical to main.py)
indices = data.edge_index
values  = torch.ones(indices.size(1))
adj = torch.sparse_coo_tensor(
    indices.to(device), values.to(device), (data.num_nodes, data.num_nodes)
)
data.adj      = adj
data.features = data.x.clone()
data          = data.to(device)

# ── Run FDMNAR at mu=0.50 ───────────────────────────────────────────
mech_params  = {"mecha": "FDMNAR", "opt": None}
p_miss_dict  = {"train": 0.5, "test": 0.5}
data_clone   = data.clone()

model_name   = "gnnmim"
mech_name    = "FDMNAR"

print(f"\nRunning: mechanism={mech_name}, model={model_name}, mu=0.50", flush=True)

data_masked = produce_NA_ood(data_clone.clone(), p_miss_dict, mech=mech_params, seeds=seeds)
real_p = torch.isnan(data_masked.masks[seeds[0]]["X_incomp"]).float().mean().item() * 100
print(f"Real missing rate (seed 0): {real_p:.2f}%", flush=True)

# ── GNNmim preparation (identical to main.py) ───────────────────────
dm_tmp = data_masked.clone()
for seed in dm_tmp.masks.keys():
    miss_flag = (~dm_tmp.masks[seed]["mask"]).double().to(device)
    x_incomp  = dm_tmp.masks[seed]["X_incomp"].to(device)
    dm_tmp.masks[seed]["X_incomp"] = torch.cat([x_incomp, miss_flag], dim=1)
    dm_tmp.masks[seed]["X_incomp"] = torch.nan_to_num(
        dm_tmp.masks[seed]["X_incomp"], nan=0.0
    ).float()
dm_tmp.num_features = dm_tmp.num_features * 2

# ── Evaluate ────────────────────────────────────────────────────────
metrics = evaluate_gcn(dm_tmp, method=None, mod=model_name)
acc, loss, f1, std = metrics[0], metrics[1], metrics[2], metrics[3]

print(f"\n{'='*60}")
print(f"FINAL RESULT")
print(f"{'='*60}")
print(f"Mechanism:    {mech_name}")
print(f"Model:        {model_name}")
print(f"mu:           {0.50}")
print(f"F1 Score:     {f1:.4f}")
print(f"F1 Std:       {std:.4f}")
print(f"Accuracy:     {acc:.4f}")
print(f"Loss:         {loss:.4f}")
print(f"Real miss %:  {real_p:.2f}%")
print(f"Seeds:        {seeds}")
print(f"{'='*60}")

# ── Also run GNNmi baseline for reference ───────────────────────────
dm2 = data_masked.clone()
metrics2 = evaluate_gcn(dm2, method=None, mod=None)
acc2, loss2, f12, std2 = metrics2[0], metrics2[1], metrics2[2], metrics2[3]
print(f"\nBaseline GNNmi: F1={f12:.4f} (+/- {std2:.4f})")
