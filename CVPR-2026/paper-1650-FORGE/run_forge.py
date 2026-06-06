import sys, os, json
sys.path.insert(0, "/repo")
import numpy as np
import torch
from forge.forge import Cfg as FORGEConfig, main as run_forge

print("Starting FORGE continual learning...")

cfg = FORGEConfig()
cfg.PT_PATH = "/repo/data/real_sites.pt"
cfg.NPZ_HOSP_PATHS = [
    "/repo/data/synth/site6.npz",
    "/repo/data/synth/site14.npz",
    "/repo/data/synth/site15.npz",
    "/repo/data/synth/site16.npz",
]
cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.SEED = 42
cfg.VERBOSE = True

# Paper settings
cfg.EPOCHS_PER_TASK = 200
cfg.WARMUP_EPOCHS = 20
cfg.PATIENCE_ON_ACC = 40
cfg.BATCH_SIZE = 32
cfg.LR = 1e-3
cfg.WEIGHT_DECAY = 5e-4
cfg.HIDDEN = 128
cfg.EMBED = 128
cfg.LAYERS = 4
cfg.DROPOUT = 0.30
cfg.ALPHA = 0.10
cfg.BETA = 0.40
cfg.GAMMA_G = 0.30
cfg.GAMMA_R = 0.15
cfg.ADJ_THRESHOLD = 0.4
cfg.REPLAY_MB_SIZE = 128
cfg.TOT_SYNTH_CAPACITY = 256
cfg.REPLAY_AFTER_FIRST = True
cfg.VAL_RATIO = 0.20

results = run_forge(cfg)

print("\n=== FORGE RESULTS ===")
summary = results["summary"]
aaa = summary["aaa"]
last_row = summary["last_row"]
print("AAA =", round(aaa, 4))
print("Last row:", [round(x, 4) for x in last_row])

# Compute FOR
metric_matrix = results["metric_matrix"]
T = len(metric_matrix)
final_row = metric_matrix[-1] if metric_matrix else []
for_values = []
for i in range(T):
    if i >= len(final_row):
        break
    max_prev = max(metric_matrix[j][i] for j in range(i, T) if i < len(metric_matrix[j]))
    for_values.append(max_prev - final_row[i])
FOR = float(np.mean(for_values)) if for_values else 0.0

print("FOR =", round(FOR, 4))
print("\nMetric matrix:")
for i, row in enumerate(metric_matrix):
    print("  After task", i+1, ":", [round(x, 4) for x in row])
