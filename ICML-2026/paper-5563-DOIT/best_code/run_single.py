import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["LD_LIBRARY_PATH"] = "/root/.mujoco/mujoco210/bin:" + os.environ.get("LD_LIBRARY_PATH", "")

import sys
sys.path.insert(0, "/repo")

import csv
import time
import traceback
import numpy as np

from pipeline import Pipeline
from search.configs import Arguments


dataset = "halfcheetah-medium-v2"
seed = 0

# Paper hyperparameters from Table 9 / Appendix C.4
eta, doob_gamma, doob_tau, doob_t_star_idx, particles = 0.2, 0.25, 0.5, 10, 4

args = Arguments()
args.dataset = dataset
args.device = "cuda:0"
args.seed = seed
args.per_sample_batch_size = particles

args.eta = eta
args.doob_M = 32
args.doob_gamma = doob_gamma
args.doob_tau = doob_tau
args.doob_t_star_idx = doob_t_star_idx
args.doob_antithetic_sampling = True

output_file = f"results/{args.dataset}/doob_single.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset", "seed", "particles", "eta", "doob_M",
            "doob_gamma", "doob_tau", "doob_t_star_idx", "mean", "std",
        ])

t0 = time.time()
print(f"[START] ds={dataset} seed={seed} "
      f"particles={particles} eta={eta} M=32 gamma={doob_gamma} "
      f"tau={doob_tau} t_star={doob_t_star_idx}", flush=True)

try:
    pipeline = Pipeline(args)
    mean, std = pipeline.eval()
except Exception as e:
    print(f"[FAIL] ds={dataset} seed={seed}: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    raise

dt = time.time() - t0
print(f"[DONE] ds={dataset} seed={seed} time={dt/60:.1f}min mean={mean:.2f} std={std:.2f}", flush=True)

with open(output_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        dataset, seed, particles, eta, args.doob_M,
        doob_gamma, doob_tau, doob_t_star_idx, mean, std,
    ])

print(f"Result: mean={mean:.2f}, std={std:.2f}")
