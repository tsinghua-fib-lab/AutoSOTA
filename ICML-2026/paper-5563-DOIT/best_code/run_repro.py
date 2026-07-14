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
eta, doob_gamma, doob_tau, doob_t_star_idx, particles = 0.2, 0.25, 0.5, 10, 64

output_file = f"results/{dataset}/doob.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "dataset", "seed", "particles", "eta", "doob_M",
        "doob_gamma", "doob_tau", "doob_t_star_idx", "mean", "std",
    ])

all_means = []
all_stds = []

for seed in range(5):
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

    t0 = time.time()
    msg = "[START] ds=%s seed=%d particles=%d eta=%.2f M=%d gamma=%.2f tau=%.2f t_star=%d" % (
        dataset, seed, particles, eta, 32, doob_gamma, doob_tau, doob_t_star_idx)
    print(msg, flush=True)

    try:
        pipeline = Pipeline(args)
        mean, std = pipeline.eval()
    except Exception as e:
        print("[FAIL] ds=%s seed=%d: %s" % (dataset, seed, e), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

    dt = time.time() - t0
    print("[DONE] ds=%s seed=%d time=%.1fmin mean=%.4f std=%.4f" % (
        dataset, seed, dt/60, mean, std), flush=True)

    all_means.append(mean)
    all_stds.append(std)

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            dataset, seed, particles, eta, args.doob_M,
            doob_gamma, doob_tau, doob_t_star_idx, mean, std,
        ])

overall_mean = np.mean(all_means)
overall_std = np.std(all_means, ddof=1)

print("\n" + "=" * 60)
print("FINAL RESULT: %s" % dataset)
means_str = ", ".join("%.4f" % m for m in all_means)
print("Per-seed means: [%s]" % means_str)
print("Overall mean: %.4f (paper scale: %.1f)" % (overall_mean, overall_mean * 100))
print("Overall std (across seeds): %.4f (paper scale: %.1f)" % (overall_std, overall_std * 100))
print("=" * 60)

with open("results/%s/summary.csv" % dataset, mode="w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value", "paper_scale"])
    w.writerow(["mean", overall_mean, overall_mean * 100])
    w.writerow(["std_across_seeds", overall_std, overall_std * 100])
    w.writerow(["seeds", means_str, ""])
