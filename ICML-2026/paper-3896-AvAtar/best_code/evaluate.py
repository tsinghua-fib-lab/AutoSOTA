#!/usr/bin/env python3
"""Reproduction eval for AvAtar (paper 3896).
Phone-Email + PARROT + AVATAR-L2: 20% prior, 20% budget, 10 rounds, 5 seeds.
"""
import subprocess, re, os

os.environ["PYTHONPATH"] = "/autosota_cache/PlanetAlign:" + os.environ.get("PYTHONPATH", "")

seeds = [0, 1, 2, 3, 4]
all_mrr = []

for seed in seeds:
    cmd = ["python3", "active_na.py",
           "--alg", "PARROT", "--dataset", "phone-email",
           "--device", "cuda", "--query_round", "10",
           "--query_portion", "0.2", "--init_train_ratio", "0.2",
           "--outIter", "10", "--modes", "sq_l2_adjoint_grad",
           "--anchor_selection_seed", str(seed)]
    result = subprocess.run(cmd, cwd="/repo/source",
                            capture_output=True, text=True, timeout=1200)
    mrr_vals = []
    for line in result.stdout.split("\n"):
        m = re.search(r"MRR:\s*([\d.]+)", line)
        if m:
            mrr_vals.append(float(m.group(1)))
    # 11 rounds x 10 epochs = 110 values.
    # Round 10/10 (query_idx=9) last value = mrr_vals[-11]
    mrr_r10 = mrr_vals[-11] if len(mrr_vals) >= 11 else (mrr_vals[-1] if mrr_vals else None)
    all_mrr.append(mrr_r10)
    print("Seed {}: MRR at Round 10 = {:.4f}".format(seed, mrr_r10))

valid = [m for m in all_mrr if m is not None]
if valid:
    avg = sum(valid) / len(valid)
    print("\n=== RESULTS ===")
    print("MRR values: {}".format([round(m, 4) for m in valid]))
    print("Average MRR at Round 10: {:.4f}".format(avg))
