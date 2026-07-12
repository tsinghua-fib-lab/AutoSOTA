import sys, os
sys.path.insert(0, "/repo")
import numpy as np
import torch
from src.experiment import run_single_seed

# Test GVALID with 1 seed on HardNonLinear8D
results = run_single_seed(
    dataset_name="HardNonLinear8D",
    sampler_name="GVALID",
    seed=42,
    N_total=350,
    B=5,
    N_init=35,
    beta=1.96,
    num_threads=1,
    n_pool=1750,
    test_ratio=0.3,
    n_candidates=500,
    gpu_batch_size=8,
    t_grid_size=101,
    init_strategy="lhs",
    validate_theory=False,
    lr_x=0.1,
    lr_t=0.1,
)

print("=== GVALID Results ===")
for r in results:
    print("N=%4d  policy_suboptimality=%.6f  E2_error=%.6f  dose_error=%.6f" % (
        r["N"], r["policy_suboptimality"], r["E2_error"], r["dose_error"]))
print("\nFinal (N=350): policy_suboptimality=%.6f" % results[-1]["policy_suboptimality"])
