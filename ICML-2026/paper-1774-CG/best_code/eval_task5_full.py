"""Full eval: Task 5 over 10 observations matching paper Table 1 convention."""
import time
import torch
from experiments.sbi.benchmark import run_task

torch.cuda.empty_cache()
vals = []
t0 = time.time()
for obs in range(1, 11):
    c2st = run_task(
        "task5",
        estimator="reinforce",
        num_steps=100,
        num_particles=1000,
        num_observation=obs,
        seed=0,
        device="cuda",
    )
    vals.append(c2st)
    print(f"obs={obs:2d}  c2st={c2st:.6f}")

v = torch.tensor(vals)
elapsed = time.time() - t0
print(f"\nmean={v.mean():.3f}  std={v.std():.3f}  paper_mean=0.525  time={elapsed:.1f}s")
