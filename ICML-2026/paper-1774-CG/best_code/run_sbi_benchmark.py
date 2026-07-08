"""Run all 5 SBI benchmark tasks with paper settings and print results."""
import torch
from experiments.sbi.benchmark import run_benchmark

torch.cuda.empty_cache()
results = run_benchmark(
    estimator="reinforce",
    num_steps=100,
    num_particles=1000,
    observations=(1,),
    seed=0,
    device="cuda",
    tasks=("task1", "task2", "task3", "task4", "task5"),
)
print()
for tk in ["task1", "task2", "task3", "task4", "task5"]:
    r = results[tk]
    print(f"C2ST {tk:8s} = {r['c2st_mean']:.3f}  (paper {r['paper']:.3f})")
avg = results["average"]
print(f"C2ST average = {avg['c2st_mean']:.3f}  (paper {avg['paper']:.3f})")
