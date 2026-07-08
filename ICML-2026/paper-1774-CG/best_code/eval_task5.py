"""Evaluation script for Task 5 (Two Moons) — the rubric target.

Reproduces C2ST for CBG (gradient-free) on the Bayesian Inference Benchmark
Task 5 at the paper settings: N=100 steps, K=1000 particles, 10^4 samples,
guidance_scale=1, seed=0.
"""
import argparse
import time
import torch
from experiments.sbi.benchmark import run_task


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBG C2ST on Task 5")
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--num-observation", type=int, default=1)
    parser.add_argument("--antithetic", action="store_true", help="Use antithetic sampling")
    parser.add_argument("--adaptive-alpha", type=float, default=0.0, help="Adaptive guidance scale alpha")
    parser.add_argument("--use-memory", action="store_true", help="Use MemoryDiffusionPosterior")
    parser.add_argument("--memory-fraction", type=float, default=0.2, help="Memory fraction for MemoryDiffusionPosterior")
    parser.add_argument("--use-cv", action="store_true", help="Use leave-one-out control variate")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale for softmax weighting")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    t0 = time.time()
    c2st = run_task(
        "task5",
        estimator="reinforce",
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        num_observation=args.num_observation,
        seed=args.seed,
        device=args.device,
        antithetic=args.antithetic,
        adaptive_alpha=args.adaptive_alpha,
        use_memory=args.use_memory,
        memory_fraction=args.memory_fraction,
        use_cv=args.use_cv,
        guidance_scale=args.guidance_scale,
    )
    elapsed = time.time() - t0
    print(f"task5_c2st={c2st:.6f}  time={elapsed:.1f}s  paper=0.525  ci=[0.497,0.553]")


if __name__ == "__main__":
    main()
