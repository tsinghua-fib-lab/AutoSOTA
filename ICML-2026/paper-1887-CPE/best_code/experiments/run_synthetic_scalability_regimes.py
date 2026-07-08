#!/usr/bin/env python3
"""
Thin wrapper around synthetic_hitl_causal_dpo.py.

Purpose:
- Keep the *exact* same algorithm/implementation as the main synthetic script.
- Only choose "regime" hyperparameters as a function of D (and density rule).
- Run a single (D, policy, seed) config per invocation (no internal sweeps).
"""

import argparse
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    S: int
    T: int
    screen_k: int
    rejuvenate_steps: int
    resample_threshold: float
    mutate_fraction: float

def pick_regime(D: int) -> Regime:
    """
    Regimes are chosen to keep:
      - expected in-degree constant (handled outside),
      - computational cost increasing but not exploding,
      - query budget scaling with graph size but capped.

    You can tune these numbers based on cluster budget.
    """
    if D <= 20:
        return Regime(S=1000, T=100, screen_k=200, rejuvenate_steps=2, resample_threshold=0.6, mutate_fraction=0.5)
    if D <= 50:
        return Regime(S=1000, T=50, screen_k=1000, rejuvenate_steps=2, resample_threshold=0.6, mutate_fraction=0.5)
    if D <= 100:
        return Regime(S=1000, T=50, screen_k=1000, rejuvenate_steps=2, resample_threshold=0.6, mutate_fraction=0.5)
    if D <= 500:
        return Regime(S=1000, T=50, screen_k=1000, rejuvenate_steps=2, resample_threshold=0.6, mutate_fraction=0.5)
    # D >= 1000
    return Regime(S=1000, T=50, screen_k=1000, rejuvenate_steps=1, resample_threshold=0.6, mutate_fraction=0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--policy", type=str, required=True,
                    choices=["eig", "uncertainty", "random", "static_eig"])
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--outroot", type=str, default="./results/scalability")
    ap.add_argument("--script", type=str, default="synthetic_hitl_causal_dpo_gpu.py")

    # Density rule: keep expected in-degree constant.
    ap.add_argument("--expected_indegree", type=float, default=4.75)

    # Keep these consistent with your current synthetic setup.
    ap.add_argument("--flip_prob", type=float, default=0.10)
    ap.add_argument("--add_remove_prob", type=float, default=0.05)
    ap.add_argument("--weight_noise", type=float, default=0.20)
    ap.add_argument("--beta_edge", type=float, default=10.0)
    ap.add_argument("--beta_dir", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=0.0)

    ap.add_argument("--mutate_fraction", type=float, default=0.1)


    # Optional: if you want to cap p_true at small D (D=10 can get dense)
    ap.add_argument("--max_edge_prob_true", type=float, default=0.25)

    args = ap.parse_args()

    D = args.D
    if D < 2:
        raise ValueError("D must be >= 2")

    # Density / expected degree rule
    p_true = float(args.expected_indegree) / float(D - 1)
    p_true = min(p_true, args.max_edge_prob_true)

    reg = pick_regime(D)

    outdir = os.path.join(args.outroot, f"D{D}", args.policy)
    os.makedirs(outdir, exist_ok=True)

    save_prefix = f"{args.policy}_D{D}_seed{args.seed}"

    cmd = [
        "python", "-u", args.script,
        "--outdir", outdir,
        "--D", str(D),
        "--S", str(reg.S),
        "--T", str(reg.T),
        "--edge_prob_true", str(p_true),
        "--flip_prob", str(args.flip_prob),
        "--add_remove_prob", str(args.add_remove_prob),
        "--weight_noise", str(args.weight_noise),
        "--beta_edge", str(args.beta_edge),
        "--beta_dir", str(args.beta_dir),
        "--lam", str(args.lam),
        "--screen_k", str(reg.screen_k),
        "--resample_threshold", str(reg.resample_threshold),
        "--mutate_fraction", str(reg.mutate_fraction),
        "--policy", args.policy,
        "--rejuvenate_samples",
        "--rejuvenate_steps", str(reg.rejuvenate_steps),
        "--seed", str(args.seed),
        "--save_prefix", save_prefix,
    ]

    print("=== Running config ===")
    print(f"D={D} policy={args.policy} seed={args.seed}")
    print(f"p_true={p_true:.6f}  S={reg.S}  T={reg.T}  screen_k={reg.screen_k}  rejuvenate_steps={reg.rejuvenate_steps}")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()