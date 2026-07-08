#!/usr/bin/env python3
# synthetic_hitl_causal_dpo_gpu.py
#
# GPU/Torch-backed version of synthetic_hitl_causal_dpo.py.
#
# Mirrors the baseline logic and output schema, but swaps in Torch-backed
# implementations for the heavy steps:
#   - ParticlePosterior_gpu (batched EIG, GPU weights)
#   - candidate_selection_gpu
#   - generation_gpu
#
# Default dtype is float64.

import os
import argparse
import json
import time
from typing import Optional, Dict

import numpy as np
import torch


from utils.utils_gpu import (entropy_categorical,
                             ess,
                             adjacency_from_weights,
                             mean_any)

from dag.dag_ops_gpu import random_dag, sample_weights  # package

from prior.prior_gpu import (
                    make_prior_particles_from_truth,
                    load_prior_particles_npz,
                    sparse_prior_logprob
)
from inference.ParticlePosterior_gpu import ParticlePosterior  # flat
from feedback.expert_gpu import simulate_expert_answer  # package
from generation.generation_gpu import screen_pairs_uncertain  # flat
from metrics.metrics_gpu import expected_true_class_prob, mean_brier_score  # flat
from metrics.structural_metrics_gpu import metrics_from_weighted_samples  # package
from inference.static_baselines_gpu import init_static_schedule  # package
from inference.candidate_selection_gpu import select_pair  # type: ignore


def run_demo(
    D: int = 8,
    policy: str = "eig",
    edge_prob_true: float = 0.18,
    S: int = 300,
    T: int = 60,
    beta_edge: float = 8.0,
    beta_dir: float = -1.5,
    lam: float = 0.0,
    screen_k: int = 100,
    resample_threshold: float = 0.5,
    seed: int = 7,
    prior_npz: Optional[str] = None,
    flip_prob: float = 0.15,
    add_remove_prob: float = 0.05,
    weight_noise: float = 0.25,
    rejuvenate_samples: bool = False,
    rejuvenate_steps: int = 1,
    mutate_fraction: float = 0.1,
    device: str = "auto",
) -> Dict:
    """Run the synthetic HiTL experiment (Torch/GPU backed)."""

    settings = dict(locals())

    rng_np = np.random.default_rng(seed)
    #torch_gen = torch.Generator(device="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    # Ground-truth DAG & weights (simulate expert)
    A_true = random_dag(D, edge_prob_true, generator=torch_gen)
    W_star = sample_weights(A_true, w_scale=1.0, generator=torch_gen)

    # Build particles for q0
    if prior_npz is None:
        particles = make_prior_particles_from_truth(
            W_star,
            S=S,
            flip_prob=flip_prob,
            add_remove_prob=add_remove_prob,
            weight_noise=weight_noise,
            generator=torch_gen,
        )
        weights = None
    else:
        particles, weights = load_prior_particles_npz(prior_npz)
        S = len(particles)
        settings["S"] = S

    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings["device"] = device

    posterior = ParticlePosterior(particles, weights, device=device, dtype=torch.float64)
    print("Using device:", posterior.device)

    logs = []
    history = []

    static_schedule = init_static_schedule(
        policy=policy,
        posterior=posterior,
        D=D,
        T=T,
        screen_k=screen_k,
        beta_edge=beta_edge,
        beta_dir=beta_dir,
        lam=lam,
        rng=rng_np,
    )

    total_pairs = D * (D - 1) // 2
    top_k = min(screen_k, total_pairs)

    visited = set()

    def mark_visited(_i: int, _j: int):
        visited.add((min(_i, _j), max(_i, _j)))

    def is_visited(_i: int, _j: int) -> bool:
        return (min(_i, _j), max(_i, _j)) in visited

    static_ptr = 0
    for t in range(1, T + 1):
        start = time.perf_counter()

        marg = posterior.edge_marginals()
        cand = screen_pairs_uncertain(marg, top_k=top_k)

        # Filter out already-visited unordered pairs
        cand = [(a, b) for (a, b) in cand if not is_visited(a, b)]

        if static_schedule is not None:
            while static_ptr < len(static_schedule) and is_visited(*static_schedule[static_ptr]):
                static_ptr += 1
            if static_ptr >= len(static_schedule):
                raise RuntimeError("Static schedule exhausted (all remaining pairs already visited).")
            i, j = static_schedule[static_ptr]
            best_eig = float("nan")
            static_ptr += 1
        else:
            if len(cand) == 0:
                # fallback random unvisited
                while True:
                    i = int(rng_np.integers(D))
                    j = int(rng_np.integers(D - 1))
                    if j >= i:
                        j += 1
                    if not is_visited(i, j):
                        best_eig = float("nan")
                        break
            else:
                (i, j), best_eig = select_pair(
                    static_schedule=None,
                    t=t,
                    cand=cand,
                    posterior=posterior,
                    policy=policy,
                    beta_edge=beta_edge,
                    beta_dir=beta_dir,
                    rng=rng_np,
                    lam=lam,
                )

        mark_visited(i, j)

        # Simulate expert answer from W_star
        y_idx, _p_star = simulate_expert_answer(
            W_star, i, j, beta_edge, beta_dir, lam, 0.0, 0.0,
            rng=rng_np,
        )

        if A_true[i, j] == 1:
            y_true = 0
        elif A_true[j, i] == 1:
            y_true = 1
        else:
            y_true = 2

        # Update and maybe resample
        posterior.update_with_observation(i, j, y_idx, beta_edge, beta_dir, lam)

        if ess(posterior.weights.detach().cpu().numpy()) / S < resample_threshold:
            posterior.resample(torch_gen=torch_gen)
            if rejuvenate_samples:
                posterior.rejuvenate_particles(
                    q0_logprob=sparse_prior_logprob,
                    expert_history=history,
                    beta_edge=beta_edge,
                    beta_dir=beta_dir,
                    lam=lam,
                    round=t,
                    n_steps=rejuvenate_steps,
                    torch_gen=torch_gen,
                    mutate_frac=mutate_fraction
                )

        history.append((i, j, y_idx))

        # Metrics (unchanged)
        marg = posterior.edge_marginals()
        A_true_np = A_true.cpu().numpy()
        # marg_bin = (marg > 0.5).astype(int)
        # mask = (marg_bin == A_true_np)
        # exist_acc = mask.astype(int).mean()
        exist_acc = float(np.mean((marg > 0.5) == (A_true_np == 1)))


        # Entropy over screened candidates, computed in a batched way
        if len(cand) > 0:
            pred = posterior.predictive_answer_dist_pairs(cand, beta_edge, beta_dir, lam)
            avg_entropy = float(np.mean([entropy_categorical(p) for p in pred]))
        else:
            avg_entropy = float("nan")

        etcp = expected_true_class_prob(posterior=posterior,
                                        beta_edge=beta_edge,
                                        beta_dir=beta_dir,
                                        lam=lam,
                                        A_true=A_true)
        brier = mean_brier_score(posterior=posterior,
                                 beta_edge=beta_edge,
                                 beta_dir=beta_dir,
                                 lam=lam,
                                 A_true=A_true)

        samp_metrics = metrics_from_weighted_samples(
            posterior.particles,
            posterior.weights,
            A_true,
        )

        end = time.perf_counter()
        elapsed_time = end - start
        logs.append(
            {
                "round": t,
                "pair": (int(i), int(j)),
                "answer_idx": int(y_idx),
                "y_true": int(y_true),
                "eig": float(best_eig),
                "exist_acc@0.5": exist_acc,
                "avg_pred_entropy": avg_entropy,
                "ess": float(ess(posterior.weights.detach().cpu().numpy())),
                "exp_true_class_prob": float(etcp),
                "brier": float(brier),
                "samp_skel_precision": float(samp_metrics["skel_precision"]),
                "samp_skel_recall": float(samp_metrics["skel_recall"]),
                "samp_skel_f1": float(samp_metrics["skel_f1"]),
                "samp_orient_precision": float(samp_metrics["orient_precision"]),
                "samp_orient_recall": float(samp_metrics["orient_recall"]),
                "samp_orient_f1": float(samp_metrics["orient_f1"]),
                "samp_shd": float(samp_metrics["shd"]),
                "elapsed_time": elapsed_time,
            }
        )

        print(
            f"Round {t} took {elapsed_time:.6f}s. "
            f"ACC={exist_acc:.3f}. SHD={samp_metrics['shd']}. "
            f"(y_true={y_true}, expert={y_idx})"
        )

    return {
        "W_star": W_star,
        "A_star": A_true,
        "posterior": posterior,
        "posterior_weights": posterior.weights.detach().cpu().numpy(),
        "posterior_marginals": posterior.edge_marginals(),
        "logs": logs,
        "settings": settings,
    }


def main():
    p = argparse.ArgumentParser(description="Human-in-the-loop causal discovery (Torch/GPU).")
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--S", type=int, default=300)
    p.add_argument("--T", type=int, default=60)
    p.add_argument("--beta_edge", type=float, default=8.0)
    p.add_argument("--beta_dir", type=float, default=-1.5)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--screen_k", type=int, default=100)
    p.add_argument("--resample_threshold", type=float, default=0.5)
    p.add_argument("--rejuvenate_samples", action="store_true")
    p.add_argument("--rejuvenate_steps", type=int, default=1)
    p.add_argument("--mutate_fraction", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--edge_prob_true", type=float, default=0.18)
    p.add_argument("--flip_prob", type=float, default=0.15)
    p.add_argument("--add_remove_prob", type=float, default=0.05)
    p.add_argument("--weight_noise", type=float, default=0.25)

    p.add_argument("--prior_npz", type=str, default=None)
    p.add_argument(
        "--policy",
        type=str,
        default="eig",
        choices=["eig", "uncertainty", "random", "static_random", "static_uncertainty", "static_eig"],
    )
    p.add_argument("--save_prefix", type=str, default="run")
    p.add_argument("--outdir", type=str, default="results")

    # GPU control
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    out = run_demo(
        D=args.D,
        edge_prob_true=args.edge_prob_true,
        S=args.S,
        T=args.T,
        beta_edge=args.beta_edge,
        beta_dir=args.beta_dir,
        lam=args.lam,
        screen_k=args.screen_k,
        resample_threshold=args.resample_threshold,
        seed=args.seed,
        flip_prob=args.flip_prob,
        add_remove_prob=args.add_remove_prob,
        weight_noise=args.weight_noise,
        prior_npz=args.prior_npz,
        policy=args.policy,
        rejuvenate_samples=args.rejuvenate_samples,
        rejuvenate_steps=args.rejuvenate_steps,
        mutate_fraction=args.mutate_fraction,
        device=args.device,
    )

    prefix = os.path.join(args.outdir, args.save_prefix)
    # np.save(f"{prefix}_W_star.npy", out["W_star"])
    # np.save(f"{prefix}_A_star.npy", out["A_star"])
    # np.save(f"{prefix}_posterior_marginals.npy", out["posterior_marginals"])

    settings = out["settings"].copy()
    if "device" in settings:
        settings["device"] = str(settings["device"])
    out["settings"] = settings

    with open(f"{prefix}_logs.json", "w") as f:
        json.dump(out["logs"], f, indent=2)
    with open(f"{prefix}_settings.json", "w") as f:
        json.dump(out["settings"], f, indent=2)

    print("Demo complete.")
    print(f"Nodes D={args.D}, rounds T={args.T}, particles S={args.S}, device={out['settings']['device']}")
    if len(out["logs"]) > 0:
        last = out["logs"][-1]
        print(
            f"Final ACC@0.5={last['exist_acc@0.5']:.3f}, "
            f"avg_pred_entropy={last['avg_pred_entropy']:.3f}, "
            f"ESS={last['ess']:.1f}"
        )
    print(
        f"Artifacts written: {prefix}_W_star.npy, {prefix}_A_star.npy, "
        f"{prefix}_logs.json, {prefix}_settings.json"
    )


if __name__ == "__main__":
    main()
