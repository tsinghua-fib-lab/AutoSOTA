#!/usr/bin/env python3
# synthetic_hitl_causal_dpo.py
# Human-in-the-loop causal discovery with Bradley-Terry expert likelihood,
# particle posterior, and EIG-based query selection.
#
# Adds "static prior" baselines:
#   - static_random: choose a fixed schedule of T unique (unordered) pairs at t=0 uniformly at random
#   - static_uncertainty: choose a fixed schedule of T pairs from the initial uncertainty-screened list
#   - static_eig: choose a fixed schedule of T pairs maximizing EIG under the initial posterior
#
# Functionality preserved: logs remain list-of-dicts, outputs unchanged.

import os
import argparse
import json
from typing import Optional, Dict, List, Tuple
import numpy as np

from utils.utils import entropy_categorical, ess
from dag.dag_ops import random_dag, sample_weights
from prior.prior import (
    make_prior_particles_from_truth,
    load_prior_particles_npz,
    sparse_prior_logprob,
)
from inference.ParticlePosterior import ParticlePosterior
from feedback.expert import simulate_expert_answer
from generation.generation import screen_pairs_uncertain
from metrics.metrics import expected_true_class_prob, mean_brier_score
from metrics.structural_metrics import metrics_from_weighted_samples
import time

from inference.static_baselines import init_static_schedule
from inference.candidate_selection import select_pair

# ---------------------- Demo experiment ----------------------------------

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
) -> Dict:
    settings = dict(locals())

    rng = np.random.default_rng(seed)

    # Ground-truth DAG & weights (simulate expert)
    A_star = random_dag(D, edge_prob_true, rng)
    W_star = sample_weights(A_star, w_scale=1.0, rng=rng)

    # Build particles for q0
    if prior_npz is None:
        particles = make_prior_particles_from_truth(
            W_star,
            S=S,
            flip_prob=flip_prob,
            add_remove_prob=add_remove_prob,
            weight_noise=weight_noise,
            rng=rng,
        )
        weights = None
    else:
        particles, weights = load_prior_particles_npz(prior_npz)
        S = len(particles)
        settings["S"] = S

    posterior = ParticlePosterior(particles, weights)

    logs = []
    A_true = (np.abs(W_star) > 1e-12).astype(int)
    history = []
    # print(f"A_true = {A_true}")

    static_schedule = init_static_schedule(policy=policy, posterior=posterior,
                                           D=D, T=T,
                                           screen_k=screen_k,
                                           beta_edge=beta_edge, beta_dir=beta_dir,
                                           lam=lam, rng=rng)

    total_pairs = D * (D - 1) // 2
    top_k = min(screen_k, total_pairs)
    for t in range(1, T + 1):
        start = time.perf_counter()

        marg = posterior.edge_marginals()
        print(f"screening {top_k} pairs from a total of {total_pairs}")
        cand = screen_pairs_uncertain(marg, top_k=top_k)

        (i,j), best_eig = select_pair(static_schedule=static_schedule,
                                      t=t,
                                      cand=cand,
                                      posterior=posterior,
                                      policy=policy,
                                      beta_edge=beta_edge,
                                      beta_dir=beta_dir,
                                      rng=rng,
                                      lam=lam)

        # Simulate expert answer from W_star
        y_idx, p_star = simulate_expert_answer(W_star, i, j, beta_edge, beta_dir, lam, 0.0, 0.0, rng)

        if A_true[i, j] == 1:
            y_true = 0
        elif A_true[j, i] == 1:
            y_true = 1
        else:
            y_true = 2
        print(f"y_true= {y_true}, Expert={y_idx}")

        # Update and maybe resample
        posterior.update_with_observation(i, j, y_idx, beta_edge, beta_dir, lam)

        if ess(posterior.weights) / S < resample_threshold:
            posterior.resample(rng=rng)
            if rejuvenate_samples:
                posterior.rejuvenate_particles(
                    q0_logprob=sparse_prior_logprob,
                    expert_history=history,
                    beta_edge=beta_edge,
                    beta_dir=beta_dir,
                    lam=lam,
                    round=t,
                    n_steps=rejuvenate_steps,
                    rng=rng,
                )

        history.append((i, j, y_idx))

        # Metrics (unchanged)
        marg = posterior.edge_marginals()
        exist_acc = float(np.mean((marg > 0.5) == (A_true == 1)))

        # Keep entropy computed over the screened candidate list (as before)
        avg_entropy = float(np.mean([
            entropy_categorical(posterior.predictive_answer_dist(a, b, beta_edge, beta_dir, lam))
            for (a, b) in cand
        ]))

        etcp = expected_true_class_prob(posterior, beta_edge, beta_dir, lam, A_star)
        brier = mean_brier_score(posterior, beta_edge, beta_dir, lam, A_star)

        samp_metrics = metrics_from_weighted_samples(posterior.particles, posterior.weights, A_true)

        logs.append({
            "round": t,
            "pair": (int(i), int(j)),
            "answer_idx": int(y_idx),
            "eig": float(best_eig),
            "exist_acc@0.5": exist_acc,
            "avg_pred_entropy": avg_entropy,
            "ess": float(ess(posterior.weights)),
            "exp_true_class_prob": float(etcp),
            "brier": float(brier),
            "samp_skel_precision": float(samp_metrics["skel_precision"]),
            "samp_skel_recall": float(samp_metrics["skel_recall"]),
            "samp_skel_f1": float(samp_metrics["skel_f1"]),
            "samp_orient_precision": float(samp_metrics["orient_precision"]),
            "samp_orient_recall": float(samp_metrics["orient_recall"]),
            "samp_orient_f1": float(samp_metrics["orient_f1"]),
            "samp_shd": float(samp_metrics["shd"]),
            "marginals": marg.tolist(),
        })

        end = time.perf_counter()
        print(f"Round {t} took {end - start:.6f} seconds. "
              f"Accuracy={exist_acc:.3f}. "
              f"SHD={samp_metrics['shd']}.")

    return {
        "W_star": W_star,
        "A_star": A_star,
        "posterior": posterior,  # keep the full object
        "posterior_weights": posterior.weights,
        "posterior_marginals": posterior.edge_marginals(),  # save matrix
        "logs": logs,
        "settings": settings
    }


# ---------------------- CLI ---------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Human-in-the-loop causal discovery (BT likelihood + EIG).")
    p.add_argument("--D", type=int, default=8, help="Number of nodes")
    p.add_argument("--S", type=int, default=300, help="Number of particles")
    p.add_argument("--T", type=int, default=60, help="Number of query rounds")
    p.add_argument("--beta_edge", type=float, default=8.0, help="Expert reliability (temperature)")
    p.add_argument("--beta_dir", type=float, default=-1.5, help="Bias term for 'none'")
    p.add_argument("--lam", type=float, default=0.0, help="Scale for structural feature phi (set >0 to enable)")
    p.add_argument("--screen_k", type=int, default=100, help="How many pairs to screen by uncertainty before EIG")
    p.add_argument("--resample_threshold", type=float, default=0.5, help="ESS/S threshold for resampling")
    p.add_argument("--rejuvenate_samples", action="store_true", help="Rejuvenate samples after resampling.")
    p.add_argument("--rejuvenate_steps", type=int, default=1, help="Rejuvenate steps after resampling.")
    p.add_argument("--seed", type=int, default=7, help="Random seed")

    # For generating prior samples that are noisy versions of the ground truth
    p.add_argument("--edge_prob_true", type=float, default=0.18,
                   help="Edge probability when sampling the ground-truth DAG")
    p.add_argument("--flip_prob", type=float, default=0.15,
                   help="Probability of flipping an existing edge orientation in true DAG for prior sampling")
    p.add_argument("--add_remove_prob", type=float, default=0.05,
                   help="Probability of adding/removing edges in prior")
    p.add_argument("--weight_noise", type=float, default=0.25,
                   help="Std dev of weight noise around truth edges in prior")

    p.add_argument("--prior_npz", type=str, default=None,
                   help="Path to .npz with arrays: particles (S,D,D), optional weights (S,)")
    p.add_argument(
        "--policy",
        type=str,
        default="eig",
        choices=["eig", "uncertainty", "random", "static_random", "static_uncertainty", "static_eig"],
        help="Query selection policy"
    )
    p.add_argument("--save_prefix", type=str, default="run", help="Prefix for output files")
    p.add_argument("--outdir", type=str, default="results", help="Directory to save all outputs")

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
    )

    prefix = os.path.join(args.outdir, args.save_prefix)
    np.save(f"{prefix}_W_star.npy", out["W_star"])
    np.save(f"{prefix}_A_star.npy", out["A_star"])
    np.save(f"{prefix}_posterior_marginals.npy", out["posterior_marginals"])
    with open(f"{prefix}_logs.json", "w") as f:
        json.dump(out["logs"], f, indent=2)
    with open(f"{prefix}_settings.json", "w") as f:
        json.dump(out["settings"], f, indent=2)

    print("Demo complete.")
    print(f"Nodes D={args.D}, rounds T={args.T}, particles S={args.S}")
    if len(out["logs"]) > 0:
        last = out["logs"][-1]
        print(f"Final ACC@0.5={last['exist_acc@0.5']:.3f}, "
              f"avg_pred_entropy={last['avg_pred_entropy']:.3f}, ESS={last['ess']:.1f}")
    print(f"Artifacts written: {prefix}_W_star.npy, {prefix}_A_star.npy, "
          f"{prefix}_logs.json, {prefix}_settings.json")


if __name__ == "__main__":
    main()
