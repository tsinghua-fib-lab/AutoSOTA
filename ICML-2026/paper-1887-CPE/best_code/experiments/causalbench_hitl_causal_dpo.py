#!/usr/bin/env python3
# causalbench_hitl_50.py
#
# HITL causal discovery on a CausalBench 50-node slice.
# - Oracle = perturbation-effect graph (A_oracle) from build_oracle_effect_graph.py
# - Evaluation = directed-edge AUPRC/AUROC + Top-K precision/recall (NOT SHD/F1-on-DAG)

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---- External metrics (probabilistic ranking metrics) ----
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- Your repo imports (match your Sachs setup) ----
from inference.ParticlePosterior import ParticlePosterior
from generation.generation import screen_pairs_uncertain
from utils.utils import ess, entropy_categorical

from inference.static_baselines import init_static_schedule
from inference.candidate_selection import select_pair

import torch # For DAG GFN
from prior.train_dag_gfn import EdgePolicy
from prior.sample_dag_gfn import sample_dag_from_gfn

try:
    from prior.prior import sparse_prior_logprob
    _HAS_PRIOR = True
except Exception:
    _HAS_PRIOR = False


# ---------------- Oracle handling ----------------

def make_oneway_oracle(A: np.ndarray) -> np.ndarray:
    """
    Keep only one-way directed effects; drop bidirectional pairs.
    This makes the 3-way label space well-defined:
      0 = i->j, 1 = j->i, 2 = none
    """
    A = (A > 0).astype(np.int8)
    both = (A == 1) & (A.T == 1)
    A[both] = 0
    return A


def oracle_label(i: int, j: int, A_ref: np.ndarray) -> int:
    """0=i->j, 1=j->i, 2=none"""
    if A_ref[i, j] == 1:
        return 0
    if A_ref[j, i] == 1:
        return 1
    return 2


# ---------------- Evaluation metrics ----------------

def _mask_offdiag(D: int) -> np.ndarray:
    return ~np.eye(D, dtype=bool)


def edge_scores_from_marginals(marg: np.ndarray) -> np.ndarray:
    """Flatten directed scores excluding diagonal."""
    D = marg.shape[0]
    mask = _mask_offdiag(D)
    return marg[mask].astype(float)


def edge_labels_from_adj(A: np.ndarray) -> np.ndarray:
    """Flatten directed 0/1 labels excluding diagonal."""
    D = A.shape[0]
    mask = _mask_offdiag(D)
    return A[mask].astype(int)


def auprc_auroc(marg: np.ndarray, A_true: np.ndarray) -> Dict[str, float]:
    y_score = edge_scores_from_marginals(marg)
    y_true = edge_labels_from_adj(A_true)

    out = {"auprc_dir": float(average_precision_score(y_true, y_score))}
    # AUROC requires both classes to be present
    if np.unique(y_true).size == 2:
        out["auroc_dir"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auroc_dir"] = float("nan")
    return out


def topk_metrics(marg: np.ndarray, A_true: np.ndarray, k: int | None = None) -> Dict[str, float]:
    """
    Evaluate how many true directed edges are captured by the top-k highest
    posterior edge probabilities (excluding diagonal).

    If k is None, we use k = number of true edges (common, fair default).
    """
    y_score = edge_scores_from_marginals(marg)
    y_true = edge_labels_from_adj(A_true)

    total_pos = int(y_true.sum())
    if k is None:
        k = total_pos
    k = int(max(1, min(k, y_score.size)))

    idx = np.argsort(-y_score)[:k]
    tp = int(y_true[idx].sum())

    prec = tp / k
    rec = tp / max(1, total_pos)
    return {"topk": int(k), "topk_prec": float(prec), "topk_rec": float(rec), "num_true_edges": int(total_pos)}


# ---------------- q0 bootstrap linear DAG (same spirit as Sachs) ----------------

def ridge_coef(Xp: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if Xp.size == 0:
        return np.zeros((0,), dtype=float)
    XtX = Xp.T @ Xp
    XtX.flat[:: XtX.shape[0] + 1] += ridge
    Xty = Xp.T @ y
    return np.linalg.solve(XtX, Xty)


def sample_linear_dag(
    X: np.ndarray,
    rng: np.random.Generator,
    max_parents: int,
    corr_screen_k: int,
    ridge: float
) -> np.ndarray:
    n, D = X.shape
    order = rng.permutation(D)
    W = np.zeros((D, D), dtype=float)

    for pos, j in enumerate(order):
        prev = order[:pos]
        if prev.size == 0:
            continue

        y = X[:, j]
        Xprev = X[:, prev]

        # quick correlation screen
        corrs = np.abs((Xprev.T @ y) / max(1, (n - 1)))
        k = min(corr_screen_k, prev.size)
        cand_idx = np.argpartition(-corrs, kth=k - 1)[:k]
        cand_parents = prev[cand_idx]

        if cand_parents.size > max_parents:
            cand_corrs = corrs[cand_idx]
            top = np.argpartition(-cand_corrs, kth=max_parents - 1)[:max_parents]
            parents = cand_parents[top]
        else:
            parents = cand_parents

        coef = ridge_coef(X[:, parents], y, ridge=ridge)
        for pnode, w in zip(parents, coef):
            if abs(w) > 1e-3:
                W[pnode, j] = float(w)

    return W


def make_q0_particles_bootstrap(
    X: np.ndarray,
    S: int,
    seed: int,
    bootstrap_n: int | None,
    max_parents: int,
    corr_screen_k: int,
    ridge: float
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n, _ = X.shape
    if bootstrap_n is None:
        bootstrap_n = n

    particles: List[np.ndarray] = []
    for _ in range(S):
        idx = rng.integers(0, n, size=bootstrap_n)
        Xb = X[idx]
        particles.append(sample_linear_dag(Xb, rng, max_parents, corr_screen_k, ridge))
    return particles

# builds DAG GFN prior
def sample_gfn(D:int, S: int, rng: np.random.Generator, max_edges: int=200):
    print("Sampling DAG GFN")
    ckpt = torch.load("prior/cb50_gfn_ckpt_std.pt", map_location="cpu")
    policy = EdgePolicy(D=D, hidden=256)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # sample S particles from GFN
    particles = [sample_dag_from_gfn(policy, D, rng, max_edges=max_edges) for _ in range(S)]

    return particles

# ---------------- HITL loop ----------------

@dataclass
class RunConfig:
    S: int
    T: int
    screen_k: int
    resample_threshold: float
    rejuvenate_samples: bool
    rejuvenate_steps: int
    max_parents: int
    corr_screen_k: int
    ridge: float
    bootstrap_n: int | None


# def select_pair(
#     posterior: ParticlePosterior,
#     cand: List[Tuple[int, int]],
#     policy: str,
#     beta_edge: float,
#     beta_dir: float,
#     lam: float,
#     rng: np.random.Generator
# ) -> Tuple[Tuple[int, int], float]:
#     if policy == "random":
#         i, j = cand[rng.integers(len(cand))]
#         return (i, j), 0.0
#     if policy == "uncertainty":
#         return cand[0], 0.0
#
#     best_pair, best_eig = None, -1.0
#     for (i, j) in cand:
#         val = posterior.eig_for_pair(i, j, beta_edge, beta_dir, lam)
#         if val > best_eig:
#             best_eig = val
#             best_pair = (i, j)
#     assert best_pair is not None
#     return best_pair, float(best_eig)


def run_once(
    X: np.ndarray,
    A_ref: np.ndarray,
    policy: str,
    cfg: RunConfig,
    beta_edge: float,
    beta_dir: float,
    lam: float,
    seed: int,
    use_dag_gfn_prior: bool=False) -> Dict:
    rng = np.random.default_rng(seed)
    D = X.shape[1]

    if use_dag_gfn_prior:
        particles = sample_gfn(D=D, S=cfg.S, rng=rng)
    else:
        particles = make_q0_particles_bootstrap(
            X, S=cfg.S, seed=seed, bootstrap_n=cfg.bootstrap_n,
            max_parents=cfg.max_parents, corr_screen_k=cfg.corr_screen_k, ridge=cfg.ridge
        )

    posterior = ParticlePosterior(particles, weights=None)

    # Init marginals + eval
    marg0 = posterior.edge_marginals()
    init_eval = {**auprc_auroc(marg0, A_ref), **topk_metrics(marg0, A_ref, k=int(A_ref.sum()))}

    static_schedule = init_static_schedule(policy=policy, posterior=posterior, D=D,
                                           T=cfg.T, screen_k=cfg.screen_k,
                                           beta_edge=beta_edge, beta_dir=beta_dir,
                                           lam=lam, rng=rng)

    asked = set()  # unordered asked pairs: (min(i,j), max(i,j))
    history: List[Tuple[int, int, int]] = []
    logs: List[Dict] = []

    for t in range(1, cfg.T + 1):
        t0 = time.perf_counter()

        marg = posterior.edge_marginals()
        top_k = min(cfg.screen_k, D * (D - 1) // 2)
        cand = screen_pairs_uncertain(marg, top_k=top_k)

        # no-repeat filter (unordered)
        cand = [(i, j) for (i, j) in cand if (min(i, j), max(i, j)) not in asked]

        if not cand:
            # fallback to all remaining unordered pairs (then expand to both directions)
            remaining = [(i, j) for i in range(D) for j in range(i + 1, D) if (i, j) not in asked]
            if not remaining:
                break
            cand = []
            for (i, j) in remaining:
                cand.append((i, j))
                cand.append((j, i))

        # (i, j), eig_val = select_pair(posterior, cand, policy, beta_edge, beta_dir, lam, rng)
        (i, j), eig_val = select_pair(static_schedule=static_schedule,
                                      t=t,
                                      cand=cand,
                                      posterior=posterior,
                                      policy=policy,
                                      beta_edge=beta_edge,
                                      beta_dir=beta_dir,
                                      rng=rng,
                                      lam=lam)

        y = oracle_label(i, j, A_ref)

        posterior.update_with_observation(i, j, y, beta_edge, beta_dir, lam)

        # mark as asked (unordered)
        asked.add((min(i, j), max(i, j)))

        if ess(posterior.weights) / cfg.S < cfg.resample_threshold:
            posterior.resample(rng=rng)
            if cfg.rejuvenate_samples:
                if not _HAS_PRIOR:
                    raise RuntimeError("rejuvenate_samples=True but prior.prior.sparse_prior_logprob not importable")
                posterior.rejuvenate_particles(
                    q0_logprob=sparse_prior_logprob,
                    expert_history=history,
                    beta_edge=beta_edge,
                    beta_dir=beta_dir,
                    lam=lam,
                    round=t,
                    n_steps=cfg.rejuvenate_steps,
                    rng=rng,
                )

        history.append((i, j, y))

        # Eval vs oracle effect graph
        marg_t = posterior.edge_marginals()
        m = auprc_auroc(marg_t, A_ref)
        tk = topk_metrics(marg_t, A_ref, k=int(A_ref.sum()))

        # Uncertainty diagnostic (cap cost)
        avg_entropy = float(np.mean([
            entropy_categorical(posterior.predictive_answer_dist(a, b, beta_edge, beta_dir, lam))
            for (a, b) in cand[: min(len(cand), 200)]
        ]))

        logs.append({
            "round": t,
            "pair": (int(i), int(j)),
            "answer_idx": int(y),
            "eig": float(eig_val),
            "avg_pred_entropy": float(avg_entropy),
            "ess": float(ess(posterior.weights)),
            "auprc_dir": float(m["auprc_dir"]),
            "auroc_dir": float(m["auroc_dir"]),
            "topk": int(tk["topk"]),
            "topk_prec": float(tk["topk_prec"]),
            "topk_rec": float(tk["topk_rec"]),
        })

        t1 = time.perf_counter()
        print(
            f"[{policy}] t={t:03d}/{cfg.T} pair=({i},{j}) y={y} "
            f"EIG={eig_val:.4f} AUPRC={m['auprc_dir']:.3f} "
            f"TopK-Prec={tk['topk_prec']:.3f} dt={t1-t0:.3f}s"
        )

    final_marg = posterior.edge_marginals()

    return {
        "init": init_eval,
        "policy": policy,
        "seed": seed,
        "settings": {"beta_edge": beta_edge, "beta_dir": beta_dir, "lam": lam, **cfg.__dict__},
        "logs": logs,
        "final": logs[-1] if logs else {},
        "posterior_marginals_init": marg0.tolist(),
        "posterior_marginals_final": final_marg.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", type=str, required=True,
                    help="output of prepare_causalbench_50.py (X, gene_names, A_oracle)")
    ap.add_argument("--outdir", type=str, default="results_causalbench50")

    # single-run controls
    ap.add_argument("--policy", type=str, default="eig",
                    choices=["eig", "uncertainty", "random",
                             "static_eig", "static_uncertainty", "static_random"],
                    help="Single policy to run.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Single RNG seed for this run (set uniquely per cluster job).")

    # inference / loop
    ap.add_argument("--S", type=int, default=800)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--screen_k", type=int, default=800)
    ap.add_argument("--resample_threshold", type=float, default=0.5)
    ap.add_argument("--rejuvenate_samples", action="store_true")
    ap.add_argument("--rejuvenate_steps", type=int, default=1)

    # q0 sampler
    ap.add_argument("--max_parents", type=int, default=2)
    ap.add_argument("--corr_screen_k", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--bootstrap_n", type=int, default=0)
    ap.add_argument("--use_dag_gfn_prior", action="store_true")

    # expert model params
    ap.add_argument("--beta_edge", type=float, default=8.0)
    ap.add_argument("--beta_dir", type=float, default=-1.5)
    ap.add_argument("--lam", type=float, default=0.0)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.dataset_npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    genes = [str(x) for x in d["gene_names"].tolist()]
    A_oracle = np.asarray(d["A_oracle"], dtype=np.int8)

    A_ref = make_oneway_oracle(A_oracle)
    bidir_dropped = int((((A_oracle > 0) & (A_oracle.T > 0)).sum()) // 2)

    # write/overwrite meta (safe even if many jobs do it)
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump({
            "genes": genes,
            "D": int(X.shape[1]),
            "N": int(X.shape[0]),
            "oracle_edges_raw": int((A_oracle > 0).sum()),
            "oracle_edges_oneway": int(A_ref.sum()),
            "oracle_density_oneway": float(A_ref.mean()),
            "bidirectional_pairs_dropped": bidir_dropped,
            "dataset_npz": args.dataset_npz,
        }, f, indent=2)

    print(
        f"[oracle stats] D={X.shape[1]} N={X.shape[0]} "
        f"edges(one-way)={int(A_ref.sum())} density={float(A_ref.mean()):.4f} "
        f"bidir_pairs_dropped={bidir_dropped}"
    )

    cfg = RunConfig(
        S=args.S,
        T=args.T,
        screen_k=args.screen_k,
        resample_threshold=args.resample_threshold,
        rejuvenate_samples=args.rejuvenate_samples,
        rejuvenate_steps=args.rejuvenate_steps,
        max_parents=args.max_parents,
        corr_screen_k=args.corr_screen_k,
        ridge=args.ridge,
        bootstrap_n=None if args.bootstrap_n == 0 else int(args.bootstrap_n),
    )

    out = run_once(
        X=X,
        A_ref=A_ref,
        policy=args.policy,
        cfg=cfg,
        beta_edge=args.beta_edge,
        beta_dir=args.beta_dir,
        lam=args.lam,
        seed=args.seed,
    )

    out_path = os.path.join(args.outdir, f"cb50_{args.policy}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()