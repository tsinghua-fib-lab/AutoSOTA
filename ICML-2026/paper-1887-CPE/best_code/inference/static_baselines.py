from generation.generation import screen_pairs_uncertain
from inference.ParticlePosterior import  ParticlePosterior
from typing import List, Tuple
import numpy as np

def init_static_schedule(policy, posterior: ParticlePosterior, D, T, screen_k,
                         beta_edge: float, beta_dir: float, lam: float, rng):

    # --- Build static schedule once (if requested) ---
    static_schedule = None
    if policy.startswith("static_"):
        static_schedule = build_static_schedule(
            posterior=posterior,
            D=D,
            T=T,
            policy=policy,
            screen_k=screen_k,
            beta_edge=beta_edge,
            beta_dir=beta_dir,
            lam=lam,
            rng=rng,
        )
        if len(static_schedule) < T:
            raise RuntimeError(f"Static schedule too short: got {len(static_schedule)} < T={T}")
        print(f"[static] built schedule for policy={policy}: {len(static_schedule)} queries (no unordered repeats).")
    return static_schedule


def build_static_schedule(
    posterior: ParticlePosterior,
    D: int,
    T: int,
    policy: str,
    screen_k: int,
    beta_edge: float,
    beta_dir: float,
    lam: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Build a fixed schedule of length T from the INITIAL posterior only.
    Enforces no repeats w.r.t. unordered pairs {i,j}.
    """
    asked = set()
    schedule: List[Tuple[int, int]] = []

    all_pairs = [(i, j) for i in range(D) for j in range(D) if i != j]

    def _add_pair(i, j):
        u = (min(i, j), max(i, j))
        if u in asked:
            return False
        asked.add(u)
        schedule.append((i, j))
        return True

    if policy == "static_random":
        rng.shuffle(all_pairs)
        for (i, j) in all_pairs:
            if _add_pair(i, j) and len(schedule) == T:
                break
        return schedule

    # For static_uncertainty/static_eig: take a candidate list from initial uncertainty screening
    marg0 = posterior.edge_marginals()
    top_k = min(screen_k, D * (D - 1) // 2)
    cand0 = screen_pairs_uncertain(marg0, top_k=top_k)  # typically ordered pairs

    # Filter to unique unordered pairs, preserving order
    filtered: List[Tuple[int, int]] = []
    for (i, j) in cand0:
        u = (min(i, j), max(i, j))
        if u in asked:
            continue
        asked.add(u)
        filtered.append((i, j))

    # Reset asked/schedule; we only used asked above to filter cand0
    asked = set()
    schedule = []

    if policy == "static_uncertainty":
        for (i, j) in filtered:
            if _add_pair(i, j) and len(schedule) == T:
                break

    elif policy == "static_eig":
        scored = []
        for (i, j) in filtered:
            val = posterior.eig_for_pair(i, j, beta_edge, beta_dir, lam)
            scored.append((float(val), i, j))
        scored.sort(reverse=True, key=lambda x: x[0])
        for (_, i, j) in scored:
            if _add_pair(i, j) and len(schedule) == T:
                break
    else:
        raise ValueError(f"Unknown static policy: {policy}")

    # If screening didn't provide enough unique pairs, fill remaining randomly (no repeats)
    if len(schedule) < T:
        rng.shuffle(all_pairs)
        for (i, j) in all_pairs:
            if _add_pair(i, j) and len(schedule) == T:
                break

    return schedule