import torch
from typing import Tuple, List

from generation.generation_gpu import screen_pairs_uncertain
import numpy as np


def static_random(
    D: int,
    rng,
) -> Tuple[int, int]:
    i = rng.integers(D)
    j = rng.integers(D - 1)
    if j >= i:
        j += 1
    return i, j


def static_uncertainty(
    posterior,
    k: int = 100,
) -> Tuple[int, int]:
    pairs = screen_pairs_uncertain(posterior, k=k)
    return pairs[0]


def static_eig(
    posterior,
    beta_edge: float,
    beta_dir: float,
    lam: float,
    k: int = 100,
) -> Tuple[int, int]:
    pairs = screen_pairs_uncertain(posterior, k=k)
    pairs_t = torch.tensor(pairs, device=posterior.device)

    eig = posterior.eig_for_pairs(
        pairs_t, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam
    )

    idx = torch.argmax(eig).item()
    return int(pairs[idx][0]), int(pairs[idx][1])

from typing import Optional, List, Tuple
import numpy as np

def init_static_schedule(
    policy: str,
    posterior,
    D: int,
    T: int,
    screen_k: int,
    beta_edge: float,
    beta_dir: float,
    lam: float,
    rng: np.random.Generator,
) -> Optional[List[Tuple[int, int]]]:
    """
    GPU-friendly static schedule initializer.

    Mirrors the CPU version:
      - if policy starts with "static_", builds a fixed schedule once
      - checks length
      - prints a status line
    """
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
    posterior,
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
    GPU/torch-friendly static schedule builder.
    Mirrors the CPU logic exactly, except uses batched EIG if available.
    Enforces no repeats w.r.t unordered pairs {i,j}.
    """
    asked = set()
    schedule: List[Tuple[int, int]] = []

    all_pairs = [(i, j) for i in range(D) for j in range(D) if i != j]

    def _add_pair(i: int, j: int) -> bool:
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

    # Your CPU uses screen_pairs_uncertain(marg0, top_k=top_k).
    # If your GPU screen function signature differs, adjust this call accordingly.
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
        # Fast path: batched scoring if posterior supports it.
        if hasattr(posterior, "eig_for_pairs"):
            device = getattr(posterior, "device", None)
            if device is None and hasattr(posterior, "particles"):
                device = posterior.particles.device
            if device is None:
                device = torch.device("cpu")

            pairs_t = torch.tensor(filtered, dtype=torch.long, device=device)  # (K,2)
            eig_cpu = posterior.eig_for_pairs(pairs_t, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)

            # Sort filtered by eig descending
            # eig_cpu = eig.detach().cpu().numpy() # already an np array
            order = np.argsort(-eig_cpu)
            for idx in order:
                i, j = filtered[int(idx)]
                if _add_pair(i, j) and len(schedule) == T:
                    break
        else:
            # Fallback: scalar loop, faithful but slower
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
