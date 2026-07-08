from typing import Tuple, List
import numpy as np
from inference.ParticlePosterior_gpu import ParticlePosterior

def select_pair_dynamic(posterior: ParticlePosterior,
                        cand: List[Tuple[int, int]],
                        policy: str,
                        beta_edge: float,
                        beta_dir: float,
                        lam: float,
                        rng: np.random.Generator) -> Tuple[Tuple[int, int], float]:
    if policy == "random":
        i, j = cand[rng.integers(len(cand))]
        return (i, j), 0.0
    if policy == "uncertainty":
        i, j = cand[0]
        return (i, j), 0.0

    # Batched EIG evaluation (fast on GPU). Falls back internally if needed.
    pairs = np.asarray(cand, dtype=np.int64)
    eig = posterior.eig_for_pairs(pairs, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
    best_idx = int(np.argmax(eig))
    best_pair = (int(pairs[best_idx, 0]), int(pairs[best_idx, 1]))
    return best_pair, float(eig[best_idx])


def select_pair(static_schedule, t,
                cand,
                posterior: ParticlePosterior, policy,
                beta_edge: float, beta_dir: float,
                rng,
                lam: float = 0.0,) -> Tuple[Tuple[int, int], float]:

    if static_schedule is not None:
        (i, j) = static_schedule[t - 1]
        try:
            best_eig = float(posterior.eig_for_pair(i, j, beta_edge, beta_dir, lam))
        except Exception:
            best_eig = 0.0
    else:
        (i, j), best_eig = select_pair_dynamic(posterior, cand, policy, beta_edge, beta_dir, lam, rng)

    print(f"selecting pair ({i}, {j})")
    return (i, j), best_eig
