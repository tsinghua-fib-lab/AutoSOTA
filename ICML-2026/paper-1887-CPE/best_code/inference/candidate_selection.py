from inference.ParticlePosterior import ParticlePosterior
from typing import Tuple, List
import numpy as np

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

    best_pair, best_eig = None, -1.0
    for (i, j) in cand:
        val = posterior.eig_for_pair(i, j, beta_edge, beta_dir, lam)
        if val > best_eig:
            best_eig = val
            best_pair = (i, j)
    assert best_pair is not None
    return best_pair, float(best_eig)


def select_pair(static_schedule, t,
                cand,
                posterior: ParticlePosterior, policy,
                beta_edge: float, beta_dir: float,
                rng,
                lam: float = 0.0,) -> Tuple[Tuple[int, int], float]:

    # --- Choose pair (adaptive vs static) ---
    if static_schedule is not None:
        (i, j) = static_schedule[t - 1]
        # For logging only; does not affect selection
        try:
            best_eig = float(posterior.eig_for_pair(i, j, beta_edge, beta_dir, lam))
        except Exception:
            best_eig = 0.0
    else:
        (i, j), best_eig = select_pair_dynamic(posterior, cand, policy, beta_edge, beta_dir, lam, rng)

    print(f"selecting pair ({i}, {j})")
    return (i, j), best_eig


