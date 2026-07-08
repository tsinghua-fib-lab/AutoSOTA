from typing import  Tuple
import numpy as np
from likelihood.preference_likelihood_threeway import bt_threeway_hier


def simulate_expert_answer(
    W_star: np.ndarray,
    i: int,
    j: int,
    beta_edge: float,
    beta_dir: float,
    lam: float,
    phi_star_ij: float,
    phi_star_ji: float,
    rng: np.random.Generator,
) -> Tuple[int, np.ndarray]:
    p = bt_threeway_hier(W_star, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam,
                         phi_ij=phi_star_ij, phi_ji=phi_star_ji)
    y = int(rng.choice(3, p=p))
    # print(f"Expert probability={p}, true W[i,j]={W_star[i,j]}, true W[j,i]={W_star[j,i]},response={y}")

    return y, p

