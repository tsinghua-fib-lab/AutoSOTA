from typing import Tuple
import numpy as np
import torch

from likelihood.preference_likelihood_threeway_gpu import bt_threeway_hier


def simulate_expert_answer(
    W_star,
    i: int,
    j: int,
    beta_edge: float,
    beta_dir: float,
    lam: float,
    phi_star_ij: float,
    phi_star_ji: float,
    rng: np.random.Generator,
) -> Tuple[int, np.ndarray]:
    """
    GPU-friendly version of simulate_expert_answer.

    Args:
        W_star: (D,D) NumPy array or torch.Tensor of true weights
        i, j: queried node indices
        beta_edge, beta_dir, lam: expert likelihood parameters
        phi_star_ij, phi_star_ji: truth scaling parameters
        rng: NumPy RNG for sampling the categorical response

    Returns:
        y: sampled response in {0,1,2}
        p: NumPy array of shape (3,) with expert probabilities
    """
    # --- ensure torch tensor ---
    if isinstance(W_star, torch.Tensor):
        Wt = W_star
    else:
        Wt = torch.as_tensor(W_star, dtype=torch.float64)

    device = Wt.device

    # --- compute expert probabilities on GPU ---
    with torch.no_grad():
        probs = bt_threeway_hier(
            W = Wt,
            i = i,
            j = j,
            beta_edge=beta_edge,
            beta_dir=beta_dir,
            lam=lam,
            phi_ij=phi_star_ij,
            phi_ji=phi_star_ji,
        )  # (3,)

        # normalize defensively
        probs = probs / probs.sum().clamp_min(1e-300)

    # --- sample using NumPy RNG (deterministic w.r.t. existing code) ---
    p_np = probs.detach().cpu().numpy()
    y = int(rng.choice(3, p=p_np))

    return y, p_np
