
from typing import List, Tuple
import numpy as np

# ---------------------- Candidate generation & screening -----------------

def screen_pairs_uncertain(
    marginals: np.ndarray, top_k: int
) -> List[Tuple[int, int]]:
    """Score unordered pairs by uncertainty, pick top_k."""
    D = marginals.shape[0]
    scored = []
    for i in range(D):
        for j in range(i + 1, D):  # <-- note: j starts at i+1
            p_ij = marginals[i, j]
            p_ji = marginals[j, i]

            # a few options; this is a reasonable one:
            # use the more uncertain direction as the pair’s score
            s_ij = p_ij * (1.0 - p_ij)
            s_ji = p_ji * (1.0 - p_ji)
            score = max(s_ij, s_ji)

            scored.append(((i, j), float(score)))

    scored.sort(key=lambda x: -x[1])
    return [ij for (ij, _) in scored[:top_k]]

def screen_pairs_uncertain_ordered(marginals: np.ndarray, top_k: int) -> List[Tuple[int, int]]:
    """Score pairs by p*(1-p), which is the variance of a Bernoulli. pick top_k.
    ordered means here that i-j and j-i are counted differently
    """
    D = marginals.shape[0]
    scored = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            p = marginals[i, j]
            scored.append(((i, j), float(p * (1.0 - p))))
    scored.sort(key=lambda x: -x[1])
    return [ij for (ij, _) in scored[:top_k]]

