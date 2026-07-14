# utils/ot_pairs.py
import numpy as np
import torch
import ot  # POT


# -------------------- Cost matrices --------------------

@torch.no_grad()
def angular_cost_matrix(phi0: torch.Tensor,
                        phi1: torch.Tensor,
                        squared: bool = True) -> np.ndarray:
    """
    Angular cost C[i,j] = (arccos(<phi0_i, phi1_j>))^{1 or 2}.
    Assumes rows are (approximately) unit-norm.
    Returns NumPy array (B,B).
    """
    dot = (phi0 @ phi1.T).clamp(-1.0, 1.0)
    ang = torch.arccos(dot)
    C = ang.square() if squared else ang
    return C.cpu().numpy()


@torch.no_grad()
def euclidean_cost_matrix(x0: torch.Tensor,
                          x1: torch.Tensor,
                          squared: bool = True) -> np.ndarray:
    """
    Euclidean cost C[i,j] = ||x0_i - x1_j||^{2}.
    Returns NumPy array (B,B).
    """
    C = torch.cdist(x0, x1)
    if squared:
        C = C.square()
    return C.cpu().numpy()


# -------------------- OT core --------------------

def _solve_plan(cost: np.ndarray,
                method: str,
                reg: float | None) -> np.ndarray:
    """
    Compute OT plan with POT given a cost matrix (NumPy).
    method: "exact" (emd) or "sinkhorn"
    reg:    float for sinkhorn, None for exact
    """
    B = cost.shape[0]
    a = np.ones(B) / B
    b = np.ones(B) / B

    if method == "exact" or (method is None and reg is None):
        P = ot.emd(a, b, cost)
    elif method == "sinkhorn":
        if reg is None:
            raise ValueError("Sinkhorn OT requires a positive reg value.")
        P = ot.sinkhorn(a, b, cost, reg)
    else:
        raise ValueError(f"Unknown OT method: {method!r}")
    return P


@torch.no_grad()
def get_ot_indices(x0: torch.Tensor,
                   x1: torch.Tensor,
                   *,
                   cost: str = "angular",
                   method: str = "exact",
                   reg: float | None = None,
                   squared: bool = True,
                   hard: bool = False,
                   normalize_rows_safeguard: bool = True) -> np.ndarray:
    """
    Build an OT plan between x0 and x1 and return pairing indices j for each i.

    Args:
      cost:    "angular" or "euclidean"
      method:  "exact" or "sinkhorn"
      reg:     entropy-regularization for sinkhorn; None for exact
      squared: use squared version of the ground cost (True recommended)
      hard:    if True, j = argmax_j π[i,j]; else sample j ~ π[i,:]/sum
      normalize_rows_safeguard: if a row has ~0 mass or non-finite, fall back to random

    Returns:
      idx: np.ndarray of shape (B,) with target index for each source row.
    """
    # Choose ground cost
    if cost == "angular":
        C = angular_cost_matrix(x0, x1, squared=squared)
    elif cost == "euclidean":
        C = euclidean_cost_matrix(x0, x1, squared=squared)
    else:
        raise ValueError(f"Unknown cost type: {cost!r}")

    # Solve for plan
    P = _solve_plan(C, method=method, reg=reg)

    B = P.shape[0]
    if hard:
        return np.argmax(P, axis=1)

    idx = []
    for i in range(B):
        row = P[i]
        s = row.sum()
        if not np.isfinite(s) or s <= 0:
            # safeguard fallback
            if normalize_rows_safeguard:
                idx.append(np.random.randint(B))
            else:
                # if you want to fail loudly instead:
                raise RuntimeError("Non-finite or zero-mass OT row encountered.")
        else:
            idx.append(np.random.choice(B, p=row / s))
    return np.array(idx)



@torch.no_grad()
def angular_ot_pairs(phi0: torch.Tensor,
                     phi1: torch.Tensor,
                     reg: float | None = None,
                     squared: bool = True,
                     hard: bool = False) -> np.ndarray:
    """
    Old API: Angular OT pairs. Uses exact if reg is None, else sinkhorn(reg).
    """
    method = "exact" if reg is None else "sinkhorn"
    return get_ot_indices(
        phi0, phi1,
        cost="angular",
        method=method,
        reg=reg,
        squared=squared,
        hard=hard
    )


@torch.no_grad()
def euclidean_ot_pairs(x0: torch.Tensor,
                       x1: torch.Tensor,
                       reg: float | None = None,
                       squared: bool = True,
                       hard: bool = False) -> np.ndarray:
    """
    New helper: Euclidean OT pairs. Uses exact if reg is None, else sinkhorn(reg).
    """
    method = "exact" if reg is None else "sinkhorn"
    return get_ot_indices(
        x0, x1,
        cost="euclidean",
        method=method,
        reg=reg,
        squared=squared,
        hard=hard
    )
