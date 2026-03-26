# OLLA_NIPS/src/utils.py

import torch

def project(v: torch.Tensor, J: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
    """
    Apply the projection matrix P = I - J @ G_inv @ J.T to a batch of vectors v
    without forming P explicitly, which is memory-efficient for large dimensions.

    Args:
        v:      A batch of vectors to be projected. Shape: (P, d)
        J:      The Jacobian of the constraints. Shape: (P, d, p)
        G_inv:  The inverse of the Gram matrix (J.T @ J). Shape: (P, p, p)

    Returns:
        The projected vectors. Shape: (P, d)
    """
    # s = J^T @ v
    s = torch.einsum('bdp,bd->bp', J, v)
    # t = G_inv @ s
    t = torch.einsum('bpq,bq->bp', G_inv, s)
    # Jt = J @ t
    Jt = torch.einsum('bdp,bp->bd', J, t)
    
    return v - Jt
