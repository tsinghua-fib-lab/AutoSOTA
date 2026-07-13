import logging

import numpy as np
import ot

from cellbridge.ot.cost import contributions_multi
from cellbridge.ot.fgw_multi import fused_gromov_wasserstein_multi

logger = logging.getLogger(__name__)


def get_unbalanced_marginals(a, b, M, reg, reg_marginals, numItermax_uot, stopThr_uot):
    """Estimate relaxed marginals with unbalanced OT."""
    # Solve unbalanced OT to get new marginals
    G_uot = ot.unbalanced.lbfgsb_unbalanced(
        a,
        b,
        M,
        reg=reg,
        reg_m=reg_marginals,
        numItermax=numItermax_uot,
        stopThr=stopThr_uot,
    )
    a_new = G_uot.sum(axis=1)
    b_new = G_uot.sum(axis=0)
    return a_new, b_new


def two_step_unbalanced_fgw_multi(
    a,
    b,
    M,
    C1,
    C2,
    alpha,
    Q=None,
    numIterFGW=30000,
    numIterEMD=200000,
    initialization=None,
    **kwargs,
):
    """Run multi-FGW after upstream unbalanced marginal adjustment."""
    logger.info("Using balanced multi-FGW with marginals adjusted upstream")
    return fgw_multi(
        a,
        b,
        M,
        C1,
        C2,
        alpha=alpha,
        Q=Q,
        numIterFGW=numIterFGW,
        numIterEMD=numIterEMD,
        initialization=initialization,
    )


def fgw_multi(
    a,
    b,
    M,
    C1,
    C2,
    alpha,
    Q=None,
    numIterFGW=30000,
    numIterEMD=200000,
    initialization=None,
):
    """Solve balanced multi-channel fused Gromov-Wasserstein."""
    # a bit of logging
    # C1 of shape (ns, ns, d)
    # C2 of shape (nt, nt, d)
    logger.info(f"alpha={alpha}")
    if Q is None:
        channel_dim = 1 if C1.ndim == 2 else C1.shape[-1]
        Q = np.eye(channel_dim)

    G = fused_gromov_wasserstein_multi(
        M,
        C1,
        C2,
        alpha=alpha,
        Q=Q,
        p=a,
        q=b,
        max_iter=numIterFGW,
        numItermaxEmd=numIterEMD,
        G0=initialization,
    )
    lin, gw = contributions_multi(M, C1, C2, G, Q)

    return G, {"linear_cost": lin, "gw_cost": gw}
