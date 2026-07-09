import logging
import sys

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def prepare_grid(
    cfg: DictConfig,
    V: np.ndarray,
    n_grid: int = 50,
    proposal: str = "unconstrained",
) -> np.ndarray:
    """Sort and coarsen grid of lik-ratio values to search over.

    Args:
        cfg: Hydra configuration object.
        V: 1-D array of likelihood-ratio values (unsorted).
        n_grid: Approximate number of values in the resulting grid.
        proposal: Proposal distribution type — ``"unconstrained"``,
            ``"safe"``, or ``"mixed"``.

    Returns:
        Sorted, coarsened grid with appropriate boundary values appended.
    """

    G = np.sort(
        np.unique(V)
    )  ## Want to search in increasing order for CPC (safest to most aggressive)

    ## Coarsen grid to approximately n_grid elements
    n_curr = len(G)
    k = max(int(n_curr / n_grid), 1)
    G = G[::k]

    if proposal == "unconstrained":
        ## For unconstrained, ensure also consider unconstrained policy in grid (np.inf)
        G = np.concatenate((G, [np.inf]))

    elif proposal == "safe":
        ## For safe, ensure that include minimum positive float value
        G = np.concatenate(([sys.float_info.min], G))

    elif proposal == "mixed":
        ## For mixed, ensure that include minimum positive float value as well as np.inf
        G = np.concatenate(([sys.float_info.min], G, [np.inf]))

    else:
        raise ValueError(f"unrecognized proposal name : {proposal}")

    return G
