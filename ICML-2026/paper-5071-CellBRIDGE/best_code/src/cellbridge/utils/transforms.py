import logging

import numpy as np
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hill_transform_autok(
    X,
    *,
    q: float = 90.0,  # percentile for K_g
    n: float = 1.0,  # Hill coefficient
    ignore_zeros: bool = True,
    min_K: float = 1e-6,
    copy: bool = True,
):
    """Apply per-gene Hill normalization."""
    logger.info("Inside the Hill transform")
    if sp.issparse(X):
        in_fmt = X.getformat()
        Y = X.copy() if copy else X
        Y = Y.tocsc(copy=False)
        indptr, data = Y.indptr, Y.data
        N, G = Y.shape[0], Y.shape[1]

        # 1) compute K_g
        K = np.empty(G, dtype=float)
        for j in range(G):
            lo, hi = indptr[j], indptr[j + 1]
            nnz = hi - lo
            if nnz == 0:
                K[j] = min_K
                continue
            col = data[lo:hi]
            if ignore_zeros:
                Kj = np.percentile(col, q)
            else:
                # include zeros analytically (no padding)
                z = N - nnz
                p = q / 100.0
                zfrac = z / N
                if p <= zfrac:
                    Kj = 0.0
                else:
                    adj_p = (p - zfrac) / (1 - zfrac)
                    Kj = np.percentile(col, adj_p * 100.0)
            if not np.isfinite(Kj) or Kj <= 0:
                Kj = min_K
            K[j] = Kj

        # 2) apply Hill per column in-place
        for j in range(G):
            lo, hi = indptr[j], indptr[j + 1]
            if lo == hi:
                continue
            d = data[lo:hi]
            u = (d / K[j]) ** n
            data[lo:hi] = u / (1.0 + u)

        return Y.asformat(in_fmt)

    # Dense path
    Y = X.copy() if copy else X
    if ignore_zeros:
        K = np.array(
            [
                np.percentile(col[col > 0], q) if np.any(col > 0) else min_K
                for col in Y.T
            ],
            dtype=float,
        )
    else:
        K = np.percentile(Y, q, axis=0)
        K[~np.isfinite(K)] = min_K
        K[K <= 0] = min_K

    U = (Y / K[None, :]) ** n
    return U / (1.0 + U)


class CCI_transform:
    """Apply a sequence of processors to an interaction/affinity matrix."""

    def __init__(self, transforms):
        if transforms is None:
            transforms = []

        self.transforms = list(transforms)

    def add(self, transform):
        """Append a processor to the pipeline.

        Accepts either a callable or a (callable, kwargs) tuple.
        """
        self.transforms.append(transform)
        return self

    def __call__(self, G, *, copy: bool = True):
        """Run the pipeline on G.

        Parameters
        ----------
        G : np.ndarray | sp.spmatrix
            Input matrix to process.
        copy : bool, default True
            If True and G is a dense ndarray, operate on a copy. For sparse
            matrices, copy semantics depend on individual processors.
        """
        if G is None:
            raise ValueError("G must not be None")

        if isinstance(G, np.ndarray) and copy:
            cur = G.copy()
        else:
            # For sparse matrices or when copy=False, pass-through
            cur = G

        if not self.transforms:
            return cur

        for _, transform in enumerate(self.transforms):
            cur = transform(cur)
        return cur
