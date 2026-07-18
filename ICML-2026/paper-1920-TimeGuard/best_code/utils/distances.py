import numpy as np

def weights_forecast_gaussian(L, F, sigma=2):
    n = L + F
    center = L  # first forecast step
    idx = np.arange(n)
    w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    return w / w.sum() 

def to_euclidean_from_pearson(X, weights=None, eps=1e-12):
    """
    X: (N, L) 
    """
    X = np.asarray(X, dtype=np.float64)
    N, L = X.shape

    if weights is None:
        # Unweighted: per-row z-norm, then scale to unit L2 norm
        mu = X.mean(axis=1, keepdims=True)
        Xc = X - mu
        var = (Xc**2).mean(axis=1, keepdims=True)
        sigma = np.sqrt(var) + eps
        Z = Xc / sigma                      # each row: mean 0, variance 1
        U = Z / np.sqrt(L)                  # each row now has unit norm
        return U

    # Weighted case
    w = np.asarray(weights, dtype=np.float64).reshape(1, L)
    w = w / (w.sum(axis=1, keepdims=True) + eps)
    sqrtw = np.sqrt(w)

    # Weighted mean/var per row
    mu = (X * w).sum(axis=1, keepdims=True)            # /sum(w)=1 already
    Xc = X - mu
    var = (w * (Xc**2)).sum(axis=1, keepdims=True)     # weighted variance
    sigma = np.sqrt(var) + eps

    # Weighted z-norm
    Z = Xc / sigma                                     # each row: w-var = 1

    U = (Z * sqrtw) / np.sqrt(w.sum() + eps)           # 
    return U