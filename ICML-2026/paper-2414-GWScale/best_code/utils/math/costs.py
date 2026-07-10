"""
Base Cost Functions

Common cost and distance functions for optimal transport and point cloud matching.
It should broadcast correctly over the two first dimensions: if X is of shape (N, 1, D) and Y is of shape (1, M, D), cost(X,Y) should be of shape (N, M).
"""

def euclidean(X, Y, p=2, tol=1e-9):
    """
    Compute Euclidean distance with numerical stability.
    
    Computes ||X - Y||p. Add tolerance before taking the root to avoid numerical issues when using autograd.
    
    Args:
        X: First point or array
        Y: Second point or array
        p: Power parameter
        tol: Small constant for numerical stability
    
    Returns:
        Distance value or array of distances
    """
    return (((X - Y) ** 2).sum(axis=-1) + tol) ** (p / 2) # Add tolerance to avoid differentiation issues


def gaussian(X, Y, sigma=1, p=2, tol=1e-9):
    """
    Compute Gaussian kernel-based distance.
    
    Converts Gaussian kernel similarity to a distance metric:
    d(X,Y) = 2 - 2*exp(-||X-Y||^p / (p*sigma^p))
    
    Args:
        X: First point or array
        Y: Second point or array
        sigma: Bandwidth parameter for Gaussian kernel
        p: Power parameter for distance computation
        tol: Small constant for numerical stability
    
    Returns:
        Kernel-based distance value or array
    """
    kernel = lambda u, v: (- euclidean(X, Y, p=p, tol=tol) / (p * sigma ** p)).exp()
    return 2 - 2 * kernel(X, Y)
