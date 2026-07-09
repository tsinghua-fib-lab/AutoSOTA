import numpy as np

def stabilize_L(
    L,
    min_eigenvalue=1e-6,
    max_eigenvalue=1e6
):
    """
    Stabilizes a lower triangular matrix L while preserving positive definiteness.
    It clips the eigenvalues of L @ L.T to ensure they lie in [min_eigenvalue, max_eigenvalue].

    Parameters:
    -----------
    L : (n x n) np.ndarray
        Lower triangular matrix to stabilize.
    min_eigenvalue : float, optional
        Minimum allowed eigenvalue to ensure positive definiteness.
    max_eigenvalue : float or None, optional
        Maximum allowed eigenvalue to prevent blowing up the matrix norm. If None, no upper clipping is performed.

    Returns:
    --------
    L_stabilized : (n x n) np.ndarray
        Stabilized lower triangular matrix (Cholesky factor).
    """
    # Ensure L is lower triangular
    L = np.tril(L)

    # Compute Gram matrix (symmetric, at least PSD if L was real)
    gram_matrix = L @ L.T

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # Clip the eigenvalues from below (and from above if max_eigenvalue is provided)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    if max_eigenvalue is not None:
        eigenvalues = np.minimum(eigenvalues, max_eigenvalue)

    # Rebuild the Gram matrix
    stabilized_gram = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Reconstruct L from the stabilized Gram matrix
    L_stabilized = np.linalg.cholesky(stabilized_gram)

    # (Optional) ensure strictly lower-triangular by zeroing any small floating-point noise above diagonal
    L_stabilized = np.tril(L_stabilized)

    return L_stabilized
