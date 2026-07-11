import numpy as np
from .utils import estimate_triangular_matrices_Ti


def compute_residuals_with_univariate_OLS(x, y):
    """
    Regress y on x for multiple views and return residuals.
    x, y: arrays of shape (m, n)
    Returns residuals of shape (m, n)
    """
    cov = np.mean(x * y, axis=1)  # shape (m,)
    var_x = np.mean(x * x, axis=1)  # shape (m,)
    beta = cov / var_x
    return y - beta[:, None] * x


def correlation_squared(a, b):
    """
    Compute squared correlation matrix between a and b.
    a, b: arrays of shape (m, n)
    Returns scalar score (Frobenius norm of the correlation matrix)
    """
    cov = a @ b.T / a.shape[1]
    std_a = a.std(axis=1, ddof=1)
    std_b = b.std(axis=1, ddof=1)
    corr = cov / np.outer(std_a, std_b)
    return np.sum(corr ** 2)


def find_direction(x, y):
    """
    Compare x → y and y → x using residual–predictor correlation.
    Lower score indicates lower correlation between residuals and predictor.
    x, y: arrays of shape (m, n)
    Returns both scores and both residuals of shape (m, n)
    """
    r_y_on_x = compute_residuals_with_univariate_OLS(x, y)
    r_x_on_y = compute_residuals_with_univariate_OLS(y, x)
    score_x_to_y = correlation_squared(x, r_y_on_x)
    score_y_to_x = correlation_squared(y, r_x_on_y)
    return score_x_to_y, score_y_to_x, r_y_on_x, r_x_on_y


def find_parent_variable(X, standardize=True):
    """
    Identify the root variable (with no parents) and get the residuals
    when regressing all the other variables on the root variable.
    X: array of shape (m, p, n) for m views, p variables, n samples
    Returns index of root variable and residuals of shape (m, p-1, n)
    """
    m, p, n = X.shape
    Xc = X - X.mean(axis=-1, keepdims=True)
    if standardize:
        Xz = Xc / Xc.std(axis=-1, keepdims=True)
    else:
        Xz = Xc

    scores = np.zeros((p, p))
    R = np.zeros((p, p, m, n))

    for i in range(p):
        for j in range(i + 1, p):
            score_ij, score_ji, r_j_on_i, r_i_on_j = find_direction(
                Xz[:, i], Xz[:, j])
            scores[i, j] = score_ji - score_ij
            scores[j, i] = score_ij - score_ji
            R[i, j] = r_j_on_i
            R[j, i] = r_i_on_j

    parent_id = np.argmin(np.sum(np.minimum(0, scores) ** 2, axis=1))
    r_all_on_parent = R[parent_id].swapaxes(0, 1)  # shape (m, p, n)
    r_all_on_parent = np.delete(r_all_on_parent, parent_id, axis=1)  # shape (m, p-1, n)
    
    return parent_id, r_all_on_parent


def estimate_causal_order(X, standardize=True):
    """
    Identify the entire ordering by estimating the root variable, 
    removing its effect on the other variables, and iterating this procedure.
    X: array of shape (m, p, n) for m views, p variables, n samples
    Returns entire ordering
    """
    m, p, n = X.shape
    X_current = X.copy()
    
    order = []
    remaining_indices = np.arange(p)
    while len(order) < p - 1:
        parent, X_current = find_parent_variable(X_current, standardize=standardize)
        order.append(remaining_indices[parent])
        remaining_indices = np.delete(remaining_indices, parent)
    order.append(remaining_indices[0])

    return order


def direct_limvam(X, standardize=True):
    """
    Assume the model xi = Bi xi + ei, where Bi are DAG matrices that share the 
    same causal ordering, and the disturbances ei are correlated across views.
    DirectLiMVAM identifies the entire causal ordering, and then estimates causal 
    weights using one-step Feasible Generalized Least Squares.
    
    Parameters
    ----------
    X: array of shape (m, p, n) for m views, p variables, n samples
    
    standardize: bool, optional (default=True)
        If True, observations are standardized to unit variance.
        If False, observations are only centered to zero mean.
    
    Returns
    -------
    B: DAG matrices (ndarray of shape (m, p, p))
    T: Strictly lower triangular matrices (ndarray of shape (m, p, p))
    P: Permutation matrix that contains the ordering (ndarray of shape (p, p))
    """
    # estimate the causal ordering using the cross-covariance-based criterion
    order = estimate_causal_order(X, standardize=standardize)
    
    # estimate causal weights with one-step Feasible Generalized Least Squares
    X_ordered = X[:, order]
    T = estimate_triangular_matrices_Ti(X_ordered)
    
    # reconstruct adjacency matrices
    P = np.eye(X.shape[1])[order]
    B = P.T @ T @ P
    
    return B, T, P
